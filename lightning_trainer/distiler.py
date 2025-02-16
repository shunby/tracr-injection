import lightning as L
import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.detach_and_roll import get_detach_and_roll_hook

class Distiler(L.LightningModule):
    def __init__(
        self,
        model,
        compiled_model,
        original_model,
        translator,
        natural_data_loss,
        detach_and_roll=True,
        algorithm_loss=True,
        unsupervised_loss=True,
        gate_dims_to_final_result=True,
        pad_communication=0,
        algorithm_loss_multiplier=1,
    ):
        super().__init__()

        self.model = model
        self.compiled_model = compiled_model
        self.compiled_model.eval()
        self.original_model = original_model
        self.tokenizer_translator = translator
        self.learnable_layers = len(compiled_model.layers)
        self.adapter = nn.Linear(model.config.hidden_size, compiled_model.model_dim)
        self.lr=1e-6
        self.adapter_lr=1e-4
        self.adapter.to(self.model.device)
        self.natural_data_loss = natural_data_loss
        self.algorithm_loss_enabled = algorithm_loss
        self.unsupervised_loss_enabled = unsupervised_loss
        self.only_result_subspace = False
        self.gate_dims_to_final_result = gate_dims_to_final_result
        self.pad_communication = pad_communication
        self.algorithm_loss_multiplier = algorithm_loss_multiplier
        if detach_and_roll and algorithm_loss:
            self.supervised_hook = get_detach_and_roll_hook(self)
        else:
            self.supervised_hook = lambda module, input, output: output
        self.get_projection_matrix()


    def training_step(self, batch, batch_idx):
        self.get_projection_matrix()
        supervised_batch, unsupervised_batch = batch

        if self.unsupervised_loss_enabled:
            unsupervised_loss = self.unsupervised_loss(unsupervised_batch)
            self.log("unsupervised_loss", unsupervised_loss, on_step=True, on_epoch=False, prog_bar=True)
        else:
            unsupervised_loss = 0

        supervised_loss, algorithm_loss = self.supervised_loss(supervised_batch)
        self.log("supervised_loss", supervised_loss, on_step=True, on_epoch=False, prog_bar=True)
        if self.algorithm_loss_enabled:
            self.log("algorithm_loss", algorithm_loss, on_step=True, on_epoch=False, prog_bar=True)

        loss = algorithm_loss + unsupervised_loss + supervised_loss
        return loss

    def unsupervised_loss(self, batch):
        input_ids = batch["input_ids"].to(self.model.device)
        outputs = self.model(input_ids=input_ids)
        logits = outputs.logits
        probs = F.log_softmax(logits, dim=-1)

        with torch.no_grad():
            outputs_original = self.original_model(input_ids=input_ids)
            logits_original = outputs_original.logits
            probs_original = F.softmax(logits_original, dim=-1)

        kl_loss = F.kl_div(probs, probs_original, reduction='batchmean', log_target=False)

        return kl_loss

    def supervised_loss(self, batch):
        input_ids = batch["input_ids"].to(self.model.device)
        loss_mask = batch["loss_mask"].to(self.model.device)

        hook_handle = self.model.transformer.h[self.learnable_layers].register_forward_hook(self.supervised_hook)
        outputs = self.model(input_ids=input_ids, output_hidden_states=True)
        hook_handle.remove()

        logits = outputs.logits

        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = input_ids[:, 1:].contiguous()
        shift_loss_mask = loss_mask[:, :-1].contiguous()

        shift_logits = shift_logits.view(-1, shift_logits.size(-1))
        shift_labels = shift_labels.view(-1)
        shift_loss_mask = shift_loss_mask.view(-1)

        loss = F.cross_entropy(shift_logits, shift_labels, reduction='none')
        loss = loss * shift_loss_mask
        loss = loss.sum() / shift_loss_mask.sum()

        if self.algorithm_loss_enabled:
            compiled_model_loss = self.algorithm_loss(outputs, input_ids) * self.algorithm_loss_multiplier
        else:
            compiled_model_loss = 0

        return loss, compiled_model_loss

    def algorithm_loss(self, outputs, input_ids):
        translated_tokens = self.tokenizer_translator(input_ids)
        self.compiled_model.embed_tokens(translated_tokens)

        total_loss = 0.0
        for i in range(min(self.learnable_layers + 1, len(outputs.hidden_states))):

            adaptation = self.adapter(outputs.hidden_states[i + self.pad_communication])
            # breakpoint()
            tensor1 = F.normalize(adaptation, p=2, dim=-1)
            tensor2 = F.normalize(self.compiled_model.residual_stream[:, 1:], p=2, dim=-1)
            loss_i = (1 - torch.sum(tensor1 * tensor2, dim=-1)).mean()
            total_loss += loss_i
            self.compiled_model.forward_one_layer()

        return total_loss

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        if dataloader_idx == 0:
            acc = self.validation_accuracy(batch)
            self.log("val_acc", acc, on_step=False, on_epoch=True, prog_bar=True)

    def validation_accuracy(self, batch):
        input_ids = batch["input_ids"].to(self.model.device)
        loss_mask = batch["loss_mask"].to(self.model.device)

        outputs = self.model(input_ids=input_ids)
        logits = outputs.logits

        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = input_ids[:, 1:].contiguous()
        shift_loss_mask = loss_mask[:, :-1].contiguous()

        predictions = shift_logits.argmax(dim=-1)

        predictions = predictions.view(-1)
        shift_labels = shift_labels.view(-1)
        shift_loss_mask = shift_loss_mask.view(-1)

        correct = (predictions == shift_labels) & (shift_loss_mask == 1)
        accuracy = correct.sum().float() / shift_loss_mask.sum()

        del input_ids, loss_mask, outputs, logits, shift_logits
        del shift_labels, shift_loss_mask, predictions
        torch.cuda.empty_cache()

        return accuracy.item()


    def get_projection_matrix(self):
        """
        Returns the projection matrix P onto the subspace spanned by
        the columns of self.adapter.weight^T.
        """
        with torch.no_grad():
            if self.gate_dims_to_final_result:
                dims = self.compiled_model.final_result_dimensions()
                W = self.adapter.weight[dims, :]
            else:
                W = self.adapter.weight
            M = W.t()
            M_pinv = torch.linalg.pinv(M)
            P = M @ M_pinv
            self.projection = P.detach()

    def configure_optimizers(self):
        return torch.optim.AdamW(
            [
                {"params": self.model.parameters(), "lr": self.lr},
                {"params": self.adapter.parameters(), "lr": self.adapter_lr},
            ]
        )
