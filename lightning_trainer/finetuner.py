import lightning as L
import torch
import torch.nn.functional as F

class Finetuner(L.LightningModule):
    def __init__(
        self,
        model,
        original_model,
        tokenizer=None,
        lr=1e-6,
        gradient_clip_val=100.0,
        hyperparams_dict={},
        natural_data_loss=False,
        residue_loss=False,
        compiled_model_loss=False,
        compiled_model=None,
        translator=None,
        adapter=None,
    ):
        super().__init__()

        self.hyperparams_dict = hyperparams_dict
        self.save_hyperparameters(ignore=['original_model', 'model', 'tokenizer', 'adapter', 'compiled_model'])

        self.model = model
        self.original_model = original_model
        self.lr = lr
        self.gradient_clip_val = gradient_clip_val
        self.tokenizer = tokenizer
        self.natural_data_loss = natural_data_loss
        self.residue_loss = residue_loss
        self.compiled_model_loss = compiled_model_loss
        self.compiled_model = compiled_model
        self.translator = translator
        self.adapter = adapter

    def supervised_loss(self, batch):
        input_ids = batch["input_ids"].to(self.model.device)
        loss_mask = batch["loss_mask"].to(self.model.device)

        outputs = self.model(input_ids=input_ids, output_hidden_states=self.compiled_model_loss)
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

        if self.compiled_model_loss:
            return loss, self.algorithm_alignment_loss(batch, outputs)

        return loss, 0

    def natural_loss(self, batch):
        input_ids = batch["input_ids"].to(self.model.device)
        loss_mask = batch["loss_mask"].to(self.model.device)

        outputs = self.model(input_ids=input_ids, output_hidden_states=self.compiled_model_loss)
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

        if self.compiled_model_loss:
            return loss, self.algorithm_alignment_loss(batch, outputs)

        return loss, 0

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

        del input_ids, outputs, logits, probs, outputs_original, logits_original, probs_original
        torch.cuda.empty_cache()

        return kl_loss

    def training_step(self, batch, batch_idx):
        supervised_batch, unsupervised_batch, natural_data_batch = batch

        supervised_loss, supervised_alignment = self.supervised_loss(supervised_batch)
        self.log("supervised_loss", supervised_loss, on_step=True, on_epoch=False, prog_bar=True)

        unsupervised_loss = self.unsupervised_loss(unsupervised_batch)
        self.log(
            "unsupervised_loss",
            unsupervised_loss,
            on_step=True,
            on_epoch=False,
            prog_bar=True
        )

        natural_data_loss = 0
        if self.natural_data_loss:
            natural_data_loss, natural_alignment = self.natural_loss(natural_data_batch)
            self.log(
                "natural_data_loss",
                natural_data_loss,
                on_step=True,
                on_epoch=False,
                prog_bar=True
            )

        if self.compiled_model_loss:
            self.log(
                "natural_alignment_loss",
                natural_alignment,
                on_step=True,
                on_epoch=False,
                prog_bar=True
            )
            self.log(
                "supervised_alignment_loss",
                supervised_alignment,
                on_step=True,
                on_epoch=False,
                prog_bar=True
            )

        total_loss = (
            supervised_loss + unsupervised_loss + natural_data_loss +
            supervised_alignment + natural_alignment
        )

        self.log("loss", total_loss, on_step=True, on_epoch=False, prog_bar=True)
        return total_loss

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        acc = self.validation_accuracy(batch)
        if dataloader_idx == 0:
            self.log("val_acc", acc, on_step=False, on_epoch=True, prog_bar=True)
        else:
            self.log("natural_acc", acc, on_step=False, on_epoch=True, prog_bar=True)

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

    def algorithm_alignment_loss(self, batch, outputs):
        translated_tokens = self.translator(batch['input_ids'])
        self.compiled_model.embed_tokens(translated_tokens)

        loss = 0
        for i in range(len(self.compiled_model.layers) + 1):
            adaptation = self.adapter(outputs.hidden_states[i])

            tensor1 = F.normalize(adaptation, p=2, dim=-1)
            tensor2 = F.normalize(self.compiled_model.residual_stream[:, 1:], p=2, dim=-1)
            cosine_similarity = torch.sum(tensor1 * tensor2, dim=-1)
            cosine_distance = 1 - cosine_similarity
            loss += cosine_distance.mean()
            self.compiled_model.forward_one_layer()
        return loss

    def configure_optimizers(self):
        if self.compiled_model_loss:
            return torch.optim.AdamW(
                [
                    {"params": self.model.parameters(), "lr": self.lr},
                    {"params": self.adapter.parameters(), "lr": self.lr},
                ]
            )
        return torch.optim.AdamW(self.model.parameters(), lr=self.lr)
