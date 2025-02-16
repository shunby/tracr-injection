import torch
import os
import pickle
import uuid

from support_model.self_attention_module import SelfAttentionModule
from support_model.cross_attention_module import CrossAttentionModule
from dotenv import load_dotenv

load_dotenv()
CACHE_DIR = os.path.expanduser(os.getenv('CACHE_DIR'))

class AddSupportModel:
    def __init__(
        self,
        main_model,
        support_model,
        tokenization_translator,
        decoder,
        communicate_every_x_layers=3,
        pad_communication=0,
        attention_heads=8,
        key_size=60,
        read_from_support=True,
        write_to_support=False,
        tanh_in_write=False,
        rescaling_factor_write=1,
    ):
        self.main_model = main_model
        self.support_model = support_model
        self.tokenization_translator = tokenization_translator
        self.decoder = decoder
        n_cross_attns = len(self.layers()) - pad_communication
        self.read_layers = [
            CrossAttentionModule(support_model.model_dim, main_model.config.hidden_size, attention_heads, key_size)
            for _ in range(n_cross_attns)
        ]
        self.write_layers = [
            CrossAttentionModule(main_model.config.hidden_size, support_model.model_dim, attention_heads, key_size)
            for _ in range(pad_communication)
        ]

        for i, layer in enumerate(self.layers()):
            original_forward = layer.forward

            if i < pad_communication:
                def write_forward(
                    *args,
                    original_forward=original_forward,
                    support_model=support_model,
                    attn=self.write_layers[i],
                    last_layer=i==pad_communication-1,
                    **kwargs
                ):
                    output = original_forward(*args, **kwargs)
                    tensor_output, *rest = output
                    support_output = support_model.residual_stream
                    support_output = support_output.to(tensor_output.dtype)

                    bos_support_output, support_output = torch.split(
                        support_output, [1, support_output.size(-2) - 1], dim=-2
                    )

                    write_output = attn(tensor_output, support_output)
                    write_output /= rescaling_factor_write
                    if tanh_in_write:
                        write_output = torch.tanh(write_output)
                    support_model.residual_stream[:, 1:, :] += write_output
                    if last_layer:
                        support_model.add_residue()

                    return output
                layer.forward = write_forward
                continue

            cross_attn_number = i - pad_communication
            if cross_attn_number >= n_cross_attns:
                continue


            def read_forward(
                *args,
                original_forward=original_forward,
                support_model=support_model,
                read_attn=self.read_layers[cross_attn_number],
                **kwargs
            ):
                output = original_forward(*args, **kwargs)
                for _ in range(communicate_every_x_layers):
                    support_output = support_model.forward_one_layer()
                    if not read_from_support:
                        support_output = torch.zeros_like(support_output)

                tensor_output, *rest = output
                support_output = support_output.to(tensor_output.dtype)

                bos_support_output, support_output = torch.split(
                    support_output, [1, support_output.size(-2) - 1], dim=-2
                )

                read_output = read_attn(support_output, tensor_output)
                main_model_output = tensor_output + read_output

                return main_model_output, *rest

            layer.forward = read_forward

    def __getattr__(self, name):
        return getattr(self.main_model, name)

    def __call__(self, *args, **kwargs):
        translated_tokens = self.tokenization_translator(kwargs['input_ids'])
        self.support_model.embed_tokens(translated_tokens)
        if 'ground_truth' in kwargs:
            translated_tokens = self.tokenization_translator(kwargs['ground_truth'])
            self.support_model.embed_ground_truth(translated_tokens)
            del kwargs['ground_truth']
        output = self.main_model(*args, **kwargs)
        return output

    def train_only_cross_attention(self):
        for param in self.main_model.parameters():
            param.requires_grad = False
        for param in self.support_model.parameters():
            param.requires_grad = False

    def trainable_parameters(self):
        read_params = [p for layer in self.read_layers for p in layer.parameters()]
        write_params = [p for layer in self.write_layers for p in layer.parameters()]
        return read_params + write_params

    def to(self, device):
        self.main_model.to(device)
        self.support_model.to(device)
        for layer in self.read_layers + self.write_layers:
            layer.to(device)

    def layers(self):
        model_name = type(self.main_model).__name__
        if model_name == 'GPT2LMHeadModel':
            return self.main_model.transformer.h
        else:
            raise 'Unrecognized model name'

    def generate(self, input_ids, max_length=50, do_sample=False):
        device = input_ids.device
        batch_size = input_ids.size(0)
        generated = input_ids

        for _ in range(max_length - input_ids.size(1)):
            output = self(input_ids=generated)
            logits = output.logits[:, -1, :]

            if do_sample:
                probabilities = torch.nn.functional.softmax(logits, dim=-1)
                next_token = torch.multinomial(probabilities, num_samples=1)
            else:
                next_token = torch.argmax(logits, dim=-1, keepdim=True)

            generated = torch.cat((generated, next_token), dim=1)

            if (next_token == self.main_model.config.eos_token_id).all():
                break

        return generated

    def save(self):
        unique_id = str(uuid.uuid4())
        save_path = os.path.join(CACHE_DIR, f"cross_attention_{unique_id}.pkl")
        with open(save_path, "wb") as f:
            pickle.dump({'read': self.read_layers, 'write': self.write_layers}, f)
        return unique_id

    def load(self, unique_id):
        load_path = os.path.join(CACHE_DIR, f"cross_attention_{unique_id}.pkl")
        if not os.path.exists(load_path):
            raise FileNotFoundError(f"No saved cross-attention layers found with ID {unique_id}")
        with open(load_path, "rb") as f:
            saved_states = pickle.load(f)
        for layer, saved_layer in zip(self.read_layers, saved_states['read']):
            layer.load_state_dict(saved_layer.state_dict())
        for layer, saved_layer in zip(self.write_layers, saved_states['write']):
            layer.load_state_dict(saved_layer.state_dict())


