import torch
import torch.nn as nn

class GatedGradientsTransformer(nn.Module):
    def __init__(self, original_block):
        super().__init__()
        self.original_block = original_block
        self.post_attn_activations = None
        self.post_mlp_activations = None

    def forward(self, hidden_states, **kwargs):
        residual = hidden_states

        # Layer Norm 1
        hidden_states = self.original_block.ln_1(hidden_states)

        # Self-Attention
        attn_outputs = self.original_block.attn(hidden_states, **kwargs)
        outputs = attn_outputs[1:]
        attn_output = attn_outputs[0]
        hidden_states = residual + attn_output
        self._capture_activation(hidden_states, 'attn')

        # Layer Norm 2
        residual = hidden_states
        hidden_states = self.original_block.ln_2(hidden_states)

        # MLP
        feed_forward_hidden_states = self.original_block.mlp(hidden_states)
        hidden_states = residual + feed_forward_hidden_states
        self._capture_activation(hidden_states, 'mlp')

        outputs = (hidden_states,) + outputs[1:]
        return outputs

    def _capture_activation(self, activation, module):
        activation_mean = activation.detach().mean(dim=(0))  # Shape: (token, hidden_size)

        if module == 'attn':
            if self.post_attn_activations is None:
                self.post_attn_activations = activation_mean.clone()
            else:
                self.post_attn_activations += activation_mean
        elif module == 'mlp':
            if self.post_mlp_activations is None:
                self.post_mlp_activations = activation_mean.clone()
            else:
                self.post_mlp_activations += activation_mean
        else:
            raise ValueError(f"Invalid module: {module}")

    def rescale_activations(self, n):
        self.post_mlp_activations /= n
        self.post_attn_activations /= n
