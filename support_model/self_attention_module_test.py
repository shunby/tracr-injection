import pytest
import torch
from .self_attention_module import SelfAttentionModule


def test_zero_effect_at_init():
    residual_stream_dim = 10
    num_heads = 3
    key_size = 7

    module = SelfAttentionModule(
        residual_stream_dim=residual_stream_dim,
        num_heads=num_heads,
        key_size=key_size
    )

    batches = 2
    sequence_len = 5
    input_tensor = torch.ones((batches, sequence_len, residual_stream_dim))
    output_tensor = module(input_tensor)

    assert(torch.allclose(input_tensor, output_tensor))
