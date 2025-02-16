import pytest
import torch
from .cross_attention_module import CrossAttentionModule


def test_zero_effect_at_init():
    read_dim = 10
    write_dim = 9
    num_heads = 3
    key_size = 7

    module = CrossAttentionModule(
        read_dim=read_dim,
        write_dim=write_dim,
        num_heads=num_heads,
        key_size=key_size
    )

    batches = 2
    sequence_len = 5
    read_stream = torch.ones((batches, sequence_len, read_dim))
    write_stream = torch.ones((batches, sequence_len, write_dim))
    output_tensor = module(read_stream, write_stream)

    assert(torch.allclose(torch.zeros_like(output_tensor), output_tensor))
