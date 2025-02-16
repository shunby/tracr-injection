import torch
import torch.nn as nn
import torch.nn.functional as F

class CrossAttentionModule(nn.Module):
    def __init__(self, read_dim, write_dim, num_heads, key_size):
        super(CrossAttentionModule, self).__init__()
        self.num_heads = num_heads
        self.key_size = key_size
        self.q_proj = nn.Linear(write_dim, num_heads * key_size)
        self.k_proj = nn.Linear(read_dim, num_heads * key_size)
        self.v_proj = nn.Linear(read_dim, num_heads * key_size)
        self.o_proj = nn.Linear(num_heads * key_size, write_dim)
        nn.init.constant_(self.o_proj.weight, 0)
        nn.init.constant_(self.o_proj.bias, 0)

    def forward(self, read_stream, write_stream):
        batch_size, seq_len, _ = read_stream.size()

        q = self.q_proj(write_stream).view(batch_size, seq_len, self.num_heads, self.key_size).transpose(1, 2)
        k = self.k_proj(read_stream).view(batch_size, seq_len, self.num_heads, self.key_size).transpose(1, 2)
        v = self.v_proj(read_stream).view(batch_size, seq_len, self.num_heads, self.key_size).transpose(1, 2)

        scale = torch.sqrt(torch.tensor(self.key_size, dtype=torch.float32))
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) / scale

        causal_mask = torch.triu(torch.ones(seq_len, seq_len, device=read_stream.device), diagonal=1).bool()
        attn_weights = attn_weights.masked_fill(causal_mask, float('-inf'))

        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_output = torch.matmul(attn_weights, v)

        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)
        attn_output = self.o_proj(attn_output)
        return attn_output

