import torch
import torch.nn as nn
import torch.nn.functional as F
from compile_model.utils import convert_to_torch

class Transformer(nn.Module):
    def __init__(
        self,
        input_dim,
        model_dim,
        num_heads,
        num_layers,
        ff_dim,
        seq_len,
        num_classes,
        key_size,
        dim_sizes,
        decoder,
        residual_stream_labels,
    ):
        super(Transformer, self).__init__()
        self.model_dim = model_dim
        self.embedding = nn.Embedding(input_dim, model_dim)
        self.positional_encoding = nn.Embedding(seq_len, model_dim)
        self.layers = nn.ModuleList([
            TransformerDecoderLayer(model_dim, num_heads, ff_dim, key_size) for _ in range(num_layers)
        ])
        self.residual_stream = None
        self.clean_residual_stream = None
        self.current_layer = None
        self.residual_stream_residue = 0
        self.dim_sizes = dim_sizes
        self.decoder = decoder
        self.residual_stream_labels = residual_stream_labels

    def forward(self, x):
        positions = torch.arange(x.size(1), device=x.device).unsqueeze(0).expand(x.size(0), -1)
        x = self.embedding(x) + self.positional_encoding(positions)
        for i, layer in enumerate(self.layers):
            x = layer(x)

        return x

    def embed_tokens(self, x):
        positions = torch.arange(x.size(1), device=x.device).unsqueeze(0).expand(x.size(0), -1)
        self.residual_stream = self.embedding(x) + self.positional_encoding(positions)
        self.current_layer = 0
        self.residual_stream_residue = 0
        return self.residual_stream

    def embed_ground_truth(self, x):
        positions = torch.arange(x.size(1), device=x.device).unsqueeze(0).expand(x.size(0), -1)
        self.clean_residual_stream = self.embedding(x) + self.positional_encoding(positions)

    def forward_one_layer(self):
        if self.current_layer >= len(self.layers):
            return self.residual_stream
        self.residual_stream = self.layers[self.current_layer](self.residual_stream)
        self.current_layer += 1
        return self.residual_stream

    def add_residue(self):
        diff = self.clean_residual_stream[:, -1] - self.residual_stream[:, -1]

        start_idx = 0
        for size in self.dim_sizes:
            end_idx = start_idx + size
            if size < 50:
                chunk_diff = diff[:, start_idx:end_idx]
                self.residual_stream_residue += torch.norm(chunk_diff)

                start_idx = end_idx

    def normalize_residual_stream(self):
        chunks_clean = torch.split(self.clean_residual_stream, self.dim_sizes, dim=-1)
        chunks_resid = torch.split(self.residual_stream, self.dim_sizes, dim=-1)

        updated_chunks = []
        for chunk_clean, chunk_resid in zip(chunks_clean, chunks_resid):
            clean_sum = chunk_clean.sum(dim=-1, keepdim=True)
            resid_sum = chunk_resid.sum(dim=-1, keepdim=True)

            scale = clean_sum / (resid_sum + 1e-8)
            scale = torch.where(clean_sum == 0, torch.ones_like(scale), scale)

            chunk_resid = chunk_resid * scale
            updated_chunks.append(chunk_resid)

        self.residual_stream = torch.cat(updated_chunks, dim=-1)

    def final_result_dimensions(self):
        indices = []
        for i, label in enumerate(self.residual_stream_labels):
            if 'final_result' in label:
                indices.append(i)

        return indices

    def set_weights(self, compiled_model_params):
        self.embedding.weight.data.copy_(convert_to_torch(compiled_model_params['token_embed']['embeddings']))
        self.positional_encoding.weight.data.copy_(convert_to_torch(compiled_model_params['pos_embed']['embeddings']))

        for i, layer in enumerate(self.layers):
            layer.q_proj.weight.data.copy_(convert_to_torch(compiled_model_params[f"transformer/layer_{i}/attn/query"]['w']).T)
            layer.k_proj.weight.data.copy_(convert_to_torch(compiled_model_params[f"transformer/layer_{i}/attn/key"]['w']).T)
            layer.v_proj.weight.data.copy_(convert_to_torch(compiled_model_params[f"transformer/layer_{i}/attn/value"]['w']).T)
            layer.o_proj.weight.data.copy_(convert_to_torch(compiled_model_params[f"transformer/layer_{i}/attn/linear"]['w']).T)
            layer.linear_1.weight.data.copy_(convert_to_torch(compiled_model_params[f"transformer/layer_{i}/mlp/linear_1"]['w']).T)
            layer.linear_2.weight.data.copy_(convert_to_torch(compiled_model_params[f"transformer/layer_{i}/mlp/linear_2"]['w']).T)

            layer.q_proj.bias.data.copy_(convert_to_torch(compiled_model_params[f"transformer/layer_{i}/attn/query"]['b']))
            layer.k_proj.bias.data.copy_(convert_to_torch(compiled_model_params[f"transformer/layer_{i}/attn/key"]['b']))
            layer.v_proj.bias.data.copy_(convert_to_torch(compiled_model_params[f"transformer/layer_{i}/attn/value"]['b']))
            layer.o_proj.bias.data.copy_(convert_to_torch(compiled_model_params[f"transformer/layer_{i}/attn/linear"]['b']))
            layer.linear_1.bias.data.copy_(convert_to_torch(compiled_model_params[f"transformer/layer_{i}/mlp/linear_1"]['b']))
            layer.linear_2.bias.data.copy_(convert_to_torch(compiled_model_params[f"transformer/layer_{i}/mlp/linear_2"]['b']))

class TransformerDecoderLayer(nn.Module):
    def __init__(self, model_dim, num_heads, ff_dim, key_size):
        super(TransformerDecoderLayer, self).__init__()
        self.num_heads = num_heads
        self.key_size = key_size
        self.model_dim = model_dim

        self.q_proj = nn.Linear(model_dim, num_heads * key_size)
        self.k_proj = nn.Linear(model_dim, num_heads * key_size)
        self.v_proj = nn.Linear(model_dim, num_heads * key_size)
        self.o_proj = nn.Linear(num_heads * key_size, model_dim)

        self.linear_1 = nn.Linear(model_dim, ff_dim)
        self.linear_2 = nn.Linear(ff_dim, model_dim)
        self.ff = nn.Sequential(self.linear_1, nn.ReLU(), self.linear_2)

    def forward(self, x):
        batch_size, seq_len, _ = x.size()

        q = self.q_proj(x).view(batch_size, seq_len, self.num_heads, self.key_size).transpose(1, 2)
        k = self.k_proj(x).view(batch_size, seq_len, self.num_heads, self.key_size).transpose(1, 2)
        v = self.v_proj(x).view(batch_size, seq_len, self.num_heads, self.key_size).transpose(1, 2)

        attn_weights = torch.matmul(q, k.transpose(-2, -1)) / torch.sqrt(torch.tensor(self.key_size, dtype=torch.float32))

        causal_mask = torch.triu(torch.ones(seq_len, seq_len, device=x.device), diagonal=1).bool()
        attn_weights = attn_weights.masked_fill(causal_mask, float('-inf'))

        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_output = torch.matmul(attn_weights, v)

        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)
        attn_output = self.o_proj(attn_output)

        x = x + attn_output
        x = x + self.ff(x)
        return x



