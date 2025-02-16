import torch

def convert_to_torch(jax_array):
    return torch.tensor(jax_array.tolist(), dtype=torch.float32)
