import torch

def get_detach_and_roll_hook(distiler):
    def detach_hook(module, input, output, distiler=distiler):
        hidden_states, *args = output
        projected = hidden_states @ distiler.projection
        non_projected = (hidden_states - projected).detach()
        rolled = torch.roll(non_projected, shifts=1, dims=0)
        hidden_states = projected + rolled
        return (hidden_states, *args)
    return detach_hook
