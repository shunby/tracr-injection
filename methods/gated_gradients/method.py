from .compute_mean_activations import compute_mean_activations

def gated_gradients(model, tokenizer, data):
    mean_activations = compute_mean_activations(model, tokenizer, data)


    # do training, gating by features


    return (model, None)

