import torch
from torch.utils.data import DataLoader
from .gated_gradients_transformer import GatedGradientsTransformer

def compute_mean_activations(model, tokenizer, data, batch_size=8):
    modify_transformer(model)
    model.eval()
    device = next(model.parameters()).device

    dataloader = DataLoader(
        data,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=lambda x: tokenizer(x, return_tensors='pt', padding=True, truncation=True)
    )
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)

    rescale_activations(model, len(data))



def modify_transformer(model):
    for i, block in enumerate(model.transformer.h):
        wrapped_block = GatedGradientsTransformer(block)
        model.transformer.h[i] = wrapped_block

def rescale_activations(model, n):
    for block in model.transformer.h:
        block.rescale_activations(n)

