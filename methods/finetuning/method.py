import copy

from lightning_trainer.train import train

def finetuning(
    model,
    tokenizer,
    data,
    unsupervised_data,
    natural_data,
    params=None,
    **kwargs,
):
    model_copy = copy.deepcopy(model)
    train(
        model,
        tokenizer,
        data,
        unsupervised_data,
        natural_data,
        original_model=model_copy,
        params=params
    )
    return model
