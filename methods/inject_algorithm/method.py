from compile_model.load_compiled_model import load_model
from support_model.create_tokenizer_translator import create_translator
from support_model.add_support_model import AddSupportModel
from lightning_trainer.train import train
import copy

def inject_algorithm(
    model,
    tokenizer,
    data,
    unsupervised_data,
    natural_data,
    params=None,
    train_enabled=True,
    compiled_model_file_name='sum-model.dill'
):
    model_copy = copy.deepcopy(model)
    compiled_model, compiled_tokenizer, decoder = load_model(filename=compiled_model_file_name)
    important_tokens = [str(i) for i in range(10)] + ['+', '=']
    translator = create_translator(tokenizer, compiled_tokenizer, important_tokens)
    wrapped_model = AddSupportModel(
        model,
        compiled_model,
        translator,
        decoder,
        read_from_support=params.read_from_support,
        write_to_support=params.write_to_support,
        tanh_in_write=params.tanh_in_write,
        communicate_every_x_layers=params.cross_attn_every_x_layers,
        key_size=params.key_size,
        attention_heads=params.attention_heads,
        rescaling_factor_write=params.rescaling_factor_write,
        pad_communication=params.pad_communication,
    )
    wrapped_model.train_only_cross_attention()
    if train_enabled:
        train(
            wrapped_model,
            tokenizer,
            data,
            unsupervised_data,
            natural_data,
            original_model=model_copy,
            params=params
        )

    return wrapped_model
