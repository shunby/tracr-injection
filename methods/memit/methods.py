import sys
import os
import contextlib

script_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(script_dir, "memit"))


from .memit.memit import apply_memit_to_model, MEMITHyperParams
from .memit.baselines.ft import apply_ft_to_model, FTHyperParams

def memit(model, tokenizer, data, unsupervised_data):
    with contextlib.redirect_stdout(open(os.devnull, 'w')):
        return apply_memit_to_model(
            model,
            tokenizer,
            build_change_requests(data),
            get_hparams_memit(model)
        )[0]

def finetuning(model, tokenizer, data):
    with contextlib.redirect_stdout(open(os.devnull, 'w')):
        return apply_ft_to_model(
            model,
            tokenizer,
            build_change_requests(data),
            get_hparams_ft(model)
        )[0]

def build_change_requests(data):
    requests = []
    for prompt, label in data:
        requests.append({
            "prompt": "{}",
            "subject": prompt,
            "target_new": {"str": label},
        })

    return requests

def get_hparams_ft(model):
    params_name = (
        './methods/memit/memit/hparams/FT/'
        f"{model.config._name_or_path.replace('/', '_')}_unconstr.json"
    )
    hparams = FTHyperParams.from_json(params_name)
    return hparams


def get_hparams_memit(model):
    params_name = (
        './methods/memit/memit/hparams/MEMIT/'
        f"{model.config._name_or_path.replace('/', '_')}.json"
    )
    hparams = MEMITHyperParams.from_json(params_name)
    return hparams


