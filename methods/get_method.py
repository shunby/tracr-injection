from .in_context_learning import in_context_learning
from .inject_algorithm.method import inject_algorithm
from .distil_algorithm.method import distil_algorithm
from .finetuning.method import finetuning

def get_method(method):
    if method == 'NO-METHOD':
        return lambda model, tokenizer, data, unsupervised_data: model
    if method == 'ICL':
        return in_context_learning
    if method == 'FT':
        return finetuning
    if method == 'INJECT':
        return inject_algorithm
    if method == 'DISTIL':
        return distil_algorithm
    raise Exception(f"Unrecognized method {method}")


