from tracr.rasp import rasp
from tracr.compiler import compiling
from tracr.compiler.lib import shift_by, make_pair_balance, make_length, make_shuffle_dyck
from functools import reduce
from operator import or_
import math
import jax
import dill
from dotenv import load_dotenv
import os

load_dotenv()
MAX_LENGTH = 40
BOS = 'bos'
PAD = 'pad'

def shuffle_dyck(
    pairs=[('(', ')'), ('{', '}'), ('[', ']')]
):
    all_diffs = []
    all_negs = []
    for left, right in pairs:
        def compare_left(x, y, left=left):
            return x == left
        starts = rasp.Select(rasp.tokens, rasp.tokens, compare_left)
        start_counts = rasp.SelectorWidth(starts).named(f"{left}_start_counts")

        def compare_right(x, y, right=right):
            return x == right
        ends = rasp.Select(rasp.tokens, rasp.tokens, compare_right)
        end_counts = rasp.SelectorWidth(ends).named(f"{right}_end_counts")

        diffs = (start_counts - end_counts).named(f"{left + right}_diffs")
        all_diffs.append(diffs)

        negs_selector = rasp.Select(diffs, rasp.tokens, lambda x, y: x < 0)
        negs_counter = rasp.SelectorWidth(negs_selector).named(f"{left + right}_negative_counters")
        all_negs.append(negs_counter)

    current_negs = all_negs[0]
    for negs in all_negs[1:]:
        current_negs = rasp.SequenceMap(lambda x, y: 1 if (x != 0 or y != 0) else 0, current_negs, negs)
    aggregated_negs = current_negs.named('aggregated_negatives')

    current_diffs = all_diffs[0]
    for i, diffs in enumerate(all_diffs[1:]):
        current_diffs = rasp.SequenceMap(lambda x, y: 1 if (x != 0 or y != 0) else 0, current_diffs, diffs).named(f"aggregated_diffs_tmp_{i}")
    aggregated_diffs = current_diffs.named('aggregated_diffs')

    result = rasp.SequenceMap(lambda x, y: x == 0 and y == 0, aggregated_negs, aggregated_diffs).named('unfiltered_result')
    filtered_result = rasp.SequenceMap(lambda x, y: x if y == 'compute' else False, result, rasp.tokens)
    return filtered_result.named('final_result')

def compile_operation(expression, max_length=MAX_LENGTH):
    model = compiling.compile_rasp_to_model(
        expression,
        vocab={'(', ')', '[', ']', '{', '}', ' ', 'compute'},
        max_seq_len=max_length,
        compiler_bos=BOS,
        compiler_pad=PAD,
        causal=True,
    )
    return model

def save_model(model, location=os.getenv('STORAGE_DIR'), filename='dyck-model.dill'):
    filepath = location + '/' + filename
    filepath = os.path.expanduser(filepath)
    directory = os.path.dirname(filepath)
    if not os.path.exists(directory):
        os.makedirs(directory)

    with open(filepath, 'wb') as f:
        dill.dump(model, f)
        print(f"model saved at: {filepath}")

if __name__ == '__main__':
    expression = shuffle_dyck()
    model = compile_operation(expression)
    model.apply([BOS, '(', ')', 'compute'])
    param_count = sum(x.size for x in jax.tree_leaves(model.params))
    save_model(model)



