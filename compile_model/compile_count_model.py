from tracr.rasp import rasp
from tracr.compiler import compiling
from tracr.compiler.lib import shift_by, make_length
import jax
from dotenv import load_dotenv
import os
import dill

load_dotenv()


ZERO_SOP = rasp.Map(lambda _: 0, rasp.tokens)
MAX_LENGTH = 120
MAX_DIGITS = 3
BOS = 'bos'
PAD = 'pad'
MLP_EXACTNESS = 10000.

def generate_count_operation():
    masked = rasp.tokens == 'yes'
    ones = rasp.Map(lambda _: 1, rasp.indices)
    prevs = rasp.Select(masked, ones, rasp.Comparison.EQ)
    count = rasp.SelectorWidth(prevs)

    digits =[]
    for d in range(MAX_DIGITS):
        def get_digit(x, d=d):
            return (x // (10 **d)) % 10
        d_result = rasp.Map(get_digit, count)
        digits.append(d_result)

    total_digits = count_total_digits(digits)
    result = assign_location_to_each_digit(digits, total_digits)

    return result.named('final_result')

def compile_operation(expression, max_length=MAX_LENGTH):
    model = compiling.compile_rasp_to_model(
        expression,
        vocab={str(i) for i in range(0, MAX_LENGTH)}.union({'yes', 'no', 'count'}),
        max_seq_len=max_length,
        compiler_bos=BOS,
        compiler_pad=PAD,
        causal=True,
        mlp_exactness=MLP_EXACTNESS
    )
    return model

def count_total_digits(final_digits):
    """
    we count how many digits does our final result have (to know where to put them later)
    [0, 2, 0, 1] -> 3
    """
    num_digits = ZERO_SOP
    has_digit = ZERO_SOP
    for d in list(range(MAX_DIGITS))[::-1]:
        has_digit = rasp.SequenceMap(lambda x, y: 1 if x + y > 0 else 0, final_digits[d], has_digit).named(f"has_{d}_digit")
        num_digits += has_digit

    return num_digits

def assign_location_to_each_digit(final_digits, num_digits):
    """
    all the digits are computed in every token. so we filter to leave only the correct digit
    according to the digits number
    """
    final_result = ZERO_SOP
    equals_selector = rasp.Select(rasp.tokens, rasp.Map(lambda x: 'count', rasp.tokens), rasp.Comparison.EQ).named('equals_selector')
    equals_pos = rasp.Aggregate(equals_selector, rasp.indices).named('equals_pos')
    for d in range(MAX_DIGITS):
        def target_location_func(x, y, d=d):
            location = x + y - d - 1
            if location >= MAX_LENGTH or location <= 0:
                return 0

            return location

        target_location = rasp.SequenceMap(target_location_func, equals_pos, num_digits).named(f"target_{d}_location")
        tmp_mask = rasp.SequenceMap(lambda x, y: int(x == y), target_location, rasp.indices).named(f"tmp_mask_{d}_digit")
        digit_in_correct_place = (final_digits[d] * tmp_mask).named(f"digit_{d}_correct_place")
        maxed_sum = rasp.SequenceMap(lambda x, y: 0 if x + y >= 10 else x + y, final_result, digit_in_correct_place)
        final_result = maxed_sum.named(f"tmp_result_{d}_digit")
    final_result = final_result.named('final_assignation')

    return final_result.named('final_assignation')

def save_model(model, location=os.getenv('STORAGE_DIR'), filename='count-model.dill'):
    filepath = location + '/' + filename
    filepath = os.path.expanduser(filepath)
    directory = os.path.dirname(filepath)
    if not os.path.exists(directory):
        os.makedirs(directory)

    with open(filepath, 'wb') as f:
        dill.dump(model, f)
        print(f"model saved at: {filepath}")

if __name__ == '__main__':
    expression = generate_count_operation()
    expression(['no'] + ['yes'] * 11 + ['count'] + ['1'])

    model = compile_operation(expression)
    model.apply([BOS, 'no'] + ['yes'] * 10 + ['count'] + ['1']).decoded
    param_count = sum(x.size for x in jax.tree_leaves(model.params))

    save_model(model)

