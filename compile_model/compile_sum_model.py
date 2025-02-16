from tracr.rasp import rasp
from tracr.compiler import compiling
from tracr.compiler.lib import shift_by
from functools import reduce
from operator import or_
import math
import jax
import dill
from dotenv import load_dotenv
import os

load_dotenv()
MAX_DIGITS = 4
MAX_LENGTH = 16
BOS = 'bos'
PAD = 'pad'

ZERO_SOP = rasp.Map(lambda _: 0, rasp.tokens)

def generate_sum_operation():
    """
    compiles a model capable of doing sums up to MAX_DIGITS digits.
    """
    digit_sums = sum_by_digits()
    final_digits = percolate_carries(digit_sums)
    num_digits = count_total_digits(final_digits)
    final_result = assign_location_to_each_digit(final_digits, num_digits)

    return final_result

def compile_operation(expression, max_length=MAX_LENGTH):
    model = compiling.compile_rasp_to_model(
        expression,
        vocab={str(i) for i in range(0, 10)}.union({'+', '=', ' '}),
        max_seq_len=max_length,
        compiler_bos=BOS,
        compiler_pad=PAD,
        causal=True,
    )
    return model

def sum_by_digits():
    """
    we do a digit by digit sum. for example:
    [2, 5, 8]
    +
    [7, 7, 7]
    =
    [2, 12, 15]
    """
    numbers = rasp.Map(lambda x: int(x) if x.isdigit() else 0, rasp.tokens).named('numbers')
    shifted_tokens = shift_by(1, rasp.tokens)
    number_just_ended = rasp.SequenceMap(lambda x, y: is_digit(x) and not is_digit(y), shifted_tokens, rasp.tokens)
    ended_numbers = rasp.Select(number_just_ended, rasp.Map(lambda x: True, rasp.tokens), rasp.Comparison.EQ)
    cumulative_ended_numbers = rasp.SelectorWidth(ended_numbers)
    second_number = rasp.SequenceMap(lambda x, mask: int(x) if x.isdigit() and mask > 0 else 0, rasp.tokens, cumulative_ended_numbers)


    shifted_first_numbers = [numbers]
    shifted_second_numbers = [second_number]
    for i in range(1, MAX_DIGITS + 1):
        current_shift = zero_shift_by(i, numbers)
        shifted_first_numbers.append(current_shift)
        current_shift = zero_shift_by(i, second_number)
        shifted_second_numbers.append(current_shift)

    shifted_first_numbers = list(map(lambda num: rasp.SequenceMap(lambda x, y: x if y else 0, num, number_just_ended), shifted_first_numbers))
    shifted_second_numbers = list(map(lambda num: rasp.SequenceMap(lambda x, y: x if y else 0, num, number_just_ended), shifted_second_numbers))

    total_numbers_counter = rasp.Select(number_just_ended, number_just_ended, rasp.Comparison.EQ)
    number_id = rasp.SelectorWidth(total_numbers_counter)
    number_id_clean = rasp.SequenceMap(lambda nid, correct: nid if correct else 0, number_id, number_just_ended)

    after_second_number = rasp.Map(lambda x: 1 if x > 1 else 0, cumulative_ended_numbers)
    grab_numbers = rasp.Select(number_id_clean, after_second_number, lambda number, equal: number == equal)
    grab_numbers_2 = rasp.Select(number_id_clean, after_second_number * 2, lambda number, equal: number == equal)

    digit_sums = []
    for i in range(1, MAX_DIGITS + 1):
        this_digit_sum = rasp.Aggregate(grab_numbers, shifted_first_numbers[i]) + rasp.Aggregate(grab_numbers_2, shifted_second_numbers[i])
        this_digit_sum = rasp.Map(lambda x: x if x > 0 else 0, this_digit_sum)
        digit_sums.append(this_digit_sum)

    return digit_sums

def percolate_carries(digit_sums):
    """
    we "percolate" the carries we obtained. for example:
    [2, 12, 15] -> [3, 3, 5]
    """
    carry_current = ZERO_SOP
    final_digits = []
    for d in range(MAX_DIGITS):
        final_digit_d = rasp.SequenceMap(
            lambda l, c: (l + c) % 10,
            digit_sums[d],
            carry_current
        ).named(f"final_{d}_digit")

        next_carry = rasp.SequenceMap(
            lambda l, c: (l + c) // 10,
            digit_sums[d],
            carry_current
        ).named(f"next_carry_{d}_digit")


        final_digits.append(final_digit_d)
        carry_current = next_carry

    return final_digits


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
    equals_selector = rasp.Select(rasp.tokens, rasp.Map(lambda x: '=', rasp.tokens), rasp.Comparison.EQ).named('equals_selector')
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
    final_result = final_result.named('final_result')

    return final_result


def save_model(model, location=os.getenv('STORAGE_DIR'), filename='sum-model.dill'):
    filepath = location + '/' + filename
    filepath = os.path.expanduser(filepath)
    directory = os.path.dirname(filepath)
    if not os.path.exists(directory):
        os.makedirs(directory)

    with open(filepath, 'wb') as f:
        dill.dump(model, f)
        print(f"model saved at: {filepath}")

def zero_shift_by(offset: int, /, sop: rasp.SOp) -> rasp.SOp:
  """Returns the sop, shifted by `offset`, Zero-padded."""
  select_off_by_offset = rasp.Select(rasp.indices, rasp.indices,
                                     lambda k, q: q == k + offset or (q - offset < 0 and k == 0))
  out = rasp.Aggregate(select_off_by_offset, sop, default=None)
  clean_out = rasp.SequenceMap(lambda value, index: value if index >= offset else 0, out, rasp.indices)
  return clean_out

def is_digit(x):
    return isinstance(x, int) or x.isdigit()

if __name__ == '__main__':
    expression = generate_sum_operation()
    model = compile_operation(expression)
    param_count = sum(x.size for x in jax.tree_leaves(model.params))
    print('param count is', param_count)
    len(model.residual_labels)
    result = model.apply([BOS, '8', '8', '+', '3', '4', '=', '1', '2']).decoded
    print('the result of adding 88 + 34 is', result[-3:])
    save_model(model)

