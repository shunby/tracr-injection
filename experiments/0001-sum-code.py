from tracr.rasp import rasp
from tracr.compiler import compiling
from tracr.compiler.lib import shift_by
from functools import reduce
from operator import or_
import math


sop = rasp.tokens
MAX_DIGITS = 5

numbers = rasp.SequenceMap(
    lambda x, i: int(x) if x.isdigit() else 0,
    sop, rasp.indices
)

plus_selector = rasp.Select(sop, rasp.Map(lambda x: '+', sop), rasp.Comparison.EQ)
plus_pos = rasp.Aggregate(plus_selector, rasp.indices)

equals_selector = rasp.Select(sop, rasp.Map(lambda x: '=', sop), rasp.Comparison.EQ)
equals_pos = rasp.Aggregate(equals_selector, rasp.indices)

indices = rasp.indices

left_last = (plus_pos - 1)
digit_sums = []
for d in range(MAX_DIGITS):
    left_idx_d = (left_last - d)
    right_idx_d = ((equals_pos - 1) - d)

    safe_left_idx_d = rasp.SequenceMap(
        lambda idx, plus: idx if idx >= 0 else plus,
        left_idx_d, plus_pos
    )

    safe_right_idx_d = rasp.SequenceMap(
        lambda idx, plus: idx if idx >= plus else plus,
        right_idx_d, plus_pos
    )

    left_digit_d = rasp.Aggregate(
        rasp.Select(indices, safe_left_idx_d, rasp.Comparison.EQ),
        numbers
    )

    right_digit_d = rasp.Aggregate(
        rasp.Select(indices, safe_right_idx_d, rasp.Comparison.EQ),
        numbers
    )

    digit_sum_d = (left_digit_d + right_digit_d)
    digit_sums.append(digit_sum_d)

digit_sums[0](['2', '8', '+', '9', '4', '=', '2', '5'])


carry_current = rasp.SequenceMap(
    lambda _, i: 0,
    sop, indices
)

final_digits = []
for d in range(MAX_DIGITS):
    sum_with_carry = (digit_sums[d] + carry_current)

    final_digit_d = rasp.SequenceMap(
        lambda val, i: val % 10,
        sum_with_carry, indices
    )

    next_carry = rasp.SequenceMap(
        lambda val, i: val // 10,
        sum_with_carry, indices
    )

    final_digits.append(final_digit_d)
    carry_current = next_carry

final_digits[0](['2', '8', '+', '9', '4', '=', '2', '5'])


num_digits = rasp.Map(lambda x: 0, rasp.tokens)
has_digit = rasp.Map(lambda x: 0, rasp.tokens)
for d in list(range(MAX_DIGITS))[::-1]:
    has_digit = rasp.SequenceMap(lambda x, y: 1 if x + y > 0 else 0, final_digits[d], has_digit)
    num_digits += has_digit


num_digits(['2', '8', '+', '9', '4', '=', '2', '5'])

final_result = rasp.Map(lambda x: 0, rasp.tokens)
for d in range(MAX_DIGITS):
    target_location = equals_pos + num_digits - d - 1
    target_location(['2', '8', '+', '9', '4', '=', '2', '5', '4'])
    tmp_mask = rasp.SequenceMap(lambda x, y: int(x == y), target_location, rasp.indices)
    tmp_mask(['2', '8', '+', '9', '4', '=', '2', '5', '4'])
    final_digits[d](['2', '8', '+', '9', '4', '=', '2', '5', '4'])
    select_off_by_offset = rasp.Select(target_location, rasp.indices, rasp.Comparison.EQ)
    digit_in_correct_place = final_digits[d] * tmp_mask
    digit_in_correct_place(['2', '8', '+', '9', '4', '=', '2', '5', '4'])
    final_result += digit_in_correct_place

final_result(['2', '8', '+', '9', '4', '=', '2', '5'])

def shift(sop: rasp.SOp, offset=1) -> rasp.SOp:
  select_off_by_offset = rasp.Select(rasp.indices, rasp.indices,
                                     lambda k, q: q == k + offset)
  out = rasp.Aggregate(select_off_by_offset, sop, default=None)
  return out

bos='B'
model = compiling.compile_rasp_to_model(
    final_result,
    vocab={str(i) for i in range(0, 10)}.union({'+', '='}),
    max_seq_len=16,
    compiler_bos=bos,
    causal=True,
)

model.apply([bos, '8', '8', '+', '3', '4', '=', '1', '2'])#.unembedded

model.model_config
model

import jax
param_count = sum(x.size for x in jax.tree_leaves(model.params))
param_count



