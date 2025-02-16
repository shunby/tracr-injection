import pytest
from .compile_sum_model import generate_sum_operation, compile_operation, PAD, BOS

@pytest.mark.parametrize("a, b, expected", [
    (['8', '8'], ['3', '4'], [1, 2, 2]),
    (['8'], ['3'], [1, 1]),
    (['9', '8', '8'], ['3', '4', '3', '4'], [4, 4, 2, 2]),
])
def test_generated_expression(a, b, expected):
    expression = generate_sum_operation()
    tokens = a + ['+'] + b + ['=', '0', '0', '0', '0']
    result = expression(tokens)
    equal_pos = len(a) + len(b) + 1
    assert(tokens[equal_pos] == '=')
    assert(result[equal_pos:equal_pos + len(expected)] == expected)


@pytest.fixture(scope="session")
def compiled_model():
    expression = generate_sum_operation()
    model = compile_operation(expression, max_length=10)
    return model


@pytest.mark.slow
@pytest.mark.parametrize("a, b, expected", [
    (['8', '8'], ['3', '4'], [1, 2, 2]),
    (['8'], ['3'], [1, 1]),
])
def test_compiled_model(compiled_model, a, b, expected):
    tokens = [BOS] + a + ['+'] + b + ['=', '0', '0', '0']
    result = compiled_model.apply(tokens).decoded
    equal_pos = len(a) + len(b) + 2
    assert(tokens[equal_pos] == '=')
    assert(result[equal_pos:equal_pos + len(expected)] == expected)
