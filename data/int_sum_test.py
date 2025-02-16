import pytest
from .int_sum import prepare_sum_dataset

@pytest.mark.parametrize("max_int, test_size, max_test_size", [
    (20, 0.2, 1000),
    (40, 0.2, 320),
])
def test_dataset_size_when_not_limited(
    max_int,
    test_size,
    max_test_size
):
    train_data, test_data = prepare_sum_dataset(
        max_int=max_int,
        test_size=test_size,
        max_test_size=max_test_size
    )
    expected_total_samples = max_int ** 2
    expected_test_size = int(expected_total_samples * test_size)
    expected_train_size = expected_total_samples - expected_test_size

    assert(expected_test_size == len(test_data))
    assert(expected_train_size == len(train_data))

@pytest.mark.parametrize("max_int, test_size, max_test_size", [
    (20, 0.2, 10),
    (40, 0.2, 100),
])
def test_dataset_size_when_limited(
    max_int,
    test_size,
    max_test_size
):
    train_data, test_data = prepare_sum_dataset(
        max_int=max_int,
        test_size=test_size,
        max_test_size=max_test_size
    )
    expected_total_samples = max_int ** 2
    expected_test_size = max_test_size
    expected_train_size = expected_total_samples - expected_test_size

    assert(expected_test_size == len(test_data))
    assert(expected_train_size == len(train_data))
