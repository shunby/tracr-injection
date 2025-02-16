import random
from torch.utils.data import Dataset, random_split

class IntSumDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        prompt, label = self.data[idx]
        return prompt, label

def create_int_sum_data(max_int, min_int=0):
    data = []
    for i in range(max_int):
        for j in range(max_int):
            result = i + j
            prompt = f"{i} + {j} ="
            label = str(result)
            data.append((prompt, label))
    return data

def create_cascade_data(digits=3):
    data = []
    for i in range(1, 10):
        target = i * (10 ** (digits - 1))
        for j in range (target):
            a = j
            b = target - a
            prompt = f"{a} + {b} ="
            label = str(target)
            data.append((prompt, label))
    return data

def create_decimal_data(max_int, data_len=4000):
    data = []
    for _ in range(data_len):
        i = random.randint(0, max_int)
        j = random.randint(0, max_int)

        i_decimal = random.randint(1, 99)
        j_decimal = random.randint(1, 99)
        float_i = i + i_decimal / 100.0
        float_j = j + j_decimal / 100.0

        total = float_i + float_j
        prompt = f"{float_i:.2f} + {float_j:.2f} ="
        label = f"{total:.2f}"

        data.append((prompt, label))
    return data

def prepare_sum_dataset(
    max_int=450,
    test_size=0.2,
    max_test_size=4000,
    length_ood=False,
    cascading_overflow=False,
    decimals=False,
):
    data = create_int_sum_data(max_int)
    dataset = IntSumDataset(data)
    t_size = min(int(test_size * len(dataset)), max_test_size)
    train_size = len(dataset) - t_size
    train_dataset, test_dataset = random_split(dataset, [train_size, t_size])

    if length_ood:
        data = create_int_sum_data(4500, min_int=1000)
        dataset = IntSumDataset(data)
        t_size = min(int(test_size * len(dataset)), max_test_size)
        train_size = len(dataset) - t_size
        _, test_dataset = random_split(dataset, [train_size, t_size])
    elif cascading_overflow:
        data = create_cascade_data()
        test_dataset = IntSumDataset(data)
    elif decimals:
        data = create_decimal_data(max_int)
        test_dataset = IntSumDataset(data)

    return train_dataset, test_dataset

if __name__ == '__main__':
    train_dataset, test_dataset = prepare_sum_dataset(decimals=True)
