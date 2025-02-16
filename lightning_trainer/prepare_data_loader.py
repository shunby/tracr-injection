import torch
from torch.utils.data import Dataset, DataLoader, random_split

class PreTokenizedDataset(Dataset):
    def __init__(self, data, tokenizer, max_length=12):
        self.data = []
        for prompt, label in data:
            combined_text = f"{prompt} {label}"
            tokenized = tokenizer(
                combined_text,
                padding="max_length",
                truncation=True,
                max_length=max_length,
                return_tensors="pt"
            )
            label_start_idx = len(tokenizer(prompt)['input_ids']) - 1
            label_end_idx = label_start_idx + len(str(label))
            loss_mask = torch.zeros_like(tokenized['input_ids'][0], dtype=torch.float32)
            loss_mask[label_start_idx:label_end_idx] = 1

            self.data.append({
                "input_ids": tokenized['input_ids'].squeeze(0),
                "loss_mask": loss_mask
            })

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

def prepare_data_loader(data, tokenizer, batch_size=12, max_length=12):
    dataset = PreTokenizedDataset(data, tokenizer, max_length)
    validation_size = 100
    train_size = len(dataset) - validation_size
    train_dataset, val_dataset = random_split(dataset, [train_size, validation_size])
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    return train_dataloader, val_dataloader
