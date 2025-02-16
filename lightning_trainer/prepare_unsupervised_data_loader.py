from torch.utils.data import DataLoader
import torch

def prepare_unsupervised_data_loader(data, tokenizer, batch_size=16, max_length=12):
    def tokenize_function(examples):
        tokens = tokenizer(
            examples['text'],
            truncation=True,
            padding='max_length',
            max_length=max_length,
            return_tensors="pt"
        )
        return {'input_ids': tokens['input_ids'].squeeze(0), 'attention_mask': tokens['attention_mask']}

    tokenized_data = data.map(tokenize_function, remove_columns=['text', 'id', 'dump', 'url', 'file_path', 'language', 'language_score', 'token_count', 'score', 'int_score'])

    return DataLoader(tokenized_data, batch_size=batch_size)

