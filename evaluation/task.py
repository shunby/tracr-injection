import torch
from tqdm import tqdm

def evaluate_task(model, tokenizer, data):
    model.eval()
    correct = 0
    total = 0

    for prompt, label in tqdm(data):
        if hasattr(model, 'prefix'):
            prompt = model.prefix + prompt
        input_ids = tokenizer.encode(prompt, return_tensors='pt').to(model.device)

        target_token = ' ' + str(label)
        target_ids = tokenizer.encode(target_token, add_special_tokens=False)
        target_len = len(target_ids)

        with torch.no_grad():
            outputs = model.generate(
                input_ids=input_ids,
                max_length=input_ids.size(1) + target_len,
                do_sample=False
            )

        generated_ids = outputs[0][input_ids.size(1):]

        if torch.equal(generated_ids, torch.tensor(target_ids).to(model.device)):
            correct += 1
        total += 1

    accuracy = correct / total if total > 0 else 0
    return accuracy

