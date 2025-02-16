import requests
import torch
import torch.nn.functional as F
from tqdm import tqdm

URL = 'https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt'

def evaluate_tiny_shakespeare(model, tokenizer, max_length=350):
    response = requests.get(URL)
    response.raise_for_status()
    data = response.text

    lines = data.splitlines()[:2000]

    total_perplexity = 0
    valid_lines = 0

    model.eval()

    with torch.no_grad():
        for line in tqdm(lines, desc='evaluating in tiny shakespeare'):
            if not line.strip():
                continue

            input_ids = tokenizer.encode(
                line,
                return_tensors="pt",
                truncation=True,
                max_length=max_length,
            ).to(model.device)

            if input_ids.size(1) < 2:
                continue

            outputs = model(input_ids=input_ids)
            logits = outputs.logits

            shifted_logits = logits[:, :-1, :]
            shifted_labels = input_ids[:, 1:]

            loss = F.cross_entropy(
                shifted_logits.reshape(-1, shifted_logits.size(-1)),
                shifted_labels.reshape(-1),
                reduction="mean"
            )

            perplexity = torch.exp(loss).item()
            total_perplexity += perplexity


            shifted_new_logits = shifted_logits

            new_probs = F.softmax(shifted_new_logits, dim=-1)

            valid_lines += 1

    if valid_lines == 0:
        return float('inf')

    avg_perplexity = total_perplexity / valid_lines

    return avg_perplexity

if __name__ == "__main__":
    from utils import get_tokenizer
    from dotenv import load_dotenv
    import os
    from transformers import GPT2LMHeadModel, GPT2Tokenizer

    load_dotenv()
    CACHE_DIR = os.path.expanduser(os.getenv('CACHE_DIR', "~/.cache/huggingface"))
    save_id = '6e6efb4b-06ca-4ac6-9759-c5fd8a75d438'
    # model_name = "gpt2-large"
    model_name = os.path.join(CACHE_DIR, save_id)
    
    tokenizer = get_tokenizer('dyck')
    # tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    model = GPT2LMHeadModel.from_pretrained(model_name, cache_dir=CACHE_DIR)
    
    # GPT‑2 doesn't have a pad token by default; set it to the EOS token.
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    
    res = evaluate_tiny_shakespeare(model, tokenizer)
    print('res', res)

