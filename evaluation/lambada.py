import json
import requests
import torch
from tqdm import tqdm
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from dotenv import load_dotenv
import os

URL = "https://openaipublic.blob.core.windows.net/gpt-2/data/lambada_test.jsonl"

def evaluate_lambada(model, tokenizer):
    response = requests.get(URL)
    response.raise_for_status()

    lines = response.text.splitlines()
    total = 0
    correct = 0

    for line in tqdm(lines, desc="Evaluating Lambada (Last Token)"):
        try:
            example = json.loads(line)
        except json.JSONDecodeError:
            continue

        full_text = example.get("text", "").strip()
        if not full_text:
            continue

        inputs = tokenizer(full_text, return_tensors="pt")
        input_ids = inputs.input_ids.to(model.device)

        if input_ids.size(1) < 2:
            continue

        context_ids = input_ids[:, :-1]
        target_token_id = input_ids[0, -1].item()

        with torch.no_grad():
            outputs = model(context_ids)
            logits = outputs.logits

        next_token_logits = logits[0, -1, :]

        predicted_token_id = torch.argmax(next_token_logits).item()

        if predicted_token_id == target_token_id:
            correct += 1
        total += 1

    accuracy = correct / total if total > 0 else 0.0
    print(f"Evaluation on Lambada (Last Token Prediction): Accuracy = {accuracy:.4f}")
    return accuracy

if __name__ == "__main__":
    from utils import get_tokenizer
    load_dotenv()
    CACHE_DIR = os.path.expanduser(os.getenv('CACHE_DIR', "~/.cache/huggingface"))
    save_id = '6e6efb4b-06ca-4ac6-9759-c5fd8a75d438'
    # model_name = "gpt2-large"  # Adjust the model variant as needed.
    model_name = os.path.join(CACHE_DIR, save_id)

    tokenizer = get_tokenizer('dyck')
    # tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    model = GPT2LMHeadModel.from_pretrained(model_name, cache_dir=CACHE_DIR)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    evaluate_lambada(model, tokenizer)

