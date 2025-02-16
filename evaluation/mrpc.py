import torch
from datasets import load_dataset
from tqdm import tqdm
from utils import get_tokenizer

def evaluate_mrpc(model, tokenizer):
    dataset = load_dataset("glue", "mrpc", split="validation")

    correct = 0
    total = 0
    predictions = []

    positive_token_id = tokenizer.encode(" yes", add_special_tokens=False)[0]
    negative_token_id = tokenizer.encode(" no", add_special_tokens=False)[0]

    for example in tqdm(dataset):
        sentence1 = example["sentence1"]
        sentence2 = example["sentence2"]
        gold_label = example["label"]

        prompt = (
            "Determine whether the following two sentences are paraphrases (i.e., have the same meaning). "
            "Answer in one word: yes or no.\n\n"
            "Sentence 1: The company reported strong earnings this quarter.\n"
            "Sentence 2: The business announced impressive profits for the quarter.\n"
            "Answer: yes\n\n"
            "Sentence 1: I love to read science fiction novels.\n"
            "Sentence 2: I prefer watching documentaries over reading books.\n"
            "Answer: no\n\n"
            f"Sentence 1: {sentence1}\n"
            f"Sentence 2: {sentence2}\n"
            "Answer:"
        )

        inputs = tokenizer(prompt, return_tensors="pt")
        input_ids = inputs.input_ids.to(model.device)

        outputs = model(input_ids)
        logits = outputs.logits

        next_token_logits = logits[0, -1, :]

        score_positive = next_token_logits[positive_token_id].item()
        score_negative = next_token_logits[negative_token_id].item()

        if score_positive > score_negative:
            predicted_label = 1
        else:
            predicted_label = 0

        predictions.append(predicted_label)
        if predicted_label == gold_label:
            correct += 1
        total += 1

    accuracy = correct / total if total > 0 else 0.0
    print(f"Evaluation on MRPC (Validation Set): Accuracy = {accuracy:.4f}")
    return accuracy

if __name__ == "__main__":
    from transformers import GPT2LMHeadModel, GPT2Tokenizer
    from dotenv import load_dotenv
    import os

    load_dotenv()
    save_id = '6e6efb4b-06ca-4ac6-9759-c5fd8a75d438'
    CACHE_DIR = os.path.expanduser(os.getenv('CACHE_DIR'))

    model_name = os.path.join(CACHE_DIR, save_id)
    # model_name = "gpt2-large"
    tokenizer = get_tokenizer('dyck')
    # tokenizer = GPT2Tokenizer.from_pretrained(model_name)

    model = GPT2LMHeadModel.from_pretrained(model_name, cache_dir=CACHE_DIR)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    evaluate_mrpc(model, tokenizer)

