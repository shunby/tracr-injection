import torch
from datasets import load_dataset
from tqdm import tqdm
from utils import get_tokenizer

def evaluate_sst2(model, tokenizer):
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    dataset = load_dataset("glue", "sst2", split="validation")

    correct = 0
    total = 0
    predictions = []
    unknown = 0

    positive_token_id = tokenizer.encode(" positive", add_special_tokens=False)[0]
    negative_token_id = tokenizer.encode(" negative", add_special_tokens=False)[0]

    for example in tqdm(dataset):
        sentence = example["sentence"]
        gold_label = example["label"]

        prompt = (
            "What is the sentiment in this review? "
            "The answer should be one word: either positive or negative.\n\n"
            f"Review: the movie was absolutely incredible.\n\n"
            "Answer: positive\n\n"
            "What is the sentiment in this review? "
            "The answer should be one word: either positive or negative.\n\n"
            f"Review: this was a shameful waste of time.\n\n"
            "Answer: negative\n\n"
            "What is the sentiment in this review? "
            "The answer should be one word: either positive or negative.\n\n"
            f"Review: {sentence}\n\n"
            "Answer:"
        )

        input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(model.device)

        inputs = tokenizer(prompt, return_tensors="pt")
        input_ids = inputs.input_ids.to(model.device)

        outputs = model(input_ids)
        logits = outputs.logits

        next_token_logits = logits[0, -1, :]

        score_positive = next_token_logits[positive_token_id].item()
        score_negative = next_token_logits[negative_token_id].item()
        most_likely_token_id = torch.argmax(next_token_logits).item()
        most_likely_token = tokenizer.decode([most_likely_token_id])
        most_likely_logit = next_token_logits[most_likely_token_id].item()

        if score_positive > score_negative:
            predicted_label = 1
        else:
            predicted_label = 0

        predictions.append(predicted_label)
        if predicted_label == gold_label:
            correct += 1
        total += 1

    accuracy = correct / total if total > 0 else 0.0
    print(f"Evaluation on SST-2 (Validation Set): Accuracy = {accuracy:.4f}")
    return accuracy


if __name__ == "__main__":
    from transformers import GPT2LMHeadModel, GPT2Tokenizer
    from dotenv import load_dotenv
    import os

    load_dotenv()

    save_id = '360b3315-4bfe-4edd-ae2d-0991ff9eb381'

    CACHE_DIR = os.path.expanduser(os.getenv('CACHE_DIR'))
    model_name = os.path.join(CACHE_DIR, save_id)
    # Specify the model name (GPT-2)
    # model_name = "gpt2-large"
    # tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    tokenizer = get_tokenizer('dyck')
    model = GPT2LMHeadModel.from_pretrained(model_name, cache_dir=CACHE_DIR)

    # GPT-2 does not have a pad token by default; set it to the EOS token.
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Move model to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Evaluate GPT-2 on SST-2
    evaluate_sst2(model, tokenizer)
