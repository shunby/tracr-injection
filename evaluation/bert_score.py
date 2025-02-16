from evaluation.prompts import PROMPTS, RESPONSES
from evaluate import load

RESPONSE_LENGTH = 50

def evaluate_bert_score(model, tokenizer, model_name):
    predictions = generate_responses(model, tokenizer)
    bertscore = load("bertscore")
    results = bertscore.compute(predictions=predictions, references=RESPONSES[model_name], model_type="distilbert-base-uncased")
    return results

def generate_responses(model, tokenizer):
    responses = []

    for prompt in PROMPTS:
        if hasattr(model, 'prefix'):
            prompt = model.prefix + prompt
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

        outputs = model.generate(
            inputs.input_ids,
            max_length=inputs.input_ids.shape[1] + RESPONSE_LENGTH,
            do_sample=False,
        )
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        continuation = response[len(prompt):]
        responses.append(continuation.strip())

    return responses
