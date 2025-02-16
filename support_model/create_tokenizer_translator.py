import torch

IMPORTANT_TOKENS = {
    'int-sum': [str(i) for i in range(10)],
    'dyck': list('()[]{}'),
    'count': [str(i) for i in range(10)] + ['x'],
}
FILLUP_TOKEN = {
    'int-sum': '+',
    'dyck': ' ',
    'count': 'no',
}
CUSTOM_MAPPING = {
    'int-sum': {},
    'dyck': { ':': 'compute' },
    'count': { ':': 'count', 'x': 'yes' },
}

def create_translator(entry_tokenizer, output_tokenizer, variations=None, task='int-sum'):
    if variations is None:
        variations = [
            lambda x: x,
            lambda x: ' ' + x,
            lambda x: x + ' ',
        ]

    mapping = {}

    fillup_token = FILLUP_TOKEN[task]
    important_tokens = IMPORTANT_TOKENS[task]
    custom_mapping = CUSTOM_MAPPING[task]
    for token in important_tokens:
        for variation in variations:
            entry_ids = entry_tokenizer.encode(variation(token), add_special_tokens=False)
            output_ids = output_tokenizer.tokenize(token, add_special_tokens=False)

            if len(entry_ids) == 1 and len(output_ids) == 1:
                mapping[entry_ids[0]] = output_ids[0].item()

    for input_token, output_token in custom_mapping.items():
        for variation in variations:
            entry_ids = entry_tokenizer.encode(variation(input_token), add_special_tokens=False)
            output_ids = output_tokenizer.tokenize(output_token, add_special_tokens=False)

            if len(entry_ids) == 1 and len(output_ids) == 1:
                mapping[entry_ids[0]] = output_ids[0].item()


    def translator(tokens):
        translated_tokens = torch.tensor(
            [mapping.get(token.item(), output_tokenizer.vocab[fillup_token]) for token in tokens.flatten()]
        ).reshape(tokens.shape).to(tokens.device)

        bos_token = output_tokenizer.vocab['bos']
        bos_tensor = torch.full((translated_tokens.shape[0], 1), bos_token, dtype=torch.long, device=tokens.device)

        translated_with_bos = torch.cat((bos_tensor, translated_tokens), dim=1)
        return translated_with_bos


    return translator

if __name__ == '__main__':
    from transformers import GPT2Tokenizer
    from compile_model.load_compiled_model import load_model


    entry_tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    model, output_tokenizer, decoder = load_model(filename='count-model.dill')

    translator = create_translator(entry_tokenizer, output_tokenizer, task='count')

    test_str = "What is the number of 'X' in this text? x x y x y y x y x y x y\nAnswer:"
    input_ids = entry_tokenizer.encode(test_str, add_special_tokens=False)
    input_tensor = torch.tensor([input_ids], dtype=torch.long)
    translated_tensor = translator(input_tensor)
