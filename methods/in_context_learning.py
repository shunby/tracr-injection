DATA_LIM = 30

def in_context_learning(model, tokenizer, data):
    model.prefix = ''.join([prompt + ' ' + label + ', ' for prompt, label in [data[i] for i in range(min(DATA_LIM, len(data)))]])
    return model, None

