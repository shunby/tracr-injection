import torch
import os
from transformers import AutoModelForCausalLM
from dotenv import load_dotenv


from utils import get_tokenizer
from compile_model.load_compiled_model import load_model
from support_model.create_tokenizer_translator import create_translator

load_dotenv()

CACHE_DIR = os.path.expanduser(os.getenv('CACHE_DIR'))
save_id = '8b2e85e7-6864-4bd0-bf64-bc6f7dbeaedf'
MODEL_NAME = os.path.join(CACHE_DIR, save_id)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, cache_dir=CACHE_DIR)
adapter_path = os.path.join(MODEL_NAME, "adapter.pth")
task = 'dyck'
tokenizer = get_tokenizer(task)

compiled_model, compiled_tokenizer, decoder = load_model(filename='dyck-model.dill')
translator = create_translator(tokenizer, compiled_tokenizer, task=task)
adapter = torch.nn.Linear(model.config.hidden_size, compiled_model.model_dim)
adapter.load_state_dict(torch.load(adapter_path))

example_case = "Are parenthesis here correctly matched? [(])\nAnswer:"
input_ids = tokenizer.encode(example_case, return_tensors='pt').to(model.device)

output = model(input_ids=input_ids, output_hidden_states=True)

layer = len(compiled_model.layers)

model_residual_stream = output.hidden_states[layer][0]

translated_residual_stream = adapter(model_residual_stream)

model_residual_stream.shape
translated_residual_stream.shape

variables = []
for i in range(translated_residual_stream.shape[0]):
    activations = translated_residual_stream[i]
    variables.append(decoder(activations))

tokens = tokenizer.convert_ids_to_tokens(input_ids[0].tolist())
clean_tokens = [token.lstrip("Ġ") for token in tokens]
clean_tokens = [token.replace("Ċ", "\n") for token in clean_tokens]

results = {
    'model_residual_stream': model_residual_stream,
    'translated_residual_stream': translated_residual_stream,
    'variables': variables,
    'tokens': clean_tokens
}

import pickle

# Save results as a .pkl file
with open('./plots/decoding_process.pkl', 'wb') as f:
    pickle.dump(results, f)
