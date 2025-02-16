from dotenv import load_dotenv
from transformers import AutoModelForCausalLM, AutoTokenizer

import os
import torch
import torch.nn as nn

load_dotenv()

cache_dir = os.path.expanduser(os.getenv('CACHE_DIR'))
model_name = "microsoft/phi-2"
model_name = "google/gemma-2b"
tokenizer = AutoTokenizer.from_pretrained(
    model_name,
    trust_remote_code=True,
    cache_dir=cache_dir
)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    trust_remote_code=True,
    cache_dir=cache_dir
)

EXTRA_DIMS = 12
original_embeddings = model.model.embed_tokens.weight.data
n_tokens, hidden_dims = original_embeddings.shape
extra_weights = torch.zeros((n_tokens, EXTRA_DIMS))
new_weights = torch.cat((original_embeddings, extra_weights), dim=1)

new_embedding_layer = nn.Embedding(n_tokens, hidden_dims + EXTRA_DIMS)
new_embedding_layer.weight.data = new_weights

model.model.embed_tokens = new_embedding_layer

model.model


original_weight = model.model.layers[0].self_attn.k_proj.weight.data
hidden_dims, = model.model.layers[0].self_attn.k_proj.weight.data.shape
hidden_dims, heads_and_head_dim 

new_weight = torch.zeros((heads_and_head_dim, hidden_dims + EXTRA_DIMS))
new_weight[:, :hidden_dims] = original_weight  # Copy old weights

model.model.layers[0].self_attn.q_proj.weight = torch.nn.Parameter(new_weight)

model.model.layers[0].self_attn.q_proj.weight.shape


def extend_hidden_dims(model, extra_dims):
    extend_embeddings(model, extra_dims)

    for layer in model.model.layers:
        extend_attn_layer(layer, extra_dims)


def extend_attn_layer(layer, extra_dims):
    original_q_weight = layer.self_attn.q_proj.weight.data
    heads_and_head_dim, hidden_dims = original_q_weight.shape
    new_q_weight = torch.zeros((heads_and_head_dim, hidden_dims + extra_dims))
    new_q_weight[:, :hidden_dims] = original_q_weight
    layer.self_attn.q_proj.weight = torch.nn.Parameter(new_q_weight)

    original_k_weight = layer.self_attn.k_proj.weight.data
    heads_and_head_dim, hidden_dims = original_k_weight.shape
    new_k_weight = torch.zeros((heads_and_head_dim, hidden_dims + extra_dims))
    new_k_weight[:, :hidden_dims] = original_k_weight
    layer.self_attn.k_proj.weight = torch.nn.Parameter(new_k_weight)

    original_v_weight = layer.self_attn.v_proj.weight.data
    heads_and_head_dim, hidden_dims = original_v_weight.shape
    new_v_weight = torch.zeros((heads_and_head_dim, hidden_dims + extra_dims))
    new_v_weight[:, :hidden_dims] = original_v_weight
    layer.self_attn.v_proj.weight = torch.nn.Parameter(new_v_weight)

    original_o_weight = layer.self_attn.o_proj.weight.data
    hidden_dims, heads_and_head_dim = original_o_weight.shape
    new_o_weight = torch.zeros((hidden_dims + extra_dims, heads_and_head_dim))
    new_o_weight[:hidden_dims, :] = original_o_weight
    layer.self_attn.o_proj.weight = torch.nn.Parameter(new_o_weight)


def extend_embeddings(model, extra_dims):
    original_embeddings = model.model.embed_tokens.weight.data
    n_tokens, hidden_dims = original_embeddings.shape
    extra_weights = torch.zeros((n_tokens, extra_dims))
    new_weights = torch.cat((original_embeddings, extra_weights), dim=1)

    new_embedding_layer = nn.Embedding(n_tokens, hidden_dims + EXTRA_DIMS)
    new_embedding_layer.weight.data = new_weights

extend_hidden_dims(model, 1)

del tokenizer.get_vocab()['21']
del tokenizer.vocab['21']

tokenizer.vocab[' +']
print(tokenizer.convert_tokens_to_ids(tokenizer.tokenize(' +')))
for key, index in tokenizer.vocab.items():
    if 963 == index:
        print('ajja', key)
    # if '+' in key:
    #     print(key)

tokenizer(' 1234 +')
tokenizer(' 1234 + 124')
tokenizer('1234 ')
tokenizer('1 234 ')
tokenizer('1 ')
tokenizer.decode([963])

prompt = "Instruct: Write a detailed analogy between mathematics and a lighthouse.\nOutput:"
inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

outputs = model.generate(**inputs, max_length=200)
generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

print(generated_text)

