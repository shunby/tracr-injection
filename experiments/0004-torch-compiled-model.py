import torch
import torch.nn as nn
import torch.nn.functional as F

from tqdm import tqdm


outputs = {}
def convert_to_torch(jax_array):
    return torch.tensor(jax_array.tolist(), dtype=torch.float32)

class Transformer(nn.Module):
    def __init__(self, input_dim, model_dim, num_heads, num_layers, ff_dim, seq_len, num_classes, key_size):
        outputs['attn'] = []
        outputs['attn_out'] = []
        outputs['q'] = []
        outputs['k'] = []
        outputs['v'] = []
        super(Transformer, self).__init__()
        self.embedding = nn.Embedding(input_dim, model_dim)
        self.positional_encoding = nn.Embedding(seq_len, model_dim)
        self.layers = nn.ModuleList([
            TransformerDecoderLayer(model_dim, num_heads, ff_dim, key_size) for _ in range(num_layers)
        ])
        self.fc = nn.Linear(model_dim, num_classes)

    def forward(self, x):
        positions = torch.arange(x.size(1), device=x.device).unsqueeze(0).expand(x.size(0), -1)
        x = self.embedding(x) + self.positional_encoding(positions)
        outputs['input'] = x.clone().detach()
        for i, layer in enumerate(self.layers):
            x = layer(x)
            outputs[i] = x.clone().detach()

        return self.fc(x)

    def set_weights(self, compiled_model_params):
        self.embedding.weight.data.copy_(convert_to_torch(compiled_model_params['token_embed']['embeddings']))
        self.positional_encoding.weight.data.copy_(convert_to_torch(compiled_model_params['pos_embed']['embeddings']))

        for i, layer in tqdm(enumerate(self.layers)):
            layer.q_proj.weight.data.copy_(convert_to_torch(compiled_model_params[f"transformer/layer_{i}/attn/query"]['w']).T)
            layer.k_proj.weight.data.copy_(convert_to_torch(compiled_model_params[f"transformer/layer_{i}/attn/key"]['w']).T)
            layer.v_proj.weight.data.copy_(convert_to_torch(compiled_model_params[f"transformer/layer_{i}/attn/value"]['w']).T)
            layer.o_proj.weight.data.copy_(convert_to_torch(compiled_model_params[f"transformer/layer_{i}/attn/linear"]['w']).T)
            layer.linear_1.weight.data.copy_(convert_to_torch(compiled_model_params[f"transformer/layer_{i}/mlp/linear_1"]['w']).T)
            layer.linear_2.weight.data.copy_(convert_to_torch(compiled_model_params[f"transformer/layer_{i}/mlp/linear_2"]['w']).T)

            layer.q_proj.bias.data.copy_(convert_to_torch(compiled_model_params[f"transformer/layer_{i}/attn/query"]['b']))
            layer.k_proj.bias.data.copy_(convert_to_torch(compiled_model_params[f"transformer/layer_{i}/attn/key"]['b']))
            layer.v_proj.bias.data.copy_(convert_to_torch(compiled_model_params[f"transformer/layer_{i}/attn/value"]['b']))
            layer.o_proj.bias.data.copy_(convert_to_torch(compiled_model_params[f"transformer/layer_{i}/attn/linear"]['b']))
            layer.linear_1.bias.data.copy_(convert_to_torch(compiled_model_params[f"transformer/layer_{i}/mlp/linear_1"]['b']))
            layer.linear_2.bias.data.copy_(convert_to_torch(compiled_model_params[f"transformer/layer_{i}/mlp/linear_2"]['b']))


class TransformerDecoderLayer(nn.Module):
    def __init__(self, model_dim, num_heads, ff_dim, key_size):
        super(TransformerDecoderLayer, self).__init__()
        self.num_heads = num_heads
        self.key_size = key_size
        self.model_dim = model_dim

        self.q_proj = nn.Linear(model_dim, num_heads * key_size)
        self.k_proj = nn.Linear(model_dim, num_heads * key_size)
        self.v_proj = nn.Linear(model_dim, num_heads * key_size)
        self.o_proj = nn.Linear(num_heads * key_size, model_dim)

        self.linear_1 = nn.Linear(model_dim, ff_dim)
        self.linear_2 = nn.Linear(ff_dim, model_dim)
        self.ff = nn.Sequential(self.linear_1, nn.ReLU(), self.linear_2)

    def forward(self, x):
        batch_size, seq_len, _ = x.size()

        q = self.q_proj(x).view(batch_size, seq_len, self.num_heads, self.key_size).transpose(1, 2)
        k = self.k_proj(x).view(batch_size, seq_len, self.num_heads, self.key_size).transpose(1, 2)
        v = self.v_proj(x).view(batch_size, seq_len, self.num_heads, self.key_size).transpose(1, 2)

        outputs['q'].append(q.clone().detach())
        outputs['k'].append(k.clone().detach())
        outputs['v'].append(v.clone().detach())

        attn_weights = torch.matmul(q, k.transpose(-2, -1)) / torch.sqrt(torch.tensor(self.key_size, dtype=torch.float32))

        causal_mask = torch.triu(torch.ones(seq_len, seq_len, device=x.device), diagonal=1).bool()
        attn_weights = attn_weights.masked_fill(causal_mask, float('-inf'))

        outputs['attn'].append(attn_weights.clone().detach())

        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_output = torch.matmul(attn_weights, v)

        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)
        attn_output = self.o_proj(attn_output)
        outputs['attn_out'].append(attn_output)

        x = x + attn_output
        x = x + self.ff(x)
        return x



from dotenv import load_dotenv
import dill
import os

load_dotenv()

def load_model(directory=os.getenv('STORAGE_DIR'), filename=os.getenv('COMPILED_MODEL')):
    filepath = directory + '/' + filename
    filepath = os.path.expanduser(filepath)
    print(filepath)
    with open(filepath, 'rb') as f:
        return dill.load(f)


compiled_model = load_model()
max_length, model_dim = compiled_model.params['pos_embed']['embeddings'].shape
input_dim, model_dim = compiled_model.params['token_embed']['embeddings'].shape

compiled_model.params


compiled_model.model_config
num_heads = compiled_model.model_config.num_heads
num_layers = compiled_model.model_config.num_layers
ff_dim = compiled_model.model_config.mlp_hidden_size
seq_len = max_length
num_classes = input_dim
key_size = compiled_model.model_config.key_size

ff_dim

####### Hyperparameters
model = Transformer(input_dim, model_dim, num_heads, num_layers, ff_dim, seq_len, num_classes, key_size)
model.set_weights(compiled_model.params)

class Tokenizer:
    def __init__(self, encoding_map,decoding_map,max_length):
        self.vocab = encoding_map
        self.decoder_vocab = {v:k for k,v in decoding_map.items()}
        self.max_length = max_length
        if sum([type(n)==int for n in self.vocab.keys()]):
            self.vocab = {str(k):v for k,v in self.vocab.items()}

    def tokenize(self, text,return_tensor='pt'):
        tokens = []
        for word in text.strip().split():

            if word in self.vocab:
                tokens.append(self.vocab[word])
        tokens = [self.vocab.get('bos')] + tokens[-self.max_length+1:]
        if return_tensor == 'pt':
            return torch.tensor(tokens)
        return tokens

    def __call__(self, text,return_tensor='pt'):
        tokens = []
        for word in text.strip().split():

            if word in self.vocab:
                tokens.append(self.vocab[word])
        tokens = [self.vocab.get('bos')] + tokens[-self.max_length+1:]
        if return_tensor == 'pt':
            return torch.tensor(tokens)
        return tokens

    def decode(self,output):
        texts = [[] * output.size(0)]
        output = output.cpu().tolist()

        for n in range(len(output)):
            texts[n] = [str(self.decoder_vocab[x]) for x in output[n]]
        return ['bos'+t[1:] for t in texts]

tokenizer = Tokenizer(
    compiled_model.input_encoder.encoding_map,
    compiled_model.output_encoder.encoding_map,
    max_length
)

model(tokenizer('2 + 2 = 4').view(1, -1))
real_outputs = compiled_model.apply(['bos', '2', '+', '2', '=', '4'])

len(real_outputs.layer_outputs)
real_outputs.input_embeddings.shape
outputs['input'].shape
torch.allclose(outputs['input'], convert_to_torch(real_outputs.input_embeddings))
mse = torch.mean((outputs['input'] - convert_to_torch(real_outputs.input_embeddings)) ** 2)
print(f"Mean Squared Error: {mse.item()}")


outputs['attn_out'][1]
real_outputs.layer_outputs[2]

for i in range(20):
    outputs['attn_out'][1]
    real_outputs.layer_outputs[2]
    print(torch.allclose(outputs['attn_out'][i], convert_to_torch(real_outputs.layer_outputs[2 * i])))

