from dotenv import load_dotenv
from transformers import AutoModelForCausalLM, AutoTokenizer
import dill
import os
import torch
from compile_model.load_compiled_model import load_model
import torch.nn as nn
import torch.nn.functional as F

load_dotenv()

cache_dir = os.path.expanduser(os.getenv('CACHE_DIR'))
# model_name = "microsoft/phi-2"
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

model.model.layers[0]

compiled_model, compiled_tokenizer = load_model()
compiled_model.model_dim
class AddSupportModel:
    def __init__(
        self,
        main_model,
        support_model,
        tokenization_translator,
        communicate_every_x_layers=3,
        attention_heads=3,
        key_size=20
    ):
        self.main_model = main_model
        self.support_model = support_model
        self.tokenization_translator = tokenization_translator
        self.cross_attention_layers = [SelfAttentionModule(main_model.config.hidden_size + compiled_model.model_dim, attention_heads, key_size) for _ in range(len(self.main_model.model.layers))]

        for i, layer in enumerate(self.main_model.model.layers):
            original_forward = layer.forward

            def modified_forward(*args, original_forward=original_forward, support_model=support_model, attn=self.cross_attention_layers[i], **kwargs):
                output = original_forward(*args, **kwargs)
                for _ in range(communicate_every_x_layers):
                    support_output = support_model.forward_one_layer()

                tensor_output, __cache = output
                concat_output = torch.cat((tensor_output, support_output), dim=-1)
                modified_output = attn(concat_output)

                return output

            layer.forward = modified_forward

    def __getattr__(self, name):
        return getattr(self.main_model, name)

    def __call__(self, *args, **kwargs):
        self.support_model.embed_tokens(translator(kwargs['input_ids']))
        output = self.main_model(*args, **kwargs)
        return output


class SelfAttentionModule(nn.Module):
    def __init__(self, residual_stream_dim, num_heads, key_size):
        super(SelfAttentionModule, self).__init__()
        self.num_heads = num_heads
        self.key_size = key_size
        self.q_proj = nn.Linear(residual_stream_dim, num_heads * key_size)
        self.k_proj = nn.Linear(residual_stream_dim, num_heads * key_size)
        self.v_proj = nn.Linear(residual_stream_dim, num_heads * key_size)
        self.o_proj = nn.Linear(num_heads * key_size, residual_stream_dim)
        nn.init.constant_(self.o_proj.weight, 0)
        nn.init.constant_(self.v_proj.bias, 0)

    def forward(self, x):
        batch_size, seq_len, _ = x.size()

        q = self.q_proj(x).view(batch_size, seq_len, self.num_heads, self.key_size).transpose(1, 2)
        k = self.k_proj(x).view(batch_size, seq_len, self.num_heads, self.key_size).transpose(1, 2)
        v = self.v_proj(x).view(batch_size, seq_len, self.num_heads, self.key_size).transpose(1, 2)

        attn_weights = torch.matmul(q, k.transpose(-2, -1)) / torch.sqrt(torch.tensor(self.key_size, dtype=torch.float32))

        causal_mask = torch.triu(torch.ones(seq_len, seq_len, device=x.device), diagonal=1).bool()
        attn_weights = attn_weights.masked_fill(causal_mask, float('-inf'))

        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_output = torch.matmul(attn_weights, v)

        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)
        attn_output = self.o_proj(attn_output)

        x = x + attn_output
        return x


compiled_tokenizer.vocab

def create_translator(entry_tokenizer, output_tokenizer, important_tokens, variations=None):
    if variations is None:
        variations = [
            lambda x: x,
            lambda x: ' ' + x,
            lambda x: x + ' ',
        ]

    mapping = {}

    for token in important_tokens:
        for variation in variations:
            try:

                entry_ids = entry_tokenizer.encode(variation(token), add_special_tokens=False)

                output_ids = output_tokenizer.tokenize(token, add_special_tokens=False)


                if len(entry_ids) == 1 and len(output_ids) == 1:
                    mapping[entry_ids[0]] = output_ids[0].item()
                else:
                    pass
                    # print(f"Warning: Token '{token}' with variation '{variation(token)}' "
                    #       f"did not produce a consistent mapping.")
            except Exception as e:
                print(f"Error processing token '{token}' with variation '{variation(token)}': {e}")

    def translator(tokens):
        return torch.tensor([mapping.get(token.item(), output_tokenizer.vocab['pad']) for token in tokens.flatten()]).reshape(tokens.shape)

    return translator

important_tokens = [str(i) for i in range(10)] + ['+', '=']

translator = create_translator(tokenizer, compiled_tokenizer, important_tokens)
compiled_model(translator(tokenizer('2 + 2 = 4', return_tensors='pt')['input_ids']))

compiled_tokenizer('2 + 2 = 4')

wrapped_model = AddSupportModel(model, compiled_model, translator)
wrapped_model(**tokenizer('2 + 2 = 4', return_tensors='pt'))
