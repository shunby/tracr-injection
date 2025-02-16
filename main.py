import comet_ml
import argparse
import torch
import pickle
import os
import uuid
import random

from dotenv import load_dotenv
from data import get_task, get_fineweb, get_mawps
from methods.get_method import get_method
from transformers import AutoModelForCausalLM
from evaluation import run_evaluations, evaluate_task, run_ood_evals
from utils import get_tokenizer, save_results, MAX_LENGTH_BY_TASK, MODEL_NAME_BY_TASK

load_dotenv()

save_id = str(uuid.uuid4())
parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=42)
parser.add_argument('--task', type=str, default='dyck')
parser.add_argument('--save_id', type=str, default=None)
parser.add_argument('--model', type=str, default='gpt2', choices=['gpt2-large', 'gpt2-xl'])
parser.add_argument('--inject', type=str, default='DISTIL')
parser.add_argument('--read_from_support', type=lambda x: str(x).lower() == 'true', default=True)
parser.add_argument('--write_to_support', type=lambda x: str(x).lower() == 'true', default=False)
parser.add_argument('--tanh_in_write', type=lambda x: str(x).lower() == 'true', default=False)
parser.add_argument('--batch_size', type=int, default=2)
parser.add_argument('--max_sequence_length', type=int, default=30)
parser.add_argument('--lr', type=float, default=1e-6)
parser.add_argument('--natural_data_loss', type=lambda x: str(x).lower() == 'true', default=False)
parser.add_argument('--residue_loss', type=lambda x: str(x).lower() == 'true', default=False)
parser.add_argument('--train_batches', type=int, default=20000)
parser.add_argument('--cross_attn_every_x_layers', type=int, default=1)
parser.add_argument('--attention_heads', type=int, default=8)
parser.add_argument('--key_size', type=int, default=60)
parser.add_argument('--rescaling_factor_write', type=int, default=1)
parser.add_argument('--pad_communication', type=int, default=0)
parser.add_argument('--compiled_model_loss', type=lambda x: str(x).lower() == 'true', default=False)
parser.add_argument('--adapter_path', type=str, default=None)
parser.add_argument('--saving_id', type=str, default=save_id)
parser.add_argument('--detach_and_roll', type=lambda x: str(x).lower() == 'true', default=True)
parser.add_argument('--algorithm_loss', type=lambda x: str(x).lower() == 'true', default=True)
parser.add_argument('--unsupervised_loss', type=lambda x: str(x).lower() == 'true', default=True)
parser.add_argument('--ood', type=lambda x: str(x).lower() == 'true', default=False)
parser.add_argument('--ood_new_token', type=lambda x: str(x).lower() == 'true', default=False)
parser.add_argument('--ood2', type=lambda x: str(x).lower() == 'true', default=False)
parser.add_argument('--ood_length', type=lambda x: str(x).lower() == 'true', default=False)
parser.add_argument('--only_result_subspace', type=lambda x: str(x).lower() == 'true', default=True)
parser.add_argument('--algorithm_loss_multiplier', type=int, default=1)
args = parser.parse_args()

seed = args.seed
torch.manual_seed(seed)
random.seed(seed)

CACHE_DIR = os.path.expanduser(os.getenv('CACHE_DIR'))
TASK = args.task
if args.save_id:
    MODEL_NAME = os.path.join(CACHE_DIR, args.save_id)
    args.adapter_path = os.path.join(MODEL_NAME, "adapter.pth")
else:
    MODEL_NAME = args.model

METHOD = args.inject

args.max_sequence_length_supervised = MAX_LENGTH_BY_TASK[TASK]
args.compiled_model_file_name = MODEL_NAME_BY_TASK[TASK]

train_dataset, test_dataset = get_task(TASK)
unsupervised_data = get_fineweb()
train_natural_data, test_natural_data = get_mawps()

model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, cache_dir=CACHE_DIR)

tokenizer = get_tokenizer(args.task)

tokenizer.pad_token = tokenizer.eos_token
method = get_method(METHOD)

model, adapter = method(
    model,
    tokenizer,
    train_dataset,
    unsupervised_data,
    train_natural_data,
    compiled_model_file_name=args.compiled_model_file_name,
    params=args
)

save_path = os.path.join(CACHE_DIR, save_id)
os.makedirs(save_path, exist_ok=True)
model.save_pretrained(save_path)
adapter_save_path = os.path.join(save_path, "adapter.pth")
torch.save(adapter.state_dict(), adapter_save_path)
print('model weights saved at', save_id)


model.to('cuda')
accuracy = evaluate_task(model, tokenizer, test_dataset)

print('task', TASK)
print('method', METHOD)
print('model', MODEL_NAME)
print('accuracy', accuracy)

save_results({
    'task': TASK,
    'method': METHOD,
    'model': MODEL_NAME,
    'accuracy': accuracy,
    'save_id': save_id
} | vars(args))

evals = run_evaluations(model, tokenizer)
ood_evals = run_ood_evals(model, tokenizer, TASK)

save_results({
    'task': TASK,
    'method': METHOD,
    'model': MODEL_NAME,
    'accuracy': accuracy,
    'save_id': save_id
} | vars(args) | evals | ood_evals)
