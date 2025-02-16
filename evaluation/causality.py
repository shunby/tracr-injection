import torch
from tqdm import tqdm
import os
from transformers import AutoModelForCausalLM, AutoTokenizer
from utils import get_tokenizer, MODEL_NAME_BY_TASK
from data import get_task
from compile_model.load_compiled_model import load_model
import random

def evaluate_task_causality(model, tokenizer, data, subspace, layer=7, noise_norm=0.3):
    model.eval()
    correct = 0
    total = 0
    projection = get_projection(subspace)
    hook = get_hook(projection, noise_norm)
    hook_handle = model.transformer.h[layer].register_forward_hook(hook)

    for prompt, label in tqdm(data):
        if hasattr(model, 'prefix'):
            prompt = model.prefix + prompt
        input_ids = tokenizer.encode(prompt, return_tensors='pt').to(model.device)

        target_token = ' ' + str(label)
        target_ids = tokenizer.encode(target_token, add_special_tokens=False)
        target_len = len(target_ids)

        with torch.no_grad():
            outputs = model.generate(
                input_ids=input_ids,
                max_length=input_ids.size(1) + target_len,
                do_sample=False
            )

        generated_ids = outputs[0][input_ids.size(1):]

        if torch.equal(generated_ids, torch.tensor(target_ids).to(model.device)):
            correct += 1
        total += 1

    hook_handle.remove()
    accuracy = correct / total if total > 0 else 0
    return accuracy



def get_projection(subspace):
    with torch.no_grad():
        W = subspace
        M = W.t()
        MtM = M.t() @ M
        eps = 1e-7
        eye = torch.eye(MtM.size(0), device=MtM.device)
        MtM_inv = torch.inverse(MtM + eps * eye)
        P = M @ MtM_inv @ M.t()
    return P

def get_hook(projection, norm, complement=False):
    def hook(module, input, output, projection=projection, norm=norm, complement=complement):
        hidden_states, *args = output
        noise = torch.randn(hidden_states.size(0),hidden_states.size(1), hidden_states.size(-1), device=hidden_states.device)
        if complement:
            noise_projected = noise - (noise @ projection)
        else:
            noise_projected = noise @ projection
        current_norms = noise_projected.norm(dim=-1, keepdim=True) + 1e-7
        noise_scaled = noise_projected * (norm / current_norms)
        hidden_states[:, :, :] = hidden_states[:, :, :] + noise_scaled
        return (hidden_states, *args)
    return hook


def get_results_for_task(save_id, task, device):
    CACHE_DIR = os.path.expanduser(os.getenv('CACHE_DIR'))
    MODEL_NAME = os.path.join(CACHE_DIR, save_id)
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, cache_dir=CACHE_DIR).to(device)
    tokenizer = get_tokenizer(task)
    compiled_model_file_name = MODEL_NAME_BY_TASK[task]
    _, test_set = get_task(task)
    compiled_model, compiled_tokenizer, decoder = load_model(filename=compiled_model_file_name)

    adapter_path = os.path.join(MODEL_NAME, "adapter.pth")
    adapter = torch.nn.Linear(model.config.hidden_size, compiled_model.model_dim)
    adapter.load_state_dict(torch.load(adapter_path))
    adapter.to(device)

    layer = len(compiled_model.layers)

    results = []
    for noise in [0, 5, 10, 30]:
        for r in [True, False]:
            dims = compiled_model.final_result_dimensions()
            subspace = adapter.weight[dims, :]
            if r:
                subspace = torch.rand(subspace.shape).to(device)

            acc = evaluate_task_causality(model, tokenizer, test_set, subspace, layer, noise)
            results.append((acc, noise, r))

    return results


if __name__ == '__main__':
    torch.manual_seed(42)
    random.seed(42)

    task = 'int-sum'
    save_id = '291d3657-f099-4e7c-a0c4-7912981efea6'
    device = 'cuda:4'

    results = get_results_for_task(save_id, task, device)

    print('results', results)
