from dotenv import load_dotenv
from transformers import AutoModelForCausalLM, AutoTokenizer
import os
import torch
from compile_model.load_compiled_model import load_model
import torch.nn as nn
import torch.nn.functional as F
from support_model.self_attention_module import SelfAttentionModule
from support_model.add_support_model import AddSupportModel
from support_model.create_tokenizer_translator import create_translator

load_dotenv()

cache_dir = os.path.expanduser(os.getenv('CACHE_DIR'))
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


compiled_model, compiled_tokenizer = load_model()


important_tokens = [str(i) for i in range(10)] + ['+', '=']
translator = create_translator(tokenizer, compiled_tokenizer, important_tokens)
compiled_model(translator(tokenizer('2 + 2 = 4', return_tensors='pt')['input_ids']))
compiled_tokenizer('2 + 2 = 4')
# wrapped_model = AddSupportModel(model, compiled_model, translator)
# wrapped_model(**tokenizer('2 + 2 = 4', return_tensors='pt'))

# wrapped_model.main_model


# wrapped_model.train_only_cross_attention()
# wrapped_model.trainable_parameters()

def count_parameters(parameters):
    return sum(p.numel() for p in parameters if p.requires_grad)

# count_parameters(wrapped_model.trainable_parameters())

########################


from tqdm import tqdm
import random

random.seed(42)

def generate_arithmetic_data(num_samples=1000):
    train_data = []
    test_data = []
    operators = ['+']
    for _ in range(int(num_samples * 0.8)):
        num1 = random.randint(0, 99)
        num2 = random.randint(0, 99)
        op = random.choice(operators)
        result = eval(f"{num1} {op} {num2}")
        train_data.append(f"{num1} {op} {num2} = {result}")
    for _ in range(int(num_samples * 0.2)):
        num1 = random.randint(0, 99)
        num2 = random.randint(0, 99)
        op = random.choice(operators)
        result = eval(f"{num1} {op} {num2}")
        test_data.append(f"{num1} {op} {num2} = {result}")
    
    return train_data, test_data

train_data, test_data = generate_arithmetic_data()

def tokenize_data(data, tokenizer):
    return tokenizer(data, return_tensors='pt', padding=True, truncation=True)

train_tokens = tokenize_data(train_data, tokenizer)
test_tokens = tokenize_data(test_data, tokenizer)

from peft import LoraConfig, get_peft_model

lora_config = LoraConfig(
    r=12,
    target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    bias="none",
    task_type="CAUSAL_LM"
)
lora_model = get_peft_model(model, lora_config)
count_parameters(lora_model.parameters())
model = lora_model

# optimizer = torch.optim.Adam(wrapped_model.trainable_parameters(), lr=5e-5)
model.train()
optimizer = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=5e-5)
criterion = nn.CrossEntropyLoss()

device = torch.device('cuda')
# device = torch.device('cpu')
# wrapped_model.to(device)
model.to(device)
epochs = 1
# wrapped_model.train()
batch_size = 16


losses = []
for epoch in range(epochs):
    epoch_loss = 0.0

    indices = torch.randperm(len(train_data))
    train_tokens = {key: val[indices] for key, val in train_tokens.items()}

    for i in tqdm(range(0, len(train_data), batch_size)):  # Batch size of 16
        inputs = {key: val[i:i+batch_size].to(model.device) for key, val in train_tokens.items()}
        labels = inputs["input_ids"]
        
        optimizer.zero_grad()
        # outputs = wrapped_model(**inputs)
        outputs = model(**inputs)
        logits = outputs.logits[:, :-1].contiguous()
        labels = labels[:, 1:].contiguous()
        
        loss = criterion(logits.view(-1, logits.size(-1)), labels.view(-1))
        loss.backward()
        optimizer.step()
        
        epoch_loss += loss.item()

        # if i %5 == 0 and i > 5:
        print(epoch_loss / (i +1))
        losses.append(epoch_loss / (i+1))

    
    print(f"Epoch {epoch + 1}/{epochs} - Loss: {epoch_loss:.4f}")


losses

with open("losses.txt", "w") as file:
    for loss in losses:
        file.write(f"{loss}\n")
