
#%%
import time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model
from datasets import load_dataset
from transformers import Trainer, TrainingArguments

# Model and device
model_id = "codellama/CodeLlama-7b-hf"
device = "cuda:0"
print(f"Using device: {device}")

# 4-bit config
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
)

# Load the model fully onto GPU
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    quantization_config=bnb_config,
    device_map=None,            # Disable auto device map
    torch_dtype=torch.bfloat16, # Could also use torch.float16
)
model.cuda()  # Force entire model to GPU

# Tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_id)
tokenizer.pad_token = tokenizer.eos_token

# LoRA config
lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

model = get_peft_model(model, lora_config)

# Load and tokenize dataset
dataset = load_dataset("json", data_files="profiling_data.jsonl", split="train")

def tokenize_function(examples):
    return tokenizer(
        examples["prompt"],
        text_target=examples["completion"],
        truncation=True,
        padding="max_length",
        max_length=512
    )

tokenized_dataset = dataset.map(tokenize_function, batched=True)
tokenized_dataset.set_format("torch", columns=["input_ids", "attention_mask", "labels"])

# Training arguments
training_args = TrainingArguments(
    output_dir="./lora_finetuned",
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,
    num_train_epochs=3,
    learning_rate=2e-4,
    fp16=True,
    logging_steps=5,
    save_steps=50,
    max_steps=100
)

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
)

# Fine-tune
time_1 = time.time()
trainer.train()
time_2 = time.time()
print(f"Fine-tuning time: {round(time_2 - time_1, 3)} seconds.")

# Save
model.save_pretrained("./lora_finetuned_model")
tokenizer.save_pretrained("./lora_finetuned_model")

#%%
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.bfloat16
)

model = AutoModelForCausalLM.from_pretrained("./lora_finetuned_model", quantization_config=bnb_config, device_map="auto")
tokenizer = AutoTokenizer.from_pretrained("./lora_finetuned_model")

prompt = "<#include <omp.h>\nint main() { int i; #pragma omp parallel for\nfor(i=0;i<1000;i++) { printf(\"%d\",i); } }, , , gcc -fopenmp -o prog main.cpp, platform(NVIDIA GPU Laptop, 4 cores, 2.5GHz, Turing)>"
inputs = tokenizer(prompt, return_tensors="pt").to("cuda:0")
outputs = model.generate(**inputs, max_new_tokens=100)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
