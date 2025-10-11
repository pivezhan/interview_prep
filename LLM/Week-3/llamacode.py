
import os
import time
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    Trainer,
    TrainingArguments
)
from peft import LoraConfig, get_peft_model
from datasets import load_dataset

# -----------------------------
# Model & environment settings
# -----------------------------
model_id = "bigcode/starcoderbase-1b"
hf_auth_token = ""  # Replace with your HF token
device = "cuda:0"

print(f"Using device: {device}")

# Clear any stale GPU memory
torch.cuda.empty_cache()
print(f"Initial GPU memory allocated: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")
print(f"Initial GPU memory reserved: {torch.cuda.memory_reserved() / 1024**2:.2f} MB")

# -----------------------------
# 4-bit quantization config
# -----------------------------
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
)

# -----------------------------
# Dataset file check
# -----------------------------
script_dir = os.path.dirname(os.path.realpath(__file__))
dataset_file = os.path.join(script_dir, "profiling_data.jsonl")

if not os.path.exists(dataset_file):
    raise FileNotFoundError(
        f"Dataset file not found at: {dataset_file}. "
        "Please ensure 'profiling_data.jsonl' is in the same directory."
    )

# -----------------------------
# Load the base model
# -----------------------------
print("Loading base model...")
try:
    base_model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=bnb_config,
        device_map=device,
        token=hf_auth_token,
    )
    print("Base model loaded successfully!")
    print(f"GPU memory after loading: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")
except Exception as e:
    print(f"Error loading model: {type(e).__name__}: {str(e)}")
    raise

# Move the base model to GPU (may be redundant if device_map already covers this)
print("Placing the base model on GPU...")
base_model.cuda()

# -----------------------------
# Load tokenizer
# -----------------------------
print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(model_id, token=hf_auth_token)
tokenizer.pad_token = tokenizer.eos_token
print("Tokenizer loaded!")

# -----------------------------
# Test the original (base) model
# -----------------------------
def test_model(model, sample_prompt, max_new_tokens=100):
    """Generate text from a given model and prompt."""
    model.eval()
    inputs = tokenizer(sample_prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            top_p=0.9,
            top_k=50
        )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

print("Testing the original (base) model on a sample prompt...")
sample_prompt = (
    "<main.cpp: '#include <omp.h>\\nint main() { /* code */ }', "
    "util.cpp: '// utility functions', "
    "util.h: '// declarations', "
    "bash.sh: 'g++ -fopenmp main.cpp util.cpp -o app && ./app', "
    "platform(Intel i7, 8 cores, 2.8GHz, type=Performance)>"
)

original_output = test_model(base_model, sample_prompt)
print("=== Original Model Output ===")
print(original_output)
print("=============================\n")

# -----------------------------
# Prepare the LoRA model for fine-tuning
# -----------------------------
print("Applying LoRA to the base model...")
lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["c_attn"],  # StarCoder lumps Q/K/V into 'c_attn'
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)
lora_model = get_peft_model(base_model, lora_config)
print("LoRA applied!")

# -----------------------------
# Load dataset
# -----------------------------
print("Loading dataset for fine-tuning...")
dataset = load_dataset("json", data_files=dataset_file, split="train")
print("Dataset loaded!")

# -----------------------------
# Tokenize dataset
# -----------------------------
def tokenize_function(examples):
    return tokenizer(
        examples["prompt"],
        text_target=examples["completion"],
        truncation=True,
        padding="max_length",
        max_length=512
    )

print("Tokenizing dataset...")
tokenized_dataset = dataset.map(tokenize_function, batched=True)
tokenized_dataset.set_format("torch", columns=["input_ids", "attention_mask", "labels"])
print("Dataset tokenized!")

# -----------------------------
# Training arguments
# -----------------------------
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

# -----------------------------
# Initialize Trainer
# -----------------------------
print("Initializing Trainer with LoRA model...")
trainer = Trainer(
    model=lora_model,
    args=training_args,
    train_dataset=tokenized_dataset
)
print("Trainer initialized!")

# -----------------------------
# Fine-tune the model
# -----------------------------
print("Starting LoRA fine-tuning...")
start_time = time.time()
trainer.train()
end_time = time.time()
print(f"Fine-tuning time: {round(end_time - start_time, 3)} seconds.")

# -----------------------------
# Save final LoRA-finetuned model
# -----------------------------
print("Saving LoRA-finetuned model...")
lora_model.save_pretrained("./lora_finetuned_model")
tokenizer.save_pretrained("./lora_finetuned_model")
print("LoRA-finetuned model saved!")

# -----------------------------
# Test the new LoRA (instruct) model
# -----------------------------
print("Testing the newly LoRA-finetuned model on the same sample prompt...")
finetuned_output = test_model(lora_model, sample_prompt)
print("=== LoRA-Finetuned Model Output ===")
print(finetuned_output)
print("====================================\n")
