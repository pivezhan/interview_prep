#%%
import os
from datasets import load_dataset
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, GenerationConfig, TrainingArguments, Trainer
import torch
import time
import evaluate
import pandas as pd
import numpy as np

# 1. Environment setup
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
# Comment or adjust this line based on your GPU setup
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # Use GPU 0 (RTX 3050)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
if not torch.cuda.is_available():
    print("CUDA not available. Check PyTorch CUDA installation.")

# 2. Clear GPU memory
torch.cuda.empty_cache()
print(f"Initial GPU memory allocated: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")

# 3. Load dataset
huggingface_dataset_name = "knkarthick/dialogsum"
dataset = load_dataset(huggingface_dataset_name)

# 4. Load model and tokenizer
model_name = 'google/flan-t5-base'
original_model = AutoModelForSeq2SeqLM.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,
    use_cache=False
).to(device)  # Move to GPU
tokenizer = AutoTokenizer.from_pretrained(model_name)
original_model.config.use_cache = False

# 5. Enable gradient checkpointing
original_model.gradient_checkpointing_enable()

# Print model stats (unchanged)
def print_number_of_trainable_model_parameters(model):
    trainable_model_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    all_model_params = sum(p.numel() for p in model.parameters())
    return (
        f"trainable model parameters: {trainable_model_params}\n"
        f"all model parameters: {all_model_params}\n"
        f"percentage of trainable model parameters: {100 * trainable_model_params / all_model_params:.2f}%"
    )

print(print_number_of_trainable_model_parameters(original_model))

# 6. Tokenization function (unchanged)
def tokenize_function(example):
    start_prompt = 'Summarize the following conversation.\n\n'
    end_prompt = '\n\nSummary: '
    prompt = [start_prompt + dialogue + end_prompt for dialogue in example["dialogue"]]
    
    tokenized_inputs = tokenizer(
        prompt,
        padding="max_length",
        max_length=256,
        truncation=True,
        return_tensors="pt"
    )
    tokenized_labels = tokenizer(
        example["summary"],
        padding="max_length",
        max_length=128,
        truncation=True,
        return_tensors="pt"
    )
    
    return {
        'input_ids': tokenized_inputs.input_ids,
        'labels': tokenized_labels.input_ids
    }

# 7. Apply tokenization (unchanged)
tokenized_datasets = dataset.map(tokenize_function, batched=True)
tokenized_datasets = tokenized_datasets.remove_columns(['id', 'dialogue', 'summary'])
tokenized_datasets = tokenized_datasets.filter(
    lambda example, idx: idx % 10 == 0,
    with_indices=True
)

print("Shapes of the datasets:")
print(f"Training: {tokenized_datasets['train'].shape}")
print(f"Validation: {tokenized_datasets['validation'].shape}")
print(f"Test: {tokenized_datasets['test'].shape}")

# # 8. Trainer setup (unchanged)
output_dir = f'./dialogue-summary-training'
num_train_epochs = 20
# training_args = TrainingArguments(
#     output_dir=output_dir,
#     learning_rate=5e-5,
#     num_train_epochs=5,
#     weight_decay=0.01,
#     logging_steps=5,
#     max_steps=10,
#     per_device_train_batch_size=1,
#     per_device_eval_batch_size=1,
#     gradient_accumulation_steps=4,
#     bf16=True,
#     gradient_checkpointing=True,
#     report_to="none"
# )

# Training arguments with validation and longer training
training_args = TrainingArguments(
    output_dir=output_dir,
    learning_rate=1e-4,              # Adjusted LR
    num_train_epochs=1,              # More epochs
    weight_decay=0.01,
    logging_steps=5,
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    gradient_accumulation_steps=4,
    bf16=True,
    gradient_checkpointing=True,
    evaluation_strategy="steps",     # Validate during training
    eval_steps=5,
    lr_scheduler_type="linear",      # Add scheduler
    warmup_steps=5,                  # Warm-up
    report_to="none"
)



trainer = Trainer(
    model=original_model,
    args=training_args,
    train_dataset=tokenized_datasets['train'],
    eval_dataset=tokenized_datasets['validation']
)

# 9. Train the model
print(f"GPU memory allocated before training: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")
trainer.train()
print("Training Finished....")
# *** SAVE the newly fine-tuned model for later use/comparison ***
trainer.save_model(f"./flan-dialogue-summary-original-checkpoint_"+str(num_train_epochs))
tokenizer.save_pretrained(f"./flan-dialogue-summary-original-checkpoint_"+str(num_train_epochs))

instruct_model = AutoModelForSeq2SeqLM.from_pretrained("./flan-dialogue-summary-checkpoint", torch_dtype=torch.bfloat16)
original_model = AutoModelForSeq2SeqLM.from_pretrained(f"./flan-dialogue-summary-original-checkpoint_"+str(num_train_epochs), torch_dtype=torch.bfloat16)

# Assuming tokenizer, original_model, instruct_model, and GenerationConfig are already defined
index = 200
dialogue = dataset['test'][index]['dialogue']
human_baseline_summary = dataset['test'][index]['summary']

prompt = f"""
Summarize the following conversation.

{dialogue}

Summary:
"""

# Tokenize the prompt and move input_ids to the GPU
input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to("cuda:0")

# Ensure the models are on the GPU (do this once during initialization, not every time)
original_model = original_model.to("cuda:0")
instruct_model = instruct_model.to("cuda:0")

# Generate outputs
original_model_outputs = original_model.generate(
    input_ids=input_ids, 
    generation_config=GenerationConfig(max_new_tokens=200, num_beams=1)
)
original_model_text_output = tokenizer.decode(original_model_outputs[0], skip_special_tokens=True)

instruct_model_outputs = instruct_model.generate(
    input_ids=input_ids, 
    generation_config=GenerationConfig(max_new_tokens=200, num_beams=1)
)
instruct_model_text_output = tokenizer.decode(instruct_model_outputs[0], skip_special_tokens=True)

# Assuming dash_line is defined somewhere, e.g., dash_line = "-" * 50
dash_line = '-'.join('' for x in range(100))
print(dash_line)
print(f'BASELINE HUMAN SUMMARY:\n{human_baseline_summary}')
print(dash_line)
print(f'ORIGINAL MODEL:\n{original_model_text_output}')
print(dash_line)
print(f'INSTRUCT MODEL:\n{instruct_model_text_output}')

rouge = evaluate.load('rouge')

import torch
import pandas as pd
from transformers import GenerationConfig

# Assuming dataset, tokenizer, original_model, and instruct_model are already defined

# Move models to GPU (do this once at initialization, not in the loop)
original_model = original_model.to("cuda:0")
instruct_model = instruct_model.to("cuda:0")

# Get dialogues and human baseline summaries
dialogues = dataset['test'][0:10]['dialogue']
human_baseline_summaries = dataset['test'][0:10]['summary']

original_model_summaries = []
instruct_model_summaries = []

# Process each dialogue
for dialogue in dialogues:  # Simplified enumeration since index isn't used
    prompt = f"""
Summarize the following conversation.

{dialogue}

Summary: """
    # Tokenize and move input_ids to GPU
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to("cuda:0")

    # Generate outputs from original model
    original_model_outputs = original_model.generate(
        input_ids=input_ids, 
        generation_config=GenerationConfig(max_new_tokens=200)
    )
    original_model_text_output = tokenizer.decode(original_model_outputs[0], skip_special_tokens=True)
    original_model_summaries.append(original_model_text_output)

    # Generate outputs from instruct model
    instruct_model_outputs = instruct_model.generate(
        input_ids=input_ids, 
        generation_config=GenerationConfig(max_new_tokens=200)
    )
    instruct_model_text_output = tokenizer.decode(instruct_model_outputs[0], skip_special_tokens=True)
    instruct_model_summaries.append(instruct_model_text_output)

# Zip summaries and create DataFrame
zipped_summaries = list(zip(human_baseline_summaries, original_model_summaries, instruct_model_summaries))
df = pd.DataFrame(zipped_summaries, columns=['human_baseline_summaries', 'original_model_summaries', 'instruct_model_summaries'])

# Display the DataFrame
print(df)


original_model_results = rouge.compute(
    predictions=original_model_summaries,
    references=human_baseline_summaries[0:len(original_model_summaries)],
    use_aggregator=True,
    use_stemmer=True,
)

instruct_model_results = rouge.compute(
    predictions=instruct_model_summaries,
    references=human_baseline_summaries[0:len(instruct_model_summaries)],
    use_aggregator=True,
    use_stemmer=True,
)

print('ORIGINAL MODEL:')
print(original_model_results)
print('INSTRUCT MODEL:')
print(instruct_model_results)


# results = pd.read_csv("data/dialogue-summary-training-results.csv")

# human_baseline_summaries = results['human_baseline_summaries'].values
# original_model_summaries = results['original_model_summaries'].values
# instruct_model_summaries = results['instruct_model_summaries'].values

# original_model_results = rouge.compute(
#     predictions=original_model_summaries,
#     references=human_baseline_summaries[0:len(original_model_summaries)],
#     use_aggregator=True,
#     use_stemmer=True,
# )

# instruct_model_results = rouge.compute(
#     predictions=instruct_model_summaries,
#     references=human_baseline_summaries[0:len(instruct_model_summaries)],
#     use_aggregator=True,
#     use_stemmer=True,
# )

# print('ORIGINAL MODEL:')
# print(original_model_results)
# print('INSTRUCT MODEL:')
# print(instruct_model_results)


print("Absolute percentage improvement of INSTRUCT MODEL over HUMAN BASELINE")

improvement = (np.array(list(instruct_model_results.values())) - np.array(list(original_model_results.values())))
for key, value in zip(instruct_model_results.keys(), improvement):
    print(f'{key}: {value*100:.2f}%')

#%%
from peft import LoraConfig, get_peft_model, TaskType

lora_config = LoraConfig(
    r=32, # Rank
    lora_alpha=32,
    target_modules=["q", "v"],
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.SEQ_2_SEQ_LM # FLAN-T5
)


peft_model = get_peft_model(original_model, 
                            lora_config)
print(print_number_of_trainable_model_parameters(peft_model))

output_dir = f'./peft-dialogue-summary-training-{str(int(time.time()))}'

peft_training_args = TrainingArguments(
    output_dir=output_dir,
    auto_find_batch_size=True,
    learning_rate=1e-3, # Higher learning rate than full fine-tuning.
    num_train_epochs=1,
    logging_steps=1,
    max_steps=1    
)
    
peft_trainer = Trainer(
    model=peft_model,
    args=peft_training_args,
    train_dataset=tokenized_datasets["train"],
)
peft_trainer.train()

peft_model_path="./peft-dialogue-summary-checkpoint-local"

peft_trainer.model.save_pretrained(peft_model_path)
tokenizer.save_pretrained(peft_model_path)


from peft import PeftModel, PeftConfig

peft_model_base = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-base", torch_dtype=torch.bfloat16)
tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-base")

peft_model = PeftModel.from_pretrained(peft_model_base, 
                                       './peft-dialogue-summary-checkpoint-from-s3/', 
                                       torch_dtype=torch.bfloat16,
                                       is_trainable=False)

print(print_number_of_trainable_model_parameters(peft_model))


from transformers import GenerationConfig
import torch

# Assuming dataset, tokenizer, original_model, instruct_model, and peft_model are already loaded
# Define dash_line (if not already defined)
dash_line = "-" * 50

# Set device (consistent with your training setup)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Move all models to the device (do this once after loading, not repeatedly)
original_model = original_model.to(device)
instruct_model = instruct_model.to(device)
peft_model = peft_model.to(device)

# Select dialogue and baseline summary
index = 200
dialogue = dataset['test'][index]['dialogue']
baseline_human_summary = dataset['test'][index]['summary']

# Create prompt
prompt = f"""
Summarize the following conversation.

{dialogue}

Summary: """

# Tokenize and move input_ids to the device
input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)

# Generate outputs
original_model_outputs = original_model.generate(
    input_ids=input_ids,
    generation_config=GenerationConfig(max_new_tokens=200, num_beams=1)
)
original_model_text_output = tokenizer.decode(original_model_outputs[0], skip_special_tokens=True)

instruct_model_outputs = instruct_model.generate(
    input_ids=input_ids,
    generation_config=GenerationConfig(max_new_tokens=200, num_beams=1)
)
instruct_model_text_output = tokenizer.decode(instruct_model_outputs[0], skip_special_tokens=True)

peft_model_outputs = peft_model.generate(
    input_ids=input_ids,
    generation_config=GenerationConfig(max_new_tokens=200, num_beams=1)
)
peft_model_text_output = tokenizer.decode(peft_model_outputs[0], skip_special_tokens=True)

# Print results
print(dash_line)
print(f'BASELINE HUMAN SUMMARY:\n{baseline_human_summary}')
print(dash_line)
print(f'ORIGINAL MODEL:\n{original_model_text_output}')
print(dash_line)
print(f'INSTRUCT MODEL:\n{instruct_model_text_output}')
print(dash_line)
print(f'PEFT MODEL:\n{peft_model_text_output}')