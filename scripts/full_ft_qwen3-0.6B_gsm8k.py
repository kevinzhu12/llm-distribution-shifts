#!/usr/bin/env python
# Full fine-tunes Qwen3-0.6B on GSM8K
# Requires: transformers>=4.39, datasets

from pathlib import Path
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    DataCollatorForLanguageModeling,
    TrainingArguments,
    Trainer,
)
import torch

MODEL_NAME = "Qwen/Qwen3-0.6B"
OUTPUT_DIR = Path("qwen3-0.6B-gsm8k-full")

# 1. Load tokenizer & model in fp16
print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME, 
    trust_remote_code=True, 
    use_fast=True
)
# Qwen doesn't define a pad token, so reuse eos
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

print("Loading model in fp16...")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    trust_remote_code=True,
    torch_dtype=torch.float16,  # Use fp16 instead of 4-bit quantization
    device_map="auto",
)

# 2. Load GSM8K and tokenize
print("Loading GSM8K dataset...")
ds = load_dataset("gsm8k", "main")  # splits: train / test

def format_example(example):
    q = example["question"].strip()
    a = example["answer"].strip()  # label includes CoT if you add it
    text = f"### Question:\n{q}\n\n### Answer:\n{a}"
    return {"text": text}

print("Tokenizing dataset...")
tok_kwargs = dict(
    truncation=True,
    max_length=1024,
    padding="max_length",
)

tokenized = (
    ds["train"]
    .map(format_example, remove_columns=ds["train"].column_names)
    .map(lambda ex: tokenizer(ex["text"], **tok_kwargs), batched=True)
)

data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)

# 3. Training arguments
# Note: Full fine-tuning requires more memory, so we use smaller batch sizes
training_args = TrainingArguments(
    output_dir=str(OUTPUT_DIR),
    bf16=True,  # Use bfloat16 instead of fp16
    fp16=False, # Make sure fp16 is off
    # Batch size configuration:
    # - per_device_train_batch_size=8: Process 8 examples at once on each GPU
    # - gradient_accumulation_steps=4: Accumulate gradients for 4 steps
    # - Effective batch size = 8 * 4 = 32 examples per update
    per_device_train_batch_size=8,  # Increased from 2 to 8
    gradient_accumulation_steps=4,   # Decreased from 16 to 4 to maintain effective batch size
    learning_rate=1e-5,  # Lower learning rate for full fine-tuning
    num_train_epochs=3,
    warmup_ratio=0.05,
    logging_steps=50,
    save_strategy="epoch",
    eval_strategy="no",
    report_to="none",
    # Add gradient checkpointing to save memory
    gradient_checkpointing=True,
    # Add weight decay to prevent overfitting
    weight_decay=0.01,
)

# 4. Trainer and train
print("Starting training...")
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized,
    data_collator=data_collator,
)
trainer.train()

# 5. Save the full model and tokenizer
print("Saving model and tokenizer...")
model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
print("Done!") 