#!/usr/bin/env python
# Fine-tunes Qwen3-0.6B on GSM8K with 4-bit QLoRA adapters
# Requires: transformers>=4.39, datasets, peft>=0.10, bitsandbytes

from pathlib import Path
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    DataCollatorForLanguageModeling,
    TrainingArguments,
    Trainer,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

MODEL_NAME = "Qwen/Qwen3-0.6B"
OUTPUT_DIR = Path("qwen3-0.6B-gsm8k-lora")

# 1. Load tokenizer & model in 4-bit
tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME, 
    trust_remote_code=True, 
    use_fast=True
)
# Qwen doesn’t define a pad token, so reuse eos
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    trust_remote_code=True,
    load_in_4bit=True,      # bitsandbytes QLoRA
    device_map="auto",
)
# Prepare for k-bit training (freeze layers, cast norms, etc.)
model = prepare_model_for_kbit_training(model)

# 2. Attach LoRA adapters
lora_cfg = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
    bias="none",
    task_type="CAUSAL_LM",
)
model = get_peft_model(model, lora_cfg)

# 3. Load GSM8K and tokenize
ds = load_dataset("gsm8k", "main")  # splits: train / test

def format_example(example):
    q = example["question"].strip()
    a = example["answer"].strip()  # label includes CoT if you add it
    text = f"### Question:\n{q}\n\n### Answer:\n{a}"
    return {"text": text}

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

# 4. Training arguments
training_args = TrainingArguments(
    output_dir=str(OUTPUT_DIR),
    fp16=True,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=8,  # effective batch size = 32
    learning_rate=2e-4,
    num_train_epochs=3,
    warmup_ratio=0.05,
    logging_steps=50,
    save_strategy="epoch",
    eval_strategy="no",
    report_to="none",
)

# 5. Trainer and train
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized,
    data_collator=data_collator,
)
trainer.train()

# 6. Save only the LoRA adapter + tokenizer
model.save_pretrained(OUTPUT_DIR / "adapter")
tokenizer.save_pretrained(OUTPUT_DIR)
