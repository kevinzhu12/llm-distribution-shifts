#!/usr/bin/env python
# Fine-tunes TinyLlama-1.1B-Chat on the GSM8K training split with LoRA
# Hugging Face  Transformers >= 4.39 and PEFT >= 0.10 are assumed.

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

MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
OUTPUT_DIR = Path("tinyllama-gsm8k-lora")

# 1. Load model & tokenizer (4-bit to keep VRAM ≤20 GB)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
tokenizer.pad_token = tokenizer.eos_token          # TinyLlama has no pad token

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    load_in_4bit=True,              # bitsandbytes 4-bit QLoRA
    device_map="auto",
)
model = prepare_model_for_kbit_training(model)

# 2. Attach LoRA adapters (rank 16 is plenty for 1 B params)
lora_cfg = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
    bias="none",
    task_type="CAUSAL_LM",
)
model = get_peft_model(model, lora_cfg)

# 3. Load GSM8K and build training texts
ds = load_dataset("gsm8k", "main")          # splits: train / test

def format_example(example):
    q = example["question"].strip()
    a = example["answer"].strip()           # answer already contains full CoT + #### 42
    return {
        "text": f"### Question:\n{q}\n\n### Answer:\n{a}"
    }

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
    output_dir=OUTPUT_DIR,
    fp16=True,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=8,     # effective 32
    learning_rate=2e-4,
    num_train_epochs=3,
    warmup_ratio=0.05,
    logging_steps=50,
    save_strategy="epoch",
    eval_strategy="no",
    report_to="none",
)

# 5. Train
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized,
    data_collator=data_collator,
)
trainer.train()

# 6. Save LoRA adapter & tokenizer only (few MB)
model.save_pretrained(OUTPUT_DIR / "adapter")
tokenizer.save_pretrained(OUTPUT_DIR)
