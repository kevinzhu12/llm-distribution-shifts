## eval
import argparse, re, torch
from pathlib import Path
from tqdm.auto import tqdm
from datasets import load_dataset

from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel, prepare_model_for_kbit_training
import math

_NUM_RE = re.compile(r"""
    (?<!\w)                # not preceded by a letter/underscore
    [-+]?                  # optional sign
    (?:\d{1,3}(?:[,_ ]\d{3})+|\d+)    # 1,234 or 1_234 or 1234
    (?:\.\d+)?             # optional decimal part
    (?!\w)                 # not followed by a letter/underscore
""", re.VERBOSE)

def last_number(text: str) -> str | None:
    """
    Extract the *last* numeric substring in `text`.

    - Accepts commas, underscores or spaces as thousands separators (e.g. '1,234,567').
    - Returns a plain digit string with separators removed, ready for float() / int().
    """
    matches = _NUM_RE.findall(text)
    if not matches:
        return None
    raw = matches[-1]
    cleaned = raw.replace(",", "").replace("_", "").replace(" ", "")  # drop separators
    # Optional: drop leading zeros so '0010' matches '10'
    cleaned = cleaned.lstrip("0") or "0"
    return cleaned

def build_svamp_prompt(ex):
    if "question_concat" in ex:          # mirror with pre-joined field
        qtext = ex["question_concat"]
    else:                                # ChilleD mirror → join Body + Question
        qtext = f"{ex['Body'].strip()} {ex['Question'].strip()}"
    return f"### Question:\n{qtext}\n\n### Answer:\n"

@torch.inference_mode()
def accuracy(model, tok, dataset, prompt_fn, gold_fn, n=None, batch_size=8):
    model.eval()
    device = next(model.parameters()).device
    total, correct = 0, 0

    iterator = dataset if not n else dataset.select(range(n))
    examples = list(iterator)
    num_examples = len(examples)
    print(f"Starting evaluation: {num_examples} examples, batch size {batch_size}")

    for start_idx in tqdm(range(0, num_examples, batch_size), desc="Evaluating"):
        batch = examples[start_idx:start_idx+batch_size]
        prompts = [prompt_fn(ex) for ex in batch]
        inputs = tok(prompts, return_tensors="pt", padding=True).to(device)

        out_ids = model.generate(
            **inputs,
            max_new_tokens=256,
            do_sample=False,
        )

        for i, ex in enumerate(batch):
            output_text = tok.decode(out_ids[i], skip_special_tokens=True)
            pred = last_number(output_text)
            gold = gold_fn(ex)
            if pred is not None and gold is not None and math.isclose(float(pred), float(gold)):
                correct += 1
            total += 1
            if (total % 50 == 0) or (total == num_examples):
                print(f"Progress: {total}/{num_examples} examples evaluated. Current accuracy: {correct/total:.4f}")

    print(f"Finished evaluation: {correct}/{total} correct. Final accuracy: {correct/total if total else 0.0:.4f}")
    return correct / total if total else 0.0



gsm8k = load_dataset("gsm8k", "main", split="test")
svamp = load_dataset("ChilleD/SVAMP", split="test") 

# 2. base model (fp16)
print("🔹 Loading TinyLlama-1.1B base …")
print("Loading TinyLlama-1.1B base model...")
base = AutoModelForCausalLM.from_pretrained(
    "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    torch_dtype=torch.float16,
    device_map="auto",
)
base_tok = AutoTokenizer.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0")
base_tok.pad_token = base_tok.eos_token
print("Base model loaded.")

# 3. LoRA‐fine-tuned (4-bit)
print("🔹 Loading LoRA adapter …")
print("Loading LoRA adapter...")
bnb_cfg = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
)
lora_base = AutoModelForCausalLM.from_pretrained(
    "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    quantization_config=bnb_cfg,
    device_map="auto",
)
lora_base = prepare_model_for_kbit_training(lora_base)

lora_model = PeftModel.from_pretrained(
    lora_base,
    "../outputs/tinyllama-gsm8k-lora/adapter",   # ← your path
    is_trainable=False,  # inference only
)
lora_tok = AutoTokenizer.from_pretrained("../outputs/tinyllama-gsm8k-lora")
lora_tok.pad_token = lora_tok.eos_token
print("LoRA adapter loaded.")


# 4. evaluate
print("\n🚀 Evaluating BASE model …")
print("Evaluating BASE model on GSM8K...")
acc_base_gsm  = accuracy(base,  base_tok, gsm8k,
                         prompt_fn=lambda ex: f"### Question:\n{ex['question'].strip()}\n\n### Answer:\n",
                        gold_fn   =lambda ex: last_number(ex["answer"]),
                        )
print(f"BASE model GSM8K accuracy: {acc_base_gsm:.4f}")

print("Evaluating BASE model on SVAMP...")
acc_base_svam = accuracy(base,  base_tok, svamp,
                         prompt_fn=build_svamp_prompt,
                         gold_fn  =lambda ex: str(ex["Answer"]).strip()
                        )
print(f"BASE model SVAMP accuracy: {acc_base_svam:.4f}")


print("\n🚀 Evaluating LoRA-FINETUNED model …")
print("Evaluating LoRA-FINETUNED model on GSM8K...")
acc_lora_gsm  = accuracy(lora_model, lora_tok, gsm8k,
                         prompt_fn=lambda ex: f"### Question:\n{ex['question'].strip()}\n\n### Answer:\n",
                        gold_fn   =lambda ex: last_number(ex["answer"]),
                        )
print(f"LoRA-FINETUNED model GSM8K accuracy: {acc_lora_gsm:.4f}")

print("Evaluating LoRA-FINETUNED model on SVAMP...")
acc_lora_svam = accuracy(lora_model, lora_tok, svamp,
                         prompt_fn=build_svamp_prompt,
                         gold_fn  =lambda ex: str(ex["Answer"]).strip()
                        )
print(f"LoRA-FINETUNED model SVAMP accuracy: {acc_lora_svam:.4f}")

# 5. report
print("\n════════════ ACCURACY ════════════")
print(f"{'Model':<15} | {'GSM8K':>7} | {'SVAMP':>7}")
print("-" * 33)
print(f"{'BASE':<15} | {acc_base_gsm*100:6.2f}% | {acc_base_svam*100:6.2f}%")
print(f"{'LoRA-FT':<15} | {acc_lora_gsm*100:6.2f}% | {acc_lora_svam*100:6.2f}%")