
import json
from pathlib import Path

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, set_seed
from peft import PeftModel
from tqdm.auto import tqdm

ROOT = Path("/kaggle/working/small-model-alignment-starter")
MODEL_ID = "HuggingFaceTB/SmolLM2-360M-Instruct"
ADAPTER = ROOT / "outputs/persona_lora/best_adapter"
DESTINATION = ROOT / "outputs/persona_lora/heldout_generations.json"

set_seed(42)

tokenizer = AutoTokenizer.from_pretrained(ADAPTER)
base = AutoModelForCausalLM.from_pretrained(
    MODEL_ID, dtype=torch.float16
).to("cuda:0")
model = PeftModel.from_pretrained(base, ADAPTER)
model.eval()
model.config.use_cache = True

rows = json.loads(
    (ROOT / "data/persona_lora/val_seed42.json").read_text()
)
assert len(rows) == 22

settings = {
    "max_new_tokens": 192,
    "do_sample": False,
    "repetition_penalty": 1.1,
    "no_repeat_ngram_size": 4,
}

@torch.inference_mode()
def generate(messages):
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    inputs = tokenizer(
        text,
        add_special_tokens=False,
        return_tensors="pt",
    ).to("cuda:0")

    output = model.generate(
        **inputs,
        **settings,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    response = tokenizer.decode(
        output[0, inputs.input_ids.shape[1]:],
        skip_special_tokens=True,
    ).strip()
    return response

metadata = {
    "base_model": MODEL_ID,
    "adapter": str(ADAPTER),
    "split": "validation",
    "seed": 42,
    "generation_settings": settings,
    "chat_template": tokenizer.chat_template,
}

# Resume completed comparisons if the script is run again.
results = []
if DESTINATION.exists():
    saved = json.loads(DESTINATION.read_text())
    assert saved["metadata"] == metadata, "Existing generation settings differ."
    results = saved["results"]
    assert len(results) <= len(rows)
    for i, result in enumerate(results):
        assert result["id"] == i
        assert result["prompt"] == rows[i]["prompt"][0]["content"]

def save_progress():
    temporary = DESTINATION.with_suffix(".tmp.json")
    temporary.write_text(
        json.dumps(
            {"metadata": metadata, "results": results},
            indent=2,
            ensure_ascii=False,
        )
    )
    temporary.replace(DESTINATION)

for i in tqdm(
    range(len(results), len(rows)),
    desc="Persona comparisons",
):
    messages = rows[i]["prompt"]

    with model.disable_adapter():
        original = generate(messages)
    adapted = generate(messages)

    results.append({
        "id": i,
        "prompt": messages[0]["content"],
        "original": original,
        "lora": adapted,
    })
    save_progress()

    print("\nPROMPT:", results[-1]["prompt"], flush=True)
    print("ORIGINAL:", original, flush=True)
    print("LORA:", adapted, flush=True)
    print("-" * 60, flush=True)

print("\nCompleted comparisons:", len(results), flush=True)
print("Empty original responses:",
      sum(not row["original"] for row in results), flush=True)
print("Empty LoRA responses:",
      sum(not row["lora"] for row in results), flush=True)
print("Saved:", DESTINATION, flush=True)
