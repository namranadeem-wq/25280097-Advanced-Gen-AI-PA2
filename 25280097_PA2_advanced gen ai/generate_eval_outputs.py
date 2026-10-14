
import argparse
import gc
import json
from pathlib import Path

import torch
import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm.auto import tqdm

ROOT = Path("/kaggle/working/small-model-alignment-starter")
INPUT = Path("/kaggle/input/datasets/namranadeem")
OUT = ROOT / "outputs/eval"
OUT.mkdir(parents=True, exist_ok=True)

SFT = INPUT / "25280097-pa2-sft-backup/outputs/sft/best"
HH = INPUT / "25280097-pa2-hh-dpo-backup/outputs/dpo_hh/best"
UF = INPUT / "25280097-pa2-uf-dpo-backup/outputs/dpo_ultrafeedback/best"

parser = argparse.ArgumentParser()
parser.add_argument("--worker", choices=["sft", "dpo"], required=True)
args = parser.parse_args()

jobs = {
    "sft": [
        ("sft_alpaca", SFT, "alpaca_eval_300_seed42.json"),
        ("sft_hh", SFT, "hh_rlhf_eval_300_seed42.json"),
    ],
    "dpo": [
        ("dpo_ultrafeedback_alpaca", UF, "alpaca_eval_300_seed42.json"),
        ("dpo_hh", HH, "hh_rlhf_eval_300_seed42.json"),
    ],
}

settings = {
    "max_new_tokens": 256,
    "do_sample": False,
    "repetition_penalty": 1.1,
    "no_repeat_ngram_size": 4,
}
BATCH_SIZE = 4
torch.manual_seed(42)

for name, checkpoint, filename in jobs[args.worker]:
    assert (checkpoint / "model.safetensors").exists(), checkpoint
    eval_path = ROOT / "eval_sets" / filename
    rows = json.loads(eval_path.read_text())
    assert len(rows) == 300
    assert len({str(row["id"]) for row in rows}) == 300

    destination = OUT / f"{name}_300_seed42.json"
    metadata = {
        "checkpoint": str(checkpoint),
        "evaluation_file": str(eval_path),
        "prompt_template": "### Instruction:\n{instruction}\n\n### Response:\n",
        "max_prompt_tokens": 256,
        "generation_settings": settings,
        "batch_size": BATCH_SIZE,
        "precision": "float16",
        "seed": 42,
        "transformers_version": transformers.__version__,
    }

    results = []
    if destination.exists():
        saved = json.loads(destination.read_text())
        assert saved["metadata"] == metadata, \
            f"Existing settings differ for {name}; do not overwrite."
        results = saved["results"]
        assert len(results) <= len(rows)
        for i, result in enumerate(results):
            assert result["id"] == rows[i]["id"]
            assert result["instruction"] == rows[i]["instruction"]

    if len(results) == len(rows):
        print(name, "already complete.", flush=True)
        continue

    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(
        checkpoint, torch_dtype=torch.float16
    ).to("cuda:0")
    model.eval()
    model.config.use_cache = True

    bar = tqdm(total=len(rows), initial=len(results), desc=name)

    for start in range(len(results), len(rows), BATCH_SIZE):
        batch_rows = rows[start:start + BATCH_SIZE]
        prompts = [
            f"### Instruction:\n{row['instruction']}\n\n### Response:\n"
            for row in batch_rows
        ]
        inputs = tokenizer(
            prompts,
            add_special_tokens=False,
            padding=True,
            truncation=True,
            max_length=256,
            return_tensors="pt",
        ).to("cuda:0")

        with torch.inference_mode():
            generated = model.generate(
                **inputs,
                **settings,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )

        responses = tokenizer.batch_decode(
            generated[:, inputs.input_ids.shape[1]:],
            skip_special_tokens=True,
        )
        for row, response in zip(batch_rows, responses):
            results.append({**row, "response": response.strip()})

        temporary = destination.with_suffix(".tmp.json")
        temporary.write_text(json.dumps(
            {"metadata": metadata, "results": results},
            indent=2, ensure_ascii=False,
        ))
        temporary.replace(destination)
        bar.update(len(batch_rows))

    bar.close()
    print(
        name, "| Saved:", len(results),
        "| Empty:", sum(not row["response"] for row in results),
        flush=True,
    )

    del model, inputs, generated
    gc.collect()
    torch.cuda.empty_cache()

print(args.worker, "worker complete.", flush=True)
