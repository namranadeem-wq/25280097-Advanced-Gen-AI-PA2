
import json
import time
from pathlib import Path

import torch
import transformers
import trl
import peft
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed
from peft import LoraConfig
from trl import DPOConfig, DPOTrainer

ROOT = Path("/kaggle/working/small-model-alignment-starter")
DATA = ROOT / "data/persona_lora"
OUT = ROOT / "outputs/persona_lora"
OUT.mkdir(parents=True, exist_ok=True)
MODEL_ID = "HuggingFaceTB/SmolLM2-360M-Instruct"
set_seed(42)

def load_split(name):
    return Dataset.from_list(
        json.loads((DATA / f"{name}_seed42.json").read_text())
    )

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
if tokenizer.pad_token_id is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    dtype=torch.float32,
)
model.config.use_cache = False

targets = ["q_proj", "k_proj", "v_proj", "o_proj"]
available = {name.rsplit(".", 1)[-1] for name, _ in model.named_modules()}
assert all(name in available for name in targets), "Missing LoRA targets."

lora = LoraConfig(
    r=8,
    lora_alpha=16,
    lora_dropout=0.05,
    target_modules=targets,
    bias="none",
    task_type="CAUSAL_LM",
)

args = DPOConfig(
    output_dir=str(OUT / "checkpoints"),
    num_train_epochs=3,
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    gradient_accumulation_steps=4,
    learning_rate=5e-5,
    lr_scheduler_type="cosine",
    warmup_ratio=0.1,
    weight_decay=0.0,
    max_grad_norm=1.0,
    optim="adamw_torch",
    beta=0.1,
    loss_type="sigmoid",
    max_length=512,
    max_prompt_length=256,
    fp16=True,
    bf16=False,
    gradient_checkpointing=True,
    gradient_checkpointing_kwargs={"use_reentrant": False},
    eval_strategy="epoch",
    save_strategy="epoch",
    save_total_limit=2,
    load_best_model_at_end=True,
    metric_for_best_model="rewards/accuracies",
    greater_is_better=True,
    logging_steps=5,
    logging_first_step=True,
    report_to="none",
    seed=42,
    data_seed=42,
    dataloader_num_workers=0,
    disable_dropout=True,
)

trainer = DPOTrainer(
    model=model,
    ref_model=None,
    args=args,
    train_dataset=load_split("train"),
    eval_dataset=load_split("val"),
    processing_class=tokenizer,
    peft_config=lora,
)

trainer.model.print_trainable_parameters()
trainable = sum(p.numel() for p in trainer.model.parameters() if p.requires_grad)
total = sum(p.numel() for p in trainer.model.parameters())

configuration = {
    "model": MODEL_ID,
    "reference": "Original model with LoRA adapters disabled",
    "train_pairs": 198,
    "validation_pairs": 22,
    "seed": 42,
    "rank": 8,
    "alpha": 16,
    "configured_lora_dropout": 0.05,
    "trainer_disable_dropout": True,
    "effective_dropout": 0.0,
    "target_modules": targets,
    "trainable_parameters": trainable,
    "total_parameters": total,
    "versions": {
        "transformers": transformers.__version__,
        "trl": trl.__version__,
        "peft": peft.__version__,
        "torch": torch.__version__,
    },
    "training_arguments": args.to_dict(),
}
(OUT / "config.json").write_text(json.dumps(configuration, indent=2, default=str))

started = time.monotonic()
result = trainer.train()

# load_best_model_at_end has restored the selected adapter.
trainer.save_model(str(OUT / "best_adapter"))
tokenizer.save_pretrained(OUT / "best_adapter")
metrics = trainer.evaluate()

trainer.state.save_to_json(str(OUT / "trainer_state.json"))
with (OUT / "metrics.jsonl").open("w") as f:
    for row in trainer.state.log_history:
        f.write(json.dumps(row) + "\n")

summary = {
    "best_checkpoint": trainer.state.best_model_checkpoint,
    "best_validation_reward_accuracy": trainer.state.best_metric,
    "selected_adapter_validation": metrics,
    "training_metrics": result.metrics,
    "optimizer_steps": trainer.state.global_step,
    "total_minutes": (time.monotonic() - started) / 60,
}
(OUT / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
print("\nPERSONA LORA TRAINING COMPLETE", flush=True)
print(json.dumps(summary, indent=2, default=str), flush=True)
