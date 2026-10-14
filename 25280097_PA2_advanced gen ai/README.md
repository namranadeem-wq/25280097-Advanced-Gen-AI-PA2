# Small-Model Alignment - PA2

**Namra Nadeem | Roll 25280097**

Repository: https://github.com/namranadeem-wq/small-model-alignment-starter

## Measured results

Anonymous pairwise judge: `google/gemini-3.6-flash`, temperature 0, seed 42.

| Candidate vs baseline | Pairs | Wins | Losses | Ties | Win rate excluding ties | Ties half credit |
|---|---:|---:|---:|---:|---:|---:|
| UltraFeedback DPO vs SFT | 300 | 103 | 138 | 59 | 42.74% | 44.17% |
| HH DPO vs SFT | 300 | 95 | 102 | 103 | 48.22% | 48.83% |
| Persona LoRA vs original SmolLM2 | 22 | 3 | 3 | 16 | 50.00% | 50.00% |

All 622 comparisons were judged; final unknown and unresolved counts are zero. Retried requests and original failures are retained. Higher validation preference accuracy did not guarantee better generated answers. Persona results use the same 22 validation prompts used for checkpoint selection and are not an independent test.

## Contents

- `alignment_core.py`: response-only batches, shifted causal log probabilities, token-mean SFT loss, summed-response DPO objective and frozen reference computation.
- `submission_adapter.py`: six unchanged grading signatures wrapping the core implementation.
- `notebooks/`: actual training and benchmark notebook exports. These contain preparation, training, recovery, judging and analysis code.
- `train_persona_lora.py`: TRL DPOTrainer/PEFT LoRA training.
- `generate_eval_outputs.py`: resumable two-worker Qwen generation.
- `generate_persona_heldout.py`: resumable original/LoRA comparisons.
- `eval_sets/`: fixed 300-prompt AlpacaEval seed-42 and synthetic HH evaluation inputs.
- `outputs/`: actual run configurations, metrics, summaries, plots and all four 300-response Qwen generation files.
- `outputs/eval/pairwise/`: anonymous inputs plus separate identity/hash mappings.
- `judge_results/`: decisions, scored results, summary tables, prompts, raw replies, retries and failures.
- `report/25280097_PA2_Report.pdf`: seven-page report with configurations, results, error analysis and assistance disclosure.
- `tests/`: adapter smoke checks transcribed from the public test contract supplied with the assignment.

## Setup and execution

Use Kaggle GPU T4 x2 for full training/generation. Keep its GPU PyTorch installation; do not replace it with CPU PyTorch. Recorded environment: PyTorch 2.11.0+cu128, Transformers 4.57.6. Persona used TRL 0.24.0 and PEFT 0.18.0.

```bash
python -m pip install -r requirements.txt
PYTHONPATH=. python -m pytest -q tests/test_public_adapter_smoke.py
```

The training notebook is an **experiment history**, including session restores and obsolete diagnostics. Do not use Run All. Cell numbers below are one-based and refer to the unmodified source-cell ordering of the export. Select the stage you need and initialize its dependencies.

| Stage | Training notebook cells | Notes |
|---|---|---|
| Baseline subset and responses | 6-9 | Creates AlpacaEval subset and Qwen baseline generations |
| Core implementation and tests | 15-17 | Writes the two root modules and runs public checks |
| SFT preparation | 20 | First user/assistant pair; 30k/1k; exact AlpacaEval prompt exclusions |
| SFT initialization and loop | 22-23 | FP32 weights, FP16 AMP; token-normalized accumulation |
| DPO data preparation | 34 | HH single-turn parsing and UF conversational pairs; exclusions and truncation filtering |
| HH initialization and loop | 36-37 | Starts from SFT best; separate frozen GPU reference |
| Accepted UF fresh-start section | 48-49 | Fixed randperm; first 50 steps and state save |
| Accepted UF restore and continuation | 55-58 | Restores step-50 file and trains remaining steps; earlier failed runs are historical |
| Persona data download and split | 61, 64 | Downloads provided 220-pair source; seeded 198/22 split |
| Persona training / generation | 65, 67 | Writes/runs standalone scripts; use current standalone scripts for all 22 generations |
| Final generation and pairing | 78, 80 | Two GPU workers; 1,200 responses; randomized separate mappings |
| Training plots | 87 | Uses attached backup metrics |

Root variables in the training history point to `/kaggle/working/small-model-alignment-starter`. Copy this project there or adjust paths before running. Data/model downloads need internet. The full 30k/15k prepared datasets and large weights are in private backup datasets, not this small source/results package; regenerate preparation from the notebook when those backups are unavailable.

Generation reads attached SFT, HH and UF checkpoints directly; adjust the three paths at the top of `generate_eval_outputs.py` if using another account. Launch separate workers with `CUDA_VISIBLE_DEVICES=0` and `CUDA_VISIBLE_DEVICES=1`, respectively, and `--worker sft` / `--worker dpo`. Saved files resume only with identical metadata.

For judging, create a notebook at https://www.kaggle.com/benchmarks/tasks/new and attach the saved pairwise files. The benchmark package is supplied by that environment. Evaluation notebook cells 4-7 contain pilot, main batches, recovery and scoring. It saves after each valid batch, sleeps 5 seconds, retries up to 3 attempts, and skips completed IDs. Cell 10 produces the evaluation plot. The original pilot/main wrapper returned a dict and emitted framework warnings; cell 6 fixes the return type for missing-pair recovery. For a new run, replace task returns with `None` and add unique request identifiers to avoid reusing a failed framework-cached invocation. Keep the original rubric and model identities hidden.

## Checkpoints and data locations

Selected checkpoints:

| Track | Selected checkpoint | Validation criterion |
|---|---|---|
| SFT | `outputs/sft/best`, step 938 | Token loss 1.37846077 |
| HH DPO | `outputs/dpo_hh/best`, step 300 | Reward accuracy 0.581 |
| UF DPO | `outputs/dpo_ultrafeedback/best`, step 400 | Reward accuracy 0.710 |
| Persona | `outputs/persona_lora/best_adapter`, source checkpoint-25 | Reward accuracy 1.0 on 22 validation pairs |

Private Kaggle backup locations used during the run:

- `namranadeem/25280097-pa2-sft-backup`
- `namranadeem/25280097-pa2-hh-dpo-backup`
- `namranadeem/25280097-pa2-uf-dpo-backup`
- `namranadeem/25280097-pa2-evaluation-progress` (persona adapter and analysis)
- `namranadeem/25280097-pa2-judge-inputs`

Private datasets require owner-granted access; these identifiers do not establish public availability. Arrange access for course staff if required. Local backups are `25280097_PA2_SFT_backup.zip`, `25280097_PA2_HH_DPO_backup.zip`, `25280097_PA2_UF_DPO_backup.zip`, and `25280097_PA2_persona_LoRA_backup.zip`. Do not upload multi-GB checkpoints or recovery states through GitHub's browser uploader.

## Configuration and limitations

All Qwen runs use `### Instruction:\n{instruction}\n\n### Response:\n`; maximum sequence length 512 and effective batch 32. SFT uses AdamW LR 2e-5, cosine decay, 29 warmup steps, weight decay 0.1 and clip 1. Both full DPO runs use fixed LR 1e-6, beta 0.1, no weight decay and clip 1. Reference is the same SFT best checkpoint. Both completed 466 successful updates from 469 attempts.

Persona LoRA uses rank 8, alpha 16, q/k/v/o attention projections, effective batch 8, LR 5e-5, cosine decay, 10% warmup and 3 epochs. There are 1,638,400 trainable parameters. Configured dropout 0.05 was disabled by TRL, so effective dropout is zero. The first checkpoint attaining 100% validation accuracy was selected.

Evaluation decoding is greedy FP16 with repetition penalty 1.1 and no-repeat 4-grams; max prompt 256 and max new tokens 256 for Qwen, 192 new tokens for persona. Comparisons match decoding settings. UF durable logs start at step 51 after state recovery. Do not fabricate missing steps. Recorded run durations exclude discarded attempts and are not total human effort.

One seed, one judge, validation reuse for persona, no human adjudication and no mirrored A/B evaluation limit conclusions. Saved HH examples demonstrate safety failures; do not claim that the model is safe.

## AI assistance

AI assistance was used to develop and debug code, plan the Kaggle workflow, analyze saved outputs, and draft the report/README. Measurements come from saved artifacts. The submitting student should review the implementation and conclusions and follow the course disclosure policy.
