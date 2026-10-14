"""Response-only SFT and DPO training primitives."""

import torch
import torch.nn.functional as F


def build_sft_batch(tokenizer, prompts, responses, max_length):
    if len(prompts) != len(responses):
        raise ValueError("Prompts and responses must have equal lengths.")
    if not prompts or max_length < 1:
        raise ValueError("Require a nonempty batch and positive max_length.")

    eos_id = getattr(tokenizer, "eos_token_id", None)
    pad_id = getattr(tokenizer, "pad_token_id", None)
    if pad_id is None:
        pad_id = eos_id
    if pad_id is None:
        raise ValueError("Tokenizer needs a pad or EOS token ID.")

    sequences, targets = [], []

    for prompt, response in zip(prompts, responses):
        prompt_ids = list(
            tokenizer.encode(prompt, add_special_tokens=False)
        )
        response_ids = list(
            tokenizer.encode(response, add_special_tokens=False)
        )

        if eos_id is not None:
            response_ids.append(eos_id)

        ids = (prompt_ids + response_ids)[:max_length]
        labels = (
            [-100] * len(prompt_ids) + response_ids
        )[:max_length]

        sequences.append(ids)
        targets.append(labels)

    # Right padding for training, independent of generation padding settings.
    width = max(1, max(len(ids) for ids in sequences))
    input_ids = torch.full(
        (len(sequences), width), pad_id, dtype=torch.long
    )
    attention_mask = torch.zeros_like(input_ids)
    labels = torch.full_like(input_ids, -100)

    for row, (ids, target) in enumerate(zip(sequences, targets)):
        length = len(ids)
        input_ids[row, :length] = torch.tensor(ids, dtype=torch.long)
        attention_mask[row, :length] = 1
        labels[row, :length] = torch.tensor(target, dtype=torch.long)

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
    }


def response_scores(model, batch):
    """Return summed log probabilities and scored token counts per row."""
    device = next(model.parameters()).device
    batch = {key: value.to(device) for key, value in batch.items()}

    outputs = model(
        input_ids=batch["input_ids"],
        attention_mask=batch["attention_mask"],
    )

    # Position t predicts the token at position t+1.
    logits = outputs.logits[:, :-1, :].float()
    targets = batch["labels"][:, 1:]
    mask = targets.ne(-100)
    safe_targets = targets.masked_fill(~mask, 0)

    token_logps = -F.cross_entropy(
        logits.transpose(1, 2),
        safe_targets,
        reduction="none",
    )
    summed_logps = token_logps.masked_fill(~mask, 0).sum(dim=1)
    token_counts = mask.sum(dim=1)
    return summed_logps, token_counts


def compute_response_logprobs(
    model, tokenizer, prompts, responses, max_length
):
    batch = build_sft_batch(
        tokenizer, prompts, responses, max_length
    )
    logps, _ = response_scores(model, batch)
    return logps


def compute_sft_loss(
    model, tokenizer, prompts, responses, max_length
):
    batch = build_sft_batch(
        tokenizer, prompts, responses, max_length
    )
    logps, counts = response_scores(model, batch)

    # Token-mean reduction across all scored responses in the batch.
    return -logps.sum() / counts.sum().clamp_min(1)


def build_dpo_batch(
    tokenizer, prompts, chosen_responses, rejected_responses, max_length
):
    return {
        "chosen": build_sft_batch(
            tokenizer, prompts, chosen_responses, max_length
        ),
        "rejected": build_sft_batch(
            tokenizer, prompts, rejected_responses, max_length
        ),
    }


def compute_dpo_loss_from_logps(
    policy_chosen_logps,
    policy_rejected_logps,
    reference_chosen_logps,
    reference_rejected_logps,
    beta,
):
    device = policy_chosen_logps.device
    reference_chosen_logps = reference_chosen_logps.to(device)
    reference_rejected_logps = reference_rejected_logps.to(device)

    relative_margin = (
        policy_chosen_logps - policy_rejected_logps
        - reference_chosen_logps + reference_rejected_logps
    )
    return -F.logsigmoid(beta * relative_margin).mean()


def compute_dpo_loss(
    policy_model,
    reference_model,
    tokenizer,
    prompts,
    chosen_responses,
    rejected_responses,
    beta,
    max_length,
):
    batch = build_dpo_batch(
        tokenizer, prompts, chosen_responses,
        rejected_responses, max_length
    )

    policy_chosen, _ = response_scores(policy_model, batch["chosen"])
    policy_rejected, _ = response_scores(policy_model, batch["rejected"])

    # Disable reference dropout temporarily and prevent reference gradients.
    was_training = reference_model.training
    reference_model.eval()
    try:
        with torch.no_grad():
            reference_chosen, _ = response_scores(
                reference_model, batch["chosen"]
            )
            reference_rejected, _ = response_scores(
                reference_model, batch["rejected"]
            )
    finally:
        reference_model.train(was_training)

    return compute_dpo_loss_from_logps(
        policy_chosen, policy_rejected,
        reference_chosen, reference_rejected, beta
    )
