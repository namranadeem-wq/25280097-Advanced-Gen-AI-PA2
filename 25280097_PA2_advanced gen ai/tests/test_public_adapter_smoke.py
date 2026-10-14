"""Public smoke-test contract transcribed from the assignment-provided tests.
Passing these checks does not guarantee private correctness or model quality.
"""
import inspect
import torch
import submission_adapter as adapter
from tests.toy_lm import ToyCausalLM, ToyTokenizer

REQUIRED_FUNCTIONS = {
 'build_sft_batch': ['tokenizer', 'prompts', 'responses', 'max_length'],
 'compute_response_logprobs': ['model', 'tokenizer', 'prompts', 'responses', 'max_length'],
 'compute_sft_loss': ['model', 'tokenizer', 'prompts', 'responses', 'max_length'],
 'build_dpo_batch': ['tokenizer', 'prompts', 'chosen_responses', 'rejected_responses', 'max_length'],
 'compute_dpo_loss_from_logps': ['policy_chosen_logps', 'policy_rejected_logps', 'reference_chosen_logps', 'reference_rejected_logps', 'beta'],
 'compute_dpo_loss': ['policy_model', 'reference_model', 'tokenizer', 'prompts', 'chosen_responses', 'rejected_responses', 'beta', 'max_length'],
}

def test_required_functions_have_the_documented_signatures():
    for name, params in REQUIRED_FUNCTIONS.items():
        function = getattr(adapter, name, None)
        assert callable(function)
        assert list(inspect.signature(function).parameters) == params

def test_sft_batch_masks_prompt_and_padding_and_includes_eos():
    tokenizer = ToyTokenizer()
    prompts, responses = ['Task: ', 'Q: '], ['ok', 'yes']
    batch = adapter.build_sft_batch(tokenizer, prompts, responses, max_length=16)
    assert set(batch) == {'input_ids', 'attention_mask', 'labels'}
    assert all(isinstance(v, torch.Tensor) for v in batch.values())
    assert batch['input_ids'].shape == batch['attention_mask'].shape == batch['labels'].shape
    assert batch['input_ids'].shape[0] == len(prompts)
    for i, (prompt, response) in enumerate(zip(prompts, responses)):
        p = len(tokenizer.encode(prompt, add_special_tokens=False))
        r = len(tokenizer.encode(response, add_special_tokens=False)) + 1
        assert batch['labels'][i, :p].eq(-100).all()
        assert batch['labels'][i, p:p+r].ne(-100).all()
        assert batch['labels'][i, p+r:].eq(-100).all()
        assert batch['labels'][i, p+r-1] == tokenizer.eos_token_id
    unmasked = batch['labels'].ne(-100)
    assert torch.equal(batch['input_ids'][unmasked], batch['labels'][unmasked])
    assert batch['attention_mask'][batch['labels'].eq(-100)].eq(0).any()

def test_response_logprobs_are_one_value_per_row_and_response_only():
    tokenizer = ToyTokenizer()
    model = ToyCausalLM(tokenizer.vocab_size, bias_scale=0.7)
    logps = adapter.compute_response_logprobs(model, tokenizer, ['aaaa: ', 'zzzz: '], ['same', 'same'], max_length=20)
    assert logps.shape == (2,)
    assert torch.isfinite(logps).all()
    torch.testing.assert_close(logps[0], logps[1])

def test_sft_loss_uses_an_accepted_reduction_and_has_policy_gradients():
    tokenizer = ToyTokenizer()
    model = ToyCausalLM(tokenizer.vocab_size, bias_scale=0.2)
    prompts, responses = ['Do: ', 'Say: '], ['a', 'longer']
    loss = adapter.compute_sft_loss(model, tokenizer, prompts, responses, max_length=16)
    logps = adapter.compute_response_logprobs(model, tokenizer, prompts, responses, max_length=16)
    batch = adapter.build_sft_batch(tokenizer, prompts, responses, max_length=16)
    count = batch['labels'][:, 1:].ne(-100).sum().clamp_min(1)
    accepted = (-logps.mean(), -logps.sum() / count)
    assert loss.ndim == 0 and torch.isfinite(loss)
    assert any(torch.allclose(loss, x, rtol=1e-5, atol=1e-6) for x in accepted)
    loss.backward()
    assert model.logit_bias.grad is not None
    assert model.logit_bias.grad.abs().sum() > 0

def test_dpo_batch_has_chosen_and_rejected_lm_batches():
    tokenizer = ToyTokenizer()
    batch = adapter.build_dpo_batch(tokenizer, ['Task: ', 'Q: '], ['good', 'yes'], ['bad', 'no'], max_length=16)
    assert set(batch) == {'chosen', 'rejected'}
    for side in batch.values():
        assert set(side) == {'input_ids', 'attention_mask', 'labels'}
        assert all(isinstance(v, torch.Tensor) for v in side.values())
        assert side['input_ids'].shape == side['attention_mask'].shape == side['labels'].shape
        assert side['input_ids'].shape[0] == 2
        assert side['labels'].ne(-100).any()

def test_dpo_loss_from_equal_logps_is_log_two():
    values = torch.tensor([-2., -5.])
    loss = adapter.compute_dpo_loss_from_logps(values, values, values, values, beta=0.2)
    assert loss.ndim == 0
    torch.testing.assert_close(loss, torch.log(torch.tensor(2.)))

def test_dpo_loss_is_scalar_and_does_not_update_reference_model():
    tokenizer = ToyTokenizer()
    policy = ToyCausalLM(tokenizer.vocab_size)
    reference = ToyCausalLM(tokenizer.vocab_size)
    loss = adapter.compute_dpo_loss(policy, reference, tokenizer, ['Pick: ', 'Choose: '], ['a', 'a'], ['b', 'b'], beta=0.2, max_length=12)
    assert loss.ndim == 0 and torch.isfinite(loss)
    loss.backward()
    assert policy.logit_bias.grad is not None
    assert policy.logit_bias.grad.abs().sum() > 0
    assert reference.logit_bias.grad is None or reference.logit_bias.grad.abs().sum() == 0
