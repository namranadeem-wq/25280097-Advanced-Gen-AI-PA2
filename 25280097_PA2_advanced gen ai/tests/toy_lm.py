"""Small offline tokenizer and causal LM used by public adapter smoke tests."""
from types import SimpleNamespace
import torch
from torch import nn

class ToyTokenizer:
    pad_token = '<pad>'
    eos_token = '<eos>'
    pad_token_id = 0
    eos_token_id = 1
    def __init__(self):
        chars = list('abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 .,!?;:-_')
        self.vocab = {self.pad_token: 0, self.eos_token: 1}
        for char in chars:
            if char not in self.vocab:
                self.vocab[char] = len(self.vocab)
        self.inv_vocab = {i: t for t, i in self.vocab.items()}
    @property
    def vocab_size(self):
        return len(self.vocab)
    def encode(self, text, add_special_tokens=False):
        ids, index = [], 0
        while index < len(text):
            if text.startswith(self.eos_token, index):
                ids.append(1); index += len(self.eos_token); continue
            if text.startswith(self.pad_token, index):
                ids.append(0); index += len(self.pad_token); continue
            ids.append(self.vocab[text[index]]); index += 1
        if add_special_tokens:
            ids.append(1)
        return ids
    def __call__(self, text, add_special_tokens=False, return_tensors=None,
                 padding=False, truncation=False, max_length=None):
        is_batch = isinstance(text, list)
        texts = text if is_batch else [text]
        encoded = [self.encode(t, add_special_tokens) for t in texts]
        if truncation and max_length is not None:
            encoded = [ids[:max_length] for ids in encoded]
        masks = [[1] * len(ids) for ids in encoded]
        if padding:
            width = max(map(len, encoded))
            if padding == 'max_length' and max_length is not None:
                width = max_length
            encoded = [ids + [0] * (width - len(ids)) for ids in encoded]
            masks = [mask + [0] * (width - len(mask)) for mask in masks]
        if return_tensors == 'pt':
            return SimpleNamespace(input_ids=torch.tensor(encoded, dtype=torch.long),
                                   attention_mask=torch.tensor(masks, dtype=torch.long))
        if is_batch:
            return {'input_ids': encoded, 'attention_mask': masks}
        return SimpleNamespace(input_ids=encoded[0], attention_mask=masks[0])

class ToyCausalLM(nn.Module):
    def __init__(self, vocab_size, bias_scale=0.0):
        super().__init__()
        self.logit_bias = nn.Parameter(torch.linspace(-0.3, 0.3, vocab_size) * bias_scale)
    def forward(self, input_ids, attention_mask=None, labels=None):
        b, length = input_ids.shape
        return SimpleNamespace(logits=self.logit_bias.view(1, 1, -1).expand(b, length, -1).contiguous())
