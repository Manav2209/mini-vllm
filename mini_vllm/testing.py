"""Deterministic offline GPT-2 fixture for tests and reproducible benchmarks."""

import torch
from transformers import GPT2Config, GPT2LMHeadModel

from .engine.model_runner import ModelRunner


class IntegerTokenizer:
    pad_token_id = 0
    eos_token_id = None

    def encode(self, prompt):
        return list(map(int, prompt.split()))

    def decode(self, tokens, skip_special_tokens=True, **kwargs):
        return "".join(f"{token} " for token in tokens)

    def __call__(self, prompt, return_tensors="pt"):
        ids = torch.tensor([self.encode(prompt)])
        return {"input_ids": ids, "attention_mask": torch.ones_like(ids)}


def tiny_runner(seed=17):
    runner = ModelRunner.__new__(ModelRunner)
    runner.device = "cpu"
    runner.tokenizer = IntegerTokenizer()
    with torch.random.fork_rng():
        torch.manual_seed(seed)
        runner.model = GPT2LMHeadModel(
            GPT2Config(
                vocab_size=97,
                n_positions=256,
                n_embd=24,
                n_layer=2,
                n_head=2,
                bos_token_id=0,
                eos_token_id=None,
            )
        ).eval()
    return runner
