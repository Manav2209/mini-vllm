
import torch

from .request import Request
from .block_manager import BlockManager
from .scheduler import Scheduler
from .model_runner import ModelRunner


class ContinuousBatchEngine:
    def __init__(self, model_name="sshleifer/tiny-gpt2"):
        self.runner = ModelRunner(model_name)

        self.block_manager = BlockManager(
            num_blocks=1024,
            block_size=16,
        )

        self.scheduler = Scheduler(
            block_manager=self.block_manager,
            max_batch_tokens=128,
        )

        self.requests = {}
        self.max_tokens = {}
        self.results = {}

    def add_request(
        self,
        request_id: str,
        prompt: str,
        max_tokens: int = 20,
    ):
        if request_id in self.requests or request_id in self.results:
            raise ValueError("Duplicate request ID")

        if max_tokens <= 0:
            raise ValueError("max_tokens must be positive")

        token_ids = self.runner.tokenizer.encode(prompt)

        if not token_ids:
            raise ValueError("Prompt is empty")

        req = Request(request_id, token_ids)

        self.requests[request_id] = req
        self.max_tokens[request_id] = max_tokens

        self.scheduler.add_request(req)

    def step(self):
        batch = self.scheduler.schedule()

        if not batch:
            return {}

        # Build the input sequences selected by scheduler.
        sequences = []

        for req, count in batch:
            end = req.num_computed_tokens + count

            # Reference implementation:
            # recompute the full processed prefix.
            sequences.append(req.token_ids[:end])

        # One model forward call for the entire batch.
        logits = self.runner.forward_batch(sequences)

        # Only commit after successful execution.
        self.scheduler.complete_step(batch)

        sampled = torch.argmax(logits, dim=-1).tolist()

        generated = {}

        for (req, count), token_id in zip(batch, sampled):

            # Don't sample during incomplete prefill.
            if req.num_computed_tokens < req.prompt_len:
                continue

            req.append_token(token_id)

            generated[req.request_id] = token_id

            reached_limit = (
                len(req.output_tokens)
                >= self.max_tokens[req.request_id]
            )

            reached_eos = (
                token_id == self.runner.tokenizer.eos_token_id
            )

            if reached_limit or reached_eos:
                self.finish(req)

        return generated

    def finish(self, req):
        text = self.runner.tokenizer.decode(
            req.output_tokens,
            skip_special_tokens=True,
        )

        self.results[req.request_id] = text

        self.scheduler.finish_request(req)

        del self.requests[req.request_id]
        del self.max_tokens[req.request_id]
