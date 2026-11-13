"""Continuous batching with per-request KV caches and one forward per step."""
from collections import deque
from dataclasses import dataclass, field
from typing import Any


@dataclass
class GenerationRequest:
    request_id: str
    token_ids: list[int]
    prompt_len: int
    max_new_tokens: int
    computed: int = 0
    cache: Any = None
    generated: list[int] = field(default_factory=list)


class CachedContinuousEngine:
    """Round-robin scheduling with a per-iteration token budget.

    Runner interface:
        encode(prompt: str) -> list[int]
        forward_batch(chunks, caches) -> list[tuple[int | None, Any]]
        decode(tokens: list[int]) -> str
        eos_token_id: int | None

    Predictions are consumed only after the entire prompt has been processed,
    or during decode. Only uncached token chunks are sent to the runner.
    """

    def __init__(self, runner, token_budget: int = 32):
        if token_budget <= 0:
            raise ValueError("token_budget must be positive")
        self.runner = runner
        self.token_budget = token_budget
        self.waiting = deque()
        self.active = deque()
        self.requests = {}
        self.results = {}

    def add_request(self, request_id: str, prompt: str, max_new_tokens: int = 16):
        if request_id in self.requests or request_id in self.results:
            raise ValueError("Duplicate request ID")
        if max_new_tokens < 1:
            raise ValueError("max_new_tokens must be positive")
        ids = self.runner.encode(prompt)
        if not ids:
            raise ValueError("Prompt must have at least one token")
        request = GenerationRequest(request_id, list(ids), len(ids), max_new_tokens)
        self.requests[request_id] = request
        self.waiting.append(request)

    def step(self) -> dict[str, int]:
        """Execute cached work for live requests. Returns newly emitted tokens."""
        # New requests can join on *any* iteration.
        self.active.extend(self.waiting)
        self.waiting.clear()
        remaining_budget = self.token_budget
        batch = []

        for req in self.active:
            if remaining_budget == 0:
                break

            pending = len(req.token_ids) - req.computed
            if pending <= 0:
                raise RuntimeError("Missing token for decode: inconsistent request state")

            # Each request gets at most one chunk per step.
            is_prefill = req.computed < req.prompt_len
            amount = min(pending, remaining_budget) if is_prefill else 1
            token_chunk = req.token_ids[req.computed:req.computed + amount]
            batch.append((req, amount, token_chunk))
            remaining_budget -= amount

        if not batch:
            return {}

        outputs = self.runner.forward_batch(
            [chunk for _, _, chunk in batch],
            [req.cache for req, _, _ in batch],
        )
        if len(outputs) != len(batch):
            raise RuntimeError("Runner must return one result per request")
        for (req, amount, _), (prediction, _) in zip(batch, outputs):
            if prediction is None and req.computed + amount >= req.prompt_len:
                raise RuntimeError("Runner must return a next-token prediction")

        # Commit only after the complete batch succeeds. Unscheduled requests
        # keep their place ahead of those that used this iteration's budget.
        selected = {req.request_id for req, _, _ in batch}
        self.active = deque(req for req in self.active if req.request_id not in selected)
        emitted = {}
        for (req, amount, _), (prediction, new_cache) in zip(batch, outputs):
            req.cache = new_cache
            req.computed += amount

            # Intermediate prefill chunks don't generate output.
            if req.computed < req.prompt_len:
                self.active.append(req)
                continue

            req.generated.append(int(prediction))
            req.token_ids.append(int(prediction))
            emitted[req.request_id] = int(prediction)

            finished = (len(req.generated) >= req.max_new_tokens or
                        prediction == self.runner.eos_token_id)
            if finished:
                self.results[req.request_id] = self.runner.decode(req.generated)
                del self.requests[req.request_id]
                req.cache = None   # release the request's HF cache
            else:
                self.active.append(req)
        return emitted


class HFRunner:
    """Hugging Face CPU runner with padded batched KV execution."""
    def __init__(self, model_name: str = "sshleifer/tiny-gpt2"):
        from .model_runner import ModelRunner

        self.runner = ModelRunner(model_name)
        self.tokenizer = self.runner.tokenizer
        self.model = self.runner.model
        self.eos_token_id = self.tokenizer.eos_token_id

    def encode(self, prompt):
        return self.tokenizer.encode(prompt)

    def decode(self, token_ids):
        return self.tokenizer.decode(token_ids, skip_special_tokens=True)

    def forward(self, token_ids, cache):
        return self.forward_batch([token_ids], [cache])[0]

    def forward_batch(self, chunks, caches):
        logits, updated_caches = self.runner.forward_cached_batch(chunks, caches)
        tokens = logits.argmax(dim=-1).tolist()
        return list(zip(tokens, updated_caches))
