"""CPU-first, per-request-KV-cache continuous request scheduler.

This is *not* one tensor-batched model forward, nor a paged KV implementation.
It provides dynamically interleaved requests with incremental cached model execution.
"""
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
        forward(token_ids: list[int], cache: Any) -> tuple[int | None, Any]
        decode(tokens: list[int]) -> str
        eos_token_id: int | None

    forward returns (next_token_prediction, updated_cache); prediction is
    consumed only after the entire prompt has been processed, or during decode.
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
        emitted = {}
        rounds = len(self.active)  # each request gets at most one turn per step

        for _ in range(rounds):
            req = self.active.popleft()
            if remaining_budget == 0:
                self.active.append(req)
                continue

            pending = len(req.token_ids) - req.computed
            if pending <= 0:
                raise RuntimeError("Missing token for decode: inconsistent request state")

            # Fair-share of *this* iteration, allows large prompts to be chunked.
            # We issue 1 token to each decode request, and available budget to prefill.
            is_prefill = req.computed < req.prompt_len
            amount = min(pending, remaining_budget) if is_prefill else 1
            token_chunk = req.token_ids[req.computed:req.computed + amount]
            prediction, new_cache = self.runner.forward(token_chunk, req.cache)
            req.cache = new_cache
            req.computed += amount
            remaining_budget -= amount

            # Intermediate prefill chunks don't generate output.
            if req.computed < req.prompt_len:
                self.active.append(req)
                continue

            if prediction is None:
                raise RuntimeError("Runner must return a next-token prediction")
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
    """Hugging Face CPU runner. One forward per request, with its own KV cache."""
    def __init__(self, model_name: str = "sshleifer/tiny-gpt2"):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        self.model.eval()
        self.eos_token_id = self.tokenizer.eos_token_id

    def encode(self, prompt):
        return self.tokenizer.encode(prompt)

    def decode(self, token_ids):
        return self.tokenizer.decode(token_ids, skip_special_tokens=True)

    def forward(self, token_ids, cache):
        torch = self.torch
        with torch.inference_mode():
            input_ids = torch.tensor([token_ids], dtype=torch.long)
            outputs = self.model(
                input_ids=input_ids,
                past_key_values=cache,
                use_cache=True,
            )
            token = int(outputs.logits[0, -1].argmax().item())
            return token, outputs.past_key_values
