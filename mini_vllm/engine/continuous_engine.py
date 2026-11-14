import hashlib
import json

import torch

from .request import Request
from .block_manager import BlockManager
from .scheduler import Scheduler
from .model_runner import ModelRunner
from .prefix_cache import PrefixCache


class ContinuousBatchEngine:
    def __init__(
        self,
        model_name="sshleifer/tiny-gpt2",
        num_blocks=1024,
        block_size=16,
        max_batch_tokens=128,
        max_active_requests=32,
        runner=None,
    ):
        self.runner = runner if runner is not None else ModelRunner(model_name)

        self.block_manager = BlockManager(
            num_blocks=num_blocks,
            block_size=block_size,
        )

        # Create prefix cache
        namespace = json.dumps(
            {
                "model": model_name,
                "revision": getattr(self.runner, "revision", None),
                "config": self.runner.model.config.to_dict(),
                "dtype": str(next(self.runner.model.parameters()).dtype),
                "block_size": block_size,
                "cache_layout": "paged-cpu-v1",
                # Pools are process-local and never exchange KV tensors. Include
                # this exact model instance so injected models cannot alias weights.
                "model_instance": id(self.runner.model),
            },
            sort_keys=True,
            default=str,
        )
        self.prefix_cache = PrefixCache(
            block_size=block_size,
            cache_namespace=hashlib.sha256(namespace.encode()).hexdigest(),
            block_manager=self.block_manager,
        )

        # Connect memory pressure to LRU eviction
        self.block_manager.on_memory_pressure = self.prefix_cache.evict

        self.scheduler = Scheduler(
            block_manager=self.block_manager,
            max_batch_tokens=max_batch_tokens,
            max_active_requests=max_active_requests,
        )

        self.requests = {}
        self.max_tokens = {}
        self.results = {}
        self.completions = {}
        self.forward_tokens = 0
        self.forward_batches = []

        self.kv_cache = self.runner.create_paged_cache(self.block_manager)

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

        context = getattr(self.runner.model.config, "max_position_embeddings", None)
        if context and len(token_ids) + max_tokens > context:
            raise ValueError("Prompt and generated tokens exceed model context")

        req = Request(request_id, token_ids)
        # Acquire ownership of reusable KV blocks.
        cached_blocks, cached_tokens = self.prefix_cache.acquire_prefix(req.token_ids)

        # The model runner needs at least one uncached
        # token to produce the next-token logits.
        if cached_tokens == len(req.token_ids):
            self.block_manager.free([cached_blocks.pop()])
            cached_tokens -= self.block_manager.block_size

        req.block_ids = cached_blocks
        req.num_cached_tokens = cached_tokens
        req.num_computed_tokens = cached_tokens
        req.block_hashes = self.prefix_cache.compute_hashes(token_ids[:cached_tokens])

        self.requests[request_id] = req
        self.max_tokens[request_id] = max_tokens

        self.scheduler.add_request(req)

    def step(self):
        original = {key: len(req.block_ids) for key, req in self.requests.items()}
        try:
            return self._step()
        except Exception:
            # Partial writes in private slots are uncommitted. Return only new
            # allocations; a retry overwrites the remaining uncommitted slots.
            for key, req in self.requests.items():
                count = original[key]
                self.block_manager.free(req.block_ids[count:])
                del req.block_ids[count:]
            raise

    def _step(self):
        batch = self.scheduler.schedule()

        if not batch:
            if self.requests:
                raise RuntimeError("No schedulable tokens: KV block pool is exhausted")
            return {}

        # Build the input sequences selected by scheduler.
        sequences = []

        for req, count in batch:
            end = req.num_computed_tokens + count

            sequences.append(req.token_ids[req.num_computed_tokens : end])

        # One model forward call for the entire batch.
        logits = self.runner.forward_paged_batch(
            sequences,
            [req for req, _ in batch],
            self.kv_cache,
        )
        self.forward_tokens += sum(map(len, sequences))
        self.forward_batches.append(tuple(map(len, sequences)))
        # Bound diagnostic history for long-running servers.
        del self.forward_batches[:-256]

        # Only commit after successful execution.
        self.scheduler.complete_step(batch)
        for req, count in batch:
            self.register_computed_blocks(req)

        sampled = torch.argmax(logits, dim=-1).tolist()

        generated = {}

        for (req, count), token_id in zip(batch, sampled):
            # Don't sample during incomplete prefill.
            if req.num_computed_tokens < req.prompt_len:
                continue

            req.append_token(token_id)

            generated[req.request_id] = token_id

            reached_limit = len(req.output_tokens) >= self.max_tokens[req.request_id]

            reached_eos = token_id == self.runner.tokenizer.eos_token_id

            if reached_limit or reached_eos:
                self.finish(req, "eos" if reached_eos else "length")

        return generated

    def finish(self, req, reason="length"):
        if req.request_id not in self.requests:
            return
        try:
            text = self.runner.tokenizer.decode(
                req.output_tokens,
                skip_special_tokens=True,
            )
        except Exception:
            self.discard_request(req.request_id)
            raise

        self.results[req.request_id] = text
        self.completions[req.request_id] = {
            "request_id": req.request_id,
            "text": text,
            "generated_tokens": len(req.output_tokens),
            "finish_reason": reason,
        }

        self.discard_request(req.request_id)

    def discard_request(self, request_id):
        """Release request ownership even when token decoding has failed."""
        req = self.requests.pop(request_id, None)
        if req is not None:
            self.scheduler.finish_request(req)
        self.max_tokens.pop(request_id, None)

    def is_complete(self, request_id):
        return request_id in self.completions

    def cancel_request(self, request_id):
        req = self.requests.get(request_id)
        if req is None:
            return False
        self.finish(req, "cancelled")
        return True

    def forget_result(self, request_id):
        self.results.pop(request_id, None)
        return self.completions.pop(request_id, None)

    def register_computed_blocks(self, req):
        block_size = self.block_manager.block_size

        # Only fully computed blocks can be shared.
        full_blocks = req.num_computed_tokens // block_size

        # Hash each committed block once. Re-hashing the whole prompt on every
        # decode step dominated Python overhead in the owner-thread profile.
        for index in range(len(req.block_hashes), full_blocks):
            block_id = req.block_ids[index]
            start = index * block_size
            block_hash = self.prefix_cache.hash_block(
                req.block_hashes[-1] if req.block_hashes else "ROOT",
                req.token_ids[start : start + block_size],
            )
            req.block_hashes.append(block_hash)

            # Already registered by this or another request.
            existing = self.prefix_cache.hash_to_block.get(block_hash)

            if existing is not None:
                continue

            # At this point the Model Runner has already
            # written valid K/V states for this block.
            self.prefix_cache.register_block(
                block_hash,
                block_id,
            )
