
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
    ):
        self.runner = ModelRunner(model_name)

        self.block_manager = BlockManager(
            num_blocks=num_blocks,
            block_size=block_size,
        )

        # Create prefix cache
        self.prefix_cache = PrefixCache(
            block_size=block_size,
            cache_namespace=model_name,
            block_manager=self.block_manager,
        )

        # Connect memory pressure to LRU eviction
        self.block_manager.on_memory_pressure = (
            self.prefix_cache.evict
        )

        self.scheduler = Scheduler(
            block_manager=self.block_manager,
            max_batch_tokens=128,
        )

        self.requests = {}
        self.max_tokens = {}
        self.results = {}

        self.kv_cache = self.runner.create_paged_cache(
            self.block_manager
        )

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
        # Acquire ownership of reusable KV blocks.
        cached_blocks, cached_tokens = (
            self.prefix_cache.acquire_prefix(req.token_ids)
        )

        # The model runner needs at least one uncached
        # token to produce the next-token logits.
        if cached_tokens == len(req.token_ids):
            self.block_manager.free([cached_blocks.pop()])
            cached_tokens -= self.block_manager.block_size

        req.block_ids = cached_blocks
        req.num_cached_tokens = cached_tokens
        req.num_computed_tokens = cached_tokens


        self.requests[request_id] = req
        self.max_tokens[request_id] = max_tokens

        self.scheduler.add_request(req)

    def step(self):
        batch = self.scheduler.schedule()

        if not batch:
            if self.requests:
                raise RuntimeError("No schedulable tokens: KV block pool is exhausted")
            return {}

        # Build the input sequences selected by scheduler.
        sequences = []

        for req, count in batch:
            end = req.num_computed_tokens + count

            sequences.append(req.token_ids[req.num_computed_tokens:end])

        # One model forward call for the entire batch.
        logits = self.runner.forward_paged_batch(
            sequences, [req for req, _ in batch], self.kv_cache,
        )

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

    def register_computed_blocks(self, req):
        block_size = self.block_manager.block_size

        # Only fully computed blocks can be shared.
        full_blocks = req.num_computed_tokens // block_size

        hashes = self.prefix_cache.compute_hashes(
            req.token_ids[:req.num_computed_tokens]
        )

        for index in range(full_blocks):
            block_id = req.block_ids[index]
            block_hash = hashes[index]

            # Already registered by this or another request.
            existing = self.prefix_cache.hash_to_block.get(
                block_hash
            )

            if existing is not None:
                continue

            # At this point the Model Runner has already
            # written valid K/V states for this block.
            self.prefix_cache.register_block(
                block_hash,
                block_id,
            )
