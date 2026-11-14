from collections import deque

from .request import Request
from .block_manager import BlockManager


class Scheduler:
    def __init__(
        self,
        block_manager: BlockManager,
        max_batch_tokens: int = 128,
        max_active_requests: int = 32,
    ):
        self.block_manager = block_manager
        self.max_batch_tokens = max_batch_tokens
        if max_batch_tokens <= 0 or max_active_requests <= 0:
            raise ValueError("Scheduler limits must be positive")
        self.max_active_requests = max_active_requests
        self._deferred = []

        self.waiting = deque()
        self.running = []

    def add_request(self, request: Request) -> None:
        self.waiting.append(request)

    def schedule(self) -> list[tuple[Request, int]]:
        budget = self.max_batch_tokens
        scheduled = []

        # Prioritize requests already running.
        # Previously budget-starved requests get the next turn.
        candidates = self._deferred + [
            req for req in list(self.running) + list(self.waiting) if req not in self._deferred
        ]

        for req in candidates:
            if budget <= 0:
                break
            if req not in self.running and len(self.running) >= self.max_active_requests:
                continue

            remaining = req.num_tokens - req.num_computed_tokens

            if remaining <= 0:
                continue

            # How many tokens can existing reserved blocks
            # plus currently free blocks accommodate?
            capacity = len(req.block_ids) * self.block_manager.block_size

            # Include cache-only evictable storage in allocation capacity.
            desired = min(remaining, budget)
            needed = max(
                0,
                (req.num_computed_tokens + desired - capacity + self.block_manager.block_size - 1)
                // self.block_manager.block_size,
            )
            if needed > self.block_manager.num_free_blocks:
                callback = self.block_manager.on_memory_pressure
                if callback is not None:
                    callback(needed - self.block_manager.num_free_blocks)

            available_capacity = (
                capacity
                - req.num_computed_tokens
                + self.block_manager.num_free_blocks * self.block_manager.block_size
            )

            num_tokens = min(
                remaining,
                budget,
                available_capacity,
            )

            if num_tokens <= 0:
                continue

            target = req.num_computed_tokens + num_tokens

            self.block_manager.reserve_for(req, target)

            if req not in self.running:
                self.waiting.remove(req)
                self.running.append(req)

            scheduled.append((req, num_tokens))
            budget -= num_tokens

        selected = {req.request_id for req, _ in scheduled}
        self._deferred = [req for req in candidates if req.request_id not in selected]
        return scheduled

    def complete_step(
        self,
        scheduled: list[tuple[Request, int]],
    ) -> None:
        # Call only after successful model execution.
        for req, num_tokens in scheduled:
            req.num_computed_tokens += num_tokens

    def finish_request(self, request: Request) -> None:
        if request in self.running:
            self.running.remove(request)
        if request in self.waiting:
            self.waiting.remove(request)
        if request in self._deferred:
            self._deferred.remove(request)
        self.block_manager.free(request.block_ids)
        request.block_ids.clear()
