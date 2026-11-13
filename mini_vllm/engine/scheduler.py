
from collections import deque

from .request import Request
from .block_manager import BlockManager


class Scheduler:
    def __init__(
        self,
        block_manager: BlockManager,
        max_batch_tokens: int = 128,
    ):
        self.block_manager = block_manager
        self.max_batch_tokens = max_batch_tokens

        self.waiting = deque()
        self.running = []

    def add_request(self, request: Request) -> None:
        self.waiting.append(request)


    def schedule(self) -> list[tuple[Request, int]]:
        budget = self.max_batch_tokens
        scheduled = []

        # Prioritize requests already running.
        candidates = list(self.running) + list(self.waiting)

        for req in candidates:
            if budget <= 0:
                break

            remaining = (
                req.num_tokens - req.num_computed_tokens
            )

            if remaining <= 0:
                continue

            # How many tokens can existing reserved blocks
            # plus currently free blocks accommodate?
            capacity = (
                len(req.block_ids) * self.block_manager.block_size
            )

            available_capacity = (
                capacity - req.num_computed_tokens
                + self.block_manager.num_free_blocks
                * self.block_manager.block_size
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

        return scheduled


    def complete_step(
        self,
        scheduled: list[tuple[Request, int]],
    ) -> None:
        # Call only after successful model execution.
        for req, num_tokens in scheduled:
            req.num_computed_tokens += num_tokens

    def finish_request(self, request: Request) -> None:
        self.running.remove(request)
        self.block_manager.free(request.block_ids)
        request.block_ids.clear()
