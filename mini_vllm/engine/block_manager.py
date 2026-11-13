
from collections import deque

from .request import Request


class BlockManager:
    def __init__(self, num_blocks: int, block_size: int):
        self.num_blocks = num_blocks
        self.block_size = block_size

        # Available physical block IDs
        self.free_blocks = deque(range(num_blocks))

    def allocate(self, num_blocks: int) -> list[int]:
        if len(self.free_blocks) < num_blocks:
            raise RuntimeError("Not enough free KV blocks")

        return [
            self.free_blocks.popleft()
            for _ in range(num_blocks)
        ]

    def free(self, block_ids: list[int]) -> None:
        self.free_blocks.extend(block_ids)
    
    def reserve(self, request: Request) -> None:
        required_blocks = (
            len(request.token_ids) + self.block_size - 1
        ) // self.block_size

        additional_blocks = required_blocks - len(request.block_ids)

        if additional_blocks <= 0:
            return

        new_blocks = self.allocate(additional_blocks)
        request.block_ids.extend(new_blocks)


    @property
    def num_free_blocks(self) -> int:
        return len(self.free_blocks)
        
    
    def reserve_for(
        self,
        request: Request,
        total_tokens: int,
    ) -> None:
        if total_tokens < 0:
            raise ValueError("Token count cannot be negative")

        required = (
            total_tokens + self.block_size - 1
        ) // self.block_size

        additional = required - len(request.block_ids)

        if additional <= 0:
            return

        new_blocks = self.allocate(additional)
        request.block_ids.extend(new_blocks)

