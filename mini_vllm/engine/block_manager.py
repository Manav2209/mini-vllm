
from collections import deque

from .request import Request


class BlockManager:
    def __init__(self, num_blocks: int, block_size: int):
        if num_blocks <= 0 or block_size <= 0:
            raise ValueError("Block count and block size must be positive")
        self.num_blocks = num_blocks
        self.block_size = block_size

        # Available physical block IDs
        self.free_blocks = deque(range(num_blocks))
        self.allocated_blocks: set[int] = set()

    def allocate(self, num_blocks: int) -> list[int]:
        if num_blocks < 0:
            raise ValueError("Block count cannot be negative")
        if len(self.free_blocks) < num_blocks:
            raise RuntimeError("Not enough free KV blocks")

        block_ids = [
            self.free_blocks.popleft()
            for _ in range(num_blocks)
        ]
        self.allocated_blocks.update(block_ids)
        return block_ids

    def free(self, block_ids: list[int]) -> None:
        if len(set(block_ids)) != len(block_ids) or any(
            block_id not in self.allocated_blocks for block_id in block_ids
        ):
            raise ValueError("Cannot free duplicate or unallocated blocks")
        self.allocated_blocks.difference_update(block_ids)
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

