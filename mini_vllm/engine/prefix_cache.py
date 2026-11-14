import hashlib
import json
from .block_manager import BlockManager
from collections import OrderedDict

class PrefixCache:
    def __init__(
        self,
        block_size: int,
        cache_namespace: str,
        block_manager: BlockManager,
    ):
        if block_size != block_manager.block_size:
            raise ValueError("Block sizes must match")

        self.block_size = block_size
        self.cache_namespace = cache_namespace
        self.block_manager = block_manager
        # Oldest cached block at the beginning.
        self.lru = OrderedDict()

        self.hash_to_block: dict[str, int] = {}
        self.block_to_hash: dict[int, str] = {}

    def hash_block(
        self ,
        previous_hash: str,
        token_ids: list[int],
    ) -> str:
        if len(token_ids) != self.block_size:
            raise ValueError("Only full blocks can be hashed")

        payload = {
            "namespace": self.cache_namespace,
            "previous_hash": previous_hash,
            "token_ids": token_ids,
        }

        encoded = json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")

        return hashlib.sha256(encoded).hexdigest()

    def compute_hashes(
        self,
        token_ids: list[int],
    ) -> list[str]:

        hashes = []
        previous_hash = "ROOT"

        num_full_blocks = (
            len(token_ids) // self.block_size
        )

        for i in range(num_full_blocks):
            start = i * self.block_size
            end = start + self.block_size

            block_tokens = token_ids[start:end]

            block_hash = self.hash_block(
                previous_hash,
                block_tokens,
            )

            hashes.append(block_hash)
            previous_hash = block_hash

        return hashes

    def register_block(
        self,
        block_hash: str,
        block_id: int,
    ) -> None:

        existing = self.hash_to_block.get(block_hash)

        if existing is not None:
            if existing != block_id:
                raise ValueError("Hash already registered")
            return

        if block_id in self.block_to_hash:
            raise ValueError("Block already cached")

        # Cache itself retains one reference.
        self.block_manager.acquire([block_id])

        self.hash_to_block[block_hash] = block_id
        self.block_to_hash[block_id] = block_hash



    def lookup(
        self,
        token_ids: list[int],
    ) -> tuple[list[int], int]:
        """
        Return the longest contiguous cached prefix.

        Returns:
            block_ids: Physical IDs of matching blocks
            cached_tokens: Number of reusable tokens
        """
        hashes = self.compute_hashes(token_ids)

        block_ids = []

        for block_hash in hashes:
            block_id = self.hash_to_block.get(block_hash)

            if block_id is None:
                break

            block_ids.append(block_id)

        cached_tokens = len(block_ids) * self.block_size

        return block_ids, cached_tokens

    def unregister_block(self, block_id: int) -> None:
        block_hash = self.block_to_hash.pop(
            block_id,
            None,
        )
        if block_hash is None:
            return
        self.hash_to_block.pop(block_hash, None)
        self.lru.pop(block_hash, None)
        # Release the cache-retention reference.
        self.block_manager.free([block_id])

    def acquire_prefix(
        self,
        token_ids: list[int],
    ) -> tuple[list[int], int]:
        block_ids, cached_tokens = self.lookup(token_ids)
        # Acquire before exposing the blocks to a request.
        self.block_manager.acquire(block_ids)
        for block_id in block_ids:
            block_hash = self.block_to_hash[block_id]
            self._touch(block_hash)
        return block_ids, cached_tokens


    def _touch(self, block_hash: str) -> None:
        """
        Mark a cached block as recently used.
        """
        if block_hash in self.lru:
            self.lru.move_to_end(block_hash)
        else:
            self.lru[block_hash] = None

    
    def evict(self, num_blocks: int = 1) -> list[int]:
        """
        Evict up to num_blocks cached entries.

        Only blocks with refcount == 1 are eligible:
        the cache owns the sole remaining reference.
        """
        if num_blocks < 0:
            raise ValueError("num_blocks cannot be negative")

        evicted = []

        for block_hash in list(self.lru.keys()):
            if len(evicted) >= num_blocks:
                break

            block_id = self.hash_to_block[block_hash]

            # Active requests may still own the block.
            if self.block_manager.ref_counts[block_id] != 1:
                continue

            # Remove metadata before releasing storage.
            self.unregister_block(block_id)
            self.lru.pop(block_hash, None)

            evicted.append(block_id)

        return evicted



