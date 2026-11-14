
import unittest

from mini_vllm.engine.block_manager import BlockManager
from mini_vllm.engine.prefix_cache import PrefixCache


class TestReferenceCounting(unittest.TestCase):

    def test_shared_blocks(self):
        manager = BlockManager(
            num_blocks=4,
            block_size=4,
        )

        cache = PrefixCache(
            block_size=4,
            cache_namespace="test-model",
            block_manager=manager,
        )

        tokens = [10, 20, 30, 40]

        # Request A allocates a block.
        block_id = manager.allocate(1)[0]

        assert manager.ref_counts[block_id] == 1

        # Simulate completed KV computation.
        block_hash = cache.compute_hashes(tokens)[0]
        cache.register_block(block_hash, block_id)

        assert manager.ref_counts[block_id] == 2

        # Request B reuses the cached prefix.
        blocks, cached_tokens = cache.acquire_prefix(tokens)

        assert blocks == [block_id]
        assert cached_tokens == 4
        assert manager.ref_counts[block_id] == 3

        # Request A finishes.
        manager.free([block_id])

        assert manager.ref_counts[block_id] == 2

        # Request B finishes.
        manager.free(blocks)

        assert manager.ref_counts[block_id] == 1

        # Cache eviction releases final reference.
        cache.unregister_block(block_id)

        assert block_id not in manager.ref_counts
        assert manager.num_free_blocks == 4


if __name__ == "__main__":
    unittest.main()
