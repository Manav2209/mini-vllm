
import unittest

from mini_vllm.engine.block_manager import BlockManager
from mini_vllm.engine.prefix_cache import PrefixCache


class TestLRU(unittest.TestCase):

    def test_eviction(self):
        manager = BlockManager(3, 4)

        cache = PrefixCache(
            block_size=4,
            cache_namespace="test-model",
            block_manager=manager,
        )

        manager.on_memory_pressure = cache.evict

        tokens_a = [1, 2, 3, 4]
        tokens_b = [5, 6, 7, 8]
        tokens_c = [9, 10, 11, 12]

        for tokens in (tokens_a, tokens_b, tokens_c):
            block = manager.allocate(1)[0]
            block_hash = cache.compute_hashes(tokens)[0]

            # Simulate a completed KV write.
            cache.register_block(block_hash, block)

            # Original request releases its reference.
            manager.free([block])

        self.assertEqual(manager.num_free_blocks, 0)

        # Refresh A, making B the oldest entry.
        block_ids, _ = cache.acquire_prefix(tokens_a)
        manager.free(block_ids)

        # The next allocation automatically evicts B.
        new_block = manager.allocate(1)[0]

        blocks, count = cache.lookup(tokens_b)
        self.assertEqual((blocks, count), ([], 0))

        self.assertEqual(
            cache.lookup(tokens_a)[1], 4
        )

        manager.free([new_block])

    def test_pinned_blocks_cannot_be_evicted(self):
        manager = BlockManager(1, 4)

        cache = PrefixCache(
            block_size=4,
            cache_namespace="test-model",
            block_manager=manager,
        )

        manager.on_memory_pressure = cache.evict

        tokens = [1, 2, 3, 4]
        block = manager.allocate(1)[0]

        cache.register_block(
            cache.compute_hashes(tokens)[0],
            block,
        )

        # Cache + original request = refcount 2.
        self.assertEqual(manager.ref_counts[block], 2)

        with self.assertRaises(RuntimeError):
            manager.allocate(1)

        # Original request releases ownership.
        manager.free([block])

        # Now the cache-only block is evictable.
        replacement = manager.allocate(1)

        self.assertEqual(len(replacement), 1)


if __name__ == "__main__":
    unittest.main()
