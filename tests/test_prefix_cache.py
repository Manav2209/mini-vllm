import unittest

from mini_vllm.engine.prefix_cache import PrefixCache
from mini_vllm.engine.block_manager import BlockManager


class TestPrefixCache(unittest.TestCase):
    def setUp(self):
        self.manager = BlockManager(16, 4)
        self.manager.allocate(16)
        self.cache = PrefixCache(
            block_size=4,
            cache_namespace="test-model-v1",
            block_manager=self.manager,
        )

    def test_identical_prefix(self):
        a = [10, 20, 30, 40, 50, 60, 70, 80]
        b = [10, 20, 30, 40, 50, 60, 90, 100]

        hashes_a = self.cache.compute_hashes(a)
        hashes_b = self.cache.compute_hashes(b)

        assert hashes_a[0] == hashes_b[0]
        assert hashes_a[1] != hashes_b[1]

    def test_partial_block(self):
        tokens = [10, 20, 30, 40, 50]

        hashes = self.cache.compute_hashes(tokens)

        assert len(hashes) == 1

    def test_different_model_namespace(self):
        another = PrefixCache(
            block_size=4,
            cache_namespace="different-model",
            block_manager=self.manager,
        )

        tokens = [10, 20, 30, 40]

        assert self.cache.compute_hashes(tokens) != another.compute_hashes(tokens)

    def test_previous_context_matters(self):
        a = [1, 2, 3, 4, 9, 10, 11, 12]
        b = [5, 6, 7, 8, 9, 10, 11, 12]

        hashes_a = self.cache.compute_hashes(a)
        hashes_b = self.cache.compute_hashes(b)

        assert hashes_a[1] != hashes_b[1]

    def test_prefix_lookup(self):
        tokens_a = [
            10,
            20,
            30,
            40,
            50,
            60,
            70,
            80,
        ]

        hashes = self.cache.compute_hashes(tokens_a)

        # Simulate two physically allocated blocks
        # whose KV tensors were already computed.
        self.cache.register_block(hashes[0], 3)
        self.cache.register_block(hashes[1], 7)

        tokens_b = [
            10,
            20,
            30,
            40,
            50,
            60,
            70,
            80,
            90,
            100,
            110,
            120,
        ]

        blocks, cached_tokens = self.cache.lookup(tokens_b)

        self.assertEqual(blocks, [3, 7])
        self.assertEqual(cached_tokens, 8)

    def test_lookup_stops_on_missing_block(self):
        tokens = [
            1,
            2,
            3,
            4,
            5,
            6,
            7,
            8,
            9,
            10,
            11,
            12,
        ]

        hashes = self.cache.compute_hashes(tokens)

        # First and third blocks exist,
        # but the second block is absent.
        self.cache.register_block(hashes[0], 2)
        self.cache.register_block(hashes[2], 9)

        blocks, count = self.cache.lookup(tokens)

        self.assertEqual(blocks, [2])
        self.assertEqual(count, 4)

    def test_unregister_block(self):
        tokens = [1, 2, 3, 4]
        block_hash = self.cache.compute_hashes(tokens)[0]

        self.cache.register_block(block_hash, 5)

        self.cache.unregister_block(5)

        blocks, count = self.cache.lookup(tokens)

        self.assertEqual(blocks, [])
        self.assertEqual(count, 0)


if __name__ == "__main__":
    unittest.main()
