import unittest

from mini_vllm.engine.continuous_engine import ContinuousBatchEngine
from mini_vllm.testing import tiny_runner


class PrefixLifecycleTests(unittest.TestCase):
    def test_live_borrowers_cancel_and_eviction(self):
        engine = ContinuousBatchEngine(runner=tiny_runner(), block_size=4, num_blocks=8)
        engine.add_request("seed", "1 2 3 4 5", 1)
        engine.step()
        for key in ("a", "b"):
            engine.add_request(key, "1 2 3 4 6", 3)
        block = engine.requests["a"].block_ids[0]
        self.assertEqual(engine.block_manager.ref_counts[block], 3)
        self.assertEqual(engine.prefix_cache.evict(8), [])
        self.assertTrue(engine.cancel_request("a"))
        self.assertFalse(engine.cancel_request("a"))
        self.assertEqual(engine.block_manager.ref_counts[block], 2)
        while engine.requests:
            engine.step()
        self.assertEqual(engine.block_manager.ref_counts[block], 1)
        engine.prefix_cache.evict(8)
        self.assertEqual(engine.block_manager.num_free_blocks, 8)
        self.assertFalse(engine.scheduler.waiting)
        self.assertFalse(engine.scheduler.running)

    def test_scheduler_reclaims_cache_only_pool(self):
        engine = ContinuousBatchEngine(runner=tiny_runner(), block_size=2, num_blocks=2)
        engine.add_request("a", "1 2 3 4", 1)
        engine.step()
        self.assertEqual(engine.block_manager.num_free_blocks, 0)
        engine.add_request("b", "5 6", 1)
        engine.step()
        self.assertIn("b", engine.results)
        engine.prefix_cache.evict(8)
        self.assertEqual(engine.block_manager.num_free_blocks, 2)

    def test_fairness_with_tiny_budget(self):
        engine = ContinuousBatchEngine(runner=tiny_runner(), max_batch_tokens=1)
        for key in ("a", "b", "c"):
            engine.add_request(key, "1", 3)
        self.assertEqual(list(engine.step()), ["a"])
        self.assertEqual(list(engine.step()), ["b"])
        self.assertEqual(list(engine.step()), ["c"])
        for key in list(engine.requests):
            engine.cancel_request(key)
