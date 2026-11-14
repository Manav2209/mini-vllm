"""Physical KV storage, address translation, and model-level reuse checks."""

import unittest
from unittest.mock import patch

import torch
from transformers import GPT2Config, GPT2LMHeadModel

from mini_vllm.engine.block_manager import BlockManager
from mini_vllm.engine.cache_engine import CachedContinuousEngine, HFRunner
from mini_vllm.engine.continuous_engine import ContinuousBatchEngine
from mini_vllm.engine.model_runner import ModelRunner
from mini_vllm.engine.paged_cache import PagedBatchCache, PagedKVCache
from mini_vllm.engine.request import Request
from tests.test_batched_cache import TokenTokenizer


class PhysicalKVTests(unittest.TestCase):
    def test_fragmented_mapping_and_cross_block_reads_and_writes(self):
        manager = BlockManager(num_blocks=4, block_size=2)
        pool = PagedKVCache(manager, num_layers=2, num_kv_heads=2, head_dim=3)
        self.assertEqual(tuple(pool.keys.shape), (2, 4, 2, 2, 3))
        pointer = pool.keys.data_ptr()
        manager.allocate(3)
        table = [2, 0, 1]
        keys = torch.arange(30, dtype=torch.float32).reshape(1, 2, 5, 3)
        values = keys + 100
        pool.write(0, table, 0, keys, values)
        self.assertEqual(
            [pool.physical_slot(table, p) for p in range(5)],
            [
                (2, 0),
                (2, 1),
                (0, 0),
                (0, 1),
                (1, 0),
            ],
        )
        for token in range(5):
            block, offset = pool.physical_slot(table, token)
            torch.testing.assert_close(pool.keys[0, block, offset], keys[0, :, token])
        for actual, expected in zip(pool.read(0, table, 0, 5), (keys, values)):
            torch.testing.assert_close(actual, expected)

        patch_keys = torch.full((1, 2, 3, 3), -7.0)
        pool.write(0, table, 1, patch_keys, patch_keys - 10)
        actual, actual_values = pool.read(0, table, 0, 5)
        torch.testing.assert_close(actual[:, :, 1:4], patch_keys)
        torch.testing.assert_close(actual_values[:, :, 1:4], patch_keys - 10)
        torch.testing.assert_close(actual[:, :, 0], keys[:, :, 0])
        torch.testing.assert_close(actual[:, :, 4], keys[:, :, 4])
        self.assertEqual(pool.keys.data_ptr(), pointer)
        self.assertEqual(torch.count_nonzero(pool.keys[1]).item(), 0)

    def test_free_reuse_and_allocation_failures_are_atomic(self):
        manager = BlockManager(num_blocks=2, block_size=2)
        pool = PagedKVCache(manager, 1, 1, 2)
        table = manager.allocate(2)
        pool.write(0, table, 0, torch.ones(1, 1, 4, 2), torch.ones(1, 1, 4, 2))
        with self.assertRaises(RuntimeError):
            manager.allocate(1)
        self.assertEqual(manager.num_free_blocks, 0)
        with self.assertRaises(ValueError):
            manager.free([table[0], table[0]])
        self.assertEqual(manager.allocated_blocks, set(table))
        manager.free(table)
        with self.assertRaises(ValueError):
            pool.read(0, table, 0, 1)
        with self.assertRaises(ValueError):
            manager.free(table)
        self.assertEqual(manager.num_free_blocks, 2)
        reused = manager.allocate(2)
        self.assertEqual(reused, table)
        new = torch.full((1, 1, 1, 2), 19.0)
        pool.write(0, reused, 0, new, -new)
        key, value = pool.read(0, reused, 0, 1)
        torch.testing.assert_close(key, new)
        torch.testing.assert_close(value, -new)
        # Reuse needs no tensor reallocation or clearing of unrelated slots.
        self.assertEqual(pool.keys[0, reused[0], 1, 0, 0].item(), 1.0)

    def test_invalid_dimensions_and_access(self):
        for blocks, size in ((0, 2), (2, 0), (-1, 2)):
            with self.assertRaises(ValueError):
                BlockManager(blocks, size)
        manager = BlockManager(2, 2)
        pool = PagedKVCache(manager, 1, 1, 2)
        table = manager.allocate(1)
        with self.assertRaises(ValueError):
            manager.allocate(-1)
        for layer, start, count in ((1, 0, 1), (0, -1, 1), (0, 0, 3)):
            with self.assertRaises(ValueError):
                pool.read(layer, table, start, count)
        with self.assertRaises(ValueError):
            pool.write(0, table, 0, torch.zeros(1, 2, 1, 2), torch.zeros(1, 2, 1, 2))
        empty, _ = pool.read(0, [], 0, 0)
        self.assertEqual(tuple(empty.shape), (1, 1, 0, 2))


class PagedModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with torch.random.fork_rng():
            torch.manual_seed(23)
            cls.model = GPT2LMHeadModel(
                GPT2Config(
                    vocab_size=97,
                    n_positions=64,
                    n_embd=24,
                    n_layer=2,
                    n_head=2,
                    bos_token_id=0,
                    eos_token_id=None,
                )
            ).eval()

    def setUp(self):
        self.runner = ModelRunner.__new__(ModelRunner)
        self.runner.device = "cpu"
        self.runner.model = self.model
        self.runner.tokenizer = TokenTokenizer()

    @torch.inference_mode()
    def assert_matches_reference(self, logits, pool, requests, prefixes):
        for row, (req, prefix) in enumerate(zip(requests, prefixes)):
            expected = self.model(input_ids=torch.tensor([prefix]), use_cache=True)
            torch.testing.assert_close(logits[row], expected.logits[0, -1], atol=2e-6, rtol=1e-5)
            for layer_idx, layer in enumerate(expected.past_key_values.layers):
                keys, values = pool.read(layer_idx, req.block_ids, 0, len(prefix))
                torch.testing.assert_close(keys, layer.keys, atol=2e-6, rtol=1e-5)
                torch.testing.assert_close(values, layer.values, atol=2e-6, rtol=1e-5)

    def test_fragmented_model_cache_mixed_chunks_and_reused_rows(self):
        manager = BlockManager(7, 2)
        pool = self.runner.create_paged_cache(manager)
        # The physical pool address stays fixed through prefill, decode and reuse.
        pointers = (pool.keys.data_ptr(), pool.values.data_ptr())
        req_a, req_b = Request("A", []), Request("B", [])
        a_blocks = manager.allocate(3)
        req_a.block_ids = [a_blocks[2], a_blocks[0], a_blocks[1]]
        req_b.block_ids = manager.allocate(1)
        chunks = [[1, 2, 3], [4]]
        with patch.object(self.model, "forward", wraps=self.model.forward) as forward:
            logits = self.runner.forward_paged_batch(chunks, [req_a, req_b], pool)
            self.assertEqual(forward.call_count, 1)
            cache = forward.call_args.kwargs["past_key_values"]
            self.assertIsInstance(cache, PagedBatchCache)
            self.assertTrue(
                all(layer.keys is None and layer.values is None for layer in cache.layers)
            )
        self.assert_matches_reference(logits, pool, [req_a, req_b], chunks)
        req_a.num_computed_tokens, req_b.num_computed_tokens = 3, 1
        a_prefix = pool.read(0, req_a.block_ids, 0, 3)
        a_prefix = tuple(tensor.clone() for tensor in a_prefix)
        req_c = Request("C", [])
        req_c.block_ids = manager.allocate(2)
        chunks = [[5], [6, 7], [8, 9, 10]]
        manager.reserve_for(req_b, 3)
        with patch.object(self.model, "forward", wraps=self.model.forward) as forward:
            logits = self.runner.forward_paged_batch(chunks, [req_a, req_b, req_c], pool)
            self.assertEqual(forward.call_count, 1)
        prefixes = [[1, 2, 3, 5], [4, 6, 7], [8, 9, 10]]
        self.assert_matches_reference(logits, pool, [req_a, req_b, req_c], prefixes)
        for current, old in zip(pool.read(0, req_a.block_ids, 0, 3), a_prefix):
            torch.testing.assert_close(current, old, rtol=0, atol=0)
        req_a.num_computed_tokens, req_c.num_computed_tokens = 4, 3

        released = list(req_b.block_ids)
        manager.free(req_b.block_ids)
        req_b.block_ids.clear()
        req_d = Request("D", [])
        req_d.block_ids = manager.allocate(2)
        self.assertEqual(req_d.block_ids, released)
        # Reordered survivors and a fresh request share a forward over reused blocks.
        with patch.object(self.model, "forward", wraps=self.model.forward) as forward:
            logits = self.runner.forward_paged_batch(
                [[13], [11], [12]], [req_c, req_d, req_a], pool
            )
            self.assertEqual(forward.call_count, 1)
        self.assert_matches_reference(
            logits,
            pool,
            [req_c, req_d, req_a],
            [
                prefixes[2] + [13],
                [11],
                prefixes[0] + [12],
            ],
        )
        self.assertEqual((pool.keys.data_ptr(), pool.values.data_ptr()), pointers)
        for req in (req_a, req_c, req_d):
            manager.free(req.block_ids)
            req.block_ids.clear()
        self.assertEqual(manager.num_free_blocks, 7)

    def test_dynamic_finish_and_reuse_in_both_engines(self):
        for engine_type in (ContinuousBatchEngine, CachedContinuousEngine):
            with self.subTest(engine=engine_type.__name__):
                if engine_type is ContinuousBatchEngine:
                    with patch(
                        "mini_vllm.engine.continuous_engine.ModelRunner", return_value=self.runner
                    ):
                        engine = engine_type(num_blocks=2, block_size=2)
                    manager, pool = engine.block_manager, engine.kv_cache
                else:
                    with patch(
                        "mini_vllm.engine.model_runner.ModelRunner", return_value=self.runner
                    ):
                        hf = HFRunner(num_blocks=2, block_size=2)
                    engine = engine_type(hf)
                    manager, pool = hf.block_manager, hf.kv_cache
                pointer = pool.keys.data_ptr()
                engine.add_request("A", "1 2", 1)
                engine.add_request("B", "3", 3)
                engine.step()
                if engine_type is ContinuousBatchEngine:
                    survivor = engine.requests["B"]
                    freed = set(range(2)) - set(survivor.block_ids)
                else:
                    survivor = engine.requests["B"].cache
                    freed = set(range(2)) - set(survivor.block_ids)
                self.assertIn("A", engine.results)
                if engine_type is ContinuousBatchEngine:
                    engine.prefix_cache.evict(2)
                self.assertEqual(manager.num_free_blocks, 1)
                engine.add_request("C", "4 5", 1)
                with patch.object(self.model, "forward", wraps=self.model.forward) as forward:
                    engine.step()
                    self.assertEqual(forward.call_count, 1)
                    batch_cache = forward.call_args.kwargs["past_key_values"]
                    self.assertEqual(set(batch_cache.layers[0].block_tables[1]), freed)
                self.assertIn("C", engine.results)
                engine.step()
                self.assertFalse(engine.requests)
                if engine_type is ContinuousBatchEngine:
                    engine.prefix_cache.evict(2)
                self.assertEqual(manager.num_free_blocks, 2)
                self.assertEqual(pool.keys.data_ptr(), pointer)
                # Reused storage produces the same output as fresh independent inference.
                for name, prompt in (("A", [1, 2]), ("C", [4, 5])):
                    with torch.inference_mode():
                        token = int(
                            self.model(input_ids=torch.tensor([prompt]), use_cache=False)
                            .logits[0, -1]
                            .argmax()
                        )
                    self.assertEqual(engine.results[name], str(token))

    def test_failed_forward_preserves_prefix_and_can_retry(self):
        with patch("mini_vllm.engine.model_runner.ModelRunner", return_value=self.runner):
            hf = HFRunner(num_blocks=2, block_size=2)
        _, cache = hf.forward([1, 2], None)
        previous = hf.kv_cache.read(0, cache.block_ids, 0, 2)[0].clone()
        original_forward = self.model.forward

        def fail_after_writes(*args, **kwargs):
            original_forward(*args, **kwargs)
            raise RuntimeError("execution failed after KV writes")

        with patch.object(self.model, "forward", side_effect=fail_after_writes):
            with self.assertRaisesRegex(RuntimeError, "after KV writes"):
                hf.forward([3], cache)
        self.assertEqual(cache.num_computed_tokens, 2)
        self.assertEqual(len(cache.block_ids), 1)
        self.assertEqual(hf.block_manager.num_free_blocks, 1)
        torch.testing.assert_close(hf.kv_cache.read(0, cache.block_ids, 0, 2)[0], previous)
        _, same = hf.forward([3], cache)
        self.assertIs(same, cache)
        self.assertEqual(cache.num_computed_tokens, 3)
        hf.release_cache(cache)
        self.assertEqual(hf.block_manager.num_free_blocks, 2)

    def test_exhaustion_returns_new_allocations_and_does_not_stall(self):
        with patch("mini_vllm.engine.model_runner.ModelRunner", return_value=self.runner):
            hf = HFRunner(num_blocks=2, block_size=2)
        _, cache = hf.forward([1, 2], None)
        with self.assertRaisesRegex(RuntimeError, "Not enough free"):
            hf.forward_batch([[3], [4, 5]], [cache, None])
        self.assertEqual(cache.num_computed_tokens, 2)
        self.assertEqual(len(cache.block_ids), 1)
        self.assertEqual(hf.block_manager.num_free_blocks, 1)
        hf.release_cache(cache)
        self.assertEqual(hf.block_manager.num_free_blocks, 2)

        with patch("mini_vllm.engine.continuous_engine.ModelRunner", return_value=self.runner):
            engine = ContinuousBatchEngine(num_blocks=1, block_size=1)
        engine.add_request("long", "1 2")
        self.assertEqual(engine.step(), {})
        with self.assertRaisesRegex(RuntimeError, "exhausted"):
            engine.step()
        engine.finish(engine.requests["long"])
        engine.prefix_cache.evict(1)
        self.assertEqual(engine.block_manager.num_free_blocks, 1)


if __name__ == "__main__":
    unittest.main()
