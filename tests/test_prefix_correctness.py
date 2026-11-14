import unittest
from unittest.mock import patch

import torch

from mini_vllm.engine.continuous_engine import ContinuousBatchEngine
from mini_vllm.testing import tiny_runner


class PrefixCorrectnessTests(unittest.TestCase):
    def setUp(self):
        self.runner = tiny_runner()
        self.engine = ContinuousBatchEngine(runner=self.runner, block_size=4, num_blocks=32)

    def run_request(self, key, tokens, count=3):
        self.engine.add_request(key, " ".join(map(str, tokens)), count)
        while key in self.engine.requests:
            self.engine.step()
        return self.engine.results[key]

    @torch.inference_mode()
    def test_logits_greedy_and_shared_integrity(self):
        self.run_request("seed", list(range(1, 10)))
        blocks, _ = self.engine.prefix_cache.lookup(list(range(1, 9)))
        saved = [
            tensor[:, blocks].clone()
            for tensor in (self.engine.kv_cache.keys, self.engine.kv_cache.values)
        ]
        with self.assertRaisesRegex(ValueError, "published shared"):
            layer_keys, layer_values = self.engine.kv_cache.read(0, [blocks[0]], 0, 1)
            self.engine.kv_cache.write(0, [blocks[0]], 0, layer_keys, layer_values)
        prompt = list(range(1, 9)) + [20, 21]
        self.engine.add_request("warm", " ".join(map(str, prompt)), 3)
        original = self.runner.forward_paged_batch
        observed = []

        def capture(*args):
            logits = original(*args)
            observed.append(logits.clone())
            return logits

        with patch.object(self.runner, "forward_paged_batch", side_effect=capture):
            while self.engine.requests:
                self.engine.step()
        generated = []
        for actual in observed:
            expected = self.runner.model(
                torch.tensor([prompt + generated]), use_cache=False
            ).logits[:, -1]
            torch.testing.assert_close(actual, expected, atol=2e-6, rtol=1e-5)
            generated.append(int(expected.argmax(-1).item()))
        self.assertEqual(self.engine.results["warm"], self.runner.tokenizer.decode(generated))
        for tensor, snapshot in zip(
            (self.engine.kv_cache.keys, self.engine.kv_cache.values), saved
        ):
            self.assertTrue(torch.equal(tensor[:, blocks], snapshot))

    def test_boundaries_and_entire_prompt(self):
        for length in (1, 3, 4, 5, 8, 12):
            with self.subTest(length=length):
                self.engine.prefix_cache.evict(32)
                tokens = list(range(1, length + 1))
                cold = self.run_request(f"cold{length}", tokens)
                self.engine.add_request(f"warm{length}", " ".join(map(str, tokens)), 3)
                req = self.engine.requests[f"warm{length}"]
                self.assertLess(req.num_cached_tokens, length)
                while self.engine.requests:
                    self.engine.step()
                self.assertEqual(cold, self.engine.results[f"warm{length}"])
        with self.assertRaises(ValueError):
            self.engine.add_request("empty", "")

    @torch.inference_mode()
    def test_divergence_at_second_block(self):
        self.run_request("seed", list(range(1, 10)), 1)
        prompt = [1, 2, 3, 4, 20, 21, 22, 23, 24]
        self.engine.add_request("divergent", " ".join(map(str, prompt)), 1)
        self.assertEqual(self.engine.requests["divergent"].num_cached_tokens, 4)
        self.engine.step()
        expected = self.runner.forward_batch([prompt]).argmax(-1).tolist()
        self.assertEqual(self.engine.results["divergent"], self.runner.tokenizer.decode(expected))

    def test_failure_after_partial_write_rolls_back_and_retries(self):
        self.engine.add_request("failure", "1 2 3 4 5", 1)
        original = self.runner.model.forward

        def fail(*args, **kwargs):
            original(*args, **kwargs)
            raise RuntimeError("after write")

        with patch.object(self.runner.model, "forward", side_effect=fail):
            with self.assertRaisesRegex(RuntimeError, "after write"):
                self.engine.step()
        self.assertFalse(self.engine.prefix_cache.hash_to_block)
        self.assertEqual(self.engine.requests["failure"].num_computed_tokens, 0)
        self.assertEqual(self.engine.block_manager.num_free_blocks, 32)
        self.engine.step()
        self.assertIn("failure", self.engine.results)

    def test_duplicate_computation_has_one_canonical_retention(self):
        for key in ("a", "b"):
            self.engine.add_request(key, "1 2 3 4 5", 1)
        self.engine.step()
        self.assertEqual(len(self.engine.prefix_cache.hash_to_block), 1)
        self.assertEqual(list(self.engine.block_manager.ref_counts.values()), [1])
        self.engine.prefix_cache.evict(32)
        self.assertEqual(self.engine.block_manager.num_free_blocks, 32)
