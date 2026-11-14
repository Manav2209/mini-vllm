"""Offline model tests: cached batches must match independent full-prefix forwards."""
import unittest
from unittest.mock import patch

import torch
from transformers import GPT2Config, GPT2LMHeadModel

from mini_vllm.engine.cache_engine import CachedContinuousEngine, HFRunner
from mini_vllm.engine.continuous_engine import ContinuousBatchEngine
from mini_vllm.engine.model_runner import ModelRunner


class TokenTokenizer:
    pad_token_id = 0
    eos_token_id = None

    def encode(self, prompt):
        return list(map(int, prompt.split()))

    def decode(self, tokens, skip_special_tokens=True):
        return ",".join(map(str, tokens))


class BatchedCacheTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with torch.random.fork_rng():
            torch.manual_seed(17)
            cls.model = GPT2LMHeadModel(GPT2Config(
                vocab_size=97, n_positions=64, n_embd=24, n_layer=2, n_head=2,
                bos_token_id=0, eos_token_id=None,
            )).eval()

    def setUp(self):
        self.runner = ModelRunner.__new__(ModelRunner)
        self.runner.device = "cpu"
        self.runner.tokenizer = TokenTokenizer()
        self.runner.model = self.model

    @torch.inference_mode()
    def assert_matches_full_prefixes(self, logits, caches, prefixes):
        for row, (cache, prefix) in enumerate(zip(caches, prefixes)):
            expected = self.model(input_ids=torch.tensor([prefix]), use_cache=True)
            torch.testing.assert_close(logits[row], expected.logits[0, -1], atol=2e-6, rtol=1e-5)
            for (key, value), layer in zip(cache, expected.past_key_values.layers):
                self.assertEqual(key.shape[-2], len(prefix))
                torch.testing.assert_close(key, layer.keys, atol=2e-6, rtol=1e-5)
                torch.testing.assert_close(value, layer.values, atol=2e-6, rtol=1e-5)
                # A cache owns only its request's states, not the whole batch.
                self.assertEqual(key.untyped_storage().nbytes(), key.numel() * key.element_size())

    def test_unequal_prefill_and_mixed_cached_chunks(self):
        prefixes = [[1, 2, 3], [4], [5, 6, 7, 8, 9]]
        with patch.object(self.model, "forward", wraps=self.model.forward) as forward:
            logits, caches = self.runner.forward_cached_batch(prefixes, [None] * 3)
            self.assertEqual(forward.call_count, 1)
            self.assertEqual(tuple(forward.call_args.kwargs["input_ids"].shape), (3, 5))
        self.assert_matches_full_prefixes(logits, caches, prefixes)
        original = [[(key.clone(), value.clone()) for key, value in cache] for cache in caches]

        chunks = [[13], [14, 15, 16], [17, 18]]
        with patch.object(self.model, "forward", wraps=self.model.forward) as forward:
            logits, updated = self.runner.forward_cached_batch(chunks, [caches[0], None, caches[2]])
            self.assertEqual(forward.call_count, 1)
            self.assertEqual(tuple(forward.call_args.kwargs["input_ids"].shape), (3, 3))
            self.assertEqual(forward.call_args.kwargs["position_ids"].tolist(), [
                [3, 0, 0], [0, 1, 2], [5, 6, 0],
            ])
            self.assertEqual(forward.call_args.kwargs["attention_mask"].tolist(), [
                [0, 0, 1, 1, 1, 1, 0, 0],
                [0, 0, 0, 0, 0, 1, 1, 1],
                [1, 1, 1, 1, 1, 1, 1, 0],
            ])
        full = [prefixes[0] + chunks[0], chunks[1], prefixes[2] + chunks[2]]
        self.assert_matches_full_prefixes(logits, updated, full)
        for cache, saved in zip(caches, original):
            for pair, previous in zip(cache, saved):
                for tensor, old in zip(pair, previous):
                    torch.testing.assert_close(tensor, old, rtol=0, atol=0)

        # Remove a finished row and reorder the survivors before decoding.
        with patch.object(self.model, "forward", wraps=self.model.forward) as forward:
            logits, caches = self.runner.forward_cached_batch([[19], [20]], [updated[2], updated[0]])
            self.assertEqual(forward.call_count, 1)
            self.assertEqual(tuple(forward.call_args.kwargs["input_ids"].shape), (2, 1))
        self.assert_matches_full_prefixes(logits, caches, [full[2] + [19], full[0] + [20]])

    def test_invalid_batches(self):
        for chunks, caches in (([], []), ([[]], [None]), ([[1]], [])):
            with self.subTest(chunks=chunks, caches=caches), self.assertRaises(ValueError):
                self.runner.forward_cached_batch(chunks, caches)

    @torch.inference_mode()
    def greedy_reference(self, prompt, count):
        tokens = self.runner.tokenizer.encode(prompt)
        generated = []
        for _ in range(count):
            token = int(self.model(input_ids=torch.tensor([tokens]), use_cache=False).logits[0, -1].argmax())
            tokens.append(token)
            generated.append(token)
        return self.runner.tokenizer.decode(generated)

    def test_both_engines_match_greedy_generation_with_dynamic_requests(self):
        specs = {"A": ("1 2", 4), "B": ("3", 2), "C": ("4 5 6 7 8", 2), "D": ("9", 1)}
        expected = {name: self.greedy_reference(*spec) for name, spec in specs.items()}
        for engine_type in (CachedContinuousEngine, ContinuousBatchEngine):
            with self.subTest(engine=engine_type.__name__):
                with patch("mini_vllm.engine.model_runner.ModelRunner", return_value=self.runner):
                    hf_runner = HFRunner()
                if engine_type is CachedContinuousEngine:
                    engine = engine_type(hf_runner, token_budget=4)
                    add = engine.add_request
                else:
                    with patch("mini_vllm.engine.continuous_engine.ModelRunner", return_value=self.runner):
                        engine = engine_type()
                    engine.scheduler.max_batch_tokens = 4
                    add = engine.add_request
                add("A", *specs["A"])
                add("B", *specs["B"])
                shapes = []
                with patch.object(self.model, "forward", wraps=self.model.forward) as forward:
                    for step in range(4):
                        if step == 1:
                            add("C", *specs["C"])
                        if step == 2:
                            add("D", *specs["D"])
                        before = forward.call_count
                        emitted = engine.step()
                        self.assertEqual(forward.call_count, before + 1)
                        shapes.append(tuple(forward.call_args.kwargs["input_ids"].shape))
                        if step == 1:
                            self.assertNotIn("C", emitted)  # incomplete prefill
                            self.assertIn("B", engine.results)
                        if step == 2:
                            self.assertIn("C", emitted)
                    self.assertEqual(engine.step(), {})
                    self.assertEqual(forward.call_count, 4)
                self.assertEqual(shapes, [(2, 2), (3, 2), (2, 3), (3, 1)])
                self.assertEqual(engine.results, expected)
                self.assertFalse(engine.requests)
                if engine_type is ContinuousBatchEngine:
                    self.assertEqual(engine.block_manager.num_free_blocks, 1024)
                else:
                    self.assertFalse(engine.active)
                    self.assertEqual(hf_runner.block_manager.num_free_blocks, 1024)


if __name__ == "__main__":
    unittest.main()
