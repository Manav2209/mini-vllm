import unittest
from unittest.mock import patch
from mini_vllm.engine.cache_engine import CachedContinuousEngine


class FakeRunner:
    eos_token_id = -1

    def __init__(self):
        self.calls = []
        self.batches = []

    def encode(self, prompt):
        return list(range(1, int(prompt) + 1))

    def decode(self, tokens):
        return ",".join(map(str, tokens))

    def forward_batch(self, chunks, caches):
        self.batches.append((chunks, caches))
        self.calls.extend((list(chunk), cache) for chunk, cache in zip(chunks, caches))
        return [(99, (cache or []) + list(chunk)) for chunk, cache in zip(chunks, caches)]


class EngineTests(unittest.TestCase):
    def test_new_request_joins_and_uses_cached_prefix(self):
        runner = FakeRunner()
        engine = CachedContinuousEngine(runner, token_budget=3)
        engine.add_request("A", "5", max_new_tokens=2)
        self.assertEqual(engine.step(), {})
        self.assertEqual(runner.calls[0], ([1, 2, 3], None))
        engine.add_request("B", "2", max_new_tokens=1)
        outputs = engine.step()
        self.assertEqual(outputs, {"A": 99})

        # A gets its remaining prompt chunk, using the previously cached prefix.
        self.assertEqual(runner.calls[1], ([4, 5], [1, 2, 3]))
        self.assertEqual(engine.step(), {"B": 99, "A": 99})
        self.assertEqual(engine.results, {"A": "99,99", "B": "99"})

        # A's first decode step only processes its newly generated token.
        self.assertIn(([99], [1, 2, 3, 4, 5]), runner.calls)

    def test_bad_inputs(self):
        engine = CachedContinuousEngine(FakeRunner())
        with self.assertRaises(ValueError):
            engine.add_request("A", "0")
        engine.add_request("A", "2")
        with self.assertRaises(ValueError):
            engine.add_request("A", "2")

    def test_mixed_batch_arrivals_and_finishes(self):
        runner = FakeRunner()
        engine = CachedContinuousEngine(runner, token_budget=8)
        engine.add_request("A", "2", max_new_tokens=2)
        engine.add_request("B", "5", max_new_tokens=4)
        req_a = engine.requests["A"]
        self.assertEqual(engine.step(), {"A": 99, "B": 99})
        self.assertEqual(len(runner.batches), 1)

        engine.add_request("C", "4", max_new_tokens=1)
        self.assertEqual(engine.step(), {"A": 99, "B": 99, "C": 99})
        chunks, caches = runner.batches[-1]
        self.assertEqual(chunks, [[99], [99], [1, 2, 3, 4]])
        self.assertEqual(caches, [[1, 2], [1, 2, 3, 4, 5], None])
        self.assertEqual(len(runner.batches), 2)
        self.assertIsNone(req_a.cache)
        self.assertEqual(set(engine.requests), {"B"})

        engine.add_request("D", "1", max_new_tokens=1)
        engine.step()
        engine.step()
        self.assertEqual(set(engine.results), {"A", "B", "C", "D"})
        self.assertFalse(engine.active)
        self.assertFalse(engine.requests)
        calls = len(runner.batches)
        self.assertEqual(engine.step(), {})
        self.assertEqual(len(runner.batches), calls)

    def test_eos_finishes_all_batch_rows(self):
        runner = FakeRunner()
        runner.eos_token_id = 99
        engine = CachedContinuousEngine(runner)
        engine.add_request("A", "2")
        engine.add_request("B", "3")
        self.assertEqual(engine.step(), {"A": 99, "B": 99})
        self.assertFalse(engine.requests)
        self.assertEqual(engine.results, {"A": "99", "B": "99"})

    def test_failed_batch_can_be_retried(self):
        runner = FakeRunner()
        engine = CachedContinuousEngine(runner, token_budget=4)
        engine.add_request("A", "2", max_new_tokens=2)
        engine.add_request("B", "1", max_new_tokens=2)
        engine.step()
        before = [(req.computed, req.cache, list(req.generated)) for req in engine.active]
        with patch.object(runner, "forward_batch", side_effect=RuntimeError("model failed")):
            with self.assertRaisesRegex(RuntimeError, "model failed"):
                engine.step()
        self.assertEqual(
            [(req.computed, req.cache, req.generated) for req in engine.active],
            before,
        )
        self.assertEqual(engine.step(), {"A": 99, "B": 99})
        self.assertFalse(engine.requests)

    def test_token_budget_rotates_unscheduled_requests(self):
        runner = FakeRunner()
        engine = CachedContinuousEngine(runner, token_budget=1)
        for request_id in ("A", "B", "C"):
            engine.add_request(request_id, "1", max_new_tokens=2)
        self.assertEqual(engine.step(), {"A": 99})
        self.assertEqual(engine.step(), {"B": 99})
        self.assertEqual(engine.step(), {"C": 99})


if __name__ == "__main__":
    unittest.main()
