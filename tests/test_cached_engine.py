import unittest
from mini_vllm.engine.cache_engine import CachedContinuousEngine


class FakeRunner:
    eos_token_id = -1
    def __init__(self):
        self.calls = []
    def encode(self, prompt):
        return list(range(1, int(prompt) + 1))
    def decode(self, tokens):
        return ",".join(map(str, tokens))
    def forward(self, token_ids, cache):
        self.calls.append((list(token_ids), cache))
        previous = cache or []
        return 99, previous + list(token_ids)


class EngineTests(unittest.TestCase):
    def test_new_request_joins_and_uses_cached_prefix(self):
        runner = FakeRunner()
        engine = CachedContinuousEngine(runner, token_budget=3)
        engine.add_request("A", "5", max_new_tokens=2)
        self.assertEqual(engine.step(), {})
        self.assertEqual(runner.calls[0], ([1,2,3], None))
        engine.add_request("B", "2", max_new_tokens=1)
        outputs = engine.step()
        self.assertEqual(outputs, {"A": 99})

        # A gets its remaining prompt chunk, using the previously cached prefix.
        self.assertEqual(runner.calls[1], ([4,5], [1,2,3]))
        self.assertEqual(engine.step(), {"B":99, "A":99})
        self.assertEqual(engine.results, {"A":"99,99", "B":"99"})
        
        # A's first decode step only processes its newly generated token.
        self.assertIn(([99], [1,2,3,4,5]), runner.calls)

    def test_bad_inputs(self):
        engine = CachedContinuousEngine(FakeRunner())
        with self.assertRaises(ValueError):
            engine.add_request("A", "0")
        engine.add_request("A", "2")
        with self.assertRaises(ValueError):
            engine.add_request("A", "2")

if __name__ == "__main__":
    unittest.main()
