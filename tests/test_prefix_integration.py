
import unittest

from mini_vllm.engine.continuous_engine import (
    ContinuousBatchEngine,
)


class TestPrefixIntegration(unittest.TestCase):

    def test_prefix_reuse(self):
        engine = ContinuousBatchEngine(
            model_name="sshleifer/tiny-gpt2",
            num_blocks=128,
            block_size=4,
        )

        prompt_a = (
            "Python is a programming language "
            "used for machine learning and web development."
        )

        prompt_b = (
            "Python is a programming language "
            "used for machine learning and data analysis."
        )

        engine.add_request("A", prompt_a, max_tokens=5)

        while "A" in engine.requests:
            engine.step()

        # Request B should find cached prefix blocks.
        engine.add_request("B", prompt_b, max_tokens=5)

        req_b = engine.requests["B"]

        self.assertGreater(req_b.num_cached_tokens, 0)
        self.assertEqual(
            req_b.num_computed_tokens,
            req_b.num_cached_tokens,
        )

        print(
            "Cached tokens:",
            req_b.num_cached_tokens,
        )

        while "B" in engine.requests:
            engine.step()

        self.assertIn("B", engine.results)


if __name__ == "__main__":
    unittest.main()
