import unittest
from unittest.mock import patch

from mini_vllm.engine.continuous_engine import ContinuousBatchEngine
from mini_vllm.testing import tiny_runner


class TestPrefixIntegration(unittest.TestCase):
    def test_real_work_skipped(self):
        engine = ContinuousBatchEngine(runner=tiny_runner(), block_size=4, num_blocks=32)
        engine.add_request("cold", "1 2 3 4 5 6 7 8 9", 1)
        engine.step()
        engine.add_request("warm", "1 2 3 4 5 6 7 8 10", 1)
        cached = engine.requests["warm"].num_cached_tokens
        self.assertEqual(cached, 8)
        with patch.object(
            engine.runner, "forward_paged_batch", wraps=engine.runner.forward_paged_batch
        ) as forward:
            engine.step()
        self.assertEqual(sum(map(len, forward.call_args.args[0])), 9 - cached)
        self.assertEqual(engine.forward_tokens, 10)
