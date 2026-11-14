import unittest

from mini_vllm.engine.request import Request
from mini_vllm.engine.block_manager import BlockManager
from mini_vllm.engine.scheduler import Scheduler


class TestScheduler(unittest.TestCase):
    def test_chunked_prefill(self):
        manager = BlockManager(
            num_blocks=100,
            block_size=4,
        )

        scheduler = Scheduler(
            block_manager=manager,
            max_batch_tokens=128,
        )

        req_a = Request("A", list(range(100)))
        req_b = Request("B", list(range(50)))
        req_c = Request("C", list(range(200)))

        scheduler.add_request(req_a)
        scheduler.add_request(req_b)
        scheduler.add_request(req_c)

        # First scheduling iteration
        batch = scheduler.schedule()

        result = [(req.request_id, count) for req, count in batch]

        self.assertEqual(result, [("A", 100), ("B", 28)])

        # Simulate successful model execution
        scheduler.complete_step(batch)

        self.assertEqual(req_a.num_computed_tokens, 100)
        self.assertEqual(req_b.num_computed_tokens, 28)
        self.assertEqual(req_c.num_computed_tokens, 0)

        # Second iteration
        batch = scheduler.schedule()

        result = [(req.request_id, count) for req, count in batch]

        self.assertEqual(result, [("C", 128)])

        print("Chunked prefill test passed!")


if __name__ == "__main__":
    unittest.main()
