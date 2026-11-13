
from mini_vllm.engine.request import Request
from mini_vllm.engine.block_manager import BlockManager
from mini_vllm.engine.scheduler import Scheduler

manager = BlockManager(
    num_blocks=100,
    block_size=16,
)

scheduler = Scheduler(
    block_manager=manager,
    max_batch_tokens=128,
)

req = Request("req-1", list(range(1000)))

scheduler.add_request(req)

# First iteration
batch = scheduler.schedule()

assert len(batch) == 1
assert batch[0][1] == 128
assert len(req.block_ids) == 8

scheduler.complete_step(batch)

print("Iteration 1:", req.num_computed_tokens)
print("Allocated blocks:", len(req.block_ids))

# Second iteration
batch = scheduler.schedule()

assert batch[0][1] == 128
assert len(req.block_ids) == 16

scheduler.complete_step(batch)

print("Iteration 2:", req.num_computed_tokens)
print("Allocated blocks:", len(req.block_ids))
