
from mini_vllm.engine.continuous_engine import (
    ContinuousBatchEngine,
)

engine = ContinuousBatchEngine()

# Request A arrives first.
engine.add_request(
    "A",
    "Explain what an operating system is.",
    max_tokens=12,
)

for i in range(3):
    print("Step:", i)
    print("Generated:", engine.step())

# Request B arrives while A is still active.
engine.add_request(
    "B",
    "What is a database?",
    max_tokens=8,
)

# Run until every request finishes.
steps = 0

while engine.requests:
    engine.step()
    steps += 1

    if steps > 100:
        raise RuntimeError("Generation stalled")

print("\nFinal outputs:")

for request_id, result in engine.results.items():
    print(request_id, ":", result)

assert "A" in engine.results
assert "B" in engine.results
assert engine.block_manager.num_free_blocks == 1024
