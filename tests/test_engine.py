from mini_vllm.engine.llm_engine import LLMEngine
from mini_vllm.testing import tiny_runner

engine = LLMEngine(
    model_name="sshleifer/tiny-gpt2",
    max_batch_tokens=4,
    runner=tiny_runner(),
)

prompt = "1 2 3 4 5 6 7 8 9 10"

result = engine.generate(
    prompt=prompt,
    max_tokens=10,
    verbose=True,
)

print("\nGenerated text:", result)
print(
    "Free blocks:",
    engine.block_manager.num_free_blocks,
)
