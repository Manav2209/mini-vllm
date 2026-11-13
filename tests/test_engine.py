
from mini_vllm.engine.llm_engine import LLMEngine

engine = LLMEngine(
    model_name="sshleifer/tiny-gpt2",
    max_batch_tokens=4,
)

prompt = (
    "Large language models generate text "
    "by predicting the next token."
)

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

