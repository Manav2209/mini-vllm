
from mini_vllm.engine.model_runner import ModelRunner

runner = ModelRunner()

token, kv_cache = runner.prefill("Hello, my name is")

generated = [token.item()]

for _ in range(10):
    token, kv_cache = runner.decode(token, kv_cache)
    generated.append(token.item())

print(runner.tokenizer.decode(generated))
