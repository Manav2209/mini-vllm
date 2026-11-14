from mini_vllm.testing import tiny_runner

runner = tiny_runner()

token, kv_cache = runner.prefill("1 2 3 4 5")

generated = [token.item()]

for _ in range(10):
    token, kv_cache = runner.decode(token, kv_cache)
    generated.append(token.item())

print(runner.tokenizer.decode(generated))
