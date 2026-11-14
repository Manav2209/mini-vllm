from mini_vllm.engine.cache_engine import CachedContinuousEngine, HFRunner

engine = CachedContinuousEngine(HFRunner(), token_budget=8)
engine.add_request("A", "What is a KV cache?", max_new_tokens=8)
for _ in range(2):
    print("tokens:", engine.step())
engine.add_request("B", "Explain batching briefly.", max_new_tokens=6)
for _ in range(100):
    if not engine.requests:
        break
    print("tokens:", engine.step())
else:
    raise RuntimeError("Generation stalled")
print(engine.results)
