"""Cold/warm prefill, verifying logits and counting real model input tokens."""

from time import perf_counter

import torch

from mini_vllm.engine.continuous_engine import ContinuousBatchEngine

from .common import load_runner, metadata, parser, prompts, save, summary


def main():
    cli = parser(__doc__)
    args = cli.parse_args()
    started = perf_counter()
    runner = load_runner(args)
    loading = perf_counter() - started
    texts = prompts(args.offline, long=True)
    runner.forward_batch([runner.tokenizer.encode(texts[0])])
    engine = ContinuousBatchEngine(
        runner=runner,
        num_blocks=args.num_blocks,
        block_size=args.block_size,
        max_batch_tokens=args.max_batch_tokens,
    )
    # Warm the paged path, not just HF's ordinary dense path.
    for index in range(2):
        for prompt in texts[:2]:
            engine.add_request("warmup", prompt, 1)
            while engine.requests:
                engine.step()
            engine.forget_result("warmup")
        engine.prefix_cache.evict(args.num_blocks)
    records = []
    for iteration in range(args.iterations):
        engine.prefix_cache.evict(args.num_blocks)
        for mode, prompt in zip(("cold", "warm"), texts[:2]):
            key = f"{iteration}-{mode}"
            engine.add_request(key, prompt, 1)
            cached = engine.requests[key].num_cached_tokens
            count = engine.requests[key].prompt_len
            captured = []
            original = runner.forward_paged_batch

            def forward(*params):
                logits = original(*params)
                captured.append(logits.clone())
                return logits

            runner.forward_paged_batch = forward
            before = engine.forward_tokens
            begin = perf_counter()
            try:
                while key in engine.requests:
                    engine.step()
            finally:
                runner.forward_paged_batch = original
            elapsed = perf_counter() - begin
            expected = runner.forward_batch([runner.tokenizer.encode(prompt)])
            torch.testing.assert_close(captured[-1], expected, atol=2e-6, rtol=1e-5)
            verified = engine.results[key] == runner.tokenizer.decode(
                expected.argmax(-1).tolist(),
                skip_special_tokens=True,
            )
            if not verified:
                raise AssertionError("Greedy output differs from full-prompt reference")
            records.append(
                {
                    "iteration": iteration,
                    "mode": mode,
                    "prompt_tokens": count,
                    "cached_tokens": cached,
                    "actual_model_input_tokens": engine.forward_tokens - before,
                    "prefill_seconds": elapsed,
                    "ttft_seconds": elapsed,
                    "cache_hit_rate": cached / count,
                    "output_verified": verified,
                }
            )
            engine.forget_result(key)
    result = {
        "environment": metadata(runner, args, loading),
        "records": records,
        "summary": {
            mode: summary([row["prefill_seconds"] for row in records if row["mode"] == mode])
            for mode in ("cold", "warm")
        },
        "note": "Prefill/TTFT includes scheduler and first sampling; downloads/loading and warmup excluded. CPU speedup is not guaranteed.",
    }
    save(result, args.output)


if __name__ == "__main__":
    main()
