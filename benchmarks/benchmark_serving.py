"""Matched sequential HF cached decode versus the shared serving worker.

This measures the worker boundary, not HTTP/network transport. All clients arrive
at once. Model loading, per-backend warmup, and warm-cache priming are excluded.
"""

import asyncio
import cProfile
import statistics
from pathlib import Path
from threading import Event, Thread
from time import perf_counter

import psutil
import torch

from mini_vllm.engine.continuous_engine import ContinuousBatchEngine
from mini_vllm.serving.worker import EngineWorker

from .common import load_runner, metadata, parser, prompts, save, summary


class RSSSampler:
    def __enter__(self):
        self.process = psutil.Process()
        self.peak = self.process.memory_info().rss
        self.stop = Event()
        self.thread = Thread(target=self.sample, daemon=True)
        self.thread.start()
        return self

    def sample(self):
        while not self.stop.wait(0.001):
            self.peak = max(self.peak, self.process.memory_info().rss)

    def __exit__(self, *args):
        self.peak = max(self.peak, self.process.memory_info().rss)
        self.stop.set()
        self.thread.join()


def metrics(records, elapsed, peak, cache_blocks=0, free_blocks=None, shapes=None):
    outputs = sum(len(row["tokens"]) for row in records)
    prompt_count = sum(row["prompt_tokens"] for row in records)
    gaps = [b - a for row in records for a, b in zip(row["token_times"], row["token_times"][1:])]
    return {
        "elapsed_seconds": elapsed,
        "requests_per_second": len(records) / elapsed,
        "output_tokens_per_second": outputs / elapsed,
        "total_tokens_per_second": (prompt_count + outputs) / elapsed,
        "ttft_seconds": summary([row["token_times"][0] for row in records]),
        "latency_seconds": summary([row["finished_at"] for row in records]),
        "mean_itl_seconds": statistics.mean(gaps) if gaps else None,
        "cached_tokens": sum(row["cached_tokens"] for row in records),
        "peak_sampled_rss_bytes": peak,
        "cache_blocks": cache_blocks,
        "free_blocks": free_blocks,
        "batch_shapes": shapes or [],
        "requests": records,
    }


@torch.inference_mode()
def sequential(runner, texts, count):
    # Cached autoregressive HF baseline, one request at a time, same weights.
    records = []
    with RSSSampler() as memory:
        start = perf_counter()
        for text in texts:
            prompt = runner.tokenizer.encode(text)
            chunk, cache, generated, times = prompt, None, [], []
            for _ in range(count):
                logits, cache = runner.forward_chunk(
                    chunk, cache, len(prompt) + len(generated) - 1 if generated else 0
                )
                token = int(logits.argmax(-1).item())
                generated.append(token)
                times.append(perf_counter() - start)
                if token == runner.tokenizer.eos_token_id:
                    break
                chunk = [token]
            records.append(
                {
                    "prompt_tokens": len(prompt),
                    "tokens": generated,
                    "token_times": times,
                    "finished_at": perf_counter() - start,
                    "cached_tokens": 0,
                }
            )
        elapsed = perf_counter() - start
    return metrics(records, elapsed, memory.peak)


async def batched(runner, texts, args, warm, profile_path=None):
    def factory():
        engine = ContinuousBatchEngine(
            runner=runner,
            num_blocks=args.num_blocks,
            block_size=args.block_size,
            max_batch_tokens=args.max_batch_tokens,
            max_active_requests=8,
        )
        for prompt in texts:
            engine.add_request("warmup", prompt, min(2, args.max_tokens))
            while engine.requests:
                engine.step()
            engine.forget_result("warmup")
        engine.prefix_cache.evict(args.num_blocks)
        if warm:
            engine.add_request("prime", texts[0], 1)
            while engine.requests:
                engine.step()
            engine.forget_result("prime")
        engine.forward_batches.clear()
        return engine

    class ProfiledWorker(EngineWorker):
        def _run(self):
            profiler = cProfile.Profile()
            try:
                profiler.runcall(super()._run)
            finally:
                profiler.dump_stats(str(profile_path))

    worker = (ProfiledWorker if profile_path else EngineWorker)(factory)
    await worker.start()
    try:
        with RSSSampler() as memory:
            begin = perf_counter()
            states = await asyncio.gather(*(worker.submit(text, args.max_tokens) for text in texts))
            await asyncio.gather(*(asyncio.wrap_future(state.result) for state in states))
            elapsed = perf_counter() - begin
        snapshot = await worker.snapshot()
        records = [
            {
                "prompt_tokens": state.prompt_tokens,
                "tokens": state.tokens,
                "token_times": [value - begin for value in state.token_times],
                "finished_at": state.finished_at - begin,
                "cached_tokens": state.cached_tokens,
            }
            for state in states
        ]
        result = metrics(
            records,
            elapsed,
            memory.peak,
            snapshot["cache_blocks"],
            snapshot["free_blocks"],
            snapshot["forward_batches"],
        )
        result["reused_blocks"] = result["cached_tokens"] // args.block_size
        if snapshot["request_references"]:
            raise AssertionError("Serving leaked request-owned blocks")
        return result
    finally:
        await worker.close()


def plot(rows, output):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(1, 2, figsize=(12, 4.5), layout="constrained")
    for backend, style in (("worker", "-o"), ("sequential_hf", "--s")):
        for workload in sorted({row["workload"] for row in rows}):
            selected = sorted(
                (row for row in rows if row["backend"] == backend and row["workload"] == workload),
                key=lambda row: row["concurrency"],
            )
            x = [row["concurrency"] for row in selected]
            axes[0].plot(
                x,
                [row["output_tokens_per_second"]["mean"] for row in selected],
                style,
                label=f"{backend}: {workload}",
            )
            axes[1].plot(x, [row["ttft_p95_seconds"]["mean"] * 1000 for row in selected], style)
    for axis in axes:
        axis.set_xlabel("Simultaneous requests")
        axis.set_xticks([1, 2, 4, 8])
        axis.grid(alpha=0.2)
    axes[0].set_ylabel("Output tokens / second")
    axes[1].set_ylabel("p95 time to first token (ms)")
    figure.legend(
        *axes[0].get_legend_handles_labels(), loc="outside lower center", ncol=2, fontsize=7
    )
    figure.savefig(output, dpi=160)
    plt.close(figure)


async def run(args):
    start = perf_counter()
    runner = load_runner(args)
    loading = perf_counter() - start
    # Exclude HF cached-decode warmup as well as paged warmup.
    sequential(runner, prompts(args.offline)[:1], args.max_tokens)
    trials = []
    workloads = ("short-cold", "long-cold", "mixed-cold", "short-warm", "long-warm")
    for workload in workloads:
        for concurrency in args.concurrency:
            short, long = prompts(args.offline), prompts(args.offline, long=True)
            texts = (long if workload.startswith("long") else short)[:concurrency]
            if workload.startswith("mixed"):
                texts = [
                    short[index] if index % 2 == 0 else long[index] for index in range(concurrency)
                ]
            for iteration in range(args.iterations):

                async def execute(backend):
                    if backend == "sequential_hf":
                        return sequential(runner, texts, args.max_tokens)
                    profile_path = None
                    if (
                        args.profile
                        and workload == "long-cold"
                        and concurrency == max(args.concurrency)
                        and iteration == 0
                    ):
                        profile_path = args.profile
                        profile_path.parent.mkdir(parents=True, exist_ok=True)
                    return await batched(
                        runner, texts, args, workload.endswith("warm"), profile_path
                    )

                pair = {}
                order = (
                    ("worker", "sequential_hf") if iteration % 2 else ("sequential_hf", "worker")
                )
                for backend in order:
                    pair[backend] = await execute(backend)
                if [row["tokens"] for row in pair["worker"]["requests"]] != [
                    row["tokens"] for row in pair["sequential_hf"]["requests"]
                ]:
                    raise AssertionError("Worker outputs differ from the matched HF baseline")
                for backend, result in pair.items():
                    trials.append(
                        {
                            "backend": backend,
                            "workload": workload,
                            "concurrency": concurrency,
                            "iteration": iteration,
                            "output_verified": True,
                            **result,
                        }
                    )
    aggregated = []
    for workload in workloads:
        for concurrency in args.concurrency:
            for backend in ("worker", "sequential_hf"):
                selected = [
                    row
                    for row in trials
                    if row["workload"] == workload
                    and row["concurrency"] == concurrency
                    and row["backend"] == backend
                ]
                aggregated.append(
                    {
                        "backend": backend,
                        "workload": workload,
                        "concurrency": concurrency,
                        **{
                            key: summary([row[key] for row in selected])
                            for key in (
                                "requests_per_second",
                                "output_tokens_per_second",
                                "total_tokens_per_second",
                                "peak_sampled_rss_bytes",
                            )
                        },
                        "ttft_p95_seconds": summary(
                            [row["ttft_seconds"]["p95"] for row in selected]
                        ),
                        "latency_p95_seconds": summary(
                            [row["latency_seconds"]["p95"] for row in selected]
                        ),
                    }
                )
    return {
        "environment": metadata(runner, args, loading),
        "max_new_tokens": args.max_tokens,
        "measurement": "In-process worker boundary; excludes HTTP and network. Concurrent arrivals share a start time, including sequential baseline queueing. RSS sampled at 1ms (may miss brief peaks). Total tokens/s counts logical prompt+output tokens, including reused tokens.",
        "trials": trials,
        "summary": aggregated,
    }


def main():
    cli = parser(__doc__)
    cli.add_argument(
        "--concurrency", nargs="+", type=int, choices=(1, 2, 4, 8), default=[1, 2, 4, 8]
    )
    cli.add_argument("--max-tokens", type=int, default=8)
    cli.add_argument(
        "--profile",
        type=Path,
        help="Save cProfile of one owner thread, including setup/warmup/idle",
    )
    args = cli.parse_args()
    if not 1 <= args.max_tokens <= 128:
        cli.error("--max-tokens must be 1–128")
    result = asyncio.run(run(args))
    save(result, args.output)
    if args.output:
        plot(result["summary"], args.output.with_suffix(".png"))


if __name__ == "__main__":
    main()
