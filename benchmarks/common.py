import argparse
import json
import platform
import statistics
from pathlib import Path

import psutil
import torch
import transformers

from mini_vllm.engine.model_runner import ModelRunner
from mini_vllm.testing import tiny_runner


def parser(description):
    result = argparse.ArgumentParser(description=description)
    result.add_argument("--model", default="sshleifer/tiny-gpt2")
    result.add_argument(
        "--offline", action="store_true", help="Seeded in-memory GPT-2, integer tokenizer"
    )
    result.add_argument("--iterations", type=int, default=5)
    result.add_argument("--threads", type=int, default=1)
    result.add_argument("--block-size", type=int, default=4)
    result.add_argument("--num-blocks", type=int, default=256)
    result.add_argument("--max-batch-tokens", type=int, default=128)
    result.add_argument("--output", type=Path)
    return result


def load_runner(args):
    if (
        min(args.iterations, args.threads, args.block_size, args.num_blocks, args.max_batch_tokens)
        < 1
    ):
        raise ValueError("Benchmark limits must be positive")
    torch.set_num_threads(args.threads)
    torch.manual_seed(17)
    return tiny_runner() if args.offline else ModelRunner(args.model)


def prompts(offline, long=False):
    if offline:
        prefix = " ".join(str(index % 80 + 1) for index in range(64 if long else 8))
        return [prefix + " " + str(value) for value in range(81, 89)]
    prefix = "A language model predicts tokens using the previous context. " * (8 if long else 1)
    return [
        prefix + suffix
        for suffix in (
            "Explain this.",
            "Give examples.",
            "Discuss memory.",
            "Discuss batching.",
            "Describe tokens.",
            "Describe caches.",
            "Explain attention.",
            "Discuss scheduling.",
        )
    ]


def metadata(runner, args, loading_seconds):
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "cpu": platform.processor(),
        "logical_cpus": psutil.cpu_count(),
        "ram_bytes": psutil.virtual_memory().total,
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "model": "in-memory GPT-2 (seed=17, 2 layers, width=24)" if args.offline else args.model,
        "model_revision": "seed=17" if args.offline else getattr(runner, "revision", None),
        "dtype": str(next(runner.model.parameters()).dtype),
        "torch_threads": torch.get_num_threads(),
        "block_size": args.block_size,
        "num_blocks": args.num_blocks,
        "max_batch_tokens": args.max_batch_tokens,
        "iterations": args.iterations,
        "decoding": "greedy",
        "loading_seconds_excluded": loading_seconds,
    }


def summary(values):
    ordered = sorted(values)
    if not ordered:
        return {"mean": None, "median": None, "p50": None, "p95": None}
    return {
        "mean": statistics.mean(ordered),
        "median": statistics.median(ordered),
        "p50": percentile(ordered, 0.5),
        "p95": percentile(ordered, 0.95),
    }


def percentile(ordered, fraction):
    position = (len(ordered) - 1) * fraction
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def save(result, path):
    encoded = json.dumps(result, indent=2)
    if path:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(encoded + "\n", encoding="utf-8")
    print(encoded)
