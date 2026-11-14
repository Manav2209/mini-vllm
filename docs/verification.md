# v0.1.0 verification record

Verified locally on Windows 11, Python 3.12.8, CPU Torch 2.15.0, Transformers
5.23.0. Linux CI is configured but was not executed in this local session.

## Starting point and baseline

The repository already contained the prefix implementation in commit `7f92fae`
(`added prefix caching`). The only pre-existing working-tree edit was the
`.gitignore` entry for `spec/`; it was preserved. No repository history was
rewritten, and no remote push/tag was performed.

The existing `.venv` referenced a removed Python installation. A portable Python
3.12.8 runtime was downloaded into the ignored `.runtime/` directory to run the
checks. The first observed suite run during early cache fixes reported 29 tests,
3 failures and 8 errors. This was a development baseline, not an untouched
historical benchmark. Problems included obsolete prefix-cache test fixtures,
tests assuming finished requests free cache-retained storage, and import-time
model smoke scripts attempting Hub access. Model smoke tests now use seeded
in-memory GPT-2, and allocation tests explicitly distinguish request release from
cache eviction. Existing dense-batch, physical-pool, fragmented-address, mixed
chunk, dynamic-batch and failure/retry tests remain covered.

## Correctness and serving

`python -m unittest discover -s tests -v` passes with 49 tests. Coverage includes:

- Cached suffix logits versus clean full-prompt logits at `atol=2e-6`,
  `rtol=1e-5`, and identical greedy output IDs.
- All-layer shared K/V snapshots with exact equality after divergent continuation;
  direct writes to published blocks are rejected.
- Exact processed-token instrumentation: a nine-token warm prompt reuses eight
  tokens and forwards one, rather than merely changing a hit counter.
- Short prompts, exact block boundaries, full-prompt reuse, multi-block prefixes,
  divergence in block two, namespace separation, and first-miss stopping.
- Borrower lifetimes, canonical duplicate retention, LRU, pinned blocks,
  allocator exhaustion, and scheduling when all free storage is cache-only.
- Failure after model writes: no partial publication, no new-allocation leaks,
  and safe retry. Decoder errors also release request ownership.
- Actual shared HTTP batched forwards in the same engine, bounded queue/history,
  context limits, conservative memory admission, and deliberate 422/429/503 errors.
- SSE ordering, exactly one terminal event, split UTF-8 reconstruction, EOS,
  queued/running cancellation, disconnect cleanup and owner-thread shutdown.
  Cancellation cleanup also waits for a saturated command buffer to drain.

`python -m ruff check mini_vllm tests benchmarks` and
`python -m ruff format --check mini_vllm tests benchmarks` pass.
The project builds a `mini_vllm-0.1.0-py3-none-any.whl` with
`python -m pip wheel --no-deps --no-build-isolation --wheel-dir dist .`.

## Fresh dependency install

All pinned requirements were installed from wheels into a separate ignored
`.fresh/` directory using `pip install --target .fresh -r requirements.txt`.
The suite was then run with Python's search path restricted to `.fresh`, the
repository and the portable runtime's standard library. Torch, Transformers and
FastAPI import paths were confirmed to point into `.fresh`; the suite passed.
The Windows sandbox required the fresh-install verification command to use the
same elevated context as installation to read the new package directories.

## Real model / real HTTP

`python -m benchmarks.smoke_http` launches the real CLI in a separate process,
waits for `/ready`, checks `/health`, performs generation, reconstructs final text
from SSE, and cancels a running stream using its `X-Request-ID`. It passed with
`sshleifer/tiny-gpt2`, four generated tokens, and verified stream/cancellation.
Worker lifecycle tests separately exercise graceful owner shutdown; the smoke
harness terminates its temporary subprocess after checking HTTP behavior.

Both benchmark scripts also ran with the real tiny GPT-2 checkpoint and with
the seeded in-memory model. Every output verification passed. The recorded
JSON, plots and interpretation are in [../benchmarks/results](../benchmarks/results).
Model downloads/loading and warmup are excluded from reported inference timing.

Version 0.1.0 is prepared locally. Creating a release commit, pushing to GitHub,
and tagging are separate optional release operations; none are claimed here.
