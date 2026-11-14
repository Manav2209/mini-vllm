"""One dedicated thread owns model, scheduler, tokenizer, cache and registry.

Handlers exchange commands/futures and bounded per-request event queues. No
handler reads or writes engine state. Cancellation happens between forwards.
"""

import asyncio
import logging
from collections import deque
from concurrent.futures import Future
from dataclasses import dataclass
from queue import Empty, Full, Queue
from threading import Thread
from time import perf_counter
from uuid import uuid4

from .registry import Registry, RequestState

log = logging.getLogger(__name__)


class ServingError(Exception):
    def __init__(self, message, status_code=503):
        super().__init__(message)
        self.status_code = status_code


@dataclass
class Command:
    action: str
    args: tuple
    reply: Future


class EngineWorker:
    def __init__(
        self,
        engine_factory,
        max_queue_size=64,
        max_active_requests=8,
        max_new_tokens=128,
        batch_wait=0.003,
        history_limit=256,
    ):
        if min(max_queue_size, max_active_requests, max_new_tokens) < 1 or batch_wait < 0:
            raise ValueError("Worker limits must be positive")
        self.engine_factory = engine_factory
        self.max_queue_size = max_queue_size
        self.max_active_requests = max_active_requests
        self.max_new_tokens = max_new_tokens
        self.batch_wait = batch_wait
        self.commands = Queue(maxsize=max_queue_size + max_active_requests + 32)
        self.registry = Registry(history_limit)
        self.pending = deque()
        self.started = Future()
        self.engine = None
        self.ready = False
        self.accepting = False
        self.stopping = False
        self.thread = Thread(target=self._run, name="mini-vllm-owner", daemon=True)

    async def start(self):
        self.thread.start()
        await asyncio.wrap_future(self.started)

    def command(self, action, *args):
        reply = Future()
        if not self.thread.is_alive():
            reply.set_exception(ServingError("Inference worker is unavailable"))
            return reply
        try:
            self.commands.put_nowait(Command(action, args, reply))
        except Full:
            reply.set_exception(ServingError("Request queue is full", 429))
        return reply

    async def submit(self, prompt, max_tokens):
        # Shield command execution so caller cancellation cannot invalidate its
        # reply future. If cancelled before receiving the ID, clean up by ID.
        key = uuid4().hex
        loop = asyncio.get_running_loop()
        future = self.command("submit", key, prompt, max_tokens)
        try:
            return await asyncio.shield(asyncio.wrap_future(future))
        except asyncio.CancelledError:

            async def cancel_abandoned():
                try:
                    await self.cancel(key)
                except ServingError:
                    # A stopped worker releases ownership during shutdown.
                    pass

            def cleanup(done):
                if not done.cancelled() and done.exception() is None:
                    try:
                        loop.call_soon_threadsafe(lambda: asyncio.create_task(cancel_abandoned()))
                    except RuntimeError:
                        pass  # Event loop shutdown also closes the owner.

            future.add_done_callback(cleanup)
            raise

    async def cancel(self, key):
        while True:
            try:
                return await asyncio.shield(asyncio.wrap_future(self.command("cancel", key)))
            except ServingError as error:
                if error.status_code != 429:
                    raise
                # Admission can be rejected under overload; cleanup must still
                # reach the owner when the bounded command buffer has room.
                await asyncio.sleep(0.001)

    async def snapshot(self):
        return await asyncio.wrap_future(self.command("snapshot"))

    async def close(self):
        self.accepting = False
        if self.thread.is_alive():
            # A shutdown command must not be dropped behind a full command queue.
            reply = Future()
            await asyncio.to_thread(self.commands.put, Command("stop", (), reply))
            await asyncio.wrap_future(reply)
            await asyncio.to_thread(self.thread.join)

    def _submit(self, key, prompt, count):
        if not self.accepting:
            raise ServingError("Inference worker is shutting down")
        if (
            not isinstance(count, int)
            or isinstance(count, bool)
            or not 1 <= count <= self.max_new_tokens
        ):
            raise ServingError(f"max_tokens must be 1–{self.max_new_tokens}", 422)
        if not isinstance(prompt, str) or not prompt.strip():
            raise ServingError("Prompt must not be empty", 422)
        if len(self.pending) >= self.max_queue_size:
            raise ServingError("Request queue is full", 429)
        try:
            tokens = self.engine.runner.tokenizer.encode(prompt)
        except (ValueError, TypeError):
            raise ServingError("Prompt could not be tokenized", 422) from None
        context = self.engine.runner.model.config.max_position_embeddings
        if not tokens or len(tokens) + count > context:
            raise ServingError("Prompt and generated tokens exceed model context", 422)
        size = self.engine.block_manager.block_size
        blocks = (len(tokens) + count - 2 + size) // size
        if blocks > self.engine.block_manager.num_blocks:
            raise ServingError("Request exceeds KV cache capacity")
        state = RequestState(key, prompt, count, blocks, prompt_tokens=len(tokens))
        # A slow/disconnected client cannot consume unbounded worker memory.
        state.events = Queue(maxsize=count + 2)
        self.registry.live[key] = state
        self.pending.append(key)
        self._admit()
        return state

    def _admit(self):
        active = [state for state in self.registry.live.values() if state.status == "running"]
        reserved = sum(state.reserved_blocks for state in active)
        while self.pending and len(active) < self.max_active_requests:
            state = self.registry.live[self.pending[0]]
            if reserved + state.reserved_blocks > self.engine.block_manager.num_blocks:
                break
            self.pending.popleft()
            try:
                self.engine.add_request(state.request_id, state.prompt, state.max_tokens)
                state.cached_tokens = self.engine.requests[state.request_id].num_cached_tokens
                state.status = "running"
                active.append(state)
                reserved += state.reserved_blocks
            except Exception:
                log.exception("request_admission_failed request_id=%s", state.request_id)
                self._fail(state)

    def _decode(self, state, final=False):
        text = self.engine.runner.tokenizer.decode(
            state.tokens,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        # GPT-2 byte tokens can split a UTF-8 character across several tokens.
        # Hold the unstable suffix until it decodes, flushing it at completion.
        if not final:
            text = text.split("\ufffd", 1)[0]
        if not text.startswith(state.emitted_text):
            raise RuntimeError("Tokenizer does not support append-only decoding")
        delta = text[len(state.emitted_text) :]
        state.emitted_text = text
        return delta

    def _terminal(self, state, reason):
        tail = self._decode(state, final=True)
        state.status = "cancelled" if reason == "cancelled" else "completed"
        state.finish_reason = reason
        state.finished_at = perf_counter()
        result = {
            "request_id": state.request_id,
            "text": state.emitted_text,
            "generated_tokens": len(state.tokens),
            "finish_reason": reason,
        }
        state.events.put_nowait(
            ("done", {"request_id": state.request_id, "finish_reason": reason, "text": tail})
        )
        state.result.set_result(result)
        self.engine.forget_result(state.request_id)
        self.registry.finish(state)
        log.info(
            "request_finished request_id=%s reason=%s generated_tokens=%s cached_tokens=%s",
            state.request_id,
            reason,
            len(state.tokens),
            state.cached_tokens,
        )

    def _cancel(self, key):
        state = self.registry.live.get(key)
        if state is None:
            return False
        if state.status == "pending":
            self.pending.remove(key)
        else:
            self.engine.discard_request(key)
        try:
            self._terminal(state, "cancelled")
        except Exception:
            log.exception("cancellation_decode_failed request_id=%s", key)
            self._fail(state)
        return True

    def _fail(self, state):
        self.engine.discard_request(state.request_id)
        self.engine.forget_result(state.request_id)
        if state.request_id in self.pending:
            self.pending.remove(state.request_id)
        state.status = "failed"
        state.error = "Inference failed"
        state.finished_at = perf_counter()
        state.events.put_nowait(("error", {"request_id": state.request_id, "error": state.error}))
        state.result.set_exception(ServingError(state.error))
        self.registry.finish(state)

    def _snapshot(self):
        manager = self.engine.block_manager
        return {
            "live_requests": len(self.registry.live),
            "pending": len(self.pending),
            "free_blocks": manager.num_free_blocks,
            "cache_blocks": len(self.engine.prefix_cache.hash_to_block),
            "request_references": sum(manager.ref_counts.values())
            - len(self.engine.prefix_cache.hash_to_block),
            "forward_batches": list(self.engine.forward_batches),
            "history_size": len(self.registry.history),
        }

    def _execute(self, command):
        try:
            if command.action == "submit":
                value = self._submit(*command.args)
            elif command.action == "cancel":
                value = self._cancel(*command.args)
            elif command.action == "snapshot":
                value = self._snapshot()
            elif command.action == "stop":
                self.accepting = False
                self.stopping = True
                for key in list(self.registry.live):
                    self._cancel(key)
                value = None
            else:
                raise ValueError("Unknown worker command")
            if not command.reply.done():
                command.reply.set_result(value)
        except Exception as error:
            if not command.reply.done():
                command.reply.set_exception(error)

    def _run(self):
        try:
            self.engine = self.engine_factory()
            self.ready = self.accepting = True
            self.started.set_result(None)
            while not self.stopping:
                if not self.registry.live:
                    # Block when idle; coalesce simultaneous arrivals briefly.
                    self._execute(self.commands.get())
                    deadline = perf_counter() + self.batch_wait
                    while not self.stopping and perf_counter() < deadline:
                        try:
                            self._execute(
                                self.commands.get(timeout=max(0, deadline - perf_counter()))
                            )
                        except Empty:
                            break
                for _ in range(self.commands.maxsize):
                    try:
                        self._execute(self.commands.get_nowait())
                    except Empty:
                        break
                if self.stopping:
                    break
                self._admit()
                if not self.engine.requests:
                    continue
                try:
                    emitted = self.engine.step()
                    for key, token in emitted.items():
                        state = self.registry.live[key]
                        state.tokens.append(token)
                        state.token_times.append(perf_counter())
                        state.events.put_nowait(
                            (
                                "token",
                                {
                                    "request_id": key,
                                    "token_id": token,
                                    "text": self._decode(state, final=self.engine.is_complete(key)),
                                },
                            )
                        )
                        if self.engine.is_complete(key):
                            self._terminal(state, self.engine.completions[key]["finish_reason"])
                except Exception:
                    log.exception("model_step_failed")
                    for state in list(self.registry.live.values()):
                        if state.status == "running":
                            self._fail(state)
        except Exception as error:
            log.exception("worker_failed")
            if not self.started.done():
                self.started.set_exception(error)
            for state in list(self.registry.live.values()):
                self._fail(state)
        finally:
            self.ready = self.accepting = False
            if self.engine is not None:
                for key in list(self.engine.requests):
                    self.engine.discard_request(key)
                self.engine.prefix_cache.evict(self.engine.block_manager.num_blocks)
            while True:
                try:
                    command = self.commands.get_nowait()
                except Empty:
                    break
                if not command.reply.done():
                    command.reply.set_exception(ServingError("Inference worker stopped"))
