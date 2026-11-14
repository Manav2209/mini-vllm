import asyncio
import threading
import time
import unittest

from mini_vllm.engine.continuous_engine import ContinuousBatchEngine
from mini_vllm.serving.worker import EngineWorker, ServingError
from mini_vllm.testing import tiny_runner


class ServingTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.worker = EngineWorker(
            lambda: ContinuousBatchEngine(runner=tiny_runner(), block_size=4, num_blocks=32),
            batch_wait=0.02,
            history_limit=2,
        )
        await self.worker.start()

    async def asyncTearDown(self):
        await self.worker.close()

    async def test_shared_forward_owner_and_bounded_history(self):
        owners = []
        original = self.worker.engine.runner.forward_paged_batch

        def forward(*args):
            owners.append(threading.get_ident())
            return original(*args)

        self.worker.engine.runner.forward_paged_batch = forward
        states = await asyncio.gather(*(self.worker.submit("1 2 3 4 5", 3) for _ in range(2)))
        results = await asyncio.gather(*(asyncio.wrap_future(state.result) for state in states))
        self.assertEqual(results[0]["text"], results[1]["text"])
        self.assertEqual(set(owners), {self.worker.thread.ident})
        snapshot = await self.worker.snapshot()
        self.assertTrue(any(len(shape) == 2 for shape in snapshot["forward_batches"]))
        self.assertEqual(snapshot["request_references"], 0)
        self.assertFalse(self.worker.engine.results)
        self.assertFalse(self.worker.engine.completions)
        self.assertFalse(await self.worker.cancel("missing"))
        third = await self.worker.submit("1", 1)
        await asyncio.wrap_future(third.result)
        self.assertEqual((await self.worker.snapshot())["history_size"], 2)

    async def test_cancel_pending_and_running_at_forward_boundary(self):
        await self.worker.close()
        entered, release = threading.Event(), threading.Event()

        def factory():
            engine = ContinuousBatchEngine(runner=tiny_runner(), num_blocks=16, block_size=4)
            original = engine.runner.forward_paged_batch

            def forward(*args):
                entered.set()
                if not release.wait(5):
                    raise RuntimeError("Test gate timed out")
                return original(*args)

            engine.runner.forward_paged_batch = forward
            return engine

        self.worker = EngineWorker(
            factory, max_active_requests=1, max_queue_size=1, batch_wait=0.02
        )
        await self.worker.start()
        a, b = await asyncio.gather(
            self.worker.submit("1 2 3 4 5", 5), self.worker.submit("6 7 8", 5)
        )
        self.assertTrue(await asyncio.to_thread(entered.wait, 3))
        overflow = self.worker.command("submit", "overflow", "1", 1)
        pending = self.worker.command("cancel", b.request_id)
        running = self.worker.command("cancel", a.request_id)
        release.set()
        with self.assertRaises(ServingError) as error:
            await asyncio.wrap_future(overflow)
        self.assertEqual(error.exception.status_code, 429)
        self.assertTrue(await asyncio.wrap_future(pending))
        self.assertTrue(await asyncio.wrap_future(running))
        for state in (a, b):
            result = await asyncio.wrap_future(state.result)
            self.assertEqual(result["finish_reason"], "cancelled")
        self.assertEqual(b.tokens, [])
        self.assertLess(len(a.tokens), 5)
        snapshot = await self.worker.snapshot()
        self.assertEqual(snapshot["request_references"], 0)
        self.assertEqual(snapshot["pending"], 0)
        self.assertFalse(self.worker.engine.scheduler.running)
        self.assertFalse(self.worker.engine.scheduler.waiting)

    async def test_shutdown_releases_ownership_and_thread(self):
        state = await self.worker.submit("1 2 3 4 5", 100)
        await self.worker.close()
        self.assertTrue(state.result.done())
        self.assertFalse(self.worker.thread.is_alive())
        self.assertFalse(self.worker.ready)
        self.assertEqual(self.worker.engine.block_manager.num_free_blocks, 32)

    async def test_queue_limits_context_memory_and_model_errors(self):
        for prompt, count, code in (
            ("", 1, 422),
            ("1", 129, 422),
            ("1 " * 256, 1, 422),
            ("1 " * 130, 1, 503),
        ):
            with self.subTest(code=code), self.assertRaises(ServingError) as error:
                await self.worker.submit(prompt, count)
            self.assertEqual(error.exception.status_code, code)
        original = self.worker.engine.runner.forward_paged_batch

        def fail(*args):
            original(*args)
            raise RuntimeError("private model details")

        self.worker.engine.runner.forward_paged_batch = fail
        state = await self.worker.submit("1 2 3 4 5", 3)
        with self.assertRaisesRegex(ServingError, "Inference failed"):
            await asyncio.wrap_future(state.result)
        snapshot = await self.worker.snapshot()
        self.assertEqual(snapshot["request_references"], 0)
        self.assertEqual(snapshot["cache_blocks"], 0)
        self.assertEqual(snapshot["free_blocks"], 32)
        self.worker.engine.runner.forward_paged_batch = original
        retry = await self.worker.submit("1 2 3 4 5", 1)
        self.assertEqual((await asyncio.wrap_future(retry.result))["finish_reason"], "length")

    async def test_memory_admission_serializes_without_deadlock(self):
        await self.worker.close()
        self.worker = EngineWorker(
            lambda: ContinuousBatchEngine(runner=tiny_runner(), num_blocks=2, block_size=4),
            max_active_requests=2,
            batch_wait=0.02,
        )
        await self.worker.start()
        states = await asyncio.gather(*(self.worker.submit("1 2 3 4 5", 2) for _ in range(2)))
        results = await asyncio.wait_for(
            asyncio.gather(*(asyncio.wrap_future(state.result) for state in states)), 3
        )
        self.assertTrue(all(result["generated_tokens"] == 2 for result in results))
        snapshot = await self.worker.snapshot()
        self.assertEqual(snapshot["request_references"], 0)
        self.assertTrue(all(len(shape) == 1 for shape in snapshot["forward_batches"]))

    async def test_decoder_failure_still_releases_request(self):
        def fail(*args, **kwargs):
            raise RuntimeError("decoder failed")

        self.worker.engine.runner.tokenizer.decode = fail
        state = await self.worker.submit("1 2 3 4 5", 1)
        with self.assertRaises(ServingError):
            await asyncio.wrap_future(state.result)
        self.assertEqual((await self.worker.snapshot())["request_references"], 0)

    async def test_cancel_is_delivered_when_command_buffer_is_full(self):
        await self.worker.close()
        entered, release = threading.Event(), threading.Event()

        def factory():
            engine = ContinuousBatchEngine(runner=tiny_runner(), num_blocks=16, block_size=4)
            original = engine.runner.forward_paged_batch

            def forward(*args):
                entered.set()
                if not release.wait(5):
                    raise RuntimeError("Test gate timed out")
                time.sleep(0.01)
                return original(*args)

            engine.runner.forward_paged_batch = forward
            return engine

        self.worker = EngineWorker(factory, max_queue_size=1, max_active_requests=1)
        await self.worker.start()
        state = await self.worker.submit("1 2 3 4 5", 5)
        self.assertTrue(await asyncio.to_thread(entered.wait, 3))
        for _ in range(self.worker.commands.maxsize):
            self.worker.command("snapshot")
        cancellation = asyncio.create_task(self.worker.cancel(state.request_id))
        await asyncio.sleep(0.005)
        self.assertFalse(cancellation.done())
        release.set()
        self.assertTrue(await asyncio.wait_for(cancellation, 3))
        self.assertEqual((await asyncio.wrap_future(state.result))["finish_reason"], "cancelled")
        self.assertEqual((await self.worker.snapshot())["request_references"], 0)
