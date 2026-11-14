import asyncio
import json
import threading
import time
import unittest

import httpx
from fastapi.testclient import TestClient

from mini_vllm.engine.continuous_engine import ContinuousBatchEngine
from mini_vllm.server import ServerConfig, create_app
from mini_vllm.testing import tiny_runner


def factory():
    return ContinuousBatchEngine(runner=tiny_runner(), num_blocks=32, block_size=4)


def parse_events(text):
    return [
        (part.splitlines()[0].split(": ", 1)[1], json.loads(part.splitlines()[1].split(": ", 1)[1]))
        for part in text.strip().split("\n\n")
    ]


class APITests(unittest.TestCase):
    def test_eos_and_startup_failure(self):
        def eos_factory():
            engine = factory()
            engine.runner.tokenizer.eos_token_id = int(
                engine.runner.forward_batch([[1]]).argmax(-1).item()
            )
            return engine

        with TestClient(create_app(engine_factory=eos_factory)) as client:
            response = client.post("/generate", json={"prompt": "1", "max_tokens": 8})
            self.assertEqual(response.json()["generated_tokens"], 1)
            self.assertEqual(response.json()["finish_reason"], "eos")

        def bad_factory():
            raise ValueError("Invalid model config")

        app = create_app(engine_factory=bad_factory)
        with self.assertRaisesRegex(ValueError, "Invalid model"):
            with TestClient(app):
                pass
        self.assertFalse(app.state.worker.ready)
        self.assertFalse(app.state.worker.thread.is_alive())

    def test_health_readiness_generation_validation_stream(self):
        app = create_app(engine_factory=factory)
        client = TestClient(app)
        self.assertEqual(client.get("/health").json(), {"status": "ok"})
        self.assertEqual(client.get("/ready").status_code, 503)
        with client:
            self.assertEqual(client.get("/ready").json(), {"ready": True})
            payload = {"prompt": "1 2 3 4 5", "max_tokens": 4}
            response = client.post("/generate", json=payload)
            self.assertEqual(response.status_code, 200, response.text)
            result = response.json()
            self.assertTrue(result["text"])
            self.assertEqual(result["generated_tokens"], 4)
            self.assertEqual(result["finish_reason"], "length")
            stream = client.post("/generate/stream", json=payload)
            self.assertIn("text/event-stream", stream.headers["content-type"])
            events = parse_events(stream.text)
            self.assertEqual([kind for kind, _ in events], ["token"] * 4 + ["done"])
            self.assertEqual("".join(data["text"] for _, data in events), result["text"])
            self.assertEqual(len({data["request_id"] for _, data in events}), 1)
            self.assertNotEqual(events[0][1]["request_id"], result["request_id"])
            for bad in (
                {},
                {"prompt": ""},
                {"prompt": "  "},
                {"prompt": 12},
                {"prompt": "1", "max_tokens": 0},
                {"prompt": "1", "max_tokens": 129},
                {"prompt": "1", "max_tokens": True},
                {"prompt": "1 " * 256},
                {"prompt": "1", "extra": True},
            ):
                self.assertEqual(client.post("/generate", json=bad).status_code, 422, bad)
            self.assertEqual(
                client.post("/generate", json={"prompt": "1 " * 130, "max_tokens": 1}).status_code,
                503,
            )
            self.assertEqual(client.post("/requests/unknown/cancel").status_code, 404)
        self.assertEqual(client.get("/ready").status_code, 503)

    def test_model_failure_normal_and_stream(self):
        def broken_factory():
            engine = factory()

            def fail(*args):
                raise RuntimeError("private details")

            engine.runner.forward_paged_batch = fail
            return engine

        with TestClient(create_app(engine_factory=broken_factory)) as client:
            response = client.post("/generate", json={"prompt": "1", "max_tokens": 1})
            self.assertEqual(response.status_code, 503)
            self.assertEqual(response.json(), {"detail": "Inference failed"})
            stream = client.post("/generate/stream", json={"prompt": "1", "max_tokens": 1})
            events = parse_events(stream.text)
            self.assertEqual([kind for kind, _ in events], ["error"])
            self.assertNotIn("private", stream.text)

    def test_utf8_split_tokens_reconstruct_final_text(self):
        def byte_factory():
            engine = factory()
            original = engine.runner.forward_paged_batch

            def decode(tokens, **kwargs):
                return bytes({1: 0xE2, 2: 0x82, 3: 0xAC}[token] for token in tokens).decode(
                    "utf-8", errors="replace"
                )

            engine.runner.tokenizer.decode = decode

            def forward(sequences, requests, pool):
                logits = original(sequences, requests, pool).clone()
                logits.fill_(-100)
                for row, req in enumerate(requests):
                    logits[row, len(req.output_tokens) + 1] = 100
                return logits

            engine.runner.forward_paged_batch = forward
            return engine

        with TestClient(create_app(engine_factory=byte_factory)) as client:
            payload = {"prompt": "4", "max_tokens": 3}
            final = client.post("/generate", json=payload).json()
            events = parse_events(client.post("/generate/stream", json=payload).text)
            self.assertEqual(final["text"], "€")
            self.assertEqual(
                [data["text"] for kind, data in events if kind == "token"], ["", "", "€"]
            )


class AsyncAPITests(unittest.IsolatedAsyncioTestCase):
    async def test_concurrent_http_requests_share_batch(self):
        both_queued = threading.Event()

        def gated_factory():
            engine = factory()
            original = engine.runner.forward_paged_batch

            def forward(*args):
                # Keep the first request live until both HTTP submissions are
                # queued, regardless of host scheduling or import latency.
                if not both_queued.wait(5):
                    raise RuntimeError("Concurrent submissions did not arrive")
                return original(*args)

            engine.runner.forward_paged_batch = forward
            return engine

        app = create_app(engine_factory=gated_factory)
        async with app.router.lifespan_context(app):
            original_command = app.state.worker.command
            submissions = 0

            def command(action, *args):
                nonlocal submissions
                result = original_command(action, *args)
                if action == "submit":
                    submissions += 1
                    if submissions == 2:
                        both_queued.set()
                return result

            app.state.worker.command = command
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app), base_url="http://test"
            ) as client:
                replies = await asyncio.gather(
                    *(
                        client.post(
                            "/generate",
                            json={
                                "prompt": "1 2 3 4 5",
                                "max_tokens": 4,
                            },
                        )
                        for _ in range(2)
                    )
                )
                self.assertTrue(all(reply.status_code == 200 for reply in replies))
            snapshot = await app.state.worker.snapshot()
            self.assertTrue(any(len(batch) == 2 for batch in snapshot["forward_batches"]))
            self.assertEqual(snapshot["request_references"], 0)

    async def test_stream_disconnect_cancels_and_releases(self):
        def slow_factory():
            engine = factory()
            original = engine.runner.forward_paged_batch

            def forward(*args):
                time.sleep(0.003)
                return original(*args)

            engine.runner.forward_paged_batch = forward
            return engine

        app = create_app(ServerConfig(max_new_tokens=128), engine_factory=slow_factory)
        async with app.router.lifespan_context(app):
            incoming = asyncio.Queue()
            await incoming.put(
                {
                    "type": "http.request",
                    "body": json.dumps(
                        {
                            "prompt": "1 2 3 4 5",
                            "max_tokens": 100,
                        }
                    ).encode(),
                    "more_body": False,
                }
            )
            disconnected = False

            async def send(message):
                nonlocal disconnected
                if message["type"] == "http.response.body" and b"event: token" in message.get(
                    "body", b""
                ):
                    if not disconnected:
                        disconnected = True
                        await incoming.put({"type": "http.disconnect"})

            scope = {
                "type": "http",
                "asgi": {"version": "3.0", "spec_version": "2.3"},
                "http_version": "1.1",
                "method": "POST",
                "scheme": "http",
                "path": "/generate/stream",
                "raw_path": b"/generate/stream",
                "query_string": b"",
                "headers": [(b"content-type", b"application/json")],
                "client": ("127.0.0.1", 1),
                "server": ("test", 80),
                "root_path": "",
            }
            await asyncio.wait_for(app(scope, incoming.get, send), 5)
            self.assertTrue(disconnected)
            for _ in range(100):
                snapshot = await app.state.worker.snapshot()
                if snapshot["live_requests"] == 0:
                    break
                await asyncio.sleep(0.005)
            self.assertEqual(snapshot["live_requests"], 0)
            self.assertEqual(snapshot["request_references"], 0)
