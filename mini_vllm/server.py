"""FastAPI front end for the canonical single-owner CPU engine."""

import asyncio
import json
import logging
import os
from contextlib import asynccontextmanager
from dataclasses import dataclass
from queue import Empty

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse

from .api.schemas import GenerateRequest, GenerateResponse
from .engine.continuous_engine import ContinuousBatchEngine
from .serving.worker import EngineWorker, ServingError

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class ServerConfig:
    model: str = "sshleifer/tiny-gpt2"
    num_blocks: int = 256
    block_size: int = 16
    max_batch_tokens: int = 128
    max_active_requests: int = 8
    max_queue_size: int = 64
    max_new_tokens: int = 128

    def __post_init__(self):
        if any(value < 1 for key, value in vars(self).items() if key != "model"):
            raise ValueError("Server limits must be positive")

    @classmethod
    def from_env(cls):
        defaults = cls()
        return cls(
            **{
                key: (str if key == "model" else int)(os.getenv(f"MINI_VLLM_{key.upper()}", value))
                for key, value in vars(defaults).items()
            }
        )


def create_app(config=None, engine_factory=None):
    config = config or ServerConfig.from_env()
    factory = engine_factory or (
        lambda: ContinuousBatchEngine(
            model_name=config.model,
            num_blocks=config.num_blocks,
            block_size=config.block_size,
            max_batch_tokens=config.max_batch_tokens,
            max_active_requests=config.max_active_requests,
        )
    )

    @asynccontextmanager
    async def lifespan(app):
        worker = EngineWorker(
            factory,
            max_queue_size=config.max_queue_size,
            max_active_requests=config.max_active_requests,
            max_new_tokens=config.max_new_tokens,
        )
        app.state.worker = worker
        try:
            await worker.start()
            log.info("worker_ready model=%s", config.model)
            yield
        finally:
            await worker.close()

    app = FastAPI(title="mini-vLLM", version="0.1.0", lifespan=lifespan)

    @app.exception_handler(ServingError)
    async def serving_error(request, error):
        return JSONResponse({"detail": str(error)}, status_code=error.status_code)

    def get_worker():
        worker = getattr(app.state, "worker", None)
        if worker is None or not worker.ready or not worker.accepting:
            raise ServingError("Inference worker is not ready")
        return worker

    @app.get("/health")
    async def health():
        return {"status": "ok"}

    @app.get("/ready")
    async def ready():
        worker = getattr(app.state, "worker", None)
        value = bool(worker and worker.ready and worker.accepting)
        return JSONResponse({"ready": value}, status_code=200 if value else 503)

    @app.post("/generate", response_model=GenerateResponse)
    async def generate(payload: GenerateRequest, request: Request):
        worker = get_worker()
        state = await worker.submit(payload.prompt, payload.max_tokens)
        future = asyncio.wrap_future(state.result)
        try:
            while not future.done():
                await asyncio.wait({future}, timeout=0.02)
                if await request.is_disconnected():
                    await worker.cancel(state.request_id)
                    raise HTTPException(499, "Client disconnected")
            return future.result()
        finally:
            if not state.result.done():
                await asyncio.shield(worker.cancel(state.request_id))

    @app.post("/generate/stream")
    async def stream(payload: GenerateRequest, request: Request):
        worker = get_worker()
        state = await worker.submit(payload.prompt, payload.max_tokens)

        async def events():
            try:
                while True:
                    if await request.is_disconnected():
                        break
                    try:
                        event, data = state.events.get_nowait()
                    except Empty:
                        await asyncio.sleep(0.005)
                        continue
                    yield f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"
                    if event in ("done", "error"):
                        break
            finally:
                if not state.result.done():
                    await asyncio.shield(worker.cancel(state.request_id))

        return StreamingResponse(
            events(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "X-Accel-Buffering": "no",
                "X-Request-ID": state.request_id,
            },
        )

    @app.post("/requests/{request_id}/cancel")
    async def cancel(request_id: str):
        cancelled = await get_worker().cancel(request_id)
        if not cancelled:
            raise HTTPException(404, "No queued or running request with this ID")
        return {"request_id": request_id, "cancelled": True}

    return app


app = create_app()
