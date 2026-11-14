from collections import OrderedDict
from concurrent.futures import Future
from dataclasses import dataclass, field
from queue import Queue
from time import perf_counter


@dataclass
class RequestState:
    request_id: str
    prompt: str
    max_tokens: int
    reserved_blocks: int
    status: str = "pending"
    result: Future = field(default_factory=Future)
    events: Queue = field(default_factory=Queue)
    tokens: list[int] = field(default_factory=list)
    emitted_text: str = ""
    cached_tokens: int = 0
    prompt_tokens: int = 0
    created_at: float = field(default_factory=perf_counter)
    token_times: list[float] = field(default_factory=list)
    finished_at: float | None = None
    error: str | None = None
    finish_reason: str | None = None


class Registry:
    """Owner-thread-only records; terminal history is bounded and contains no KV."""

    def __init__(self, history_limit=256):
        if history_limit < 0:
            raise ValueError("History limit cannot be negative")
        self.live = {}
        self.history = OrderedDict()
        self.history_limit = history_limit

    def finish(self, state):
        self.live.pop(state.request_id, None)
        self.history[state.request_id] = {
            "status": state.status,
            "finish_reason": state.finish_reason,
            "error": state.error,
        }
        while len(self.history) > self.history_limit:
            self.history.popitem(last=False)
