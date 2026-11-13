
import torch

from .request import Request
from .block_manager import BlockManager
from .scheduler import Scheduler
from .model_runner import ModelRunner


class LLMEngine:
    def __init__(
        self,
        model_name="sshleifer/tiny-gpt2",
        max_batch_tokens=16,
    ):
        self.runner = ModelRunner(model_name)

        self.block_manager = BlockManager(
            num_blocks=1024,
            block_size=16,
        )

        self.scheduler = Scheduler(
            block_manager=self.block_manager,
            max_batch_tokens=max_batch_tokens,
        )

    def generate(
        self,
        prompt: str,
        max_tokens: int = 20,
        verbose: bool = False,
    ) -> str:

        if max_tokens <= 0:
            return ""

        prompt_ids = self.runner.tokenizer.encode(prompt)

        if not prompt_ids:
            raise ValueError("Prompt must not be empty")

        request = Request("req-1", prompt_ids)
        self.scheduler.add_request(request)

        kv_cache = None

        try:
            while len(request.output_tokens) < max_tokens:

                batch = self.scheduler.schedule()

                if not batch:
                    raise RuntimeError("No schedulable tokens")

                req, count = batch[0]

                start = req.num_computed_tokens
                end = start + count

                chunk = req.token_ids[start:end]

                # Execute the selected chunk
                logits, kv_cache = self.runner.forward_chunk(
                    token_ids=chunk,
                    past_key_values=kv_cache,
                    previous_tokens=start,
                )

                # Update progress after execution
                self.scheduler.complete_step(batch)

                if verbose:
                    print(
                        f"Processed {count} tokens, "
                        f"total={req.num_computed_tokens}, "
                        f"KV blocks={len(req.block_ids)}"
                    )

                # If prefill is incomplete, schedule
                # another chunk before sampling.
                if req.num_computed_tokens < req.prompt_len:
                    continue

                next_token = torch.argmax(
                    logits,
                    dim=-1,
                ).item()

                req.append_token(next_token)

                # Stop after EOS or output limit
                if next_token == self.runner.tokenizer.eos_token_id:
                    break

        finally:
            if request in self.scheduler.running:
                self.scheduler.finish_request(request)

            elif request in self.scheduler.waiting:
                self.scheduler.waiting.remove(request)

        return self.runner.tokenizer.decode(
            request.output_tokens,
            skip_special_tokens=True,
        )
