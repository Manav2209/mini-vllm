
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


class ModelRunner:
    def __init__(self, model_name="sshleifer/tiny-gpt2"):
        self.device = "cpu"

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name
        )

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name
        )

        self.model.to(self.device)
        self.model.eval()
    

    @torch.inference_mode()
    def forward_chunk(
        self,
        token_ids: list[int],
        past_key_values=None,
        previous_tokens: int = 0,
    ):
        if not token_ids:
            raise ValueError("Cannot execute an empty chunk")

        input_ids = torch.tensor(
            [token_ids],
            dtype=torch.long,
            device=self.device,
        )

        total_length = previous_tokens + len(token_ids)

        attention_mask = torch.ones(
            (1, total_length),
            dtype=torch.long,
            device=self.device,
        )

        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            use_cache=True,
        )

        # Scores for the next token
        logits = outputs.logits[:, -1, :]

        return logits, outputs.past_key_values


    @torch.inference_mode()
    def prefill(self, prompt: str):
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
        )

        outputs = self.model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            use_cache=True,
        )

        logits = outputs.logits[:, -1, :]
        next_token = torch.argmax(logits, dim=-1)

        return next_token, outputs.past_key_values

    @torch.inference_mode()
    def decode(self, token_id, past_key_values):
        outputs = self.model(
            input_ids=token_id.reshape(1, 1),
            past_key_values=past_key_values,
            use_cache=True,
        )

        logits = outputs.logits[:, -1, :]
        next_token = torch.argmax(logits, dim=-1)

        return next_token, outputs.past_key_values

    
    @torch.inference_mode()
    def forward_batch(
        self,
        sequences: list[list[int]],
    ) -> torch.Tensor:
        if not sequences or any(not seq for seq in sequences):
            raise ValueError("Sequences must be non-empty")

        # Find the longest sequence in this batch.
        max_len = max(len(seq) for seq in sequences)

        # GPT-2 requires a valid padding token.
        pad_id = self.tokenizer.eos_token_id

        padded = []
        masks = []

        for seq in sequences:
            padding = max_len - len(seq)

            # Left padding keeps the final token at the same
            # position across every row in the batch.
            padded.append([pad_id] * padding + seq)
            masks.append([0] * padding + [1] * len(seq))

        input_ids = torch.tensor(
            padded,
            dtype=torch.long,
            device=self.device,
        )

        attention_mask = torch.tensor(
            masks,
            dtype=torch.long,
            device=self.device,
        )

        # Correct positional indices for left-padded inputs.
        position_ids = attention_mask.long().cumsum(-1) - 1
        position_ids.clamp_(min=0)

        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            use_cache=False,
        )

        return outputs.logits[:, -1, :]

