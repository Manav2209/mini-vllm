
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache


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
    def forward_cached_batch(self, sequences, caches):
        """Run uncached chunks together, retaining unpadded full-attention KV pairs.

        Each request cache is a tuple of (key, value) tensors per layer, with
        shape [1, heads, processed_tokens, head_dim]. Input caches are not mutated.
        """
        if not sequences or any(not seq for seq in sequences):
            raise ValueError("Sequences must be non-empty")
        if len(sequences) != len(caches):
            raise ValueError("Each sequence must have one cache entry")

        lengths = [len(seq) for seq in sequences]
        past_lengths = [cache[0][0].shape[-2] if cache else 0 for cache in caches]
        max_past = max(past_lengths)
        max_chunk = max(lengths)
        batch_size = len(sequences)
        pad_id = self.tokenizer.pad_token_id
        if pad_id is None:
            pad_id = self.tokenizer.eos_token_id
        if pad_id is None:
            pad_id = 0

        input_ids = torch.full(
            (batch_size, max_chunk), pad_id, dtype=torch.long, device=self.device,
        )
        attention_mask = torch.zeros(
            (batch_size, max_past + max_chunk), dtype=torch.long, device=self.device,
        )
        position_ids = torch.zeros_like(input_ids)
        for row, (seq, past, length) in enumerate(zip(sequences, past_lengths, lengths)):
            input_ids[row, :length] = torch.tensor(seq, device=self.device)
            attention_mask[row, max_past - past:max_past + length] = 1
            position_ids[row, :length] = torch.arange(past, past + length, device=self.device)

        # Align real past tokens at the right of a shared prefix. Current chunks
        # start after that prefix and are right-padded independently.
        batched_cache = None
        if max_past:
            template = next(cache for cache in caches if cache)
            packed = []
            for layer_idx, (key, value) in enumerate(template):
                shape = (batch_size, key.shape[1], max_past, key.shape[-1])
                keys = key.new_zeros(shape)
                values = value.new_zeros(shape)
                for row, (cache, past) in enumerate(zip(caches, past_lengths)):
                    if past:
                        keys[row:row + 1, :, max_past - past:] = cache[layer_idx][0]
                        values[row:row + 1, :, max_past - past:] = cache[layer_idx][1]
                packed.append((keys, values))
            batched_cache = DynamicCache(packed)

        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=batched_cache,
            use_cache=True,
        )
        rows = torch.arange(batch_size, device=self.device)
        last = torch.tensor(lengths, device=self.device) - 1
        logits = outputs.logits[rows, last, :]

        # Copy only real KV states into independent request caches. No finished
        # request or padding remains pinned by a view into the batched tensors.
        updated_caches = []
        for row, (past, length) in enumerate(zip(past_lengths, lengths)):
            layers = []
            for layer in outputs.past_key_values.layers:
                pair = tuple(
                    torch.cat((
                        tensor[row:row + 1, :, max_past - past:max_past],
                        tensor[row:row + 1, :, max_past:max_past + length],
                    ), dim=-2)
                    for tensor in (layer.keys, layer.values)
                )
                layers.append(pair)
            updated_caches.append(tuple(layers))
        return logits, updated_caches

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

