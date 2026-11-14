"""Fixed physical KV storage with per-request logical block tables.

The model-facing adapter gathers dense attention inputs on CPU. KV ownership,
allocation and persistent writes belong to this pool, not DynamicCache.
"""

import torch
from transformers.cache_utils import Cache, CacheLayerMixin

from .block_manager import BlockManager


class PagedKVCache:
    def __init__(
        self,
        block_manager: BlockManager,
        num_layers: int,
        num_kv_heads: int,
        head_dim: int,
        dtype=torch.float32,
        device="cpu",
    ):
        if min(num_layers, num_kv_heads, head_dim) <= 0:
            raise ValueError("KV tensor dimensions must be positive")
        self.block_manager = block_manager
        self.block_size = block_manager.block_size
        shape = (num_layers, block_manager.num_blocks, self.block_size, num_kv_heads, head_dim)
        self.keys = torch.zeros(shape, dtype=dtype, device=device)
        self.values = torch.zeros_like(self.keys)

    def physical_slot(self, block_table: list[int], position: int) -> tuple[int, int]:
        if position < 0 or position >= len(block_table) * self.block_size:
            raise ValueError("Token position is outside the block table")
        block_id = block_table[position // self.block_size]
        if block_id not in self.block_manager.allocated_blocks:
            raise ValueError("Block is not allocated")
        return block_id, position % self.block_size

    def _slots(self, block_table, start, count):
        if start < 0 or count < 0 or start + count > len(block_table) * self.block_size:
            raise ValueError("KV range is outside the block table")
        if count:
            first = start // self.block_size
            last = (start + count - 1) // self.block_size
            if any(
                block not in self.block_manager.allocated_blocks
                for block in block_table[first : last + 1]
            ):
                raise ValueError("Block is not allocated")
        positions = torch.arange(start, start + count, device=self.keys.device)
        table = torch.tensor(block_table, dtype=torch.long, device=self.keys.device)
        return table[positions // self.block_size], positions % self.block_size

    def _check_layer(self, layer_idx):
        if not 0 <= layer_idx < self.keys.shape[0]:
            raise ValueError("Invalid KV layer index")

    @torch.inference_mode()
    def write(self, layer_idx, block_table, start, keys, values):
        """Scatter [1, heads, tokens, head_dim] states into physical blocks."""
        self._check_layer(layer_idx)
        if (
            keys.ndim != 4
            or keys.shape != values.shape
            or keys.shape[0] != 1
            or keys.shape[1] != self.keys.shape[3]
            or keys.shape[3] != self.keys.shape[4]
        ):
            raise ValueError("KV states must match the pool's head dimensions")
        blocks, offsets = self._slots(block_table, start, keys.shape[-2])
        if any(block in self.block_manager.shared_blocks for block in blocks.unique().tolist()):
            raise ValueError("Cannot write to a published shared KV block")
        self.keys[layer_idx, blocks, offsets] = keys[0].transpose(0, 1)
        self.values[layer_idx, blocks, offsets] = values[0].transpose(0, 1)

    def read(self, layer_idx, block_table, start, count):
        """Gather states in logical token order; only read committed token ranges."""
        self._check_layer(layer_idx)
        blocks, offsets = self._slots(block_table, start, count)
        return tuple(
            tensor[layer_idx, blocks, offsets].transpose(0, 1).unsqueeze(0)
            for tensor in (self.keys, self.values)
        )


class _PagedLayer(CacheLayerMixin):
    is_sliding = False
    supports_early_init = False

    def __init__(self, pool, layer_idx, block_tables, past_lengths, chunk_lengths):
        super().__init__()
        self.pool = pool
        self.layer_idx = layer_idx
        self.block_tables = block_tables
        self.past_lengths = past_lengths
        self.chunk_lengths = chunk_lengths
        self.max_past = max(past_lengths)
        self.updated = False

    def lazy_initialization(self, key_states, value_states):
        # Persistent tensors were allocated once by PagedKVCache.
        self.is_initialized = True

    def get_seq_length(self):
        return self.max_past + (max(self.chunk_lengths) if self.updated else 0)

    def get_mask_sizes(self, query_length):
        return self.max_past + query_length, 0

    def get_max_length(self):
        return self.pool.block_manager.num_blocks * self.pool.block_size

    def update(self, key_states, value_states, *args, **kwargs):
        self.lazy_initialization(key_states, value_states)
        shape = (*key_states.shape[:2], self.max_past + key_states.shape[-2], key_states.shape[-1])
        keys = key_states.new_zeros(shape)
        values = value_states.new_zeros(shape)
        for row, (table, past, count) in enumerate(
            zip(
                self.block_tables,
                self.past_lengths,
                self.chunk_lengths,
            )
        ):
            # Padded query positions never consume physical KV slots.
            self.pool.write(
                self.layer_idx,
                table,
                past,
                key_states[row : row + 1, :, :count],
                value_states[row : row + 1, :, :count],
            )
            previous = self.pool.read(self.layer_idx, table, 0, past)
            for dense, old, current in zip((keys, values), previous, (key_states, value_states)):
                dense[row : row + 1, :, self.max_past - past : self.max_past] = old
                dense[row : row + 1, :, self.max_past : self.max_past + count] = current[
                    row : row + 1, :, :count
                ]
        self.updated = True
        # Dense inputs are temporary attention workspaces, never stored here.
        return keys, values


class PagedBatchCache(Cache):
    """Model cache interface for a single batch over an external physical pool."""

    def __init__(self, pool, block_tables, past_lengths, chunk_lengths):
        block_tables = [list(table) for table in block_tables]
        super().__init__(
            layers=[
                _PagedLayer(pool, layer_idx, block_tables, past_lengths, chunk_lengths)
                for layer_idx in range(pool.keys.shape[0])
            ]
        )
