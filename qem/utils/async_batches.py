"""Asynchronous data loading for fit_stochastic().

Wraps random batch generation in a torch.utils.data.DataLoader with
num_workers>0 so CPU batch indexing overlaps GPU computation.
"""

from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


class _RandomBatchDataset(Dataset):
    """Dataset that yields pre-shuffled batch-index tensors.

    Each item is a 1-D LongTensor of indices for one batch.
    """

    def __init__(self, total_examples: int, batch_size: int, seed: int = 42) -> None:
        self.total_examples = int(total_examples)
        self.batch_size = int(batch_size)
        self.seed = int(seed)
        # Pre-shuffle once; DataLoader workers will slice this deterministically.
        rng = np.random.default_rng(self.seed)
        self._indices = rng.permutation(self.total_examples)
        self._batches = [
            self._indices[i : i + self.batch_size]
            for i in range(0, self.total_examples, self.batch_size)
        ]

    def __len__(self) -> int:
        return len(self._batches)

    def __getitem__(self, idx: int) -> torch.Tensor:
        return torch.from_numpy(self._batches[idx].copy())


def get_async_batches(
    total_examples: int,
    batch_size: int,
    *,
    num_workers: int = 2,
    prefetch_factor: int = 2,
    seed: int = 42,
    device: torch.device | str | None = None,
) -> DataLoader:
    """Build a DataLoader that asynchronously yields batch-index tensors.

    Args:
        total_examples: number of coordinates / examples to split into batches.
        batch_size: batch size.
        num_workers: number of CPU worker processes (0 = synchronous).
        prefetch_factor: batches to prefetch per worker.
        seed: RNG seed for deterministic shuffling.
        device: target device for the returned tensors (DataLoader default
            collate will keep them on CPU; callers should .to(device)).

    Returns:
        A DataLoader whose iterations yield ``torch.LongTensor`` batches.
    """
    if num_workers <= 0:
        # Synchronous fallback — same API, no overhead.
        ds = _RandomBatchDataset(total_examples, batch_size, seed=seed)
        return DataLoader(ds, batch_size=None, num_workers=0)

    ds = _RandomBatchDataset(total_examples, batch_size, seed=seed)
    return DataLoader(
        ds,
        batch_size=None,
        num_workers=num_workers,
        prefetch_factor=prefetch_factor,
        persistent_workers=True,
        pin_memory=(device is not None and str(device).startswith("cuda")),
    )
