"""Backend setup — pure PyTorch.

QEM is PyTorch-only. This module sets a sensible default dtype on
import; that's it. Device auto-detection lives in
:func:`qem.utils.tensors.best_device`.
"""

from __future__ import annotations

import os

import torch

torch.set_default_dtype(torch.float32)

# TF32 matmul precision on Ampere+ (SM80+) GPUs — ~1.5–2× fp32 throughput.
# Gated by QEM_TF32=0 so users can opt out for strict reproducibility.
if os.getenv("QEM_TF32", "1") != "0":
    if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8:
        torch.set_float32_matmul_precision("high")

__all__: list[str] = []
