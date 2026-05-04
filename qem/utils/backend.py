"""Backend setup — pure PyTorch.

QEM is PyTorch-only. This module sets a sensible default dtype on
import; that's it. Device auto-detection lives in
:func:`qem.utils.tensors.best_device`.
"""

from __future__ import annotations

import torch

torch.set_default_dtype(torch.float32)


__all__: list[str] = []
