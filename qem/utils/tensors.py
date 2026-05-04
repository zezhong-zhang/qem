"""Small tensor / numpy interop helpers.

Replaces the old ``qem/utils/{backend,params,torch_compat}.py`` surface
with a few focused functions that do exactly what the rest of the
codebase needs.  Pure PyTorch — no Keras shim.
"""

from __future__ import annotations

import copy
from typing import Any

import numpy as np
import torch


_DTYPE_MAP: dict[str, torch.dtype] = {
    "float32": torch.float32,
    "float64": torch.float64,
    "float16": torch.float16,
    "int32": torch.int32,
    "int64": torch.int64,
    "bool": torch.bool,
}


def _resolve_dtype(dtype: Any) -> torch.dtype | None:
    """Map a dtype identifier (str / np.dtype / torch.dtype / None) to torch."""
    if dtype is None or isinstance(dtype, torch.dtype):
        return dtype
    if isinstance(dtype, str):
        return _DTYPE_MAP[dtype]
    name = np.dtype(dtype).name
    return _DTYPE_MAP[name]


def to_numpy(value: Any) -> np.ndarray:
    """Convert tensors / arrays / scalars to ``np.ndarray``."""
    if isinstance(value, np.ndarray):
        return value
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def to_tensor(value: Any, dtype: Any = "float32") -> torch.Tensor:
    """Convert array-likes to a torch tensor on CPU.

    Already-tensor inputs are returned unchanged when no dtype cast is
    requested, otherwise cast in-place via ``.to``.
    """
    target = _resolve_dtype(dtype)
    if isinstance(value, torch.Tensor):
        return value if target is None else value.to(dtype=target)
    return torch.as_tensor(value, dtype=target)


def stop_grad(value: Any) -> Any:
    """Detach a tensor (no-op for non-tensors)."""
    if isinstance(value, torch.Tensor):
        return value.detach()
    return value


def clone_params(params: dict[str, Any]) -> dict[str, Any]:
    """Deep-copy a dict whose values are tensors and/or plain Python."""
    out: dict[str, Any] = {}
    for key, value in params.items():
        if isinstance(value, torch.Tensor):
            out[key] = value.detach().clone()
        else:
            out[key] = copy.deepcopy(value)
    return out


def release_memory() -> None:
    """Release accelerator cached memory if available; no-op on CPU."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    mps = getattr(torch.backends, "mps", None)
    if mps is not None and mps.is_available():
        empty_cache = getattr(getattr(torch, "mps", None), "empty_cache", None)
        if empty_cache is not None:
            empty_cache()


def best_device() -> torch.device:
    """Auto-detect the best torch device for QEM fitting.

    Priority: CUDA (covers NVIDIA + AMD/ROCm via the same API) → CPU.
    Apple's MPS backend is **not** picked automatically because its
    ``scatter_add`` reduction has precision issues that hurt fitting
    quality (~14% worse residuals on the StatSTEM Au benchmark).

    Override with the ``QEM_DEVICE`` environment variable
    (``cuda`` / ``mps`` / ``cpu``).
    """
    import os
    override = os.environ.get("QEM_DEVICE", "").strip().lower()
    if override:
        return torch.device(override)
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


__all__ = [
    "to_numpy",
    "to_tensor",
    "stop_grad",
    "clone_params",
    "release_memory",
    "best_device",
]
