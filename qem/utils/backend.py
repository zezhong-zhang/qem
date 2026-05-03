"""PyTorch backend utilities for QEM."""

from __future__ import annotations

import copy
import importlib.util
import os
from typing import Any

import numpy as np


def safe_convert_to_numpy(tensor: Any) -> np.ndarray:
    """Convert tensors, parameters, arrays, and scalars to NumPy."""
    if isinstance(tensor, np.ndarray):
        return tensor
    if hasattr(tensor, "detach"):
        return tensor.detach().cpu().numpy()
    return np.asarray(tensor)


def safe_convert_to_tensor(array: Any, dtype: str = "float32"):
    """Convert input data to a PyTorch tensor."""
    from qem.utils import torch_compat as keras

    return keras.ops.convert_to_tensor(array, dtype=dtype)


def release_backend_memory() -> None:
    """Best-effort release of cached CUDA memory."""
    try:
        import torch
    except ImportError:
        return
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def safe_deepcopy_params(params: dict[str, Any]) -> dict[str, Any]:
    """Deep-copy a parameter dictionary containing PyTorch tensors."""
    copied_params: dict[str, Any] = {}
    for key, value in params.items():
        if hasattr(value, "detach"):
            copied_params[key] = value.detach().clone()
        elif hasattr(value, "shape"):
            copied_params[key] = safe_convert_to_tensor(safe_convert_to_numpy(value))
        else:
            copied_params[key] = copy.deepcopy(value)
    return copied_params


def detect_available_backends() -> list[str]:
    """Return ``['torch']`` when PyTorch is importable, else an empty list."""
    if importlib.util.find_spec("torch") is None:
        return []
    return ["torch"]


def get_best_backend() -> str:
    """Return the only supported backend, PyTorch."""
    if not detect_available_backends():
        raise RuntimeError("PyTorch is required. Install qem[torch] or torch>=2.0.")
    return "torch"


def configure_backend(backend_name: str | None = None, force: bool = False) -> str:
    """Configure QEM for native PyTorch execution."""
    backend_name = backend_name or "torch"
    if backend_name != "torch":
        raise ValueError("QEM now supports only the native PyTorch backend.")
    get_best_backend()
    os.environ["QEM_BACKEND"] = "torch"
    backend_specific_config("torch")
    return "torch"


def setup_test_backend() -> str | None:
    """Set up the PyTorch backend for tests."""
    try:
        backend = configure_backend("torch")
        print(f"Using PyTorch backend: {backend}")
        return backend
    except Exception as exc:
        print(f"Warning: Failed to configure backend: {exc}")
        return None


def backend_specific_config(backend_name: str) -> None:
    """Apply PyTorch defaults used by QEM."""
    if backend_name != "torch":
        return
    try:
        import torch
    except ImportError:
        return
    torch.set_default_dtype(torch.float32)


def auto_configure() -> str | None:
    """Auto-configure the PyTorch backend if available."""
    if detect_available_backends():
        return configure_backend("torch")
    return None


if __name__ == "__main__":
    auto_configure()
