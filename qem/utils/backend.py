"""Backend helpers — pure PyTorch, no Keras shim.

Historical surface preserved as thin re-exports of :mod:`qem.utils.tensors`.
New code should import from ``qem.utils.tensors`` directly.
"""

from __future__ import annotations

import os

from .tensors import (
    best_device,
    clone_params as safe_deepcopy_params,
    release_memory as release_backend_memory,
    to_numpy as safe_convert_to_numpy,
    to_tensor as safe_convert_to_tensor,
)


def detect_available_backends() -> list[str]:
    """QEM is PyTorch-only; report ``['torch']`` if importable, else ``[]``."""
    import importlib.util
    if importlib.util.find_spec("torch") is None:
        return []
    return ["torch"]


def get_best_backend() -> str:
    if not detect_available_backends():
        raise RuntimeError("PyTorch is required (install qem[torch] or torch>=2.0).")
    return "torch"


def configure_backend(backend_name: str | None = None, force: bool = False) -> str:
    """No-op other than confirming PyTorch is available + setting a default dtype."""
    backend_name = backend_name or "torch"
    if backend_name != "torch":
        raise ValueError("QEM supports only the PyTorch backend.")
    get_best_backend()
    os.environ["QEM_BACKEND"] = "torch"
    import torch
    torch.set_default_dtype(torch.float32)
    return "torch"


def setup_test_backend() -> str | None:
    try:
        return configure_backend("torch")
    except Exception as exc:  # pragma: no cover
        print(f"Warning: Failed to configure backend: {exc}")
        return None


def backend_specific_config(backend_name: str) -> None:  # pragma: no cover
    """Kept for back-compat — :func:`configure_backend` already does the work."""
    if backend_name == "torch":
        configure_backend(backend_name)


def auto_configure() -> str | None:
    return configure_backend("torch") if detect_available_backends() else None


__all__ = [
    "safe_convert_to_numpy",
    "safe_convert_to_tensor",
    "safe_deepcopy_params",
    "release_backend_memory",
    "best_device",
    "detect_available_backends",
    "get_best_backend",
    "configure_backend",
    "setup_test_backend",
    "backend_specific_config",
    "auto_configure",
]


if __name__ == "__main__":
    auto_configure()
