"""Parameter-dict helpers for QEM fitting.

Re-exports the tensor-conversion helpers under their historical names
so existing call sites keep working with one fewer import line.
"""

from __future__ import annotations

import h5py

from .tensors import (
    clone_params as safe_deepcopy_params,
    stop_grad as safe_stop_gradient,
    to_numpy as safe_convert_to_numpy,
    to_tensor as safe_convert_to_tensor,
)


def export_params(params: dict, filename: str) -> None:
    """Write a flat parameter dict to an HDF5 file."""
    with h5py.File(filename, "w") as f:
        for key, value in params.items():
            f.create_dataset(key, data=safe_convert_to_numpy(value))


__all__ = [
    "export_params",
    "safe_convert_to_numpy",
    "safe_convert_to_tensor",
    "safe_stop_gradient",
    "safe_deepcopy_params",
]
