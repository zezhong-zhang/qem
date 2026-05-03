"""Persistence helpers for fusion results."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Union

import numpy as np

from .route_b_joint_ls import FusionResult


def save_fusion_result(result: FusionResult, path: Union[str, Path]) -> None:
    """Save a fusion result to a compressed ``.npz`` file."""

    target = Path(path)
    np.savez_compressed(
        str(target),
        concentrations=result.concentrations,
        elements=np.asarray(result.elements, dtype=object),
        cost_history=json.dumps(result.cost_history),
        metadata=json.dumps(result.metadata),
    )


def load_fusion_result(path: Union[str, Path]) -> FusionResult:
    """Load a fusion result saved by :func:`save_fusion_result`."""

    with np.load(str(path), allow_pickle=True) as data:
        return FusionResult(
            concentrations=data["concentrations"],
            elements=[str(element) for element in data["elements"].tolist()],
            cost_history=json.loads(str(data["cost_history"])),
            metadata=json.loads(str(data["metadata"])),
        )
