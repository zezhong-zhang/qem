"""Edge-handling helpers used by :func:`qem.fit.optimization.loss`.

Two concerns live here:

* **Soft boundary penalty** — keeps atoms from drifting outside the image
  by adding a smooth quadratic penalty proportional to how far past the
  edge they are. Activated by setting ``fitter.boundary_strength > 0``.

* **Adaptive edge loss** — boosts the gradient on peaks that are partially
  outside the image (visibility < 1.0) so the optimiser still gets a
  signal. Activated by setting ``fitter.adaptive_edge_loss = True``.

These are configured by *plain attribute assignment*, not by
``enable_X`` / ``disable_X`` methods (those were a stateful API smell —
a flag with five lines of logging dressed up as a method). Defaults are
declared on :class:`FitterEdgeMixin` below; override them on the
instance::

    fitter.boundary_strength = 0.05      # turn on penalty
    fitter.adaptive_edge_loss = True     # gradient boost
    fitter.window = np.ones_like(...)    # disable Butterworth dampening
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    pass


def calculate_peak_visibility(self, pos_x, pos_y, width):
    """Fraction of each peak's 3σ disk that's inside the image.

    Args:
        pos_x, pos_y: peak centres (tensors).
        width: peak σ (tensor).

    Returns:
        ``visibility ∈ [0.01, 1.0]`` per peak.
    """
    h, w = self.image.shape
    radius = 3.0 * width
    x_min = torch.maximum(pos_x - radius, torch.zeros_like(pos_x))
    x_max = torch.minimum(pos_x + radius, torch.full_like(pos_x, w - 1))
    y_min = torch.maximum(pos_y - radius, torch.zeros_like(pos_y))
    y_max = torch.minimum(pos_y + radius, torch.full_like(pos_y, h - 1))
    visible = torch.maximum(x_max - x_min, torch.zeros_like(x_min)) * torch.maximum(
        y_max - y_min, torch.zeros_like(y_min)
    )
    total = (2 * radius) * (2 * radius)
    return torch.clamp(visible / total, 0.01, 1.0)


def calculate_boundary_penalty(self, pos_x, pos_y, width, max_distance: float = 3.0):
    """Smooth quadratic penalty for atoms outside the image bounds.

    Atoms may sit up to ``max_distance · σ`` outside the image without
    paying any penalty (the Gaussian still has visible signal there).
    Beyond that, a quadratic ramp kicks in.

    Returns a scalar penalty.
    """
    h, w = self.image.shape
    dist_left = -pos_x
    dist_right = pos_x - (w - 1)
    dist_top = -pos_y
    dist_bottom = pos_y - (h - 1)
    allowed = max_distance * width
    zero = torch.zeros_like(pos_x)
    return torch.sum(
        torch.maximum(dist_left - allowed, zero) ** 2
        + torch.maximum(dist_right - allowed, zero) ** 2
        + torch.maximum(dist_top - allowed, zero) ** 2
        + torch.maximum(dist_bottom - allowed, zero) ** 2
    )


class FitterEdgeMixin:
    """Edge-handling configuration + helper methods for :class:`Fitter`.

    Class-level attribute defaults — ``Fitter`` instances inherit them
    and can override per-instance::

        fitter.boundary_strength = 0.05
        fitter.adaptive_edge_loss = True
    """

    # ---------- attribute defaults (instance-overridable) -----------------
    boundary_strength: float = 0.0           # 0 disables the penalty
    boundary_margin: float = 2.0
    adaptive_edge_loss: bool = False

    # ---------- helper methods --------------------------------------------
    calculate_peak_visibility = calculate_peak_visibility
    calculate_boundary_penalty = calculate_boundary_penalty


__all__ = [
    "FitterEdgeMixin",
    "calculate_peak_visibility",
    "calculate_boundary_penalty",
]
