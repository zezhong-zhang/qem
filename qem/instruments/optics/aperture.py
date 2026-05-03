"""Probe-forming aperture functions (hard and soft).

Both helpers operate on the angular grid (``alpha`` in radians) used by
the rest of the optics package; they're independent of grid sampling
except for the soft-edge variant.
"""

from __future__ import annotations

import torch


def hard_aperture(
    alpha: torch.Tensor,
    semiangle_cutoff_mrad: float,
) -> torch.Tensor:
    """1 inside the cutoff, 0 outside.  Matches abtem ``hard_aperture``."""
    cutoff = semiangle_cutoff_mrad * 1e-3
    return (alpha <= cutoff).to(alpha.dtype)


def soft_aperture(
    alpha: torch.Tensor,
    phi: torch.Tensor,
    semiangle_cutoff_mrad: float,
    angular_sampling_mrad: tuple[float, float],
) -> torch.Tensor:
    """Aperture with a one-pixel-wide soft edge.

    Mirrors abtem ``soft_aperture``: rolls off linearly from 1 to 0
    across the cell at the cutoff, with the rolloff width set by the
    angular sampling so the antialiased disc is sub-pixel-precise.
    """
    cutoff = semiangle_cutoff_mrad * 1e-3
    dy_rad = angular_sampling_mrad[0] * 1e-3
    dx_rad = angular_sampling_mrad[1] * 1e-3
    denom = torch.sqrt(
        (torch.cos(phi) * dy_rad) ** 2 + (torch.sin(phi) * dx_rad) ** 2
    )
    # Avoid division by zero at the centre pixel; the value there is
    # set to 1 explicitly below.
    denom = torch.where(denom == 0, torch.ones_like(denom), denom)
    out = torch.clamp((cutoff - alpha) / denom + 0.5, min=0.0, max=1.0)
    # Force unity at the DC pixel(s) — anywhere alpha is exactly zero.
    out = torch.where(alpha == 0, torch.ones_like(out), out)
    return out
