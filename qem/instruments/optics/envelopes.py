"""Quasi-coherent partial-coherence envelopes.

Both envelopes match abtem (verified numerically against
``abtem.transfer.{TemporalEnvelope,SpatialEnvelope}``).
"""

from __future__ import annotations

import math

import torch

from .aberrations import Aberrations
from .chi import grad_chi


def temporal_envelope(
    alpha: torch.Tensor,
    *,
    wavelength: float,
    focal_spread: float,
) -> torch.Tensor:
    """Temporal coherence damping ``exp(-(π λ Δ_f k²) ² / 4)``.

    ``focal_spread`` is the 1/e half-width Δ_f of the Gaussian focal-spread
    distribution (Kirkland convention).  The same number is what abtem
    calls ``focal_spread``.  Use :func:`focal_spread_from_chromatic` to
    derive it from chromatic aberration and energy spread.
    """
    if focal_spread == 0.0:
        return torch.ones_like(alpha)
    arg = 0.5 * math.pi / wavelength * focal_spread * alpha * alpha
    return torch.exp(-(arg * arg))


def spatial_envelope(
    alpha: torch.Tensor,
    phi: torch.Tensor,
    *,
    wavelength: float,
    aberrations: Aberrations,
    angular_spread_mrad: float,
) -> torch.Tensor:
    """Spatial coherence damping ``exp(-(σ_α/2)² · |∇χ|²)``.

    ``angular_spread_mrad`` is the source angular spread (the σ_α of the
    Gaussian source distribution at the sample plane).  For an
    unaberrated probe ``∇χ = 0`` and the envelope is identically 1; use
    :func:`temporal_envelope` if you want temporal damping without
    aberrations.
    """
    if angular_spread_mrad == 0.0 or aberrations.is_zero():
        return torch.ones_like(alpha)
    sigma = angular_spread_mrad * 1e-3
    dk, dphi = grad_chi(alpha, phi, wavelength=wavelength, aberrations=aberrations)
    return torch.exp(-((sigma * 0.5) ** 2) * (dk * dk + dphi * dphi))


def partial_coherence_envelope(
    alpha: torch.Tensor,
    phi: torch.Tensor,
    *,
    wavelength: float,
    aberrations: Aberrations,
    focal_spread: float = 0.0,
    angular_spread_mrad: float = 0.0,
) -> torch.Tensor:
    """Product of temporal and spatial envelopes."""
    e_t = temporal_envelope(alpha, wavelength=wavelength, focal_spread=focal_spread)
    e_s = spatial_envelope(
        alpha, phi,
        wavelength=wavelength,
        aberrations=aberrations,
        angular_spread_mrad=angular_spread_mrad,
    )
    return e_t * e_s


def focal_spread_from_chromatic(
    Cc_A: float,
    delta_E_eV: float,
    energy_eV: float,
    convention: str = "1/e",
) -> float:
    """Chromatic focal spread Δ_f = Cc · ΔE / E.

    The energy-spread convention determines whether ``delta_E_eV`` is
    converted before being plugged into the Kirkland formula:

    - ``"1/e"``: ΔE is the 1/e half-width.  Pass-through (default).
    - ``"FWHM"``: divide by 2·sqrt(ln 2) to convert to 1/e half-width.
    - ``"std"``: multiply by sqrt(2) (1/e half-width = σ·√2).
    """
    if convention == "1/e":
        delta_E_1e = delta_E_eV
    elif convention == "FWHM":
        delta_E_1e = delta_E_eV / (2.0 * math.sqrt(math.log(2.0)))
    elif convention == "std":
        delta_E_1e = delta_E_eV * math.sqrt(2.0)
    else:
        raise ValueError(
            f"Unknown delta_E convention {convention!r}; "
            "expected '1/e', 'FWHM', or 'std'."
        )
    return Cc_A * delta_E_1e / energy_eV
