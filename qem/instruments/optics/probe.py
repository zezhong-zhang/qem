"""Probe parameters and the focused-probe wave function.

A :class:`Probe` is a frozen dataclass with everything you need to build
the q-space probe wave function and the partial-coherence envelopes.
:func:`probe_wave` returns the real-space probe (centered).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch

from .aberrations import Aberrations
from .aperture import hard_aperture, soft_aperture
from .chi import chi
from .constants import wavelength as _wavelength
from .grid import Grid


@dataclass(frozen=True)
class Probe:
    """Microscope probe parameters.

    Parameters
    ----------
    energy : float
        Beam energy in eV.
    aperture : float
        Convergence semi-angle in mrad.  ``inf`` ⇒ no aperture (HRTEM).
    aberrations : Aberrations
        Polar coefficients (defaults to all zeros).
    focal_spread : float
        Δ_f in Å, 1/e half-width of the focal spread.  Set 0 to disable
        the temporal-coherence envelope.  See
        :func:`qem.instruments.optics.envelopes.focal_spread_from_chromatic`
        for the Cc + ΔE → Δ_f conversion.
    angular_spread : float
        σ_α in mrad, source angular spread.  Set 0 to disable the
        spatial-coherence envelope.
    soft_aperture : bool
        If True, use a one-pixel soft edge on the aperture.
    """

    energy: float
    aperture: float = float("inf")
    aberrations: Aberrations = field(default_factory=Aberrations)
    focal_spread: float = 0.0
    angular_spread: float = 0.0
    soft_aperture: bool = True

    @property
    def wavelength(self) -> float:
        """Electron wavelength in Å."""
        return _wavelength(self.energy)


def probe_wave_q(
    grid: Grid,
    probe: Probe,
    *,
    device: torch.device | str | None = None,
    dtype: torch.dtype | None = None,
) -> torch.Tensor:
    """Probe wave function in q-space (fft-natural ordering).

    ``ψ(q) = aperture(q) · exp(-i χ(q))``.  Returns a complex tensor
    with shape ``(grid.ny, grid.nx)``.
    """
    dtype = dtype or torch.get_default_dtype()
    lam = probe.wavelength
    alpha, phi = grid.alpha_phi(lam, device=device, dtype=dtype)
    if probe.aperture == float("inf"):
        ap = torch.ones_like(alpha)
    elif probe.soft_aperture:
        ap = soft_aperture(
            alpha, phi, probe.aperture, grid.angular_sampling(lam)
        )
    else:
        ap = hard_aperture(alpha, probe.aperture)
    phase = chi(alpha, phi, wavelength=lam, aberrations=probe.aberrations)
    # ψ(q) = aperture · exp(-i χ).  Build directly as complex.
    real = ap * torch.cos(phase)
    imag = -ap * torch.sin(phase)
    return torch.complex(real, imag)


def probe_wave(
    grid: Grid,
    probe: Probe,
    *,
    centered: bool = True,
    normalize: bool = True,
    device: torch.device | str | None = None,
    dtype: torch.dtype | None = None,
) -> torch.Tensor:
    """Focused probe wave function in real space.

    Parameters
    ----------
    centered
        If True (default), return the wave with the probe peak at the
        array centre (fftshift'd).  If False, leave it in fft-natural
        ordering with the peak at ``[0, 0]``.
    normalize
        If True (default), normalise so that ``Σ |ψ(r)|² = 1``.
    """
    psi_q = probe_wave_q(grid, probe, device=device, dtype=dtype)
    psi_r = torch.fft.ifft2(psi_q)
    if normalize:
        norm = torch.sqrt(torch.sum(psi_r.real ** 2 + psi_r.imag ** 2))
        if norm > 0:
            psi_r = psi_r / norm
    if centered:
        psi_r = torch.fft.fftshift(psi_r)
    return psi_r
