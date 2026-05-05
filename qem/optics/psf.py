"""PSF / CTF builders + image-analysis helpers.

**Builders** (``Grid + Probe → PSF``).  Each imaging mode exposes two
functions:

- ``*_psf(grid, probe)`` returns the real-space PSF, fftshift-centered.
- ``*_ctf(grid, probe)`` returns the q-space transfer function in
  fft-natural ordering (DC at ``[0, 0]``).

The four supported modes:

- ``ssb``  — analytical Hofer–Pennycook formula for SSB ptychography.
- ``adf``  — incoherent annular dark field, ``|ψ_probe(r)|²``.
- ``icom`` — integrated centre-of-mass (with optional high-pass filter).
- ``epie`` — analytical approximation for ePIE ptychography.

All builders return real PSFs (``torch.float``) or complex CTFs
(``torch.complex``).  Partial-coherence envelopes derived from
``probe.focal_spread`` and ``probe.angular_spread`` are applied to the
q-space CTF and inverse-transformed.

**Image-analysis helpers** (numpy in / numpy out):

- :func:`calculate_psf_width` — second-moment width of an existing PSF.
- :func:`extract_psf_from_atom_image` — recenter + normalise a single
  isolated atom from an experimental image into a PSF estimate.
"""

from __future__ import annotations

import math
import os
from typing import Optional

import numpy as np
import torch

from .aberrations import Aberrations  # noqa: F401  (re-exported in __init__)
from .chi import chi
from .envelopes import partial_coherence_envelope
from .grid import Grid
from .probe import Probe, probe_wave_q


# torch.compile gating — skip on MPS (graph compilation not supported) and
# respect QEM_COMPILE=0 opt-out.
_CUDA_AVAILABLE = torch.cuda.is_available()
_COMPILE_ENABLED = os.getenv("QEM_COMPILE", "1") != "0" and _CUDA_AVAILABLE


def _maybe_compile(fn):
    """Decorator: apply torch.compile on CUDA, passthrough everywhere else."""
    if _COMPILE_ENABLED:
        return torch.compile(fn, mode="default")
    return fn


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _envelope(grid, probe, alpha, phi):
    return partial_coherence_envelope(
        alpha, phi,
        wavelength=probe.wavelength,
        aberrations=probe.aberrations,
        focal_spread=probe.focal_spread,
        angular_spread_mrad=probe.angular_spread,
    )


def _ctf_to_psf(ctf: torch.Tensor) -> torch.Tensor:
    """fft-natural CTF → fftshift-centered real-space PSF."""
    return torch.real(torch.fft.fftshift(torch.fft.ifft2(ctf)))


def _psf_to_ctf(psf: torch.Tensor) -> torch.Tensor:
    """fftshift-centered real-space PSF → fft-natural q-space CTF."""
    return torch.fft.fft2(torch.fft.ifftshift(psf))


# ---------------------------------------------------------------------------
# SSB ptychography
# ---------------------------------------------------------------------------


@_maybe_compile
def ssb_ctf(
    grid: Grid,
    probe: Probe,
    *,
    device: torch.device | str | None = None,
    dtype: torch.dtype | None = None,
) -> torch.Tensor:
    """Analytical SSB transfer function (Hofer & Pennycook 2023, Eq. 11).

    Returns a complex tensor in fft-natural ordering.
    """
    dtype = dtype or torch.get_default_dtype()
    lam = probe.wavelength
    alpha, phi = grid.alpha_phi(lam, device=device, dtype=dtype)
    # q normalised by the convergence semi-angle.
    if not math.isfinite(probe.aperture) or probe.aperture <= 0:
        raise ValueError("SSB CTF requires a finite, positive aperture.")
    alpha_cut = probe.aperture * 1e-3
    qn = alpha / alpha_cut

    real = torch.zeros_like(alpha)
    # Region 1: 0 ≤ qn ≤ 1   (overlap of two aperture discs).
    m1 = (qn >= 0) & (qn <= 1)
    if m1.any():
        x = qn[m1].clamp(0.0, 0.9999)
        real_m1 = (
            torch.arccos(x / 2)
            - (x / 2) * torch.sqrt(1 - (x / 2) ** 2)
            - torch.arccos(x)
            + x * torch.sqrt(1 - x ** 2)
        )
        real = real.masked_scatter(m1, real_m1 * (4.0 / math.pi))
    # Region 2: 1 < qn ≤ 2.
    m2 = (qn > 1) & (qn <= 2)
    if m2.any():
        x = qn[m2].clamp(0.0, 0.9999)
        real_m2 = torch.arccos(x / 2) - (x / 2) * torch.sqrt(1 - (x / 2) ** 2)
        real = real.masked_scatter(m2, real_m2 * (4.0 / math.pi))

    # Aberration phase factor + partial-coherence envelope.
    phase = chi(alpha, phi, wavelength=lam, aberrations=probe.aberrations)
    env = _envelope(grid, probe, alpha, phi)
    amplitude = real * env
    return torch.complex(amplitude * torch.cos(-phase), amplitude * torch.sin(-phase))


def ssb_psf(grid: Grid, probe: Probe, **kw) -> torch.Tensor:
    """Real-space SSB PSF — centered, dtype matches ``Grid.q_grid``."""
    return _ctf_to_psf(ssb_ctf(grid, probe, **kw))


# ---------------------------------------------------------------------------
# ADF (incoherent)
# ---------------------------------------------------------------------------


@_maybe_compile
def adf_psf(
    grid: Grid,
    probe: Probe,
    *,
    device: torch.device | str | None = None,
    dtype: torch.dtype | None = None,
) -> torch.Tensor:
    """Incoherent ADF PSF = ``|ψ_probe(r)|²``.

    Partial coherence is applied to the probe wave function in q-space
    before the ψ → |ψ|² step.  Detector inner/outer angles only set the
    overall signal scaling for thin specimens (Z-contrast); they don't
    shape the PSF and therefore aren't taken here.
    """
    dtype = dtype or torch.get_default_dtype()
    lam = probe.wavelength
    psi_q = probe_wave_q(grid, probe, device=device, dtype=dtype)
    if probe.focal_spread or probe.angular_spread:
        alpha, phi = grid.alpha_phi(lam, device=device, dtype=dtype)
        env = _envelope(grid, probe, alpha, phi)
        psi_q = psi_q * env
    psi_r = torch.fft.ifft2(psi_q)
    psf = psi_r.real ** 2 + psi_r.imag ** 2
    psf = torch.fft.fftshift(psf)
    total = psf.sum()
    if total > 0:
        psf = psf / total
    return psf


def adf_ctf(grid: Grid, probe: Probe, **kw) -> torch.Tensor:
    """ADF CTF — fft of ``adf_psf`` (autocorrelation of ψ_probe)."""
    return _psf_to_ctf(adf_psf(grid, probe, **kw)).to(torch.complex64)


# ---------------------------------------------------------------------------
# iCoM
# ---------------------------------------------------------------------------


@_maybe_compile
def icom_ctf(
    grid: Grid,
    probe: Probe,
    *,
    high_pass_mrad: float | None = None,
    high_pass_order: int = 2,
    device: torch.device | str | None = None,
    dtype: torch.dtype | None = None,
) -> torch.Tensor:
    """iCoM transfer function with optional Butterworth high-pass.

    Without a high-pass filter the CTF is identically 1 (and the PSF
    is a δ).  ``high_pass_mrad`` rolls off low frequencies so atomic
    contrast is recovered.
    """
    dtype = dtype or torch.get_default_dtype()
    lam = probe.wavelength
    alpha, phi = grid.alpha_phi(lam, device=device, dtype=dtype)
    if high_pass_mrad is None:
        amp = torch.ones_like(alpha)
    else:
        cutoff = high_pass_mrad * 1e-3
        order = max(1, int(high_pass_order))
        amp = 1.0 - 1.0 / (1.0 + (alpha / cutoff) ** (2 * order))
    phase = chi(alpha, phi, wavelength=lam, aberrations=probe.aberrations)
    env = _envelope(grid, probe, alpha, phi)
    amplitude = amp * env
    return torch.complex(amplitude * torch.cos(-phase), amplitude * torch.sin(-phase))


def icom_psf(grid: Grid, probe: Probe, **kw) -> torch.Tensor:
    return _ctf_to_psf(icom_ctf(grid, probe, **kw))


# ---------------------------------------------------------------------------
# ePIE (analytical approximation)
# ---------------------------------------------------------------------------


@_maybe_compile
def epie_ctf(
    grid: Grid,
    probe: Probe,
    *,
    defocus_filter_nm: float = 1.0,
    device: torch.device | str | None = None,
    dtype: torch.dtype | None = None,
) -> torch.Tensor:
    """Analytical approximation to the ePIE transfer function.

    ePIE doesn't have a closed-form CTF; this is the same approximation
    that was in the legacy code (single-disc overlap with a heuristic
    defocus rolloff).  Useful for fitting / qualitative work, not for
    quantitative phase reconstruction.
    """
    dtype = dtype or torch.get_default_dtype()
    lam = probe.wavelength
    alpha, phi = grid.alpha_phi(lam, device=device, dtype=dtype)
    if not math.isfinite(probe.aperture) or probe.aperture <= 0:
        raise ValueError("ePIE CTF requires a finite, positive aperture.")
    alpha_cut = probe.aperture * 1e-3
    qn = alpha / alpha_cut
    real = torch.zeros_like(alpha)
    mask = (qn >= 0) & (qn <= 2)
    if mask.any():
        x = qn[mask].clamp(0.0, 1.9999)
        x2 = (x / 2).clamp(0.0, 0.9999)
        ctf_val = (4.0 / math.pi) * (
            torch.arccos(x2) - x2 * torch.sqrt(1 - x2 ** 2)
        )
        # Heuristic defocus rolloff: 1/(1 + (defocus·q/10)²)
        q_invA = alpha[mask] / lam
        df_factor = 1.0 / (1.0 + (defocus_filter_nm * q_invA / 10.0) ** 2)
        real = real.masked_scatter(mask, ctf_val * df_factor)
    phase = chi(alpha, phi, wavelength=lam, aberrations=probe.aberrations)
    env = _envelope(grid, probe, alpha, phi)
    amplitude = real * env
    return torch.complex(amplitude * torch.cos(-phase), amplitude * torch.sin(-phase))


def epie_psf(grid: Grid, probe: Probe, **kw) -> torch.Tensor:
    return _ctf_to_psf(epie_ctf(grid, probe, **kw))


# ---------------------------------------------------------------------------
# Image-analysis helpers (numpy in / numpy out)
# ---------------------------------------------------------------------------


def calculate_psf_width(psf: np.ndarray, dx: float = 1.0) -> float:
    """Effective PSF width: second-moment radius of ``|psf|``.

    Parameters
    ----------
    psf
        2D PSF array (real or signed; for SSB-like PSFs with a negative
        halo, the absolute value is used so the halo contributes to the
        width).
    dx
        Pixel size in Å (kept for API back-compat; the returned width is
        in *pixels* — multiply by ``dx`` for Å).

    Returns
    -------
    Width in pixels.
    """
    abs_psf = np.abs(psf)
    total = abs_psf.sum()
    if total == 0:
        return 1.0
    yy, xx = np.indices(psf.shape)
    x_c = (xx * abs_psf).sum() / total
    y_c = (yy * abs_psf).sum() / total
    var_x = ((xx - x_c) ** 2 * abs_psf).sum() / total
    var_y = ((yy - y_c) ** 2 * abs_psf).sum() / total
    return float(np.sqrt(var_x + var_y))


def extract_psf_from_atom_image(
    atom_image: np.ndarray,
    background: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Estimate a PSF from a single isolated atom in an image.

    Subtracts an optional background, clips negatives, recenters by
    centre-of-mass, and normalises to unit sum.
    """
    psf = atom_image.copy().astype(float)
    if background is not None:
        psf = psf - background
    psf = np.maximum(psf, 0.0)

    ny, nx = psf.shape
    cy, cx = ny // 2, nx // 2
    total = psf.sum()
    if total > 0:
        yy, xx = np.indices(psf.shape)
        x_c = (xx * psf).sum() / total
        y_c = (yy * psf).sum() / total
        psf = np.roll(psf, int(round(cy - y_c)), axis=0)
        psf = np.roll(psf, int(round(cx - x_c)), axis=1)
        psf = psf / psf.sum()
    return psf
