"""Image-analysis helpers for point-spread functions.

Lives outside :mod:`qem.instruments.optics` because these don't generate
PSFs — they measure or extract them from real (or simulated) image data.
"""

from __future__ import annotations

from typing import Optional

import numpy as np


def calculate_psf_width(psf: np.ndarray, dx: float = 1.0) -> float:
    """Effective PSF width: second-moment radius of ``|psf|``.

    Parameters
    ----------
    psf
        2D PSF array (real or signed; for SSB-like PSFs with a negative
        halo, the absolute value is used so the halo contributes to the
        width).
    dx
        Pixel size in Å (units affect the returned width).

    Returns
    -------
    Width in *pixels* (multiply by ``dx`` for Å).
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


__all__ = ["calculate_psf_width", "extract_psf_from_atom_image"]
