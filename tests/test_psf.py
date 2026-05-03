"""High-level PSF builder tests for qem.optics.

Covers shapes, centering, dtype, partial-coherence behavior, and
imaging-mode-specific invariants (SSB negative halo, ADF positivity,
iCoM filter, ePIE smoothing).  Numerical equivalence with abtem is
verified separately in :mod:`tests.test_optics_vs_abtem`.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from qem.optics import (
    Aberrations,
    Grid,
    Probe,
    adf_psf,
    epie_psf,
    focal_spread_from_chromatic,
    icom_psf,
    ssb_psf,
)
from qem.processing.psf import calculate_psf_width, extract_psf_from_atom_image


@pytest.fixture
def grid() -> Grid:
    return Grid(pixels=(64, 64), extent=(64.0, 64.0))


@pytest.fixture
def probe() -> Probe:
    return Probe(energy=60e3, aperture=20.0)


# ---------------------------------------------------------------------------
# SSB
# ---------------------------------------------------------------------------


class TestSSB:
    def test_shape_and_dtype(self, grid, probe):
        psf = ssb_psf(grid, probe)
        assert psf.shape == grid.pixels
        assert psf.dtype.is_floating_point
        assert isinstance(psf, torch.Tensor)

    def test_psf_centered(self, grid, probe):
        psf = ssb_psf(grid, probe)
        peak = tuple(int(x) for x in (psf == psf.max()).nonzero()[0])
        assert peak == (32, 32)

    def test_psf_has_negative_halo(self, grid, probe):
        """SSB's signature: a ring of negative phase contrast around the peak."""
        psf = ssb_psf(grid, probe)
        assert psf.min().item() < -0.01

    def test_alpha_changes_psf_width(self, grid):
        """Larger convergence angle → tighter PSF (smaller width)."""
        psf_small = ssb_psf(grid, Probe(energy=60e3, aperture=10.0))
        psf_large = ssb_psf(grid, Probe(energy=60e3, aperture=30.0))
        w_small = calculate_psf_width(psf_small.cpu().numpy())
        w_large = calculate_psf_width(psf_large.cpu().numpy())
        assert w_large < w_small


# ---------------------------------------------------------------------------
# ADF
# ---------------------------------------------------------------------------


class TestADF:
    def test_shape(self, grid, probe):
        psf = adf_psf(grid, probe)
        assert psf.shape == grid.pixels

    def test_psf_centered(self, grid, probe):
        psf = adf_psf(grid, probe)
        peak = tuple(int(x) for x in (psf == psf.max()).nonzero()[0])
        assert peak == (32, 32)

    def test_psf_non_negative(self, grid, probe):
        """ADF is incoherent → PSF = |probe|² is non-negative."""
        psf = adf_psf(grid, probe)
        assert psf.min().item() >= 0.0

    def test_psf_normalized(self, grid, probe):
        """ADF PSF is normalized to unit sum."""
        psf = adf_psf(grid, probe)
        assert psf.sum().item() == pytest.approx(1.0, abs=1e-6)


# ---------------------------------------------------------------------------
# iCoM
# ---------------------------------------------------------------------------


class TestICoM:
    def test_no_filter_is_unit_ctf(self, grid, probe):
        """Without a high-pass filter the CTF is identically 1 → PSF is δ."""
        psf = icom_psf(grid, probe)
        assert psf.shape == grid.pixels
        # δ at (32, 32) when fftshift'd; max should clearly dominate
        assert psf.max().item() > 0.99

    def test_high_pass_filter_changes_psf(self, grid, probe):
        psf_no_hp = icom_psf(grid, probe)
        psf_hp = icom_psf(grid, probe, high_pass_mrad=5.0)
        # Filter should redistribute weight away from the central peak
        assert psf_hp.max().item() < psf_no_hp.max().item()


# ---------------------------------------------------------------------------
# ePIE
# ---------------------------------------------------------------------------


class TestEPIE:
    def test_shape(self, grid, probe):
        psf = epie_psf(grid, probe)
        assert psf.shape == grid.pixels

    def test_psf_centered(self, grid, probe):
        psf = epie_psf(grid, probe)
        peak = tuple(int(x) for x in (psf == psf.max()).nonzero()[0])
        assert peak == (32, 32)


# ---------------------------------------------------------------------------
# Partial coherence: temporal + spatial envelopes
# ---------------------------------------------------------------------------


class TestPartialCoherence:
    def test_temporal_coherence_dampens_high_q(self, grid):
        """Focal spread dampens high spatial frequencies in the q-space CTF.

        Test directly against the spectrum (a robust, monotone signal),
        not the second-moment width of the real-space PSF (which can drift
        either way for small effects on a coarse grid).
        """
        from qem.optics import ssb_ctf

        ctf_clean = ssb_ctf(grid, Probe(energy=60e3, aperture=20.0))
        ctf_damp = ssb_ctf(
            grid,
            Probe(
                energy=60e3, aperture=20.0,
                focal_spread=focal_spread_from_chromatic(2e7, 0.5, 60e3),
            ),
        )
        # Sum of the magnitude must shrink under damping.
        assert ctf_damp.abs().sum().item() < ctf_clean.abs().sum().item()

    def test_spatial_envelope_unit_for_unaberrated(self, grid):
        """∇χ = 0 for unaberrated probe → spatial envelope ≡ 1.

        Source size on its own (no aberrations) leaves the PSF unchanged.
        """
        clean = ssb_psf(grid, Probe(energy=60e3, aperture=20.0))
        with_src = ssb_psf(
            grid, Probe(energy=60e3, aperture=20.0, angular_spread=1.0),
        )
        torch.testing.assert_close(clean, with_src)

    def test_spatial_envelope_with_aberration_dampens_ctf(self, grid):
        """With aberrations, source size damps the q-space CTF magnitude."""
        from qem.optics import ssb_ctf

        ab = Aberrations(defocus=50.0, Cs=1e7)  # uncorrected STEM
        ctf_clean = ssb_ctf(grid, Probe(energy=60e3, aperture=20.0, aberrations=ab))
        ctf_damp = ssb_ctf(
            grid,
            Probe(
                energy=60e3, aperture=20.0, aberrations=ab,
                angular_spread=1.0,
            ),
        )
        assert ctf_damp.abs().sum().item() < ctf_clean.abs().sum().item()


# ---------------------------------------------------------------------------
# Image-analysis helpers (qem/processing/psf.py)
# ---------------------------------------------------------------------------


class TestPSFHelpers:
    def test_calculate_psf_width_gaussian(self):
        y, x = np.indices((32, 32))
        sigma = 2.0
        psf = np.exp(-((x - 16.0) ** 2 + (y - 16.0) ** 2) / (2 * sigma ** 2))
        psf /= psf.sum()
        # 2D Gaussian: total width = sqrt(σ² + σ²) = σ·sqrt(2)
        width = calculate_psf_width(psf)
        assert width == pytest.approx(sigma * np.sqrt(2), rel=0.05)

    def test_calculate_psf_width_handles_negative_halo(self):
        """For SSB-like PSFs with negatives, |psf| moments still give a width."""
        y, x = np.indices((32, 32))
        r = np.sqrt((x - 16.0) ** 2 + (y - 16.0) ** 2)
        psf = np.exp(-r ** 2 / 8.0) - 0.1 * np.exp(-r ** 2 / 50.0)
        assert calculate_psf_width(psf) > 0

    def test_extract_psf_recenters_and_normalizes(self):
        atom = np.zeros((32, 32))
        atom[10, 14] = 1.0          # off-center peak
        psf = extract_psf_from_atom_image(atom)
        assert psf.shape == atom.shape
        assert psf.sum() == pytest.approx(1.0, abs=1e-6)
        peak = np.unravel_index(np.argmax(psf), psf.shape)
        assert peak == (16, 16)     # recentered on grid centre

    def test_extract_psf_subtracts_background(self):
        atom = np.ones((32, 32)) * 0.1   # uniform background
        atom[16, 16] = 1.1               # plus a peak
        bg = np.ones_like(atom) * 0.1
        psf = extract_psf_from_atom_image(atom, background=bg)
        assert psf.min() >= 0.0
