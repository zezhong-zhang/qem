"""Numerical equivalence of the new optics package with abtem.

The reference values are computed from the abtem source code (extracted
into standalone closed-form functions in this file — abtem itself
requires Python 3.11+ and isn't a runtime dependency of qem).  Each test
asserts agreement to fp64 tolerance.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from qem.optics import (
    Aberrations,
    Grid,
    Probe,
    chi,
    focal_spread_from_chromatic,
    grad_chi,
    spatial_envelope,
    temporal_envelope,
    wavelength,
)


# ---------------------------------------------------------------------------
# abtem reference closed forms (from abtem.transfer)
# ---------------------------------------------------------------------------

_ABTEM_POLAR_DEFAULTS = {
    "C10": 0.0, "C12": 0.0, "phi12": 0.0,
    "C21": 0.0, "phi21": 0.0,
    "C23": 0.0, "phi23": 0.0,
    "C30": 0.0,
    "C32": 0.0, "phi32": 0.0,
    "C34": 0.0, "phi34": 0.0,
    "C41": 0.0, "phi41": 0.0,
    "C43": 0.0, "phi43": 0.0,
    "C45": 0.0, "phi45": 0.0,
    "C50": 0.0,
    "C52": 0.0, "phi52": 0.0,
    "C54": 0.0, "phi54": 0.0,
    "C56": 0.0, "phi56": 0.0,
}


def abtem_chi(alpha, phi, *, wavelength_A, **coefs):
    """Mirror of abtem.transfer.Aberrations._evaluate_from_angular_grid."""
    p = dict(_ABTEM_POLAR_DEFAULTS)
    p.update(coefs)
    arr = np.zeros_like(alpha)
    arr += 0.5 * alpha**2 * (p["C10"] + p["C12"] * np.cos(2 * (phi - p["phi12"])))
    arr += (1 / 3) * alpha**3 * (
        p["C21"] * np.cos(phi - p["phi21"])
        + p["C23"] * np.cos(3 * (phi - p["phi23"]))
    )
    arr += 0.25 * alpha**4 * (
        p["C30"]
        + p["C32"] * np.cos(2 * (phi - p["phi32"]))
        + p["C34"] * np.cos(4 * (phi - p["phi34"]))
    )
    arr += 0.2 * alpha**5 * (
        p["C41"] * np.cos(phi - p["phi41"])
        + p["C43"] * np.cos(3 * (phi - p["phi43"]))
        + p["C45"] * np.cos(5 * (phi - p["phi45"]))
    )
    arr += (1 / 6) * alpha**6 * (
        p["C50"]
        + p["C52"] * np.cos(2 * (phi - p["phi52"]))
        + p["C54"] * np.cos(4 * (phi - p["phi54"]))
        + p["C56"] * np.cos(6 * (phi - p["phi56"]))
    )
    return 2 * np.pi / wavelength_A * arr


def abtem_temporal_envelope(alpha, *, wavelength_A, focal_spread):
    """Mirror of abtem.transfer.TemporalEnvelope._evaluate_from_angular_grid."""
    return np.exp(-((0.5 * np.pi / wavelength_A * focal_spread * alpha**2) ** 2))


def abtem_spatial_envelope(
    alpha, phi, *, wavelength_A, angular_spread_mrad, **coefs
):
    """Mirror of abtem.transfer.SpatialEnvelope._evaluate_from_angular_grid.

    Same polynomial as ``abtem_chi`` differentiated; inlined here so the
    test is self-contained.
    """
    p = dict(_ABTEM_POLAR_DEFAULTS)
    p.update(coefs)
    angular_spread = angular_spread_mrad * 1e-3
    pre = 2 * np.pi / wavelength_A
    dchi_dk = pre * (
        ((p["C12"] * np.cos(2 * (phi - p["phi12"])) + p["C10"]) * alpha)
        + ((p["C23"] * np.cos(3 * (phi - p["phi23"]))
            + p["C21"] * np.cos(phi - p["phi21"])) * alpha**2)
        + ((p["C30"]
            + p["C32"] * np.cos(2 * (phi - p["phi32"]))
            + p["C34"] * np.cos(4 * (phi - p["phi34"]))) * alpha**3)
        + ((p["C45"] * np.cos(5 * (phi - p["phi45"]))
            + p["C43"] * np.cos(3 * (phi - p["phi43"]))
            + p["C41"] * np.cos(phi - p["phi41"])) * alpha**4)
        + ((p["C56"] * np.cos(6 * (phi - p["phi56"]))
            + p["C54"] * np.cos(4 * (phi - p["phi54"]))
            + p["C52"] * np.cos(2 * (phi - p["phi52"])) + p["C50"]) * alpha**5)
    )
    dchi_dphi = -pre * (
        0.5 * (2 * p["C12"] * np.sin(2 * (phi - p["phi12"]))) * alpha
        + (1 / 3) * (
            3 * p["C23"] * np.sin(3 * (phi - p["phi23"]))
            + p["C21"] * np.sin(phi - p["phi21"])
        ) * alpha**2
        + (1 / 4) * (
            4 * p["C34"] * np.sin(4 * (phi - p["phi34"]))
            + 2 * p["C32"] * np.sin(2 * (phi - p["phi32"]))
        ) * alpha**3
        + (1 / 5) * (
            5 * p["C45"] * np.sin(5 * (phi - p["phi45"]))
            + 3 * p["C43"] * np.sin(3 * (phi - p["phi43"]))
            + p["C41"] * np.sin(phi - p["phi41"])
        ) * alpha**4
        + (1 / 6) * (
            6 * p["C56"] * np.sin(6 * (phi - p["phi56"]))
            + 4 * p["C54"] * np.sin(4 * (phi - p["phi54"]))
            + 2 * p["C52"] * np.sin(2 * (phi - p["phi52"]))
        ) * alpha**5
    )
    return np.exp(-((angular_spread / 2) ** 2) * (dchi_dk**2 + dchi_dphi**2))


# ---------------------------------------------------------------------------
# fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def fp64_setup():
    prev = torch.get_default_dtype()
    torch.set_default_dtype(torch.float64)
    yield
    torch.set_default_dtype(prev)


def _alpha_phi(n_alpha=21, n_phi=12):
    a = np.linspace(0, 0.030, n_alpha)        # 0..30 mrad in rad
    p = np.linspace(0, 2 * np.pi, n_phi, endpoint=False)
    A, P = np.meshgrid(a, p, indexing="ij")
    return A, P


def _torch_xy(A, P):
    return torch.from_numpy(A), torch.from_numpy(P)


# ---------------------------------------------------------------------------
# chi tests — different aberration combinations
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("coefs", [
    {"C10": -50.0},                           # defocus only
    {"C30": 1e7},                              # Cs only
    {"C12": 10.0, "phi12": 0.5},               # 2-fold astigmatism
    {"C23": 50.0, "phi23": 1.2},               # 3-fold astigmatism
    {"C21": 30.0, "phi21": 0.7},               # axial coma
    {"C10": -50.0, "C30": 1e7, "C12": 10.0},   # combined
])
def test_chi_matches_abtem(fp64_setup, coefs):
    A, P = _alpha_phi()
    a_t, p_t = _torch_xy(A, P)
    lam = wavelength(60e3)

    qem_chi = chi(a_t, p_t, wavelength=lam, aberrations=Aberrations(**coefs)).cpu().numpy()
    ref = abtem_chi(A, P, wavelength_A=lam, **coefs)

    np.testing.assert_allclose(qem_chi, ref, rtol=1e-12, atol=1e-13)


def test_grad_chi_matches_finite_difference(fp64_setup):
    """grad_chi returns (∂χ/∂α, (1/α)·∂χ/∂φ) — verify both via finite diff."""
    coefs = {"C10": -50.0, "C30": 1e7, "C12": 10.0, "phi12": 0.5}
    lam = wavelength(60e3)
    a, p = 0.020, 0.7
    h = 1e-7

    a_t = torch.tensor([a])
    p_t = torch.tensor([p])
    g_alpha, g_phi = grad_chi(a_t, p_t, wavelength=lam,
                              aberrations=Aberrations(**coefs))

    chi_p = chi(torch.tensor([a + h]), p_t, wavelength=lam,
                aberrations=Aberrations(**coefs)).item()
    chi_m = chi(torch.tensor([a - h]), p_t, wavelength=lam,
                aberrations=Aberrations(**coefs)).item()
    fd_dalpha = (chi_p - chi_m) / (2 * h)
    assert abs(g_alpha.item() - fd_dalpha) < 1e-3

    chi_p = chi(a_t, torch.tensor([p + h]), wavelength=lam,
                aberrations=Aberrations(**coefs)).item()
    chi_m = chi(a_t, torch.tensor([p - h]), wavelength=lam,
                aberrations=Aberrations(**coefs)).item()
    fd_dphi_over_alpha = (chi_p - chi_m) / (2 * h) / a
    assert abs(g_phi.item() - fd_dphi_over_alpha) < 1e-3


# ---------------------------------------------------------------------------
# envelope tests
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("focal_spread", [0.0, 10.0, 30.0, 100.0])
def test_temporal_envelope_matches_abtem(fp64_setup, focal_spread):
    A, _ = _alpha_phi()
    a_t = torch.from_numpy(A[:, 0])
    lam = wavelength(60e3)
    qem = temporal_envelope(a_t, wavelength=lam, focal_spread=focal_spread).cpu().numpy()
    ref = abtem_temporal_envelope(A[:, 0], wavelength_A=lam, focal_spread=focal_spread)
    np.testing.assert_allclose(qem, ref, rtol=1e-12, atol=1e-13)


@pytest.mark.parametrize("coefs,sigma", [
    ({"C10": -50.0}, 0.5),
    ({"C30": 1e7}, 0.1),
    ({"C12": 10.0, "phi12": 0.5}, 1.0),
    ({"C10": -50.0, "C30": 1e7}, 0.1),
])
def test_spatial_envelope_matches_abtem(fp64_setup, coefs, sigma):
    A, P = _alpha_phi()
    a_t, p_t = _torch_xy(A, P)
    lam = wavelength(60e3)
    qem = spatial_envelope(
        a_t, p_t,
        wavelength=lam,
        aberrations=Aberrations(**coefs),
        angular_spread_mrad=sigma,
    ).cpu().numpy()
    ref = abtem_spatial_envelope(
        A, P, wavelength_A=lam, angular_spread_mrad=sigma, **coefs
    )
    np.testing.assert_allclose(qem, ref, rtol=1e-12, atol=1e-13)


def test_spatial_envelope_unit_for_unaberrated(fp64_setup):
    """No aberrations ⇒ ∇χ = 0 ⇒ envelope ≡ 1."""
    A, P = _alpha_phi()
    a_t, p_t = _torch_xy(A, P)
    lam = wavelength(60e3)
    qem = spatial_envelope(
        a_t, p_t, wavelength=lam,
        aberrations=Aberrations(),
        angular_spread_mrad=1.0,
    ).cpu().numpy()
    np.testing.assert_array_equal(qem, np.ones_like(qem))


def test_focal_spread_from_chromatic_conventions(fp64_setup):
    """Δ_f = Cc · ΔE / E, with σ-mode and FWHM-mode conversions."""
    Cc, E = 2e7, 60e3
    dE_1e = 0.5  # 1/e half-width
    dE_std = dE_1e / math.sqrt(2)
    dE_fwhm = dE_1e * 2 * math.sqrt(math.log(2))

    fs_1e = focal_spread_from_chromatic(Cc, dE_1e, E, "1/e")
    fs_std = focal_spread_from_chromatic(Cc, dE_std, E, "std")
    fs_fwhm = focal_spread_from_chromatic(Cc, dE_fwhm, E, "FWHM")

    assert math.isclose(fs_1e, Cc * dE_1e / E)
    assert math.isclose(fs_std, fs_1e)
    assert math.isclose(fs_fwhm, fs_1e, rel_tol=1e-12)


# ---------------------------------------------------------------------------
# Probe / Aberrations sign-convention sanity
# ---------------------------------------------------------------------------


def test_defocus_alias_negates_C10(fp64_setup):
    """Aberrations(defocus=50) must store C10 = -50."""
    ab = Aberrations(defocus=50.0)
    assert ab.C10 == -50.0
    assert ab.defocus == 50.0


def test_chi_invariant_under_alias(fp64_setup):
    """chi(defocus=50) == chi(C10=-50)."""
    A, P = _alpha_phi()
    a_t, p_t = _torch_xy(A, P)
    lam = wavelength(60e3)
    chi_alias = chi(a_t, p_t, wavelength=lam, aberrations=Aberrations(defocus=50.0))
    chi_direct = chi(a_t, p_t, wavelength=lam, aberrations=Aberrations(C10=-50.0))
    np.testing.assert_array_equal(chi_alias.cpu().numpy(), chi_direct.cpu().numpy())


# ---------------------------------------------------------------------------
# Probe wave function: norm + centering
# ---------------------------------------------------------------------------


def test_probe_wave_normalized_and_centered(fp64_setup):
    grid = Grid(pixels=(64, 64), extent=(64.0, 64.0))
    probe = Probe(energy=60e3, aperture=20.0)
    psi = probe_wave_real_space(grid, probe)
    intensity = psi.real**2 + psi.imag**2
    assert math.isclose(float(intensity.sum()), 1.0, abs_tol=1e-12)
    peak = tuple(int(x) for x in (intensity == intensity.max()).nonzero()[0])
    assert peak == (32, 32)


def probe_wave_real_space(grid, probe):
    from qem.optics import probe_wave  # late import to keep top tidy
    return probe_wave(grid, probe)
