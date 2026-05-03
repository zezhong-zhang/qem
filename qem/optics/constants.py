"""Physical constants and unit conversions for electron optics.

Pure functions, no state.  Inputs are scalars or any array-like that
participates in standard arithmetic (numpy arrays, torch tensors, plain
floats).  Returns match the input type.
"""

from __future__ import annotations

import math

# Planck's constant times speed of light, eV·Å.
_HC = 1.23984193e4
# Electron rest mass energy, eV.
_M0C2 = 5.109989461e5


def wavev(energy_eV):
    """Relativistically-corrected electron wavenumber k = 1/λ in Å⁻¹.

    Kirkland Eq. 2.5.
    """
    return ((energy_eV * (energy_eV + 2.0 * _M0C2)) ** 0.5) / _HC


def wavelength(energy_eV):
    """Relativistic electron wavelength λ in Å."""
    return _HC / ((energy_eV * (energy_eV + 2.0 * _M0C2)) ** 0.5)


def relativistic_mass_correction(energy_eV):
    """γ = (m₀c² + E)/m₀c² (Kirkland Eq. 2.2)."""
    return (_M0C2 + energy_eV) / _M0C2


def mrad_to_invA(angle_mrad, energy_eV):
    """Convert a scattering angle in mrad to spatial frequency in Å⁻¹.

    α [rad] = q / k  ⇒  q [Å⁻¹] = (α [mrad] · 1e-3) · k(E).
    """
    return angle_mrad * 1e-3 * wavev(energy_eV)


def invA_to_mrad(q_invA, energy_eV):
    """Inverse of :func:`mrad_to_invA`."""
    return q_invA / wavev(energy_eV) * 1e3


def scherzer_defocus(Cs_A, energy_eV):
    """Scherzer defocus in Å (positive ⇒ under-focus, abtem convention).

    Δf = sqrt(1.5 · Cs · λ).  See Kirkland Eq. 5.21.
    """
    return math.copysign(1.0, Cs_A) * math.sqrt(abs(1.5 * Cs_A * wavelength(energy_eV)))


def point_resolution(Cs_A, energy_eV):
    """Scherzer point resolution in Å.  See Kirkland Eq. 5.23."""
    lam = wavelength(energy_eV)
    return 0.66 * (abs(Cs_A) ** 0.25) * (lam ** 0.75)
