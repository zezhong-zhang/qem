"""Electron-optics primitives: aberrations, χ, envelopes, aperture, PSF.

Public API — everything imported here is part of the supported surface.

Quick reference
---------------

    from qem.instruments.optics import Aberrations, Probe, Grid, ssb_psf

    grid  = Grid(pixels=(128, 128), extent=(64.0, 64.0))   # Å
    probe = Probe(
        energy=60e3, aperture=20.0,
        aberrations=Aberrations(defocus=50, Cs=1e7),
        focal_spread=focal_spread_from_chromatic(2e7, 0.5, 60e3),
        angular_spread=0.1,            # mrad
    )
    psf   = ssb_psf(grid, probe)       # torch.Tensor, real, fftshift-centered

Units (everything in Å / radians / eV inside; mrad at the user surface):

- ``Probe.energy``       eV
- ``Probe.aperture``     mrad (semi-angle cutoff)
- ``Probe.focal_spread`` Å (1/e half-width Δ_f)
- ``Probe.angular_spread`` mrad (source angular σ)
- ``Aberrations.Cnm``    Å
- ``Aberrations.phinm``  radians
- ``Grid.extent``        Å

Sign convention: ``defocus = -C10`` (abtem); positive ``defocus`` ⇒
under-focus.  The wave function is ``ψ(k) = aperture · exp(-i χ(k))``.
"""

from __future__ import annotations

# Constants and unit conversions
from .constants import (
    invA_to_mrad,
    mrad_to_invA,
    point_resolution,
    relativistic_mass_correction,
    scherzer_defocus,
    wavelength,
    wavev,
)

# Aberrations
from .aberrations import (
    ALL_SYMBOLS,
    MAGNITUDE_SYMBOLS,
    PHASE_SYMBOLS,
    Aberrations,
)

# Phase function and its gradient
from .chi import chi, grad_chi

# Coherence envelopes
from .envelopes import (
    focal_spread_from_chromatic,
    partial_coherence_envelope,
    spatial_envelope,
    temporal_envelope,
)

# Aperture functions
from .aperture import hard_aperture, soft_aperture

# Grid + Probe data classes
from .grid import Grid
from .probe import Probe, probe_wave, probe_wave_q

# PSF / CTF builders
from .psf import (
    adf_ctf,
    adf_psf,
    epie_ctf,
    epie_psf,
    icom_ctf,
    icom_psf,
    ssb_ctf,
    ssb_psf,
)

__all__ = [
    # Constants
    "wavev",
    "wavelength",
    "relativistic_mass_correction",
    "mrad_to_invA",
    "invA_to_mrad",
    "scherzer_defocus",
    "point_resolution",
    # Aberrations
    "Aberrations",
    "ALL_SYMBOLS",
    "MAGNITUDE_SYMBOLS",
    "PHASE_SYMBOLS",
    # Chi
    "chi",
    "grad_chi",
    # Envelopes
    "temporal_envelope",
    "spatial_envelope",
    "partial_coherence_envelope",
    "focal_spread_from_chromatic",
    # Aperture
    "hard_aperture",
    "soft_aperture",
    # Data containers
    "Grid",
    "Probe",
    # Probe wave function
    "probe_wave",
    "probe_wave_q",
    # PSF / CTF builders
    "ssb_psf",
    "ssb_ctf",
    "adf_psf",
    "adf_ctf",
    "icom_psf",
    "icom_ctf",
    "epie_psf",
    "epie_ctf",
]
