"""Electron-optics primitives: aberrations, χ, envelopes, aperture, PSF.

Public API — everything imported here is part of the supported surface.

Quick reference
---------------

    from qem.optics import Aberrations, Probe, Grid, ssb_psf

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

Sample tilt (:func:`tilt_from_affine`, :class:`SampleTilt`) and the
numpy multislice helpers (:func:`focused_probe`,
:func:`make_contrast_transfer_function`, …) live next door in
:mod:`qem.optics.tilt` and :mod:`qem.optics.wave`.
"""

from __future__ import annotations

# Aberrations
from .aberrations import (
    ALL_SYMBOLS,
    MAGNITUDE_SYMBOLS,
    PHASE_SYMBOLS,
    Aberrations,
)

# Aperture functions
from .aperture import hard_aperture, soft_aperture

# Phase function and its gradient
from .chi import chi, grad_chi

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

# Coherence envelopes
from .envelopes import (
    focal_spread_from_chromatic,
    partial_coherence_envelope,
    spatial_envelope,
    temporal_envelope,
)

# Grid + Probe data classes
from .grid import Grid
from .probe import Probe, probe_wave, probe_wave_q

# PSF / CTF builders + image-analysis helpers
from .psf import (
    adf_ctf,
    adf_psf,
    calculate_psf_width,
    epie_ctf,
    epie_psf,
    extract_psf_from_atom_image,
    icom_ctf,
    icom_psf,
    ssb_ctf,
    ssb_psf,
)

# Sample tilt
from .tilt import SampleTilt, tilt_from_affine

# Numpy multislice helpers
from .wave import (
    Cc_defocus_spread,
    Cc_integration_points,
    convert_deltaE,
    convert_tilt_angles,
    depth_of_field,
    focused_probe,
    make_contrast_transfer_function,
    plane_wave_illumination,
    simulation_result_with_Cc,
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
    # PSF image-analysis helpers
    "calculate_psf_width",
    "extract_psf_from_atom_image",
    # Sample tilt
    "SampleTilt",
    "tilt_from_affine",
    # Multislice helpers
    "depth_of_field",
    "convert_tilt_angles",
    "make_contrast_transfer_function",
    "focused_probe",
    "plane_wave_illumination",
    "convert_deltaE",
    "Cc_integration_points",
    "Cc_defocus_spread",
    "simulation_result_with_Cc",
]
