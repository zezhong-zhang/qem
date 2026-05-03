"""Instrument-specific models and corrections for QEM.

The new optics public API lives under :mod:`qem.instruments.optics`:

    from qem.instruments.optics import Aberrations, Probe, Grid, ssb_psf

For backward compatibility, the legacy CTF/probe class API
(``SSB_CTF``, ``ADF_CTF``, ``Aberration``, ``Probe`` etc.) is still
re-exported from this module via :mod:`qem.instruments._legacy`.
Calling those wrappers emits a ``DeprecationWarning``; they delegate
to the new functional core under the hood.
"""

from __future__ import annotations

# --- new public API (preferred) ------------------------------------------
from . import optics
from .optics import (
    Aberrations,
    Grid,
    Probe,
    adf_ctf,
    adf_psf,
    chi,
    epie_ctf,
    epie_psf,
    focal_spread_from_chromatic,
    grad_chi,
    hard_aperture,
    icom_ctf,
    icom_psf,
    invA_to_mrad,
    mrad_to_invA,
    partial_coherence_envelope,
    point_resolution,
    probe_wave,
    probe_wave_q,
    relativistic_mass_correction,
    scherzer_defocus,
    soft_aperture,
    spatial_envelope,
    ssb_ctf,
    ssb_psf,
    temporal_envelope,
    wavelength,
    wavev,
)

# --- detector + tilt (unchanged) -----------------------------------------
from .detector import Calibrate_Detector, Calibrate_Dose, Detector
from .tilt import SampleTilt, tilt_from_affine

# --- wave / multislice helpers (carved out of the old probe.py) ----------
from . import wave
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

# --- legacy back-compat shim ---------------------------------------------
from ._legacy import (
    ADF_CTF,
    Aberration,
    LegacyProbe as _LegacyProbe,
    ProbeParameters,
    SSB_CTF,
    aberration_starter_pack,
    calculate_psf_width,
    chi as _legacy_chi,                       # noqa: F401  (back-compat re-export)
    create_aberration_list,
    create_probe_parameters,
    ePIE_CTF,
    extract_psf_from_atom_image,
    iCoM_CTF,
)

# Old code did `from qem.instruments import Probe`; it expected the legacy
# probe class with envelope methods, not the new dataclass.  Provide both
# under different names so callers can migrate gradually.
LegacyProbe = _LegacyProbe

__all__ = [
    # ---- new optics surface --------------------------------------------
    "optics",
    "Aberrations",
    "Probe",
    "Grid",
    "chi",
    "grad_chi",
    "temporal_envelope",
    "spatial_envelope",
    "partial_coherence_envelope",
    "focal_spread_from_chromatic",
    "hard_aperture",
    "soft_aperture",
    "probe_wave",
    "probe_wave_q",
    "ssb_psf",
    "ssb_ctf",
    "adf_psf",
    "adf_ctf",
    "icom_psf",
    "icom_ctf",
    "epie_psf",
    "epie_ctf",
    "wavev",
    "wavelength",
    "relativistic_mass_correction",
    "mrad_to_invA",
    "invA_to_mrad",
    "scherzer_defocus",
    "point_resolution",
    # ---- detector + tilt -----------------------------------------------
    "Detector",
    "Calibrate_Dose",
    "Calibrate_Detector",
    "SampleTilt",
    "tilt_from_affine",
    # ---- legacy / deprecated -------------------------------------------
    "Aberration",
    "aberration_starter_pack",
    "LegacyProbe",
    "ProbeParameters",
    "create_probe_parameters",
    "create_aberration_list",
    "SSB_CTF",
    "ADF_CTF",
    "ePIE_CTF",
    "iCoM_CTF",
    # ---- moved to qem/processing/psf.py (re-exported for back-compat) --
    "calculate_psf_width",
    "extract_psf_from_atom_image",
    # ---- wave / multislice helpers (qem.instruments.wave) --------------
    "wave",
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
