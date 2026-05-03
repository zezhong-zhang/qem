"""Hardware instrument models — detector, tilt, multislice wave helpers.

Electron-optics primitives (Aberrations, Probe, Grid, PSF builders) live
in :mod:`qem.optics`, not here.
"""

from __future__ import annotations

from .detector import Calibrate_Detector, Calibrate_Dose, Detector
from .tilt import SampleTilt, tilt_from_affine
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
    # detector
    "Detector",
    "Calibrate_Dose",
    "Calibrate_Detector",
    # tilt
    "SampleTilt",
    "tilt_from_affine",
    # wave / multislice helpers
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
