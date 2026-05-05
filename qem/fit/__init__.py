"""Fitting algorithms and models for QEM.

`Fitter` is lazy-loaded to dodge fitter.py's heavy import cascade
(h5py, matscipy, GMM, ASE, crystal_analyzer). Use `from qem.fit import
Fitter` or `from qem import Fitter` and the deferral kicks in.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from .background import Background, estimate_background
from .solver import (
    build_local_peaks,
    build_sparse_matrix,
    solve_system,
    validate_params,
    validate_solution,
    process_height_scaling,
    process_background,
    linear_estimator,
)
from .model import GaussianKernel, GaussianModel, ImageModel, LorentzianModel, VoigtModel
from .potential import (
    ConvolutionImageModel,
    PointPotentialModel,
    calculate_residual,
    correlation_coefficient,
    normalized_root_mean_square_error,
)
from .ptycho import PtychoOptimizer
from .refine import calculate_center_of_mass, fit_gaussian, gauss2d
from .voronoi import voronoi_integrate, voronoi_point_record


# Lazy attributes — these all transitively require fitter.py
# (ConvFit / AdfConvFit / PtychoFit subclass Fitter), so we defer
# their import until first access. Saves ~40 transitive deps on
# `import qem` for headless users.
_LAZY = {
    "Fitter": ("qem.fit.fitter", "Fitter"),
    "ConvFit": ("qem.fit.convolve", "ConvFit"),
    "AdfConvFit": ("qem.fit.convolve", "AdfConvFit"),
    "PtychoFit": ("qem.fit.convolve", "PtychoFit"),
    "OptimizationResult": ("qem.fit.convolve", "OptimizationResult"),
    "fit_adf_image": ("qem.fit.convolve", "fit_adf_image"),
    "fit_ssb_ptychography": ("qem.fit.convolve", "fit_ssb_ptychography"),
}


def __getattr__(name: str):
    target = _LAZY.get(name)
    if target is None:
        raise AttributeError(f"module 'qem.fit' has no attribute {name!r}")
    import importlib
    return getattr(importlib.import_module(target[0]), target[1])


if TYPE_CHECKING:
    from .convolve import (  # noqa: F401
        AdfConvFit,
        ConvFit,
        OptimizationResult,
        PtychoFit,
        fit_adf_image,
        fit_ssb_ptychography,
    )
    from .fitter import Fitter as Fitter  # noqa: F401

__all__ = [
    # Core models
    "ImageModel",
    "GaussianModel",
    "LorentzianModel",
    "VoigtModel",
    "GaussianKernel",

    # Main fitting class
    "Fitter",

    # Linear solver pipeline (functions, not classes — see qem.fit.solver)
    "build_local_peaks",
    "build_sparse_matrix",
    "solve_system",
    "validate_params",
    "validate_solution",
    "process_height_scaling",
    "process_background",
    "linear_estimator",

    # Ptychography / convolution-based fitting
    "PtychoOptimizer",
    "ConvFit",
    "AdfConvFit",
    "PtychoFit",
    "OptimizationResult",
    "fit_ssb_ptychography",
    "fit_adf_image",

    # Point-potential and convolution models
    "PointPotentialModel",
    "ConvolutionImageModel",
    "correlation_coefficient",
    "normalized_root_mean_square_error",
    "calculate_residual",

    # Refinement helpers
    "calculate_center_of_mass",
    "fit_gaussian",
    "gauss2d",

    # Background estimation
    "Background",
    "estimate_background",

    # Voronoi integration
    "voronoi_integrate",
    "voronoi_point_record",
]
