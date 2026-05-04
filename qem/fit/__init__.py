"""Fitting algorithms and models for QEM."""

from .background import Background, estimate_background
from .convolve import (
    AdfConvFit,
    ConvFit,
    OptimizationResult,
    PtychoFit,
    fit_adf_image,
    fit_ssb_ptychography,
)
from .fitter import Fitter
from .solver import (
    DesignMatrixBuilder,
    LinearSystemSolver,
    ParameterValidator,
    SolutionProcessor,
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

__all__ = [
    # Core models
    "ImageModel",
    "GaussianModel",
    "LorentzianModel",
    "VoigtModel",
    "GaussianKernel",

    # Main fitting class
    "Fitter",

    # Linear solver components
    "LinearSystemSolver",
    "DesignMatrixBuilder",
    "ParameterValidator",
    "SolutionProcessor",

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
