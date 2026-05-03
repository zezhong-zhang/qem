"""Fitting algorithms and models for QEM."""

from .background import BackgroundEstimator, estimate_background
from .convolve_fitting import (
    ADFConvolutionFitting,
    ConvolutionFitting,
    OptimizationResult,
    PtychographyFitting,
    fit_adf_image,
    fit_ssb_ptychography,
)
from .image_fitting import ImageFitting
from .linear_solver import (
    DesignMatrixBuilder,
    LinearSystemSolver,
    MemoryInfo,
    ParameterValidator,
    SolutionProcessor,
)
from .model import GaussianKernel, GaussianModel, ImageModel, LorentzianModel, VoigtModel
from .point_potential import (
    ConvolutionImageModel,
    PointPotentialModel,
    calculate_residual,
    correlation_coefficient,
    normalized_root_mean_square_error,
)
from .ptychography_optimization import PtychographyOptimizer
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
    "ImageFitting",

    # Linear solver components
    "LinearSystemSolver",
    "DesignMatrixBuilder",
    "ParameterValidator",
    "SolutionProcessor",
    "MemoryInfo",

    # Ptychography / convolution-based fitting
    "PtychographyOptimizer",
    "ConvolutionFitting",
    "ADFConvolutionFitting",
    "PtychographyFitting",
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
    "BackgroundEstimator",
    "estimate_background",

    # Voronoi integration
    "voronoi_integrate",
    "voronoi_point_record",
]
