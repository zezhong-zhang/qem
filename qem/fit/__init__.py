"""Fitting algorithms and models for QEM."""

from .background import BackgroundEstimator, estimate_background
from .convolve_fitting import ADFConvolutionFitting, ConvolutionFitting, PtychographyFitting
from .image_fitting import ImageFitting
from .linear_solver import (
    DesignMatrixBuilder,
    LinearSystemSolver,
    MemoryInfo,
    ParameterValidator,
    SolutionProcessor,
)
from .model import GaussianModel, ImageModel, LorentzianModel, VoigtModel
from .ptychography_optimization import PtychographyOptimizer
from .refine import calculate_center_of_mass, fit_gaussian, gauss2d
from .voronoi import voronoi_integrate

__all__ = [
    # Core models
    "ImageModel",
    "GaussianModel",
    "LorentzianModel",
    "VoigtModel",

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

    # Refinement helpers
    "calculate_center_of_mass",
    "fit_gaussian",
    "gauss2d",

    # Background estimation
    "BackgroundEstimator",
    "estimate_background",

    # Voronoi integration
    "voronoi_integrate",
]
