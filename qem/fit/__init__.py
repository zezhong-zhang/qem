"""
Fitting algorithms and models for QEM.
"""

from .model import *
from .image_fitting import ImageFitting
from .linear_solver import *
from .refine import *
from .background import BackgroundEstimator, estimate_background
from .voronoi import *

__all__ = [
    # Core models
    'ImageModel',
    
    # Main fitting class
    'ImageFitting',
    
    # Linear solver components
    'LinearSystemSolver',
    'DesignMatrixBuilder',
    'ParameterValidator', 
    'SolutionProcessor',
    'MemoryInfo',
    
    # Refinement functions
    'calculate_center_of_mass',
    'fit_gaussian',
    'gauss2d',
    
    # GMM
    'GaussianMixtureModel',
    
    # Background estimation
    'BackgroundEstimator',
    'estimate_background',
    
    # Voronoi integration
    'voronoi_integrate',
]