"""
Fitting algorithms and models for QEM.
"""

from .model import *
from .image_fitting import ImageFitting
from .linear_solver import *
from .refine import *
from .background_estimator import *
# from .background_2d import *
from .voronoi import *

__all__ = [
    # Core models
    'ImageModel',
    
    # Main fitting class
    'ImageFitting',
    
    # Linear solver components
    'ParameterValidator',
    'DesignMatrixBuilder',
    'LinearSystemSolver',
    'SolutionProcessor',
    'MemoryEstimator',
    
    # Refinement functions
    'calculate_center_of_mass',
    'fit_gaussian',
    'gauss2d',
    
    # GMM
    'GaussianMixtureModel',
    
    # Background estimation
    'RobustBackgroundEstimator',
    'background_estimation',
    
    # Voronoi integration
    'voronoi_integrate',
]