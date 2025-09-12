"""
QEM - Quantitative Electron Microscopy Analysis Package

A comprehensive package for analyzing atomic-resolution electron microscopy images.
"""

__version__ = "0.1.0"

# Core modules - import these directly as they're commonly used
from . import io
from . import fit
from .fit.image_fitting import ImageFitting
from . import analysis
from . import visualization
from . import processing
from . import instruments
from . import utils
from . import schema

# Main application
from .app import *

__all__ = [
    'data',
    'fit', 
    'analysis',
    'visualization',
    'processing',
    'instruments',
    'utils',
    'schema',
]