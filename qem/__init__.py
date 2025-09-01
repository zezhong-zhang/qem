"""
QEM - Quantitative Electron Microscopy Analysis Package

A comprehensive package for analyzing atomic-resolution electron microscopy images.
"""

__version__ = "0.1.0"

# Core modules - import these directly as they're commonly used
from . import io
from . import utils

# Only import backend if needed to avoid initialization issues
try:
    from . import backend
    __all__ = ['io', 'utils', 'backend']
except ImportError:
    __all__ = ['io', 'utils']