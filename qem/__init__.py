"""QEM - Quantitative Electron Microscopy Analysis Package.

Pure-PyTorch library for atomic-resolution STEM image quantification.
"""

__version__ = "0.1.0"

# Core modules - import these directly as they're commonly used
from . import io
from . import fit
from .fit.fitter import Fitter
from . import analysis
from . import viz
from . import processing
from . import detector
from . import optics
from . import utils

__all__ = [
    'Fitter',
    'fit',
    'analysis',
    'viz',
    'processing',
    'detector',
    'optics',
    'utils',
    'io',
]
