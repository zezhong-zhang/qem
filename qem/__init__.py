"""
QEM - Quantitative Electron Microscopy Analysis Package

A comprehensive package for analyzing atomic-resolution electron microscopy images.

The Streamlit GUI lives in :mod:`qem.app` and is intentionally **not** imported
here, so ``import qem`` is safe in headless, library, and docs-build contexts.
Launch the GUI with the ``qem-app`` console script (see :mod:`qem.cli`) or
``streamlit run -m qem.app``.
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
from . import optimizers
from . import utils
from . import schema

__all__ = [
    'ImageFitting',
    'fit',
    'analysis',
    'visualization',
    'processing',
    'instruments',
    'optimizers',
    'utils',
    'schema',
    'io',
]