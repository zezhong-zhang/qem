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
from .fit.fitter import Fitter
from . import analysis
from . import viz
from . import processing
from . import instruments
from . import utils

__all__ = [
    'Fitter',
    'fit',
    'analysis',
    'viz',
    'processing',
    'instruments',
    'utils',
    'io',
]
