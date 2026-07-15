"""QEM - Quantitative Electron Microscopy Analysis Package.

Pure-PyTorch library for atomic-resolution STEM image quantification.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

__version__ = "0.1.0"

# Subpackages are lightweight enough to import eagerly. The heavy
# Fitter class (h5py + matscipy + GMM + ASE + crystal_analyzer) is
# loaded lazily via __getattr__ below.
from . import analysis, detector, fit, io, optics, processing, utils, viz

__all__ = [
    "Fitter",
    "fit",
    "analysis",
    "viz",
    "processing",
    "detector",
    "optics",
    "utils",
    "io",
]


def __getattr__(name: str):
    if name == "Fitter":
        from .fit.fitter import Fitter as _Fitter
        return _Fitter
    raise AttributeError(f"module 'qem' has no attribute {name!r}")


if TYPE_CHECKING:
    # IDE-only re-export so autocomplete still works.
    from .fit.fitter import Fitter as Fitter  # noqa: F401
