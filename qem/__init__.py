"""
QEM - Quantitative Electron Microscopy Analysis Package

A comprehensive package for analyzing atomic-resolution electron microscopy images.

The Streamlit GUI lives in :mod:`qem.app` and is intentionally **not** imported
here, so ``import qem`` is safe in headless, library, and docs-build contexts.
Launch the GUI with the ``qem-app`` console script (see :mod:`qem.cli`) or
``streamlit run -m qem.app``.
"""

__version__ = "0.1.0"

# Select a Keras backend before any qem submodule imports keras. Without this,
# Keras 3 defaults to TensorFlow and a bare ``pip install qem`` install raises
# ``ModuleNotFoundError`` on ``import qem``. We respect any user-set
# ``KERAS_BACKEND`` and otherwise pick the first installed accelerated backend,
# falling back to the always-available NumPy backend so the import never fails.
def _autoselect_keras_backend() -> None:
    import os
    import importlib.util

    if os.environ.get("KERAS_BACKEND"):
        return
    for name in ("torch", "jax", "tensorflow"):
        if importlib.util.find_spec(name) is not None:
            os.environ["KERAS_BACKEND"] = name
            return
    os.environ["KERAS_BACKEND"] = "numpy"


_autoselect_keras_backend()
del _autoselect_keras_backend

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