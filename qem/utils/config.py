"""
Configuration management for QEM.
Handles precision settings and other configurable parameters.
"""

import os
import numpy as np
from typing import Union, Type


def _env_bool(name: str, default: bool) -> bool:
    """Parse a boolean env var. Disabled by '0', 'false', 'False'; enabled otherwise."""
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw not in ("0", "false", "False", "")


class PrecisionConfig:
    """Manages precision settings for QEM calculations."""

    # Default precision settings
    DEFAULT_PRECISION = "float32"
    DEFAULT_LINEAR_SOLVER_PRECISION = "float32"
    DEFAULT_ENABLE_TF32 = True
    DEFAULT_ENABLE_COMPILE = False

    # Supported precision types
    SUPPORTED_PRECISIONS = {
        "float32": np.float32,
        "float64": np.float64,
    }

    def __init__(self):
        """Initialize precision configuration from environment variables."""
        self._load_from_env()

    def _load_from_env(self):
        """Load configuration from environment variables."""
        # Load precision settings
        self.precision = os.getenv("QEM_PRECISION", self.DEFAULT_PRECISION)
        self.linear_solver_precision = os.getenv(
            "QEM_LINEAR_SOLVER_PRECISION",
            self.DEFAULT_LINEAR_SOLVER_PRECISION
        )

        # Performance flags. Default: TF32 on (free perf on Ampere+),
        # torch.compile off (opt-in because it interacts unpredictably with
        # backend-dispatching keras.ops chains).
        self.enable_tf32 = _env_bool("QEM_TF32", self.DEFAULT_ENABLE_TF32)
        self.enable_compile = _env_bool("QEM_COMPILE", self.DEFAULT_ENABLE_COMPILE)

        # Validate precision settings
        self._validate_precision()
    
    def _validate_precision(self):
        """Validate that precision settings are supported."""
        if self.precision not in self.SUPPORTED_PRECISIONS:
            raise ValueError(
                f"Unsupported precision '{self.precision}'. "
                f"Supported: {list(self.SUPPORTED_PRECISIONS.keys())}"
            )
        
        if self.linear_solver_precision not in self.SUPPORTED_PRECISIONS:
            raise ValueError(
                f"Unsupported linear solver precision '{self.linear_solver_precision}'. "
                f"Supported: {list(self.SUPPORTED_PRECISIONS.keys())}"
            )
    
    @property
    def numpy_dtype(self) -> Type[np.floating]:
        """Get numpy dtype for general calculations."""
        return self.SUPPORTED_PRECISIONS[self.precision]
    
    @property
    def linear_solver_numpy_dtype(self) -> Type[np.floating]:
        """Get numpy dtype for linear solver calculations."""
        return self.SUPPORTED_PRECISIONS[self.linear_solver_precision]
    
    @property
    def keras_dtype(self) -> str:
        """Get Keras dtype string for general calculations."""
        return self.precision
    
    @property
    def linear_solver_keras_dtype(self) -> str:
        """Get Keras dtype string for linear solver calculations."""
        return self.linear_solver_precision
    
    def get_numpy_array(self, data, dtype: str = None) -> np.ndarray:
        """Create numpy array with configured precision."""
        if dtype is None:
            dtype = self.precision
        
        target_dtype = self.SUPPORTED_PRECISIONS[dtype]
        return np.asarray(data, dtype=target_dtype)
    
    def get_linear_solver_array(self, data) -> np.ndarray:
        """Create numpy array with linear solver precision."""
        return self.get_numpy_array(data, self.linear_solver_precision)
    
    def convert_to_precision(self, array: np.ndarray, dtype: str = None) -> np.ndarray:
        """Convert array to specified precision."""
        if dtype is None:
            dtype = self.precision
        
        target_dtype = self.SUPPORTED_PRECISIONS[dtype]
        return array.astype(target_dtype)
    
    def is_float64_supported(self) -> bool:
        """Check if float64 is supported by current backend."""
        try:
            import keras
            backend = keras.backend.backend()
            
            if backend == "torch":
                import torch
                # Check if we're on MPS (Apple Silicon)
                if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                    # MPS doesn't support float64
                    return False
            
            return True
        except ImportError:
            return True
    
    def get_safe_precision(self, requested_precision: str = None) -> str:
        """Get safe precision for current backend."""
        if requested_precision is None:
            requested_precision = self.precision
        
        # If float64 is requested but not supported, fallback to float32
        if requested_precision == "float64" and not self.is_float64_supported():
            return "float32"
        
        return requested_precision
    
    def __repr__(self):
        return (
            f"PrecisionConfig(precision='{self.precision}', "
            f"linear_solver_precision='{self.linear_solver_precision}')"
        )


# Global configuration instance
_config = None

def get_config() -> PrecisionConfig:
    """Get global configuration instance."""
    global _config
    if _config is None:
        _config = PrecisionConfig()
    return _config

def reload_config():
    """Reload configuration from environment variables."""
    global _config
    _config = PrecisionConfig()
    return _config

# Convenience functions
def get_precision() -> str:
    """Get current precision setting."""
    return get_config().precision

def get_linear_solver_precision() -> str:
    """Get current linear solver precision setting."""
    return get_config().linear_solver_precision

def get_numpy_dtype() -> Type[np.floating]:
    """Get numpy dtype for general calculations."""
    return get_config().numpy_dtype

def get_linear_solver_numpy_dtype() -> Type[np.floating]:
    """Get numpy dtype for linear solver calculations."""
    return get_config().linear_solver_numpy_dtype

def create_array(data, precision: str = None) -> np.ndarray:
    """Create numpy array with configured precision."""
    return get_config().get_numpy_array(data, precision)

def create_linear_solver_array(data) -> np.ndarray:
    """Create numpy array with linear solver precision."""
    return get_config().get_linear_solver_array(data)


def maybe_compile(fn, *, mode: str = "default"):
    """Wrap ``fn`` in :func:`torch.compile` when conditions are met.

    Returns ``torch.compile(fn, mode=mode)`` when all of:

    - ``PrecisionConfig.enable_compile`` is true (env: ``QEM_COMPILE=1``)
    - The active Keras backend is ``torch``
    - CUDA is available (we don't compile on MPS — fragile on Apple Silicon —
      and CPU compile rarely pays off for QEM's small kernels)

    Otherwise returns ``fn`` unchanged. Compilation errors are swallowed and
    the original function is returned, so this is always safe to wrap around
    a callable.
    """
    cfg = get_config()
    if not cfg.enable_compile:
        return fn
    try:
        import keras
        if keras.backend.backend() != "torch":
            return fn
        import torch
        if not torch.cuda.is_available():
            return fn
        return torch.compile(fn, mode=mode)
    except Exception:
        return fn