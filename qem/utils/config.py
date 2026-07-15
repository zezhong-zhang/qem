"""
Configuration management for QEM.
Handles precision settings and other configurable parameters.
"""

import os

import numpy as np


class PrecisionConfig:
    """Manages precision settings for QEM calculations."""

    # Default precision settings
    DEFAULT_PRECISION = "float32"
    DEFAULT_LINEAR_SOLVER_PRECISION = "float32"

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

        # Performance flags (gated by env vars, default OFF to preserve back-compat)
        self.enable_tf32 = os.getenv("QEM_TF32", "0") != "0"
        self.enable_compile = os.getenv("QEM_COMPILE", "0") != "0"

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
    def numpy_dtype(self) -> type[np.floating]:
        """Get numpy dtype for general calculations."""
        return self.SUPPORTED_PRECISIONS[self.precision]

    @property
    def linear_solver_numpy_dtype(self) -> type[np.floating]:
        """Get numpy dtype for linear solver calculations."""
        return self.SUPPORTED_PRECISIONS[self.linear_solver_precision]

    @property
    def torch_dtype(self) -> str:
        """Get PyTorch dtype string for general calculations."""
        return self.precision

    @property
    def linear_solver_torch_dtype(self) -> str:
        """Get PyTorch dtype string for linear solver calculations."""
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
            f"linear_solver_precision='{self.linear_solver_precision}', "
            f"enable_tf32={self.enable_tf32}, enable_compile={self.enable_compile})"
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

def get_numpy_dtype() -> type[np.floating]:
    """Get numpy dtype for general calculations."""
    return get_config().numpy_dtype

def get_linear_solver_numpy_dtype() -> type[np.floating]:
    """Get numpy dtype for linear solver calculations."""
    return get_config().linear_solver_numpy_dtype

def create_array(data, precision: str = None) -> np.ndarray:
    """Create numpy array with configured precision."""
    return get_config().get_numpy_array(data, precision)

def create_linear_solver_array(data) -> np.ndarray:
    """Create numpy array with linear solver precision."""
    return get_config().get_linear_solver_array(data)
