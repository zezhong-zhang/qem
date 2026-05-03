"""Backend detection and configuration utilities."""

from __future__ import annotations

import logging
import os
from collections.abc import Iterable

logger = logging.getLogger(__name__)

BACKEND_PREFERENCE = ("torch", "jax", "tensorflow")


def _backend_imports(backend_name: str) -> Iterable[str]:
    """Return import names needed to verify a Keras backend."""
    if backend_name == "jax":
        return ("jax", "jaxlib")
    if backend_name == "torch":
        return ("torch",)
    if backend_name == "tensorflow":
        return ("tensorflow",)
    raise ValueError(f"Unknown backend: {backend_name}")


def _can_import_backend(backend_name: str) -> bool:
    """Return whether a backend imports cleanly in this environment."""
    for module_name in _backend_imports(backend_name):
        try:
            __import__(module_name)
        except Exception as exc:
            logger.debug("Keras backend %s unavailable via %s: %s", backend_name, module_name, exc)
            return False
    return True


def detect_available_backends() -> list[str]:
    """
    Detect which Keras backends are available in the current environment.
    
    Returns:
        list: List of available backend names in order of preference
    """
    return [backend for backend in BACKEND_PREFERENCE if _can_import_backend(backend)]


def get_best_backend() -> str:
    """
    Get the best available backend for the current environment.
    
    Returns:
        str: Name of the best available backend
        
    Raises:
        RuntimeError: If no backends are available
    """
    available = detect_available_backends()
    
    if not available:
        raise RuntimeError(
            "No Keras backends available. Please install at least one of: "
            "jax, torch, or tensorflow"
        )
    
    for backend in BACKEND_PREFERENCE:
        if backend in available:
            return backend
    
    # Fallback to first available
    return available[0]


def configure_backend(backend_name: str | None = None, force: bool = False) -> str:
    """
    Configure Keras to use the specified backend.
    
    Args:
        backend_name (str, optional): Backend to use. If None, auto-detect best.
        force (bool): Whether to force reconfiguration even if already set.
        
    Returns:
        str: Name of the configured backend
    """
    if backend_name is None:
        backend_name = get_best_backend()
    
    # Check if backend is available
    available = detect_available_backends()
    if backend_name not in available:
        raise ValueError(
            f"Backend '{backend_name}' is not available. "
            f"Available backends: {available}"
        )
    
    # Set environment variable
    current_backend = os.environ.get("KERAS_BACKEND")
    if current_backend != backend_name or force:
        os.environ["KERAS_BACKEND"] = backend_name
        
        # Clear any existing Keras session
        try:
            import keras

            keras.backend.clear_session()
        except Exception as exc:
            logger.debug("Unable to clear Keras session while configuring backend: %s", exc)

    return backend_name


def setup_test_backend() -> str | None:
    """
    Set up the best available backend for testing.
    
    Returns:
        str: Name of the configured backend
    """
    try:
        backend = configure_backend()
        backend_specific_config(backend)
        logger.info("Using Keras backend: %s", backend)
        return backend
    except Exception as e:
        logger.warning("Failed to configure Keras backend: %s", e)
        # Try to use whatever is available
        available = detect_available_backends()
        if available:
            backend = available[0]
            os.environ["KERAS_BACKEND"] = backend
            backend_specific_config(backend)
            logger.info("Fallback Keras backend: %s", backend)
            return backend
        return None


def backend_specific_config(backend_name: str) -> None:
    """
    Apply backend-specific configurations.
    
    Args:
        backend_name (str): Name of the backend to configure
    """
    if backend_name == 'jax':
        try:
            import jax
            
            # Read environment variables for JAX configuration
            jax_platforms = os.environ.get('JAX_PLATFORMS', 'cpu')
            jax_enable_x64 = os.environ.get('JAX_ENABLE_X64', 'true').lower() == 'true'
            jax_disable_jit = os.environ.get('JAX_DISABLE_JIT', 'true').lower() == 'true'
            
            # Apply JAX configurations
            jax.config.update('jax_platforms', jax_platforms)
            jax.config.update("jax_enable_x64", jax_enable_x64)
            jax.config.update('jax_disable_jit', jax_disable_jit)
            
            # Set memory preallocation to avoid memory issues
            if 'XLA_PYTHON_CLIENT_PREALLOCATE' not in os.environ:
                os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
                
        except Exception as exc:
            logger.debug("Unable to apply JAX backend configuration: %s", exc)
    
    elif backend_name == 'torch':
        try:
            import torch
            # Set default tensor type to float32
            torch.set_default_dtype(torch.float32)
            # Use CPU if CUDA is not available
            if not torch.cuda.is_available():
                torch.set_default_device('cpu')
        except Exception as exc:
            logger.debug("Unable to apply PyTorch backend configuration: %s", exc)
    
    elif backend_name == 'tensorflow':
        try:
            import tensorflow as tf
            # Suppress TensorFlow warnings
            tf.get_logger().setLevel('ERROR')
            os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
            # Use CPU if GPU is not available
            tf.config.set_visible_devices([], 'GPU')
        except Exception as exc:
            logger.debug("Unable to apply TensorFlow backend configuration: %s", exc)


# Auto-configure on import only if explicitly requested
def auto_configure() -> str | None:
    """Auto-configure backend if none is set."""
    try:
        if not os.environ.get("KERAS_BACKEND"):
            available = detect_available_backends()
            if available:
                backend = available[0]  # Use first available
                os.environ["KERAS_BACKEND"] = backend
                backend_specific_config(backend)
                return backend
    except Exception as exc:
        logger.debug("Unable to auto-configure Keras backend: %s", exc)
    return None

# Only auto-configure if this module is run directly
if __name__ == "__main__":
    auto_configure()
