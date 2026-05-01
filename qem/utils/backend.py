"""
Backend-safe tensor conversion utilities for QEM.
"""

import numpy as np


def safe_convert_to_numpy(tensor):
    """
    Safely convert a Keras tensor to numpy array, handling different backends.
    
    Args:
        tensor: Keras tensor or numpy array
        
    Returns:
        numpy.ndarray: The tensor converted to numpy array
    """
    import keras
    from keras import ops
    
    if isinstance(tensor, np.ndarray):
        return tensor
    
    backend = keras.backend.backend()
    
    if backend == "torch":
        # PyTorch backend requires detach().cpu().numpy()
        if hasattr(tensor, 'detach'):
            return tensor.cpu().detach().numpy()
        else:
            # Fallback to ops.convert_to_numpy for non-gradient tensors
            return ops.convert_to_numpy(tensor)
    elif backend == "jax":
        # JAX backend can use ops.convert_to_numpy directly
        return ops.convert_to_numpy(tensor)
    else:
        # TensorFlow backend
        return ops.convert_to_numpy(tensor)


def safe_convert_to_tensor(array, dtype="float32"):
    """
    Safely convert a numpy array to Keras tensor.
    
    Args:
        array: numpy array or tensor
        dtype: target dtype for the tensor
        
    Returns:
        Keras tensor
    """
    from keras import ops
    return ops.convert_to_tensor(array, dtype=dtype)



def release_backend_memory():
    """
    Best-effort release of cached device memory for the active Keras backend.

    Safe to call from backend-agnostic code: it's a no-op when the active
    backend is not torch, when torch isn't importable, or when no GPU is
    available. Use this instead of calling ``torch.cuda.empty_cache()``
    directly so that CPU/MPS/JAX/TF code paths don't crash.
    """
    try:
        import keras
        if keras.backend.backend() != "torch":
            return
    except Exception:
        return

    try:
        import torch
    except ImportError:
        return

    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def safe_deepcopy_params(params):
    """
    Safely deep copy a parameter dictionary containing tensors.
    
    Args:
        params: Dictionary containing tensors and other values
        
    Returns:
        Dictionary with safely copied parameters
    """
    import keras
    from keras import ops
    import copy
    
    backend = keras.backend.backend()
    copied_params = {}
    
    for key, value in params.items():
        if hasattr(value, 'shape'):  # It's a tensor
            if backend == "torch":
                # For PyTorch, detach and clone to create a leaf tensor
                if hasattr(value, 'detach'):
                    copied_params[key] = value.detach().clone()
                else:
                    copied_params[key] = ops.convert_to_tensor(safe_convert_to_numpy(value))
            else:
                # For JAX and TensorFlow, convert to tensor
                copied_params[key] = ops.convert_to_tensor(safe_convert_to_numpy(value))
        else:
            # For non-tensors, use regular deepcopy
            copied_params[key] = copy.deepcopy(value)
    
    return copied_params

"""Backend detection and configuration utilities."""
import os
import warnings


def detect_available_backends():
    """
    Detect which Keras backends are available in the current environment.
    
    Returns:
        list: List of available backend names in order of preference
    """
    available_backends = []
    
    # Check for JAX
    try:
        import jax
        import jaxlib
        available_backends.append('jax')
    except ImportError:
        pass
    
    # Check for PyTorch
    try:
        import torch
        available_backends.append('torch')
    except ImportError:
        pass
    
    # Check for TensorFlow
    try:
        import tensorflow
        available_backends.append('tensorflow')
    except ImportError:
        pass
    
    return available_backends


def get_best_backend():
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
    
    # Preference order: PyTorch > JAX > TensorFlow (PyTorch is more stable for our use case)
    preference_order = ['torch', 'jax', 'tensorflow']
    
    for backend in preference_order:
        if backend in available:
            return backend
    
    # Fallback to first available
    return available[0]


def configure_backend(backend_name=None, force=False):
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
        except ImportError:
            pass
    
    return backend_name


def setup_test_backend():
    """
    Set up the best available backend for testing.
    
    Returns:
        str: Name of the configured backend
    """
    try:
        backend = configure_backend()
        backend_specific_config(backend)
        print(f"Using Keras backend: {backend}")
        return backend
    except Exception as e:
        print(f"Warning: Failed to configure backend: {e}")
        # Try to use whatever is available
        available = detect_available_backends()
        if available:
            backend = available[0]
            os.environ["KERAS_BACKEND"] = backend
            backend_specific_config(backend)
            print(f"Fallback to: {backend}")
            return backend
        return None


def backend_specific_config(backend_name):
    """
    Apply backend-specific configurations.
    
    Args:
        backend_name (str): Name of the backend to configure
    """
    if backend_name == 'jax':
        try:
            import jax
            import jax.numpy as jnp
            
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
                
        except (ImportError, Exception):
            pass
    
    elif backend_name == 'torch':
        try:
            import torch
            # Set default tensor type to float32
            torch.set_default_dtype(torch.float32)
            # Use CPU if CUDA is not available
            if not torch.cuda.is_available():
                torch.set_default_device('cpu')
        except (ImportError, Exception):
            pass
    
    elif backend_name == 'tensorflow':
        try:
            import tensorflow as tf
            # Suppress TensorFlow warnings
            tf.get_logger().setLevel('ERROR')
            os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
            # Use CPU if GPU is not available
            tf.config.set_visible_devices([], 'GPU')
        except (ImportError, Exception):
            pass


# Auto-configure on import only if explicitly requested
def auto_configure():
    """Auto-configure backend if none is set."""
    try:
        if not os.environ.get("KERAS_BACKEND"):
            available = detect_available_backends()
            if available:
                backend = available[0]  # Use first available
                os.environ["KERAS_BACKEND"] = backend
                backend_specific_config(backend)
                return backend
    except Exception:
        pass
    return None

# Only auto-configure if this module is run directly
if __name__ == "__main__":
    auto_configure()