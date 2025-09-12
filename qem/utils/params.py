"""
Backend-safe utilities and I/O functions for QEM.
"""

import numpy as np


def export_params(params, filename):
    """
    Export the parameters to a file.

    Parameters:
    - params: Dictionary of parameters to export.
    - filename: Name of the file to export to.
    """
    with open(filename, "w") as f:
        for key, value in params.items():
            f.create_dataset(key, data=value)
    f.close()

# Backend-safe tensor conversion utilities
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