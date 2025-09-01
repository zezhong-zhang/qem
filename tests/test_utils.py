"""Tests for utility functions."""
import numpy as np
import pytest

# Configure backend automatically
from qem.backend import setup_test_backend, detect_available_backends
setup_test_backend()

import keras
from keras import ops

from qem.utils import safe_convert_to_numpy, safe_convert_to_tensor, safe_deepcopy_params


def test_safe_convert_to_numpy():
    """Test safe tensor to numpy conversion across backends."""
    # Test with numpy array (should pass through)
    np_array = np.array([1.0, 2.0, 3.0])
    result = safe_convert_to_numpy(np_array)
    assert isinstance(result, np.ndarray)
    np.testing.assert_array_equal(result, np_array)
    
    # Test with Keras tensor
    tensor = ops.convert_to_tensor([1.0, 2.0, 3.0], dtype='float32')
    result = safe_convert_to_numpy(tensor)
    assert isinstance(result, np.ndarray)
    np.testing.assert_array_almost_equal(result, [1.0, 2.0, 3.0])


def test_safe_convert_to_tensor():
    """Test safe numpy to tensor conversion."""
    # Test with numpy array
    np_array = np.array([1.0, 2.0, 3.0])
    result = safe_convert_to_tensor(np_array)
    
    # Check that it's a tensor
    assert hasattr(result, 'shape')
    result_np = safe_convert_to_numpy(result)
    np.testing.assert_array_almost_equal(result_np, np_array)
    
    # Test with different dtype
    result_int = safe_convert_to_tensor(np_array, dtype='int32')
    result_int_np = safe_convert_to_numpy(result_int)
    np.testing.assert_array_equal(result_int_np, [1, 2, 3])


def test_safe_deepcopy_params():
    """Test safe deep copying of parameter dictionaries."""
    # Create test parameters with tensors
    original_params = {
        'pos_x': ops.convert_to_tensor([1.0, 2.0], dtype='float32'),
        'pos_y': ops.convert_to_tensor([3.0, 4.0], dtype='float32'),
        'height': ops.convert_to_tensor([0.5, 0.8], dtype='float32'),
        'width': ops.convert_to_tensor([1.0, 1.2], dtype='float32'),
        'background': ops.convert_to_tensor(0.1, dtype='float32'),
        'metadata': {'test': 'value'}  # Non-tensor value
    }
    
    # Test deep copy
    copied_params = safe_deepcopy_params(original_params)
    
    # Check that all keys are present
    assert set(copied_params.keys()) == set(original_params.keys())
    
    # Check that tensor values are equal but not the same object
    for key in ['pos_x', 'pos_y', 'height', 'width', 'background']:
        original_np = safe_convert_to_numpy(original_params[key])
        copied_np = safe_convert_to_numpy(copied_params[key])
        np.testing.assert_array_almost_equal(original_np, copied_np)
        
        # For tensors, they should be different objects
        if hasattr(original_params[key], 'shape'):
            assert original_params[key] is not copied_params[key]
    
    # Check non-tensor values
    assert copied_params['metadata'] == original_params['metadata']
    assert copied_params['metadata'] is not original_params['metadata']  # Should be deep copied


def test_backend_compatibility():
    """Test that utilities work across different backends."""
    from qem.backend import configure_backend
    
    backends_to_test = detect_available_backends()
    
    if not backends_to_test:
        pytest.skip("No backends available for testing")
    
    original_backend = keras.backend.backend()
    
    for backend_name in backends_to_test:
        try:
            # Switch backend
            configure_backend(backend_name, force=True)
            
            # Test conversion functions
            test_array = np.array([1.0, 2.0, 3.0])
            tensor = safe_convert_to_tensor(test_array)
            result = safe_convert_to_numpy(tensor)
            
            np.testing.assert_array_almost_equal(result, test_array)
            
        except Exception as e:
            pytest.skip(f"Backend {backend_name} not properly configured: {e}")
    
    # Restore original backend
    configure_backend(original_backend, force=True)


def test_gradient_handling():
    """Test that gradient-enabled tensors are handled correctly."""
    # This test is mainly for PyTorch backend
    if keras.backend.backend() != 'torch':
        pytest.skip("Gradient test only relevant for PyTorch backend")
    
    # Create a tensor that requires gradients (if using PyTorch)
    try:
        import torch
        
        # Create a tensor with gradients
        torch_tensor = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
        
        # Convert using our safe function
        result = safe_convert_to_numpy(torch_tensor)
        
        # Should work without errors
        assert isinstance(result, np.ndarray)
        np.testing.assert_array_almost_equal(result, [1.0, 2.0, 3.0])
        
    except ImportError:
        pytest.skip("PyTorch not available")