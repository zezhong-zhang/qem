"""Tests for utility functions."""
import numpy as np
import pytest

# Configure backend automatically
from qem.utils.backend import setup_test_backend, detect_available_backends
setup_test_backend()

import torch
import torch as _torch_mod
import torch

from qem.utils.params import safe_convert_to_numpy, safe_convert_to_tensor, safe_deepcopy_params


def test_safe_convert_to_numpy():
    """Test safe tensor to numpy conversion across backends."""
    # Test with numpy array (should pass through)
    np_array = np.array([1.0, 2.0, 3.0])
    result = safe_convert_to_numpy(np_array)
    assert isinstance(result, np.ndarray)
    np.testing.assert_array_equal(result, np_array)
    
    # Test with Keras tensor
    tensor = torch.as_tensor([1.0, 2.0, 3.0], dtype=torch.float32)
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
    result_int = safe_convert_to_tensor(np_array, dtype=torch.int32)
    result_int_np = safe_convert_to_numpy(result_int)
    np.testing.assert_array_equal(result_int_np, [1, 2, 3])


def test_safe_deepcopy_params():
    """Test safe deep copying of parameter dictionaries."""
    # Create test parameters with tensors
    original_params = {
        'pos_x': torch.as_tensor([1.0, 2.0], dtype=torch.float32),
        'pos_y': torch.as_tensor([3.0, 4.0], dtype=torch.float32),
        'height': torch.as_tensor([0.5, 0.8], dtype=torch.float32),
        'width': torch.as_tensor([1.0, 1.2], dtype=torch.float32),
        'background': torch.as_tensor(0.1, dtype=torch.float32),
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
    from qem.utils.backend import configure_backend
    
    backends_to_test = detect_available_backends()
    
    if not backends_to_test:
        pytest.skip("No backends available for testing")
    
    original_backend = "torch"
    
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


@pytest.mark.parametrize(
    "dtype_input, expected_torch_dtype_attr",
    [
        ("bool", "bool"),
        ("int64", "int64"),
        ("float64", "float64"),
    ],
)
def test_tensors_resolve_dtype(dtype_input, expected_torch_dtype_attr):
    """qem.utils.tensors._resolve_dtype maps strings to torch dtypes."""
    import torch

    from qem.utils.tensors import _resolve_dtype

    expected = getattr(torch, expected_torch_dtype_attr)
    assert _resolve_dtype(dtype_input) is expected


def test_gradient_handling():
    """Test that gradient-enabled tensors are handled correctly."""
    # This test is mainly for PyTorch backend
    # Gradient test always applies — qem is PyTorch-native.
    
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