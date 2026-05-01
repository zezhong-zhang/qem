"""Basic tests for core functionality."""
import numpy as np
import pytest

# Configure backend automatically
from qem.utils.backend import setup_test_backend
backend = setup_test_backend()

from keras import ops
from qem.fit.model import GaussianModel
from qem.utils.params import safe_convert_to_numpy, safe_convert_to_tensor
from qem.fit.image_fitting import ImageFitting


def test_backend_detection():
    """Test that backend detection works."""
    from qem.utils.backend import detect_available_backends
    available = detect_available_backends()
    assert len(available) > 0, "No backends available"
    print(f"Available backends: {available}")


def test_safe_conversions():
    """Test safe tensor conversions."""
    # Test numpy to tensor
    np_array = np.array([1.0, 2.0, 3.0])
    tensor = safe_convert_to_tensor(np_array)
    
    # Test tensor to numpy
    result = safe_convert_to_numpy(tensor)
    
    np.testing.assert_array_almost_equal(result, np_array)


def test_gaussian_model_basic():
    """Test basic Gaussian model functionality."""
    # Create a simple grid
    size = 20
    x_grid = ops.arange(size, dtype='float32')
    y_grid = ops.arange(size, dtype='float32')
    x_grid, y_grid = ops.meshgrid(x_grid, y_grid)
    
    # Create test parameters for a single peak
    params = {
        "pos_x": ops.convert_to_tensor([10.0], dtype='float32'),
        "pos_y": ops.convert_to_tensor([10.0], dtype='float32'),
        "height": ops.convert_to_tensor([1.0], dtype='float32'),
        "width": ops.convert_to_tensor([2.0], dtype='float32'),
        "background": ops.convert_to_tensor(0.1, dtype='float32')
    }
    
    # Test model
    model = GaussianModel(dx=1.0)
    model.set_params(params)
    model.build()
    
    # Test sum method
    result = model.sum(x_grid, y_grid, local=False)
    result_np = safe_convert_to_numpy(result)
    
    # Basic checks
    assert result_np.shape == (size, size)
    assert result_np.max() > 0.1  # Should be above background
    assert result_np.min() >= 0.1  # Should be at least background level
    
    # Test volume calculation
    volumes = model.volume(params)
    volumes_np = safe_convert_to_numpy(volumes)
    assert len(volumes_np) == 1
    assert volumes_np[0] > 0


def test_image_fitting_basic():
    """Test basic ImageFitting functionality."""
    # Create a simple synthetic image
    size = 20
    image = np.zeros((size, size))
    
    # Add a simple Gaussian-like peak manually
    center = size // 2
    for i in range(size):
        for j in range(size):
            r2 = (i - center)**2 + (j - center)**2
            image[i, j] = np.exp(-r2 / (2 * 2**2)) + 0.1
    
    # Initialize ImageFitting
    fitter = ImageFitting(
        image=image,
        dx=1.0,
        model_type="gaussian"
    )
    
    # Set a single coordinate
    fitter.coordinates = np.array([[center, center]], dtype=float)
    
    # Initialize parameters
    params = fitter.init_params(atom_size=2.0)
    
    # Basic checks
    assert params is not None
    assert 'pos_x' in params
    assert 'pos_y' in params
    assert 'height' in params
    assert 'width' in params
    
    # Check that coordinates were set correctly
    assert fitter.num_coordinates == 1
    
    # Test prediction
    prediction = fitter.predict(local=False)
    prediction_np = safe_convert_to_numpy(prediction)
    assert prediction_np.shape == image.shape


@pytest.mark.skipif(backend != 'torch', reason="PyTorch-specific test")
def test_pytorch_specific():
    """Test PyTorch-specific functionality."""
    import torch
    
    # Test that we can create a tensor that requires gradients
    tensor = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
    
    # Test safe conversion
    result = safe_convert_to_numpy(tensor)
    np.testing.assert_array_almost_equal(result, [1.0, 2.0, 3.0])


@pytest.mark.skipif(backend != 'jax', reason="JAX-specific test")
def test_jax_specific():
    """Test JAX-specific functionality."""
    import jax.numpy as jnp
    
    # Test JAX array creation
    array = jnp.array([1.0, 2.0, 3.0])
    
    # Test safe conversion
    result = safe_convert_to_numpy(array)
    np.testing.assert_array_almost_equal(result, [1.0, 2.0, 3.0])


if __name__ == "__main__":
    # Run basic tests
    test_backend_detection()
    test_safe_conversions()
    test_gaussian_model_basic()
    test_image_fitting_basic()
    
    if backend == 'torch':
        test_pytorch_specific()
    elif backend == 'jax':
        test_jax_specific()
    
    print("All basic tests passed!")