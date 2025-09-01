"""Tests for the new Keras-based model implementation."""
import numpy as np
import pytest

# Configure backend automatically
from qem.backend import setup_test_backend
setup_test_backend()

from keras import ops

# Import the model classes
from qem.model import (
    GaussianModel,
    LorentzianModel,
    VoigtModel,
    GaussianKernel
)
from qem.utils import safe_convert_to_numpy



@pytest.fixture
def grid_2d():
    """Create a 2D coordinate grid for testing."""
    x = ops.arange(50, dtype='float32')
    y = ops.arange(50, dtype='float32')
    x_grid, y_grid = ops.meshgrid(x, y)
    return x_grid, y_grid

@pytest.fixture
def peak_params():
    """Create test parameters for peaks."""
    return {
        "pos_x": ops.convert_to_tensor([25.0, 35.0], dtype='float32'),
        "pos_y": ops.convert_to_tensor([25.0, 15.0], dtype='float32'),
        "height": ops.convert_to_tensor([1.0, 0.5], dtype='float32'),
        "width": ops.convert_to_tensor([2.0, 2.0], dtype='float32'),
        "ratio": ops.convert_to_tensor([0.5, 0.5], dtype='float32'),
        "background": ops.convert_to_tensor(0.1, dtype='float32')
    }

def test_gaussian_model(grid_2d, peak_params):
    """Test the new Keras-based Gaussian model."""
    x_grid, y_grid = grid_2d
    model = GaussianModel(dx=1.0)
    
    # Set up the model
    model.set_params(peak_params)
    # Build the model
    model.build()
    
    # Test sum method
    result = model.sum(x_grid, y_grid, local=False)
    result_np = safe_convert_to_numpy(result)
    
    # Check shape
    assert result_np.shape == (50, 50)
    
    # Check that peaks are above background
    background_val = safe_convert_to_numpy(peak_params["background"])
    assert np.all(result_np >= background_val)
    
    # Test volume calculation
    volumes = model.volume(peak_params)
    volumes_np = safe_convert_to_numpy(volumes)
    assert len(volumes_np) == 2
    assert np.all(volumes_np > 0)
    
    # Test local vs global calculation
    result_local = model.sum(x_grid, y_grid, local=False)
    result_local_np = safe_convert_to_numpy(result_local)
    
    # Results should be close
    np.testing.assert_allclose(result_np, result_local_np, rtol=1e-3, atol=1e-3)

def test_lorentzian_model(grid_2d, peak_params):
    """Test the new Keras-based Lorentzian model."""
    x_grid, y_grid = grid_2d
    model = LorentzianModel(dx=1.0)
    
    # Set up the model
    model.set_params(peak_params)
    model.build()
    
    # Test sum method
    result =model.sum(x_grid, y_grid, local=False)
    result_np = safe_convert_to_numpy(result)
    
    # Check shape
    assert result_np.shape == (50, 50)
    
    # Check that peaks are above background
    background_val = safe_convert_to_numpy(peak_params["background"])
    assert np.all(result_np >= background_val)
    
    # Test volume calculation
    volumes = model.volume(peak_params)
    volumes_np = safe_convert_to_numpy(volumes)
    assert len(volumes_np) == 2
    assert np.all(volumes_np > 0)

def test_voigt_model(grid_2d, peak_params):
    """Test the new Keras-based Voigt model."""
    x_grid, y_grid = grid_2d
    model = VoigtModel(dx=1.0)
    
    # Set up the model
    model.set_params(peak_params)
    model.build()
    
    # Test sum method
    result = model.sum(x_grid, y_grid, local=False)
    result_np = safe_convert_to_numpy(result)
    
    # Check shape
    assert result_np.shape == (50, 50)
    
    # Check that peaks are above background
    background_val = safe_convert_to_numpy(peak_params["background"])
    assert np.all(result_np >= background_val)
    
    # Test volume calculation
    volumes = model.volume(peak_params)
    volumes_np = safe_convert_to_numpy(volumes)
    assert len(volumes_np) == 2
    assert np.all(volumes_np > 0)

def test_gaussian_kernel():
    """Test the Gaussian kernel functionality."""
    kernel = GaussianKernel()
    sigma = 1.5
    
    # Test kernel creation
    kernel_array = kernel.gaussian_kernel(sigma)
    kernel_np = safe_convert_to_numpy(kernel_array)
    
    # Check kernel is symmetric
    np.testing.assert_allclose(kernel_np, kernel_np.T, rtol=1e-5, atol=1e-8)
    
    # Check kernel sums to approximately 1
    assert np.abs(np.sum(kernel_np) - 1.0) < 1e-5
    
    # Test Gaussian filter
    image = ops.zeros((20, 20), dtype='float32')
    # Create a simple point source at center
    image_np = safe_convert_to_numpy(image)
    image_np[10, 10] = 1.0
    image = ops.convert_to_tensor(image_np, dtype='float32')
    
    filtered = kernel.gaussian_filter(image, sigma)
    filtered_np = safe_convert_to_numpy(filtered)
    
    # Check shape preserved
    assert filtered_np.shape == (20, 20)
    
    # Check smoothing occurred (peak value should be less than original)
    assert filtered_np[10, 10] < 1.0
    
    # Check total intensity approximately preserved
    assert np.abs(np.sum(filtered_np) - 1.0) < 1e-5
