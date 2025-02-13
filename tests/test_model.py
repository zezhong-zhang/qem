import os
import numpy as np
import jax.numpy as jnp
import pytest
from qem.model import (
    gaussian_2d_numba, lorentzian_2d_numba, voigt_2d_numba,
    gaussian_2d_jax, butterworth_window, gaussian_kernel,
    gaussian_filter_jax, mask_grads
)

# Configure JAX
import jax
if os.environ.get('JAX_PLATFORMS') == 'cuda':
    jax.config.update('jax_platform_name', 'gpu')
    jax.config.update('jax_enable_x64', True)

@pytest.fixture
def grid_2d():
    x = np.linspace(-5, 5, 50)
    y = np.linspace(-5, 5, 50)
    X, Y = np.meshgrid(x, y)
    return X, Y

@pytest.fixture
def peak_params():
    pos_x = np.array([0.0, 2.0])
    pos_y = np.array([0.0, -2.0])
    height = np.array([1.0, 0.5])
    width = 1.0
    gamma = 1.0
    ratio = 0.5
    return pos_x, pos_y, height, width, gamma, ratio

def test_gaussian_2d_numba(grid_2d, peak_params):
    X, Y = grid_2d
    pos_x, pos_y, height, width, _, _ = peak_params
    
    result = gaussian_2d_numba(X, Y, pos_x, pos_y, height, width)
    
    # Check shape
    assert result.shape == (50, 50, 2)
    
    # Check peak heights at specified positions with relaxed tolerance
    for i in range(len(pos_x)):
        x_idx = np.abs(X[0, :] - pos_x[i]).argmin()
        y_idx = np.abs(Y[:, 0] - pos_y[i]).argmin()
        assert np.abs(result[y_idx, x_idx, i] - height[i]) < 0.05

def test_lorentzian_2d_numba(grid_2d, peak_params):
    X, Y = grid_2d
    pos_x, pos_y, height, _, gamma, _ = peak_params
    
    result = lorentzian_2d_numba(X, Y, pos_x, pos_y, height, gamma)
    
    # Check shape
    assert result.shape == (50, 50, 2)
    
    # Check peak heights at specified positions with relaxed tolerance
    for i in range(len(pos_x)):
        x_idx = np.abs(X[0, :] - pos_x[i]).argmin()
        y_idx = np.abs(Y[:, 0] - pos_y[i]).argmin()
        assert np.abs(result[y_idx, x_idx, i] - height[i]) < 0.05

def test_voigt_2d_numba(grid_2d, peak_params):
    X, Y = grid_2d
    pos_x, pos_y, height, width, gamma, ratio = peak_params
    
    result = voigt_2d_numba(X, Y, pos_x, pos_y, height, width, gamma, ratio)
    
    # Check shape
    assert result.shape == (50, 50, 2)
    
    # Check peak heights at specified positions with relaxed tolerance
    for i in range(len(pos_x)):
        x_idx = np.abs(X[0, :] - pos_x[i]).argmin()
        y_idx = np.abs(Y[:, 0] - pos_y[i]).argmin()
        assert np.abs(result[y_idx, x_idx, i] - height[i]) < 0.05

def test_gaussian_2d_jax(grid_2d, peak_params):
    X, Y = grid_2d
    pos_x, pos_y, height, width, _, _ = peak_params
    
    # Convert inputs to jax arrays with explicit dtype
    dtype = jnp.float64 if jax.config.x64_enabled else jnp.float32
    X_jax = jnp.asarray(X, dtype=dtype)
    Y_jax = jnp.asarray(Y, dtype=dtype)
    pos_x_jax = jnp.asarray(pos_x, dtype=dtype)
    pos_y_jax = jnp.asarray(pos_y, dtype=dtype)
    height_jax = jnp.asarray(height, dtype=dtype)
    width_jax = jnp.asarray(width, dtype=dtype)
    
    result = gaussian_2d_jax(X_jax, Y_jax, pos_x_jax, pos_y_jax, height_jax, width_jax)
    
    # Convert result back to numpy for testing
    result = np.array(result)
    
    # Check shape
    assert result.shape == (50, 50, 2)
    
    # Check peak heights at specified positions with appropriate tolerance
    rtol = 0.02  # 2% relative tolerance
    for i in range(len(pos_x)):
        x_idx = np.abs(X[0, :] - pos_x[i]).argmin()
        y_idx = np.abs(Y[:, 0] - pos_y[i]).argmin()
        # Get a small neighborhood around the peak to account for discretization
        peak_region = result[max(0, y_idx-1):min(50, y_idx+2), 
                           max(0, x_idx-1):min(50, x_idx+2), i]
        max_value = np.max(peak_region)
        # For a non-normalized Gaussian, the peak height should be exactly height[i]
        assert np.isclose(max_value, height[i], rtol=rtol)
        
        # Check relative decay at specific distances
        distance = np.sqrt((X - pos_x[i])**2 + (Y - pos_y[i])**2)
        # At one standard deviation (width), value should be height * exp(-0.5)
        one_sigma_mask = np.abs(distance - width) < 0.2
        if np.any(one_sigma_mask):
            one_sigma_values = result[..., i][one_sigma_mask]
            expected_value = height[i] * np.exp(-0.5)
            assert np.any(np.abs(one_sigma_values - expected_value) < rtol * height[i])
        
        # At three standard deviations, value should be very small
        three_sigma_mask = (distance > 3 * width)
        assert np.all(result[..., i][three_sigma_mask] < 0.05 * height[i])

def test_butterworth_window():
    shape = (32, 32)
    cutoff_radius_ftr = 0.25
    order = 2
    
    window = butterworth_window(shape, cutoff_radius_ftr, order)
    
    # Check shape
    assert window.shape == shape
    
    # Check values are between 0 and 1
    assert np.all(window >= 0)
    assert np.all(window <= 1)
    
    # Check center value is 1
    center_y, center_x = shape[0] // 2, shape[1] // 2
    assert np.abs(window[center_y, center_x] - 1.0) < 1e-6

def test_gaussian_kernel():
    sigma = 1.5
    kernel = gaussian_kernel(sigma)
    
    # Check kernel is symmetric with appropriate tolerance
    rtol = 1e-5  # relative tolerance
    atol = 1e-8  # absolute tolerance
    assert np.allclose(kernel, kernel.T, rtol=rtol, atol=atol)
    
    # Check kernel sums to approximately 1
    assert np.abs(np.sum(kernel) - 1.0) < rtol
    
    # Check peak at center
    center = kernel.shape[0] // 2
    assert np.abs(kernel[center, center] - np.max(kernel)) < atol

def test_gaussian_filter_jax():
    # Create test image
    image = np.zeros((20, 20))
    image[10, 10] = 1.0
    sigma = 1.5
    
    # Convert to jax array with explicit dtype
    dtype = jnp.float64 if jax.config.x64_enabled else jnp.float32
    image_jax = jnp.asarray(image, dtype=dtype)
    
    # Apply filter
    filtered = gaussian_filter_jax(image_jax, sigma)
    filtered = np.array(filtered)
    
    # Check shape preserved
    assert filtered.shape == image.shape
    
    # Check smoothing occurred (peak value should be less than original)
    assert filtered[10, 10] < 1.0
    
    # Check total intensity approximately preserved
    rtol = 1e-5  # relative tolerance
    assert np.isclose(np.sum(filtered), np.sum(image), rtol=rtol)

def test_mask_grads():
    # Create dummy gradients dictionary
    grads = {
        'param1': np.array([1.0, 2.0, 3.0]),
        'param2': np.array([4.0, 5.0, 6.0]),
        'param3': np.array([7.0, 8.0, 9.0])
    }
    
    keys_to_mask = ['param1', 'param3']
    
    masked_grads = mask_grads(grads, keys_to_mask)
    
    # Check masked parameters are zero
    assert np.all(masked_grads['param1'] == 0)
    assert np.all(masked_grads['param3'] == 0)
    
    # Check non-masked parameter unchanged
    assert np.all(masked_grads['param2'] == grads['param2'])
