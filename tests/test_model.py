import os
import numpy as np
import jax.numpy as jnp
import pytest
from qem.model import (
    gaussian_2d_numba, lorentzian_2d_numba, voigt_2d_numba,
    gaussian_2d_jax, butterworth_window, gaussian_kernel,
    gaussian_filter_jax, mask_grads
)

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
    
    # Convert inputs to jax arrays
    X_jax = jnp.array(X, dtype=jnp.float32)
    Y_jax = jnp.array(Y, dtype=jnp.float32)
    pos_x_jax = jnp.array(pos_x, dtype=jnp.float32)
    pos_y_jax = jnp.array(pos_y, dtype=jnp.float32)
    height_jax = jnp.array(height, dtype=jnp.float32)
    
    result = gaussian_2d_jax(X_jax, Y_jax, pos_x_jax, pos_y_jax, height_jax, width)
    
    # Convert result back to numpy for testing
    result = np.array(result)
    
    # Check shape
    assert result.shape == (50, 50, 2)
    
    # Check peak heights at specified positions with relaxed tolerance
    for i in range(len(pos_x)):
        x_idx = np.abs(X[0, :] - pos_x[i]).argmin()
        y_idx = np.abs(Y[:, 0] - pos_y[i]).argmin()
        assert np.abs(result[y_idx, x_idx, i] - height[i]) < 0.05

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
    
    # Check kernel is symmetric
    assert np.allclose(kernel, kernel.T)
    
    # Check kernel sums to approximately 1
    assert np.abs(np.sum(kernel) - 1.0) < 1e-6
    
    # Check peak at center
    center = kernel.shape[0] // 2
    assert kernel[center, center] == np.max(kernel)

def test_gaussian_filter_jax():
    # Create test image
    image = np.zeros((20, 20))
    image[10, 10] = 1.0
    sigma = 1.5
    
    # Convert to jax array
    image_jax = jnp.array(image)
    
    # Apply filter
    filtered = gaussian_filter_jax(image_jax, sigma)
    filtered = np.array(filtered)
    
    # Check shape preserved
    assert filtered.shape == image.shape
    
    # Check smoothing occurred (peak value should be less than original)
    assert filtered[10, 10] < 1.0
    
    # Check total intensity approximately preserved
    assert np.abs(np.sum(filtered) - np.sum(image)) < 1e-6

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
