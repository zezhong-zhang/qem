import os

# Configure JAX to use CPU if CUDA is not available
if not os.environ.get('JAX_PLATFORMS'):
    os.environ["JAX_PLATFORMS"] = "cpu"
    os.environ["JAX_ENABLE_X64"] = "True"

import jax
import numpy as np
import pytest

# Import the model classes
from qem.model import (
    set_backend,
    GaussianModel,
    LorentzianModel,
    VoigtModel,
    GaussianKernel
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
    ratio = 0.5
    return {"pos_x": pos_x, "pos_y": pos_y, "height": height, "width": width, "ratio": ratio}

def test_set_backend():
    # Test setting valid backend
    backend = set_backend('jax')
    assert backend in ['tensorflow', 'pytorch', 'jax']
    
    # Test invalid backend
    with pytest.raises(ValueError):
        set_backend('invalid')

def test_gaussian_model(grid_2d, peak_params):
    X, Y = grid_2d
    model = GaussianModel(dx=1.0, background=0.1)
    
    # Test sum method
    result = model.sum(
        X, Y,
        peak_params["pos_x"],
        peak_params["pos_y"],
        peak_params["height"],
        peak_params["width"]
    )
    
    # Check shape and background
    assert result.shape == (50, 50)
    assert np.all(result >= model.background)
    
    # Check peak heights
    for i in range(len(peak_params["pos_x"])):
        x_idx = np.abs(X[0, :] - peak_params["pos_x"][i]).argmin()
        y_idx = np.abs(Y[:, 0] - peak_params["pos_y"][i]).argmin()
        expected_height = peak_params["height"][i] + model.background
        assert np.abs(result[y_idx, x_idx] - expected_height) < 0.05
    
    # Test volume calculation
    volumes = model.volume(peak_params)
    assert len(volumes) == len(peak_params["pos_x"])
    for vol in volumes:
        assert vol > 0

def test_gaussian_model_local(grid_2d, peak_params):
    X, Y = grid_2d
    model = GaussianModel(dx=1.0, background=0.1)
        
    print("\nGrid info:")
    print("X range:", X[0, 0], "to", X[0, -1])
    print("Y range:", Y[0, 0], "to", Y[-1, 0])
    print("Physical pos_x:", peak_params["pos_x"])
    print("Physical pos_y:", peak_params["pos_y"])

    # Test global sum method
    result_global = model.sum(
        X, Y,
        peak_params["pos_x"],
        peak_params["pos_y"],
        peak_params["height"],
        peak_params["width"]
    )

    # Test local sum method
    result_local = model.sum(
        X, Y,
        peak_params["pos_x"],
        peak_params["pos_y"],
        peak_params["height"],
        peak_params["width"],
        local=True
    )

    # Print some debug info
    print("\nGlobal shape:", result_global.shape)
    print("Local shape:", result_local.shape)
    print("Global max:", result_global.max())
    print("Local max:", result_local.max())
    print("Global min:", result_global.min())
    print("Local min:", result_local.min())
    print("Global mean:", result_global.mean())
    print("Local mean:", result_local.mean())
        
    # Results should be very close
    assert np.allclose(result_global, result_local, rtol=1e-4, atol=1e-4)
    
    # Check shape and background
    assert result_local.shape == (50, 50)
    assert np.all(result_local >= model.background)
    
    # Check peak heights
    for i in range(len(peak_params["pos_x"])):
        x_idx = np.abs(X[0, :] - peak_params["pos_x"][i]).argmin()
        y_idx = np.abs(Y[:, 0] - peak_params["pos_y"][i]).argmin()
        expected_height = peak_params["height"][i] + model.background
        assert np.abs(result_local[y_idx, x_idx] - expected_height) < 0.05
    
    # Test with peaks far apart to ensure local calculation works
    far_peaks = {
        "pos_x": np.array([-20.0, 20.0]),
        "pos_y": np.array([-20.0, 20.0]),
        "height": np.array([1.0, 1.0]),
        "width": peak_params["width"],
        "ratio": peak_params["ratio"]
    }
    
    # Calculate with both methods
    result_global_far = model.sum(
        X, Y,
        far_peaks["pos_x"],
        far_peaks["pos_y"],
        far_peaks["height"],
        far_peaks["width"]
    )
    
    result_local_far = model.sum(
        X, Y,
        far_peaks["pos_x"],
        far_peaks["pos_y"],
        far_peaks["height"],
        far_peaks["width"],
        local=True
    )
    
    # For far peaks, both results should be close to background in the center
    center_idx = len(X) // 2
    assert np.abs(result_global_far[center_idx, center_idx] - model.background) < 1e-5
    assert np.abs(result_local_far[center_idx, center_idx] - model.background) < 1e-5

def test_lorentzian_model(grid_2d, peak_params):
    X, Y = grid_2d
    model = LorentzianModel(dx=1.0, background=0.1)
    
    # Test sum method
    result = model.sum(
        X, Y,
        peak_params["pos_x"],
        peak_params["pos_y"],
        peak_params["height"],
        peak_params["width"]
    )
    
    # Check shape and background
    assert result.shape == (50, 50)
    assert np.all(result >= model.background)
    
    # Check peak heights
    for i in range(len(peak_params["pos_x"])):
        x_idx = np.abs(X[0, :] - peak_params["pos_x"][i]).argmin()
        y_idx = np.abs(Y[:, 0] - peak_params["pos_y"][i]).argmin()
        expected_height = peak_params["height"][i] + model.background
        assert np.abs(result[y_idx, x_idx] - expected_height) < 0.15
    
    # Test volume calculation
    volumes = model.volume(peak_params)
    assert len(volumes) == len(peak_params["pos_x"])
    for vol in volumes:
        assert vol > 0

def test_voigt_model(grid_2d, peak_params):
    X, Y = grid_2d
    model = VoigtModel(dx=1.0, background=0.1)
    
    # Test sum method
    result = model.sum(
        X, Y,
        peak_params["pos_x"],
        peak_params["pos_y"],
        peak_params["height"],
        peak_params["width"],
        peak_params["ratio"]
    )
    
    # Check shape and background
    assert result.shape == (50, 50)
    assert np.all(result >= model.background)
    
    # Check peak heights
    for i in range(len(peak_params["pos_x"])):
        x_idx = np.abs(X[0, :] - peak_params["pos_x"][i]).argmin()
        y_idx = np.abs(Y[:, 0] - peak_params["pos_y"][i]).argmin()
        expected_height = peak_params["height"][i] + model.background
        assert np.abs(result[y_idx, x_idx] - expected_height) < 0.05
    
    # Test volume calculation
    volumes = model.volume(peak_params)
    assert len(volumes) == len(peak_params["pos_x"])
    for vol in volumes:
        assert vol > 0

def test_gaussian_kernel():
    kernel = GaussianKernel()
    sigma = 1.5
    
    # Test kernel creation
    kernel_array = kernel.gaussian_kernel(sigma)
    
    # Check kernel is symmetric
    assert np.allclose(kernel_array, kernel_array.T, rtol=1e-5, atol=1e-8)
    
    # Check kernel sums to approximately 1
    assert np.abs(np.sum(kernel_array) - 1.0) < 1e-5
    
    # Test Gaussian filter
    image = np.zeros((20, 20))
    image[10, 10] = 1.0
    
    filtered = kernel.gaussian_filter(image, sigma)
    
    # Check shape preserved
    assert filtered.shape == image.shape
    
    # Check smoothing occurred (peak value should be less than original)
    assert filtered[10, 10] < 1.0
    
    # Check total intensity approximately preserved
    assert np.abs(np.sum(filtered) - np.sum(image)) < 1e-5
