import os
os.environ["JAX_PLATFORMS"] = "cpu"  # Force JAX to use CPU

import numpy as np
import pytest
from qem.image_fitting import ImageModelFitting
from qem.model import gaussian_sum_parallel


def test_gaussian_peak_fitting():
    # Test parameters
    image_size = 100
    dx = 0.1  # pixel size in Angstroms
    noise_level = 0.02
    
    # Ground truth parameters for multiple Gaussian peaks
    true_positions = np.array([
        [25, 25],
        [25, 75],
        [75, 25],
        [75, 75]
    ])
    true_amplitudes = np.array([1.0, 1.2, 0.8, 1.1])
    true_sigmas = np.array([3.0, 2.8, 3.2, 2.9])
    
    # Create synthetic image
    x = np.arange(image_size)
    y = np.arange(image_size)
    X, Y = np.meshgrid(x, y)
    
    # Create synthetic image using gaussian_sum_parallel
    synthetic_image = gaussian_sum_parallel(
        X, Y,
        pos_x=true_positions[:, 0],
        pos_y=true_positions[:, 1],
        height=true_amplitudes,
        width=true_sigmas.mean(),  # using mean sigma as width
        background=0.0
    )
    
    # Add noise
    np.random.seed(42)  # for reproducibility
    noise = np.random.normal(0, noise_level, synthetic_image.shape)
    synthetic_image += noise
    
    # Initialize image fitting
    fitter = ImageModelFitting(
        image=synthetic_image,
        dx=dx,
        units="A",
        elements=["X"]  # Using a single dummy element type
    )
    
    # Initial guess - slightly perturbed from true positions
    initial_positions = true_positions + np.random.normal(0, 1, true_positions.shape)
    
    # Set coordinates and initialize parameters
    fitter.coordinates = initial_positions
    fitter.init_params(atom_size=true_sigmas.mean())
    
    # Fit the image using global optimization
    fitter.fit_global(maxiter=10000, tol=1e-4, step_size=0.1)
    
    # Get fitted parameters
    fitted_params = fitter.params
    
    # Assert positions are within tolerance
    position_tolerance = 0.5  # pixels
    np.testing.assert_allclose(
        fitted_params['pos_x'], 
        true_positions[:, 0], 
        atol=position_tolerance,
        err_msg="Refined positions do not match ground truth"
    )
    np.testing.assert_allclose(
        fitted_params['pos_y'], 
        true_positions[:, 1], 
        atol=position_tolerance,
        err_msg="Refined positions do not match ground truth"
    )
    
    # Assert amplitudes are within tolerance
    amplitude_tolerance = 0.2
    np.testing.assert_allclose(
        fitted_params['height'], 
        true_amplitudes, 
        rtol=amplitude_tolerance,
        err_msg="Refined amplitudes do not match ground truth"
    )
    
    # Assert sigmas are within tolerance
    sigma_tolerance = 0.5
    np.testing.assert_allclose(
        fitted_params['sigma'], 
        np.full_like(fitted_params['sigma'], true_sigmas.mean()),  # compare with mean sigma since we used it for generation
        atol=sigma_tolerance,
        err_msg="Refined sigmas do not match ground truth"
    )
    
    # Optional: Check the residual
    model_image = fitter.predict(fitter.params, fitter.X, fitter.Y)
    residual = np.mean(np.abs(synthetic_image - model_image))
    assert residual < 0.1, f"Residual {residual} is too large"
