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
    position_atol = 0.5  # Position accuracy within 0.5 pixels
    np.testing.assert_allclose(
        fitted_params['pos_x'], 
        true_positions[:, 0], 
        atol=position_atol,
        err_msg="Refined positions do not match ground truth"
    )
    np.testing.assert_allclose(
        fitted_params['pos_y'], 
        true_positions[:, 1], 
        atol=position_atol,
        err_msg="Refined positions do not match ground truth"
    )
    
    # Assert amplitudes are within tolerance
    height_atol = 0.1  # Height accuracy within 0.1 units (relative to peak heights of 0.5-1.0)
    np.testing.assert_allclose(
        fitted_params['height'], 
        true_amplitudes, 
        atol=height_atol,
        err_msg="Refined amplitudes do not match ground truth"
    )
    
    # Assert sigmas are within tolerance
    width_atol = 0.2  # Width parameter accuracy within 0.2 units
    np.testing.assert_allclose(
        fitted_params['sigma'], 
        np.full_like(fitted_params['sigma'], true_sigmas.mean()),  # compare with mean sigma since we used it for generation
        atol=width_atol,
        err_msg="Refined sigmas do not match ground truth"
    )
    
    # Optional: Check the residual
    model_image = fitter.predict(fitter.params, fitter.X, fitter.Y)
    residual = np.mean(np.abs(synthetic_image - model_image))
    assert residual < 0.1, f"Residual {residual} is too large"


import numpy as np
import jax
import pytest

from qem.model import (
    gaussian_sum_parallel,
    lorentzian_sum_parallel,
    voigt_sum_parallel
)
from qem.image_fitting import ImageModelFitting

# Set JAX to use CPU since no GPU is available
jax.config.update('jax_platforms', 'cpu')

@pytest.mark.parametrize("peak_type,gen_func", [
    ("gaussian", gaussian_sum_parallel),
    ("lorentzian", lorentzian_sum_parallel),
    ("voigt", voigt_sum_parallel)
])
def test_peak_fitting(peak_type, gen_func):
    # Generate synthetic image parameters
    image_size = (100, 100)
    num_peaks = 5
    
    # Generate random peak positions
    rng = np.random.default_rng(42)
    pos_x = rng.uniform(20, 80, num_peaks)
    pos_y = rng.uniform(20, 80, num_peaks)
    heights = rng.uniform(0.5, 1.0, num_peaks)
    
    # Parameters specific to each peak type
    mean_sigma = 2.0
    sigmas = np.full(num_peaks, mean_sigma)
    mean_gamma = mean_sigma / np.sqrt(2 * np.log(2))  # Convert sigma to gamma
    gammas = np.full(num_peaks, mean_gamma)
    mean_ratio = 0.9  # Default ratio in init_params
    ratios = np.full(num_peaks, mean_ratio)
    
    # Create synthetic image
    x = np.arange(image_size[1])
    y = np.arange(image_size[0])
    X, Y = np.meshgrid(x, y, indexing='xy')
    
    # Ground truth parameters
    true_params = {
        'pos_x': pos_x,
        'pos_y': pos_y,
        'height': heights,
    }
    
    # Generate synthetic image based on peak type
    if peak_type == "gaussian":
        image = gen_func(X, Y, pos_x, pos_y, heights, sigmas, 0.0)
        true_params['sigma'] = sigmas
    elif peak_type == "lorentzian":
        image = gen_func(X, Y, pos_x, pos_y, heights, gammas, 0.0)
        true_params['gamma'] = gammas
    else:  # voigt
        image = gen_func(X, Y, pos_x, pos_y, heights, sigmas, gammas, ratios, 0.0)
        true_params['sigma'] = sigmas
        true_params['gamma'] = gammas
        true_params['ratio'] = ratios
    
    # Add some noise
    noise_level = 0.01
    image += rng.normal(0, noise_level, image.shape)
    
    # Initialize ImageModelFitting
    fitter = ImageModelFitting(image=image, dx=1.0)
    fitter.model_type = peak_type
    fitter.same_width = False  # Allow independent widths for each peak
    
    # Set coordinates and initialize parameters
    coordinates = np.column_stack((pos_x, pos_y))
    fitter.coordinates = coordinates
    
    # Initialize parameters with appropriate size for peak type
    if peak_type == "gaussian":
        fitter.init_params(atom_size=mean_sigma)
    elif peak_type == "lorentzian":
        fitter.init_params(atom_size=mean_gamma * np.sqrt(2 * np.log(2)))  # Convert gamma back to sigma for init
    else:  # voigt
        # For Voigt, initialize with the true values to ensure proper starting point
        fitter.init_params(atom_size=mean_sigma)
        # Update gamma and ratio with true values since they're coupled parameters
        fitter.params['gamma'] = gammas.copy()
        fitter.params['ratio'] = ratios.copy()
    
    # Fit the image using global optimization
    fitter.fit_global(maxiter=1000, tol=1e-4, step_size=0.01)
    
    # Get fitted parameters
    fitted_params = fitter.params
    
    # Set tolerances for different parameter types
    position_atol = 0.3  # Position accuracy within 0.3 pixels (sub-pixel accuracy)
    height_rtol = 0.05  # Height relative accuracy within 5%
    width_rtol = 0.1   # Width parameter relative accuracy within 10%
    ratio_atol = 0.05  # Ratio absolute accuracy within 0.05 (for Voigt)
    
    # Check positions and heights
    np.testing.assert_allclose(
        fitted_params['pos_x'], true_params['pos_x'],
        atol=position_atol,
        err_msg=f"{peak_type}: X positions do not match ground truth"
    )
    np.testing.assert_allclose(
        fitted_params['pos_y'], true_params['pos_y'],
        atol=position_atol,
        err_msg=f"{peak_type}: Y positions do not match ground truth"
    )
    np.testing.assert_allclose(
        fitted_params['height'], true_params['height'],
        rtol=height_rtol,
        err_msg=f"{peak_type}: Heights do not match ground truth"
    )
    
    # Check peak-type specific parameters
    if peak_type == "gaussian":
        np.testing.assert_allclose(
            fitted_params['sigma'], true_params['sigma'],
            rtol=width_rtol,
            err_msg="Gaussian: Sigma values do not match ground truth"
        )
    elif peak_type == "lorentzian":
        np.testing.assert_allclose(
            fitted_params['gamma'], true_params['gamma'],
            rtol=width_rtol,
            err_msg="Lorentzian: Gamma values do not match ground truth"
        )
    else:  # voigt
        np.testing.assert_allclose(
            fitted_params['sigma'], true_params['sigma'],
            rtol=width_rtol,
            err_msg="Voigt: Sigma values do not match ground truth"
        )
        np.testing.assert_allclose(
            fitted_params['gamma'], true_params['gamma'],
            rtol=width_rtol,
            err_msg="Voigt: Gamma values do not match ground truth"
        )
        np.testing.assert_allclose(
            fitted_params['ratio'], true_params['ratio'],
            atol=ratio_atol,
            err_msg="Voigt: Ratio values do not match ground truth"
        )
