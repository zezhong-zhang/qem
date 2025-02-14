import numpy as np
import pytest
from qem.refine import calculate_center_of_mass, fit_gaussian, gauss2d

def test_gaussian_fitting():
    # Create a synthetic image with a known 2D Gaussian peak
    image_size = (50, 50)
    # Ground truth parameters
    true_amplitude = 1.0
    true_x0 = 25.0
    true_y0 = 25.0
    true_sigma_x = 2.0
    true_sigma_y = 2.5
    true_theta = np.pi / 6  # 30 degrees rotation
    true_offset = 0.1
    
    # Create coordinate grid
    x = np.arange(image_size[1], dtype=np.float64)
    y = np.arange(image_size[0], dtype=np.float64)
    xy_meshgrid = np.array(np.meshgrid(x, y))  # Ensure it's a numpy array
    
    # Generate synthetic image
    true_params = (true_amplitude, true_x0, true_y0, true_sigma_x, true_sigma_y, true_theta, true_offset)
    image = gauss2d(xy_meshgrid, *true_params).reshape(image_size)
    
    # Add some noise
    rng = np.random.default_rng(42)
    noise_level = 0.01
    noisy_image = image + rng.normal(0, noise_level, image.shape)
    
    # Test center of mass calculation
    com = calculate_center_of_mass(noisy_image)
    # Center of mass should be close to true center for a symmetric Gaussian
    np.testing.assert_allclose(com[0], [true_y0, true_x0], atol=0.5)
    
    # Test Gaussian fitting
    # Use slightly offset initial guess to test convergence
    init_x0 = true_x0 + 0.5
    init_y0 = true_y0 - 0.5
    init_sigma_x = true_sigma_x * 1.2
    init_sigma_y = true_sigma_y * 1.2
    init_theta = true_theta * 0.8
    init_offset = true_offset * 0.9
    
    # Perform the fit
    fitted_params, cov_matrix, success = fit_gaussian(
        init_x0, init_y0, init_sigma_x, init_sigma_y, init_theta, init_offset,
        noisy_image, plot=False
    )
    
    # Check that fit was successful
    assert success, "Gaussian fitting failed to converge"
    
    # Define tolerances for different parameters
    position_tol = 0.3  # Position accuracy within 0.3 pixels
    sigma_rtol = 0.1   # Width relative accuracy within 10%
    angle_tol = 0.1    # Angle accuracy within ~6 degrees
    ampl_rtol = 0.1    # Amplitude relative accuracy within 10%
    offset_tol = 0.02  # Offset absolute accuracy within 0.02
    
    # Check fitted parameters against ground truth
    np.testing.assert_allclose(fitted_params[0], true_amplitude, rtol=ampl_rtol,
                             err_msg="Amplitude does not match ground truth")
    np.testing.assert_allclose(fitted_params[1], true_x0, atol=position_tol,
                             err_msg="X position does not match ground truth")
    np.testing.assert_allclose(fitted_params[2], true_y0, atol=position_tol,
                             err_msg="Y position does not match ground truth")
    np.testing.assert_allclose(fitted_params[3], true_sigma_x, rtol=sigma_rtol,
                             err_msg="Sigma X does not match ground truth")
    np.testing.assert_allclose(fitted_params[4], true_sigma_y, rtol=sigma_rtol,
                             err_msg="Sigma Y does not match ground truth")
    np.testing.assert_allclose(fitted_params[5], true_theta, atol=angle_tol,
                             err_msg="Rotation angle does not match ground truth")
    np.testing.assert_allclose(fitted_params[6], true_offset, atol=offset_tol,
                             err_msg="Offset does not match ground truth")
    
    # Check that covariance matrix is reasonable (no huge uncertainties)
    assert cov_matrix is not None, "Covariance matrix was not returned"
    param_uncertainties = np.sqrt(np.diag(cov_matrix))
    # Relative uncertainties should be small for a good fit
    relative_uncertainties = param_uncertainties / np.abs(fitted_params)
    assert np.all(relative_uncertainties < 0.5), "Parameter uncertainties are too large"

def test_center_of_mass_simple():
    # Test center of mass with a simple case
    image = np.zeros((5, 5))
    image[2, 2] = 1.0  # Single pixel at center
    
    com = calculate_center_of_mass(image)
    np.testing.assert_allclose(com[0], [2, 2], atol=1e-10)
    
    # Test with multiple peaks
    image = np.zeros((5, 5))
    image[1, 1] = 1.0
    image[3, 3] = 1.0
    
    com = calculate_center_of_mass(image)
    np.testing.assert_allclose(com[0], [2, 2], atol=1e-10)

def test_gaussian_fit_edge_cases():
    # Test fitting with extreme initial guesses
    image_size = (30, 30)
    x = np.arange(image_size[1], dtype=np.float64)
    y = np.arange(image_size[0], dtype=np.float64)
    xy_meshgrid = np.array(np.meshgrid(x, y))  # Ensure it's a numpy array
    
    # Generate a simple Gaussian
    true_params = (1.0, 15.0, 15.0, 2.0, 2.0, 0.0, 0.0)
    image = gauss2d(xy_meshgrid, *true_params).reshape(image_size)
    
    # Test with very poor initial guess
    fitted_params, _, success = fit_gaussian(
        10.0, 10.0, 1.0, 1.0, np.pi/2, 0.5,
        image, plot=False, verbose=False
    )
    
    # Should still find approximately correct position
    assert success, "Fitting failed with poor initial guess"
    np.testing.assert_allclose(fitted_params[1:3], [15.0, 15.0], rtol=0.1, atol=0.5)
    
    # Test with very narrow initial guess
    fitted_params, _, success = fit_gaussian(
        15.0, 15.0, 0.1, 0.1, 0.0, 0.0,
        image, plot=False, verbose=False
    )
    
    # Should still find approximately correct width
    assert success, "Fitting failed with narrow initial guess"
    np.testing.assert_allclose(fitted_params[3:5], [2.0, 2.0], rtol=0.2)
