"""
2D background estimation methods for QEM image fitting.

This module provides 2D background estimation using rolling ball and photutils Background2D methods,
with optimization of background scaling parameters.
"""

import logging
import numpy as np
from scipy import ndimage
from scipy.optimize import minimize_scalar
from typing import Tuple, Optional, Union, Dict, Any
import warnings

try:
    from photutils import Background2D, MedianBackground, SExtractorBackground
    PHOTUTILS_AVAILABLE = True
except ImportError:
    PHOTUTILS_AVAILABLE = False
    warnings.warn("photutils not available. Background2D method will be unavailable.")

from qem.utils.params import safe_convert_to_numpy


class Background2DEstimator:
    """
    2D background estimation with rolling ball and Background2D methods.
    
    This class provides spatially varying background estimation with scaling optimization,
    suitable for cases where background has spatial variations.
    """
    
    def __init__(self, image: np.ndarray, dx: float = 1.0, mask: Optional[np.ndarray] = None):
        """
        Initialize the 2D background estimator.
        
        Args:
            image: Input image array
            dx: Pixel size (default: 1.0)
            mask: Optional mask array for excluded regions
        """
        self.image = safe_convert_to_numpy(image)
        self.dx = dx
        self.mask = mask
        self.ny, self.nx = self.image.shape
        
    def estimate_background_rolling_ball(self, 
                                     radius: int = 50,
                                     smoothing: bool = True) -> np.ndarray:
        """
        Estimate background using rolling ball algorithm.
        
        This method simulates a rolling ball (structuring element) moving over
        the image surface to estimate the background.
        
        Args:
            radius: Radius of the rolling ball (structuring element)
            smoothing: Whether to apply smoothing to the background
            
        Returns:
            2D background array
        """
        # Create circular structuring element
        y, x = np.ogrid[-radius:radius+1, -radius:radius+1]
        mask = x**2 + y**2 <= radius**2
        
        # Apply morphological opening (rolling ball)
        background = ndimage.grey_opening(self.image, footprint=mask)
        
        if smoothing:
            # Apply additional smoothing to reduce artifacts
            background = ndimage.gaussian_filter(background, sigma=radius//4)
            
        logging.info(f"Rolling ball background estimation completed, radius: {radius}")
        return background
        
    def estimate_background_photutils(self,
                                   box_size: Tuple[int, int] = (50, 50),
                                   filter_size: Tuple[int, int] = (3, 3),
                                   method: str = 'median') -> np.ndarray:
        """
        Estimate background using photutils Background2D.
        
        Args:
            box_size: Size of the background mesh boxes
            filter_size: Size of the median filter applied to the low-resolution background
            method: Background estimation method ('median' or 'sextractor')
            
        Returns:
            2D background array
        """
        if not PHOTUTILS_AVAILABLE:
            raise ImportError("photutils is required for Background2D method")
            
        # Select background estimator
        if method == 'median':
            bkg_estimator = MedianBackground()
        elif method == 'sextractor':
            bkg_estimator = SExtractorBackground()
        else:
            raise ValueError(f"Unknown method: {method}")
            
        # Create Background2D object
        bkg = Background2D(
            self.image,
            box_size=box_size,
            filter_size=filter_size,
            bkg_estimator=bkg_estimator,
            mask=self.mask
        )
        
        logging.info(f"Background2D estimation completed: {bkg.background_median:.3f} ± {bkg.background_rms:.3f}")
        return bkg.background
        
    def optimize_background_scaling(self,
                                  background_2d: np.ndarray,
                                  method: str = 'likelihood') -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Optimize the scaling of the 2D background.
        
        This method finds the optimal scaling factor for the 2D background
        to minimize the difference between the scaled background and the actual image.
        
        Args:
            background_2d: 2D background array to optimize
            method: Optimization method ('likelihood', 'mse', or 'robust')
            
        Returns:
            Tuple of (optimized_background_2d, optimization_info)
        """
        def objective_function(scale: float) -> float:
            """Objective function for optimization."""
            scaled_background = scale * background_2d
            residual = self.image - scaled_background
            
            if method == 'likelihood':
                # Negative log-likelihood assuming Gaussian noise
                return 0.5 * np.sum(residual**2)
            elif method == 'mse':
                # Mean squared error
                return np.mean(residual**2)
            elif method == 'robust':
                # Robust loss (Huber loss approximation)
                abs_residual = np.abs(residual)
                huber_threshold = 3.0 * np.median(abs_residual)
                loss = np.where(abs_residual <= huber_threshold,
                               0.5 * residual**2,
                               huber_threshold * (abs_residual - 0.5 * huber_threshold))
                return np.mean(loss)
            else:
                raise ValueError(f"Unknown optimization method: {method}")
                
        # Initial scale estimate
        initial_scale = np.median(self.image) / np.median(background_2d)
        
        # Optimize scaling parameter
        result = minimize_scalar(
            objective_function,
            bounds=(0.1, 10.0),
            method='bounded'
        )
        
        optimal_scale = result.x
        optimized_background = optimal_scale * background_2d
        
        # Calculate optimization metrics
        residual = self.image - optimized_background
        rms_error = np.sqrt(np.mean(residual**2))
        
        optimization_info = {
            'optimal_scale': optimal_scale,
            'initial_scale': initial_scale,
            'rms_error': rms_error,
            'success': result.success,
            'iterations': result.nit,
            'method': method
        }
        
        logging.info(f"Background scaling optimization completed: scale={optimal_scale:.3f}, RMS={rms_error:.3f}")
        return optimized_background, optimization_info
        
    def estimate_background_2d(self, 
                             method: str = 'rolling_ball',
                             optimize_scaling: bool = True,
                             **kwargs) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Comprehensive 2D background estimation.
        
        Args:
            method: Background estimation method ('rolling_ball', 'photutils')
            optimize_scaling: Whether to optimize background scaling
            **kwargs: Additional parameters for specific methods
            
        Returns:
            Tuple of (background_2d, estimation_info)
        """
        # Step 1: Initial background estimation
        if method == 'rolling_ball':
            background_2d = self.estimate_background_rolling_ball(**kwargs)
        elif method == 'photutils':
            background_2d = self.estimate_background_photutils(**kwargs)
        else:
            raise ValueError(f"Unknown method: {method}")
            
        estimation_info = {
            'method': method,
            'original_shape': background_2d.shape,
            'original_median': float(np.median(background_2d)),
            'original_mean': float(np.mean(background_2d))
        }
        
        # Step 2: Optimize scaling if requested
        if optimize_scaling:
            optimized_background, opt_info = self.optimize_background_scaling(
                background_2d, method=kwargs.get('optimization_method', 'likelihood')
            )
            estimation_info.update({
                'optimized': True,
                'optimization': opt_info,
                'optimized_median': float(np.median(optimized_background)),
                'optimized_mean': float(np.mean(optimized_background))
            })
            background_2d = optimized_background
        else:
            estimation_info['optimized'] = False
            
        return background_2d, estimation_info
        
    def validate_2d_background(self,
                             background_2d: np.ndarray,
                             validation_method: str = 'statistical') -> Dict[str, Any]:
        """
        Validate the 2D background estimate.
        
        Args:
            background_2d: 2D background array to validate
            validation_method: Validation approach ('statistical', 'residual', 'spatial')
            
        Returns:
            Validation metrics and confidence scores
        """
        residual = self.image - background_2d
        
        validation_results = {
            'validation_method': validation_method,
            'residual_stats': {
                'mean': float(np.mean(residual)),
                'std': float(np.std(residual)),
                'min': float(np.min(residual)),
                'max': float(np.max(residual))
            }
        }
        
        if validation_method == 'statistical':
            # Check if residual is approximately Gaussian
            residual_mad = np.median(np.abs(residual - np.median(residual)))
            residual_iqr = np.percentile(residual, 75) - np.percentile(residual, 25)
            
            # Estimate noise level
            expected_noise = residual_mad / 0.6745  # MAD to std conversion
            
            validation_results.update({
                'noise_estimate': float(expected_noise),
                'residual_mad': float(residual_mad),
                'residual_iqr': float(residual_iqr),
                'skewness': float(((residual - np.mean(residual))**3).mean() / (residual.std()**3 + 1e-10)),
                'kurtosis': float(((residual - np.mean(residual))**4).mean() / (residual.std()**4 + 1e-10) - 3)
            })
            
        elif validation_method == 'spatial':
            # Check spatial consistency
            background_gradient = np.gradient(background_2d)
            gradient_magnitude = np.sqrt(background_gradient[0]**2 + background_gradient[1]**2)
            
            validation_results.update({
                'max_gradient': float(np.max(gradient_magnitude)),
                'mean_gradient': float(np.mean(gradient_magnitude)),
                'gradient_std': float(np.std(gradient_magnitude))
            })
            
        # Calculate overall confidence score
        residual_std = validation_results['residual_stats']['std']
        image_std = np.std(self.image)
        confidence = max(0.0, 1.0 - (residual_std / (image_std + 1e-10)))
        
        validation_results['confidence'] = float(confidence)
        
        return validation_results


class Background2DIntegrator:
    """Integration class for 2D background estimation into QEM workflows."""
    
    @staticmethod
    def integrate_2d_background(image_fitting,
                              method: str = 'rolling_ball',
                              optimize_scaling: bool = True,
                              **kwargs) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Integrate 2D background estimation into ImageFitting workflow.
        
        Args:
            image_fitting: ImageFitting instance
            method: Background estimation method
            optimize_scaling: Whether to optimize background scaling
            **kwargs: Additional parameters
            
        Returns:
            Tuple of (background_2d, integration_info)
        """
        # Create 2D background estimator
        estimator = Background2DEstimator(
            image=image_fitting.image,
            dx=image_fitting.dx
        )
        
        # Estimate 2D background
        background_2d, estimation_info = estimator.estimate_background_2d(
            method=method,
            optimize_scaling=optimize_scaling,
            **kwargs
        )
        
        # Validate the estimate
        validation = estimator.validate_2d_background(background_2d)
        
        integration_info = {
            'background_2d': background_2d,
            'estimation_info': estimation_info,
            'validation': validation,
            'method': method,
            'optimize_scaling': optimize_scaling
        }
        
        logging.info(f"2D background estimation integrated: {method}")
        logging.info(f"Validation confidence: {validation['confidence']:.3f}")
        
        return background_2d, integration_info