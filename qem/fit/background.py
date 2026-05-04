import torch
"""
Unified background estimation for QEM image fitting.

This module provides robust background estimation methods optimized for fitting
and linear estimation, combining 1D and 2D background approaches.
"""

import logging
from typing import TYPE_CHECKING, Tuple, Dict, Any, Union, Optional

import numpy as np
from scipy.optimize import minimize_scalar

if TYPE_CHECKING:
    from qem.fit.fitter import Fitter  # noqa: F401

try:
    from photutils.background import Background2D, SExtractorBackground
    PHOTUTILS_AVAILABLE = True
except ImportError:
    PHOTUTILS_AVAILABLE = False
    logging.debug("photutils is not installed; falling back to median/polynomial background methods.")

from qem.utils.tensors import to_numpy


class Background:
    """
    Unified background estimator for QEM image fitting.
    
    Provides both 1D (scalar) and 2D (spatially varying) background estimation
    optimized for integration with linear estimation and fitting workflows.
    """
    
    def __init__(self, image: np.ndarray, dx: float = 1.0, mask: Optional[np.ndarray] = None):
        """
        Initialize the background estimator.
        
        Args:
            image: Input image array
            dx: Pixel size (default: 1.0)
            mask: Optional mask array for excluded regions
        """
        self.image = to_numpy(image)
        self.dx = dx
        self.mask = mask
        self.ny, self.nx = self.image.shape
        
        # State for 2D background
        self.background_2d = None
        self.background_scale = 1.0
        self.use_2d_background = False
        
    def estimate_scalar_background(self, method: str = 'robust') -> float:
        """
        Estimate scalar (1D) background value.
        
        Args:
            method: Estimation method ('robust', 'percentile', 'median')
            
        Returns:
            Estimated background value
        """
        if method == 'robust':
            # Use multiple robust statistics and take conservative estimate
            median_val = np.median(self.image)
            percentile_5 = np.percentile(self.image, 5)
            percentile_10 = np.percentile(self.image, 10)
            
            # Take the minimum to avoid overestimation
            background = min(median_val * 0.8, percentile_5, percentile_10)
            
        elif method == 'percentile':
            background = np.percentile(self.image, 5)
            
        elif method == 'median':
            background = np.median(self.image)
            
        else:
            raise ValueError(f"Unknown scalar method: {method}")
        
        logging.info("Scalar background: %.3f (method: %s)", background, method)
        return float(background)
    
    def estimate_2d_background(self, 
                             method: str = 'photutils',
                             box_size: Tuple[int, int] = None,
                             filter_size: Tuple[int, int] = (3, 3)) -> np.ndarray:
        """
        Estimate 2D (spatially varying) background.
        
        Args:
            method: Estimation method ('photutils', 'median', 'polynomial')
            box_size: Size of background mesh boxes (auto-determined if None)
            filter_size: Size of median filter for smoothing
            
        Returns:
            2D background array
        """
        if method == 'photutils' and PHOTUTILS_AVAILABLE:
            return self._estimate_photutils_2d(box_size, filter_size)
        elif method == 'median':
            return self._estimate_median_2d()
        elif method == 'polynomial':
            return self._estimate_polynomial_2d()
        else:
            logging.warning("Falling back to median 2D background")
            return self._estimate_median_2d()
    
    def _estimate_photutils_2d(self, 
                              box_size: Tuple[int, int] = None,
                              filter_size: Tuple[int, int] = (3, 3)) -> np.ndarray:
        """Estimate 2D background using photutils."""
        if box_size is None:
            # Auto-determine box size: ~1/10 of image size, minimum 20 pixels
            box_x = max(20, self.nx // 10)
            box_y = max(20, self.ny // 10)
            box_size = (box_y, box_x)
        
        try:
            bkg = Background2D(
                self.image,
                box_size=box_size,
                filter_size=filter_size,
                bkg_estimator=SExtractorBackground(),
                mask=self.mask
            )
            logging.info("Photutils 2D background: box_size=%s", box_size)
            return bkg.background
            
        except Exception as e:
            logging.warning("Photutils failed: %s, using median fallback", e)
            return self._estimate_median_2d()
    
    def _estimate_median_2d(self) -> np.ndarray:
        """Fallback median 2D background."""
        median_value = np.median(self.image)
        background = np.full_like(self.image, median_value)
        logging.info("Median 2D background: %.3f", median_value)
        return background
    
    def _estimate_polynomial_2d(self, degree: int = 1) -> np.ndarray:
        """Estimate 2D background using polynomial fitting."""
        # Create coordinate grids
        y_coords, x_coords = np.mgrid[0:self.ny, 0:self.nx]
        x_flat = x_coords.flatten()
        y_flat = y_coords.flatten()
        image_flat = self.image.flatten()
        
        # Create polynomial terms (simplified for fitting)
        terms = [np.ones_like(x_flat)]  # constant term
        if degree >= 1:
            terms.extend([x_flat, y_flat])  # linear terms
        if degree >= 2:
            terms.extend([x_flat**2, y_flat**2, x_flat*y_flat])  # quadratic terms
        
        A = np.column_stack(terms)
        
        try:
            # Robust fitting using a subset of data
            n_samples = min(5000, len(image_flat))
            indices = np.random.choice(len(image_flat), n_samples, replace=False)
            coeffs = np.linalg.lstsq(A[indices], image_flat[indices], rcond=None)[0]
            background = (A @ coeffs).reshape(self.ny, self.nx)
            
        except np.linalg.LinAlgError:
            logging.warning("Polynomial fitting failed, using median fallback")
            background = self._estimate_median_2d()
        
        logging.info("Polynomial 2D background: degree=%d", degree)
        return background
    
    def get_initial_scale_estimate(self, background_2d: np.ndarray) -> float:
        """
        Get a simple initial scale estimate for 2D background.
        
        This provides a reasonable starting point for linear estimation,
        but the actual optimization is done by the linear solver.
        
        Args:
            background_2d: 2D background array
            
        Returns:
            Initial scale estimate
        """
        # Simple ratio-based initial estimate
        image_median = np.median(self.image)
        bg_median = np.median(background_2d)
        
        if bg_median > 1e-10:
            initial_scale = image_median / bg_median
        else:
            initial_scale = 1.0
        
        # Keep within reasonable bounds
        initial_scale = max(0.1, min(10.0, initial_scale))
        
        logging.info("Initial 2D background scale estimate: %.3f", initial_scale)
        return float(initial_scale)
    
    def optimize_2d_background_scale(self, background_2d: np.ndarray) -> float:
        """
        Optimize scaling factor for 2D background for linear estimation.
        
        This method finds the optimal scaling factor for the 2D background
        that will be used as a parameter in the linear estimation system.
        The 2D background pattern itself remains fixed - only the scale changes.
        
        Args:
            background_2d: 2D background array to scale
            
        Returns:
            Optimal scaling factor
        """
        def objective(scale: float) -> float:
            """Robust objective function using Huber loss."""
            scaled_bg = scale * background_2d
            residual = self.image - scaled_bg
            
            # Huber loss for robustness
            abs_residual = np.abs(residual)
            threshold = 2.0 * np.median(abs_residual)
            
            loss = np.where(
                abs_residual <= threshold,
                0.5 * residual**2,
                threshold * (abs_residual - 0.5 * threshold)
            )
            return np.mean(loss)
        
        # Initial scale estimate
        initial_scale = self.get_initial_scale_estimate(background_2d)
        
        # Optimize with reasonable bounds
        result = minimize_scalar(objective, bounds=(0.01, 100.0), method='bounded')
        optimal_scale = result.x
        
        logging.info("2D background scale optimization: %.3f -> %.3f", 
                    initial_scale, optimal_scale)
        return float(optimal_scale)
    
    def enable_2d_background(self, 
                           method: str = 'photutils',
                           **kwargs) -> Dict[str, Any]:
        """
        Enable 2D background mode for fitting and linear estimation.
        
        IMPORTANT: In 2D background mode, the background pattern is estimated once
        and remains FIXED. During gradient descent and linear estimation, only the
        SCALING FACTOR is optimized, not the entire 2D matrix. This ensures:
        1. Computational efficiency (optimizing 1 parameter vs N×M parameters)
        2. Numerical stability 
        3. Physical meaningfulness (background shape is preserved)
        
        Args:
            method: Background estimation method
            **kwargs: Additional parameters for background estimation
            
        Returns:
            Dictionary with background information
        """
        # Estimate 2D background
        self.background_2d = self.estimate_2d_background(method=method, **kwargs)
        
        # Optimize initial scaling factor (the 2D pattern stays fixed during fitting)
        self.background_scale = self.optimize_2d_background_scale(self.background_2d)
        
        # Enable 2D mode
        self.use_2d_background = True
        
        info = {
            'method': method,
            'background_shape': self.background_2d.shape,
            'initial_scale': self.background_scale,
            'background_range': (float(np.min(self.background_2d)), 
                               float(np.max(self.background_2d))),
            'enabled': True
        }
        
        logging.info("2D background enabled: method=%s, initial_scale=%.3f", method, self.background_scale)
        return info
    
    def disable_2d_background(self):
        """Disable 2D background mode and revert to scalar background."""
        self.use_2d_background = False
        self.background_2d = None
        self.background_scale = 1.0
        logging.info("2D background disabled")
    
    def get_current_background(self, scalar_value: float = None) -> np.ndarray:
        """
        Get current background as 2D array.
        
        Args:
            scalar_value: Scalar background value (for 1D mode)
            
        Returns:
            2D background array
        """
        if self.use_2d_background and self.background_2d is not None:
            return self.background_scale * self.background_2d
        else:
            # Return scalar background as 2D array
            if scalar_value is None:
                scalar_value = self.estimate_scalar_background()
            return np.full((self.ny, self.nx), scalar_value)
    
    def update_2d_background_scale(self, new_scale: float):
        """Update the 2D background scaling factor."""
        if self.use_2d_background:
            self.background_scale = new_scale
            logging.debug("Updated 2D background scale: %.3f", new_scale)
    
    def get_background_for_linear_estimation(self) -> Union[float, np.ndarray]:
        """
        Get background in the format needed for linear estimation.
        
        CRITICAL: For 2D background, this returns the UNSCALED background pattern.
        The linear solver will optimize the scaling factor, treating the 2D pattern
        as a fixed basis function. This is computationally efficient and ensures
        the background shape remains physically meaningful.
        
        Returns:
            Scalar value for 1D mode, unscaled 2D array for 2D mode
        """
        if self.use_2d_background and self.background_2d is not None:
            return self.background_2d  # UNSCALED - scaling factor is optimized separately
        else:
            return self.estimate_scalar_background()
    
    def validate_background_quality(self) -> Dict[str, Any]:
        """
        Validate background quality and provide metrics.
        
        Returns:
            Dictionary with quality metrics
        """
        if self.use_2d_background and self.background_2d is not None:
            # 2D background validation
            current_bg = self.get_current_background()
            residual = self.image - current_bg
            
            residual_std = np.std(residual)
            image_std = np.std(self.image)
            
            # Spatial smoothness check
            gradient = np.gradient(self.background_2d)
            gradient_mag = np.sqrt(gradient[0]**2 + gradient[1]**2)
            
            quality_score = max(0.0, 1.0 - (residual_std / (image_std + 1e-10)))
            smoothness_score = 1.0 / (1.0 + np.mean(gradient_mag))
            
            return {
                'mode': '2d',
                'residual_std': float(residual_std),
                'quality_score': float(quality_score),
                'smoothness_score': float(smoothness_score),
                'overall_score': float(0.7 * quality_score + 0.3 * smoothness_score),
                'background_scale': float(self.background_scale)
            }
        else:
            # 1D background validation
            scalar_bg = self.estimate_scalar_background()
            below_bg = np.sum(self.image <= scalar_bg)
            bg_ratio = below_bg / self.image.size
            
            # Simple quality check
            image_min = np.min(self.image)
            image_median = np.median(self.image)
            is_reasonable = image_min <= scalar_bg <= image_median
            
            return {
                'mode': '1d',
                'background_value': float(scalar_bg),
                'background_ratio': float(bg_ratio),
                'is_reasonable': bool(is_reasonable),
                'quality_score': float(1.0 if is_reasonable else 0.5)
            }


def estimate_background(image: np.ndarray, 
                                  dx: float = 1.0,
                                  method: str = 'robust',
                                  use_2d: bool = False,
                                  **kwargs) -> Union[float, np.ndarray]:
    """
    Convenience function for background estimation in fitting workflows.
    
    The background scaling optimization is handled by the linear estimation
    and fitting processes, not here.
    
    Args:
        image: Input image array
        dx: Pixel size
        method: Background estimation method
        use_2d: Whether to use 2D background
        **kwargs: Additional parameters
        
    Returns:
        Scalar background value (1D) or 2D background pattern (unscaled)
    """
    estimator = Background(image, dx)
    
    if use_2d:
        estimator.enable_2d_background(method=method, **kwargs)
        return estimator.background_2d  # Return unscaled pattern
    else:
        return estimator.estimate_scalar_background(method=method)


def enable_2d_background(self,
                       method: str = 'photutils',
                       **kwargs) -> dict:
    """
    Enable 2D background estimation for the image fitting.
    
    Args:
        method: Background estimation method ('photutils', 'median', 'polynomial')
        **kwargs: Additional parameters for background estimation
        
    Returns:
        Dictionary with background estimation information
    """
    logging.info("Enabling 2D background estimation with method: %s", method)
    
    # Enable 2D background in the estimator
    info = self.background_estimator.enable_2d_background(method=method, **kwargs)
    
    # Update fit_background to use 2D mode
    self.fit_background = True
    
    logging.info("2D background estimation completed: scale=%.3f", info['initial_scale'])
    return info

def disable_2d_background(self):
    """Disable 2D background estimation and revert to scalar background."""
    self.background_estimator.disable_2d_background()
    logging.info("2D background estimation disabled")

def get_current_background(self) -> np.ndarray:
    """
    Get the current background (2D or scalar).
    
    Returns:
        Background array (2D if enabled, otherwise scalar broadcast to 2D)
    """
    if self.background_estimator.use_2d_background:
        return self.background_estimator.get_current_background()
    else:
        # Get scalar background value
        bg_value = getattr(self, 'init_background', 0.0)
        if self.params is not None and 'background' in self.params:
            bg_value = to_numpy(self.params['background'])
            if np.isscalar(bg_value):
                bg_value = float(bg_value)
            else:
                bg_value = float(bg_value.item()) if bg_value.size == 1 else float(bg_value[0])
        return self.background_estimator.get_current_background(bg_value)

def update_2d_background_scale(self, new_scale: float):
    """Update the 2D background scaling factor."""
    self.background_estimator.update_2d_background_scale(new_scale)

def optimize_2d_background_scale(self) -> float:
    """
    Optimize the 2D background scaling factor for the current image.
    
    This method finds the optimal scaling factor for the 2D background
    that minimizes the residual between the scaled background and the image.
    
    Returns:
        Optimal scaling factor
    """
    if not self.background_estimator.use_2d_background or self.background_estimator.background_2d is None:
        raise ValueError("2D background not enabled or not estimated")
    
    from scipy.optimize import minimize_scalar
    
    background_2d = self.background_estimator.background_2d
    
    def objective(scale: float) -> float:
        """Objective function using robust loss."""
        scaled_bg = scale * background_2d
        residual = self.image - scaled_bg
        
        # Use robust loss (Huber loss)
        abs_residual = np.abs(residual)
        threshold = 2.0 * np.median(abs_residual)
        
        loss = np.where(
            abs_residual <= threshold,
            0.5 * residual**2,
            threshold * (abs_residual - 0.5 * threshold)
        )
        return np.mean(loss)
    
    # Get initial estimate
    initial_scale = self.background_estimator.background_scale
    
    # Optimize with reasonable bounds
    result = minimize_scalar(objective, bounds=(0.01, 100.0), method='bounded')
    optimal_scale = result.x
    
    # Update the background estimator
    self.update_2d_background_scale(optimal_scale)
    
    logging.info("2D background scale optimized: %.3f -> %.3f", initial_scale, optimal_scale)
    return float(optimal_scale)



def _bind(cls) -> None:
    """Attach extracted methods back onto Fitter at class-load time."""
    cls.enable_2d_background = enable_2d_background
    cls.disable_2d_background = disable_2d_background
    cls.get_current_background = get_current_background
    cls.update_2d_background_scale = update_2d_background_scale
    cls.optimize_2d_background_scale = optimize_2d_background_scale


__all__ = [
    "enable_2d_background",
    "disable_2d_background",
    "get_current_background",
    "update_2d_background_scale",
    "optimize_2d_background_scale",
    "_bind",
]

