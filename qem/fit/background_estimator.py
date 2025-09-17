"""
Robust background estimation module for QEM image fitting.

This module provides improved background estimation methods that address the stability
issues in the linear estimator, particularly for cases with small vacuum background
regions and samples with uniform thickness.
"""

import logging
import numpy as np
from scipy import ndimage
from scipy.stats import mode, median_abs_deviation
from sklearn.cluster import KMeans
from typing import Tuple, Optional, Union

from qem.utils.params import safe_convert_to_numpy


class RobustBackgroundEstimator:
    """
    Robust background estimation using multiple strategies.
    
    This class implements several background estimation methods that are more
    robust than simple mean-based estimation, particularly for cases where:
    - There's a small vacuum background region
    - Sample has uniform thickness
    - Background estimation tends to be overestimated
    """
    
    def __init__(self, image: np.ndarray, dx: float = 1.0):
        """
        Initialize the background estimator.
        
        Args:
            image: Input image array
            dx: Pixel size (default: 1.0)
        """
        self.image = safe_convert_to_numpy(image)
        self.dx = dx
        self.ny, self.nx = self.image.shape
        
    def estimate_background_mad(self, percentile: float = 5.0) -> float:
        """
        Estimate background using Median Absolute Deviation (MAD).
        
        This method is robust to outliers and works well when the background
        region is small compared to the sample area.
        
        Args:
            percentile: Percentile to use for background estimation (default: 5th percentile)
            
        Returns:
            Estimated background value
        """
        # Flatten the image for analysis
        flat_image = self.image.ravel()
        
        # Calculate median and MAD
        median = np.median(flat_image)
        mad = median_abs_deviation(flat_image, scale='normal')
        
        # Identify background pixels using MAD-based threshold
        # Background pixels should be in the lower intensity range
        background_mask = flat_image < (median + 2 * mad)
        background_pixels = flat_image[background_mask]
        
        # Use the specified percentile of background pixels
        if len(background_pixels) > 0:
            background = np.percentile(background_pixels, percentile)
        else:
            # Fallback to global minimum
            background = np.min(self.image)
            
        logging.info(f"MAD-based background estimation: {background:.3f}")
        return float(background)
        
    def estimate_background_kmeans(self, n_clusters: int = 2) -> float:
        """
        Estimate background using K-means clustering.
        
        This method clusters pixel intensities and identifies the cluster
        with the lowest intensity as background.
        
        Args:
            n_clusters: Number of clusters for K-means (default: 3)
            
        Returns:
            Estimated background value
        """
        # Flatten the image
        flat_image = self.image.ravel().reshape(-1, 1)
        
        # Perform K-means clustering
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        kmeans.fit(flat_image)
        
        # Get cluster centers and labels
        centers = kmeans.cluster_centers_.flatten()
        labels = kmeans.labels_
        
        # Identify the cluster with the lowest intensity as background
        background_cluster = np.argmin(centers)
        background_pixels = flat_image[labels == background_cluster].flatten()
        
        # Use the mode of the background cluster
        if len(background_pixels) > 0:
            background = mode(background_pixels).mode
            # background = np.percentile(background_pixels, 10)  # Conservative estimate
        else:
            background = np.min(self.image)
            
        logging.info(f"K-means background estimation: {background:.3f}")
        return float(background)
        
    def estimate_background_morphological(self, kernel_size: int = 5) -> np.ndarray:
        """
        Estimate background using morphological operations.
        
        This method estimates a spatially varying background using morphological
        opening, which is effective for removing small foreground objects.
        
        Args:
            kernel_size: Size of the structuring element (default: 5)
            
        Returns:
            2D background array
        """
        # Create structuring element
        kernel = np.ones((kernel_size, kernel_size))
        
        # Apply morphological opening to estimate background
        background_2d = ndimage.grey_opening(self.image, footprint=kernel)
        
        logging.info(f"Morphological background estimation completed, kernel size: {kernel_size}")
        return background_2d
        
    def estimate_background_adaptive(self, 
                                   window_size: int = 50,
                                   percentile: float = 10.0) -> np.ndarray:
        """
        Estimate background using adaptive windowing.
        
        This method divides the image into windows and estimates background
        locally, then interpolates to get a smooth background estimate.
        
        Args:
            window_size: Size of the local window (default: 50)
            percentile: Percentile to use within each window (default: 10th percentile)
            
        Returns:
            2D background array
        """
        # Calculate window parameters
        step = max(1, window_size // 2)
        
        # Initialize background grid
        background_grid = np.zeros((self.ny // step + 1, self.nx // step + 1))
        
        # Calculate background for each window
        for i in range(0, self.ny, step):
            for j in range(0, self.nx, step):
                # Define window boundaries
                y_start = max(0, i)
                y_end = min(self.ny, i + window_size)
                x_start = max(0, j)
                x_end = min(self.nx, j + window_size)
                
                # Extract window
                window = self.image[y_start:y_end, x_start:x_end]
                
                # Use percentile as background estimate for this window
                if window.size > 0:
                    background_grid[i // step, j // step] = np.percentile(window, percentile)
                else:
                    background_grid[i // step, j // step] = 0
        
        # Interpolate to get full background
        from scipy.interpolate import RegularGridInterpolator
        
        # Create grid coordinates
        y_coords = np.arange(0, self.ny, step)
        x_coords = np.arange(0, self.nx, step)
        
        # Ensure grid matches actual dimensions
        if len(y_coords) != background_grid.shape[0]:
            y_coords = np.linspace(0, self.ny-1, background_grid.shape[0])
        if len(x_coords) != background_grid.shape[1]:
            x_coords = np.linspace(0, self.nx-1, background_grid.shape[1])
        
        # Create interpolator
        interpolator = RegularGridInterpolator((y_coords, x_coords), background_grid,
                                             bounds_error=False, fill_value=None)
        
        # Create full coordinate grids
        y_full, x_full = np.meshgrid(np.arange(self.ny), np.arange(self.nx), indexing='ij')
        points = np.stack([y_full.ravel(), x_full.ravel()], axis=1)
        
        # Interpolate
        background_2d = interpolator(points).reshape(self.ny, self.nx)
        
        logging.info(f"Adaptive background estimation completed, window size: {window_size}")
        return background_2d
        
    def estimate_background_edge_detection(self, 
                                       edge_threshold: float = 0.1,
                                       low_intensity_ratio: float = 0.3) -> float:
        """
        Estimate background using edge detection and low-intensity region analysis.
        
        This method identifies low-intensity regions that are likely to be background
        by analyzing edges and intensity distributions.
        
        Args:
            edge_threshold: Threshold for edge detection (default: 0.1)
            low_intensity_ratio: Ratio of pixels to consider as background (default: 0.3)
            
        Returns:
            Estimated background value
        """
        # Calculate gradient magnitude for edge detection
        gy, gx = np.gradient(self.image)
        gradient_magnitude = np.sqrt(gx**2 + gy**2)
        
        # Normalize gradient
        gradient_magnitude = gradient_magnitude / np.max(gradient_magnitude)
        
        # Identify smooth regions (likely background)
        smooth_mask = gradient_magnitude < edge_threshold
        
        # Get intensity values in smooth regions
        smooth_intensities = self.image[smooth_mask]
        
        if len(smooth_intensities) > 0:
            # Sort intensities and take lowest portion
            sorted_intensities = np.sort(smooth_intensities)
            n_background = max(1, int(len(sorted_intensities) * low_intensity_ratio))
            background_pixels = sorted_intensities[:n_background]
            
            # Use robust statistics on background pixels
            background = np.median(background_pixels)
        else:
            # Fallback to global minimum
            background = np.min(self.image)
            
        logging.info(f"Edge-based background estimation: {background:.3f}")
        return float(background)
        
    def estimate_background_robust(self, 
                                 method: str = 'combined',
                                 **kwargs) -> Union[float, np.ndarray]:
        """
        Robust background estimation using the best available method.
        
        Args:
            method: Estimation method ('mad', 'kmeans', 'morphological', 'adaptive', 'edge', 'combined')
            **kwargs: Additional parameters for specific methods
            
        Returns:
            Estimated background value or 2D background array
        """
        if method == 'combined':
            # Use multiple methods and take the most conservative estimate
            estimates = []
            
            try:
                mad_bg = self.estimate_background_mad(**kwargs.get('mad', {}))
                estimates.append(mad_bg)
            except Exception as e:
                logging.warning(f"MAD method failed: {e}")
                
            try:
                kmeans_bg = self.estimate_background_kmeans(**kwargs.get('kmeans', {}))
                estimates.append(kmeans_bg)
            except Exception as e:
                logging.warning(f"K-means method failed: {e}")
                
            try:
                edge_bg = self.estimate_background_edge_detection(**kwargs.get('edge', {}))
                estimates.append(edge_bg)
            except Exception as e:
                logging.warning(f"Edge detection method failed: {e}")
            
            if estimates:
                # Use the minimum estimate to avoid overestimation
                background = np.mean(estimates)
            else:
                # Fallback to global minimum
                background = np.min(self.image)
                
        elif method == 'mad':
            background = self.estimate_background_mad(**kwargs)
        elif method == 'kmeans':
            background = self.estimate_background_kmeans(**kwargs)
        elif method == 'morphological':
            background = self.estimate_background_morphological(**kwargs)
        elif method == 'adaptive':
            background = self.estimate_background_adaptive(**kwargs)
        elif method == 'edge':
            background = self.estimate_background_edge_detection(**kwargs)
        else:
            raise ValueError(f"Unknown method: {method}")
            
        return background


class BackgroundValidator:
    """
    Validate background estimates and provide confidence metrics.
    """
    
    @staticmethod
    def validate_background_estimate(image: np.ndarray, 
                                   background: Union[float, np.ndarray],
                                   method: str = 'statistical') -> dict:
        """
        Validate background estimate using statistical methods.
        
        Args:
            image: Original image
            background: Estimated background value or array
            method: Validation method ('statistical', 'consistency', 'visual')
            
        Returns:
            Dictionary with validation metrics
        """
        validation_results = {}
        
        if isinstance(background, (int, float)):
            background_value = background
            
            # Calculate metrics
            image_min = np.min(image)
            image_max = np.max(image)
            image_median = np.median(image)
            
            # Check if background is reasonable
            is_reasonable = image_min <= background_value <= image_median
            
            # Calculate background proportion
            below_background = np.sum(image <= background_value)
            background_ratio = below_background / image.size
            
            validation_results = {
                'background_value': background_value,
                'image_min': image_min,
                'image_max': image_max,
                'image_median': image_median,
                'is_reasonable': is_reasonable,
                'background_ratio': background_ratio,
                'confidence': 1.0 - abs(background_value - image_median) / (image_max - image_min + 1e-10)
            }
            
        else:  # 2D background
            background_mean = np.mean(background)
            image_mean = np.mean(image)
            
            # Calculate relative difference
            relative_diff = abs(background_mean - image_mean) / (image_mean + 1e-10)
            
            validation_results = {
                'background_mean': background_mean,
                'image_mean': image_mean,
                'relative_difference': relative_diff,
                'confidence': 1.0 - min(relative_diff, 1.0)
            }
            
        return validation_results


def background_estimation(image_fitting, 
                                  background_method: str = 'combined',
                                  **kwargs) -> float:
    """
    Integrate robust background estimation into existing ImageFitting workflow.
    
    Args:
        image_fitting: ImageFitting instance
        background_method: Background estimation method to use
        **kwargs: Additional parameters for background estimation
        
    Returns:
        Estimated background value
    """
    # Create background estimator
    estimator = RobustBackgroundEstimator(
        image=image_fitting.image,
        dx=image_fitting.dx
    )
    
    # Estimate background
    background = estimator.estimate_background_robust(
        method=background_method,
        **kwargs
    )
    
    # Validate the estimate
    validator = BackgroundValidator()
    validation = validator.validate_background_estimate(
        image_fitting.image, background
    )
    
    logging.info(f"Background estimation completed: {background:.3f}")
    logging.info(f"Validation confidence: {validation.get('confidence', 0):.3f}")
    
    return background