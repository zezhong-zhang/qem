"""Complex-domain analysis — extracted from qem.fit.fitter (Linus #9).

Methods are bound back onto the Fitter class via `_bind(Fitter)` from
qem.fit.fitter, so `fitter.estimate_complex_domains(...)` keeps working.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.path import Path
from scipy.ndimage import binary_dilation, binary_erosion, gaussian_filter, laplace, sobel
from skimage.measure import find_contours
from skimage.morphology import label, remove_small_objects

if TYPE_CHECKING:
    from qem.fit.fitter import Fitter  # noqa: F401


def estimate_complex_domains(
    self,
    domain_separation_method: str = "intensity_gradient",
    min_domain_size: int = 200,
    domain_threshold: int = 15,  # Percentile threshold for domain boundary detection
    plot_analysis: bool = True,
    vacuum_threshold: float = 0.05,  # Threshold for vacuum detection
    polygon_enclosure: bool = True,  # Enable polygon enclosure
):
    """
    Enhanced peak position estimation for complex STO domains with comprehensive
    domain analysis, polygon enclosure, and robust peak detection.
    
    This enhanced method addresses several critical issues:
    1. Separates vacuum from interface regions before gradient calculation
    2. Creates continuous mask regions instead of lattice patterns
    3. Automatically encloses each domain using polygons with indexing
    4. Implements robust multi-scale algorithm for strong/weak peak detection
    
    Args:
        domain_separation_method: Method to separate domains ('intensity_gradient', 'laplacian', 'sobel')
        min_domain_size: Minimum size for a domain to be considered bulk
        plot_analysis: Whether to plot the analysis results
        vacuum_threshold: Threshold for vacuum region detection
        polygon_enclosure: Whether to use automatic polygon enclosure
        
    Returns:
        dict: Dictionary containing enhanced peak coordinates, region classifications, and polygon data
    """
    
    # Convert interface width from Angstroms to pixels
    
    # Step 1: Vacuum separation and preprocessing
    vacuum_mask, clean_image = self._separate_vacuum_and_sample(
        vacuum_threshold=vacuum_threshold
    )
    
    
    # Step 2: Enhanced domain boundary identification
    sample_mask, boundary_strength, domain_regions, domain_label = self._identify_domain_boundaries(
        method=domain_separation_method,
        min_domain_size=min_domain_size,
        domain_threshold=domain_threshold,
        vacuum_mask=vacuum_mask,
        clean_image=clean_image
    )
    
    # Step 3: Automatic polygon enclosure with indexing
    polygon_data = {}
    if polygon_enclosure:
        polygon_data = self._create_polygon_enclosures(domain_regions)        

    # Step 4: plotting
    if plot_analysis:
        self._plot_domain_analysis(vacuum_mask,  boundary_strength, polygon_data, domain_label)
    
    results = {
        'bulk_mask': sample_mask,
        'boundary_strength': boundary_strength,
        'domain_regions': domain_regions,
        'polygon_data': polygon_data,
        'vacuum_mask': vacuum_mask
    }
    
    
    return results

def _separate_vacuum_and_sample(self, vacuum_threshold: float = 0.05):
    """
    Separate vacuum regions from interface regions using intensity-based thresholding.
    
    Args:
        vacuum_threshold: Threshold for identifying vacuum regions (low intensity)
        
    Returns:
        tuple: (vacuum_mask, clean_image) where vacuum_mask identifies vacuum regions
               and clean_image has vacuum regions masked out
    """

    
    # Create intensity histogram to identify vacuum threshold
    image_flat = self.image.flatten()
    # Use median absolute deviation for robust threshold estimation
    median_intensity = np.median(image_flat)
    mad = np.median(np.abs(image_flat - median_intensity))
    
    vacuum_threshold_abs = np.percentile(image_flat, vacuum_threshold*100)
    # Adaptive vacuum threshold based on image statistics
    adaptive_threshold = min(vacuum_threshold_abs, median_intensity - 2 * mad)
    
    # Detect vacuum regions
    vacuum_mask = self.image < adaptive_threshold
    
    # Clean up vacuum mask to remove noise
    vacuum_mask = gaussian_filter(vacuum_mask.astype(float), 10) > 0.95
    vacuum_mask = remove_small_objects(vacuum_mask)
    # vacuum_mask = binary_dilation(vacuum_mask, iterations=5)

    # Create clean image with vacuum masked out
    clean_image = self.image.copy()
    clean_image[vacuum_mask] = np.median(self.image[~vacuum_mask])
    
    return vacuum_mask, clean_image

def _identify_domain_boundaries(self, method="intensity_gradient", min_domain_size=50, domain_threshold = 15, vacuum_mask=None, clean_image=None):
    """
    Enhanced domain boundary identification with continuous regions and vacuum separation.
    
    Args:
        method: Method for boundary detection
        min_domain_size: Minimum size for bulk regions
        vacuum_mask: Mask identifying vacuum regions
        clean_image: Pre-processed image with vacuum removed
        
    Returns:
        tuple: (bulk_mask, interface_mask, boundary_strength, domain_regions)
    """

    
    if clean_image is None:
        clean_image = self.image
    
    # Apply different boundary detection methods on clean image
    if method == "intensity_gradient":
        # Use gradient magnitude to identify boundaries
        grad_x = sobel(gaussian_filter(clean_image, 2), axis=1)
        grad_y = sobel(gaussian_filter(clean_image, 2), axis=0)
        boundary_strength = np.sqrt(grad_x**2 + grad_y**2)
        
    elif method == "laplacian":
        # Use Laplacian to identify rapid intensity changes
        boundary_strength = np.abs(laplace(gaussian_filter(clean_image, 1.5)))
        
    elif method == "sobel":
        # Use Sobel operator for edge detection
        boundary_strength = sobel(gaussian_filter(clean_image, 2))
        
    else:
        raise ValueError(f"Unknown boundary detection method: {method}")
    
    # Normalize boundary strength
    boundary_strength = boundary_strength / boundary_strength.max()
    boundary_strength = gaussian_filter(boundary_strength, sigma=20.0)


    sample_threshold = np.percentile(gaussian_filter(self.image, 5), 5)
    sample_mask = gaussian_filter(self.image, 5) > sample_threshold
    sample_mask = gaussian_filter(remove_small_objects(sample_mask), 5) > 0.5
    # # Create boundary mask using adaptive threshold
    domain_threshold_abs = np.percentile(boundary_strength, domain_threshold)  
    
    domain_mask = (boundary_strength < domain_threshold_abs) & (~vacuum_mask) & sample_mask
    domain_mask = remove_small_objects(domain_mask, min_size=min_domain_size)
    
    # Remove small bulk regions
    domain_label = label(domain_mask)

    # Identify continuous bulk regions
    unique_regions = np.unique(domain_label)
    unique_regions = unique_regions[unique_regions != 0]  # Remove background
    
    domain_regions = {}
    
    for region_id in unique_regions:
        region_mask = domain_label == region_id
        region_size = np.sum(region_mask)
        
        if region_size >= min_domain_size:
            domain_regions[region_id] = {
                'mask': region_mask,
                'size': region_size,
                'centroid': np.array(np.where(region_mask)).mean(axis=1)
            }
    return sample_mask, boundary_strength, domain_regions, domain_label

def _create_polygon_enclosures(self, domain_regions):
    """
    Automatically create polygon enclosures for each identified domain.
    
    Args:
        domain_regions: Dictionary of domain regions
        interface_mask: Mask of interface regions
        
    Returns:
        dict: Polygon data with indices and boundaries
    """

    
    polygon_data = {}
    
    # Create polygon for each domain region
    for region_id, region_info in domain_regions.items():
        mask = region_info['mask']
        
        # Find contours for this region
        contours = find_contours(mask.astype(float), 0.5)
        
        if len(contours) > 0:
            # Use the largest contour
            largest_contour = max(contours, key=len)
            
            # Create polygon path
            polygon_path = Path(largest_contour)
            
            polygon_data[region_id] = {
                'vertices': largest_contour,
                'path': polygon_path,
                'centroid': region_info['centroid'],
                'area': region_info['size'],
            }

    return polygon_data



def _bind(cls) -> None:
    """Attach extracted methods back onto Fitter at class-load time."""
    cls.estimate_complex_domains = estimate_complex_domains
    cls._separate_vacuum_and_sample = _separate_vacuum_and_sample
    cls._identify_domain_boundaries = _identify_domain_boundaries
    cls._create_polygon_enclosures = _create_polygon_enclosures


__all__ = [
    "estimate_complex_domains",
    "_separate_vacuum_and_sample",
    "_identify_domain_boundaries",
    "_create_polygon_enclosures",
    "_bind",
]
