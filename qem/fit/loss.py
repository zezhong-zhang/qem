"""Loss / boundary-penalty / edge-loss helpers — extracted from
qem.fit.fitter (Linus #9). Bound back onto Fitter via _bind(Fitter)
from qem.fit.fitter so existing fitter.loss / fitter.calculate_*
call sites keep working.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np
import torch

if TYPE_CHECKING:
    from qem.fit.fitter import Fitter  # noqa: F401


def enable_boundary_penalty(self, margin: float = 2.0, strength: float = 0.01):
    """
    Enable soft boundary penalty to improve edge atom fitting.
    
    This adds a smooth penalty term to the loss function that gently pushes
    atoms back when they get too close to image boundaries, without hard clipping
    that would zero out gradients.
    
    Args:
        margin: Distance from edge (in pixels) where penalty starts. Default 2.0
        strength: Penalty strength multiplier. Higher = stronger constraint. Default 0.01
                 Recommended range: 0.001 to 0.1
    
    Example:
        >>> fitter.enable_boundary_penalty(margin=3.0, strength=0.01)
        >>> fitter.fit_global()  # Edge atoms will be constrained
    """
    self.use_boundary_penalty = True
    self.boundary_margin = margin
    self.boundary_strength = strength
    
    logging.info(f"Boundary penalty enabled: margin={margin}, strength={strength}")

def disable_boundary_penalty(self):
    """Disable boundary penalty constraint."""
    self.use_boundary_penalty = False
    
    logging.info("Boundary penalty disabled")

def enable_adaptive_edge_loss(self):
    """
    Enable adaptive gradient boosting for edge peaks.
    
    This amplifies the gradient signal for peaks with low visibility
    (near or outside image boundaries), helping the optimizer converge
    to the correct position even when most of the peak is invisible.
    
    Example:
        >>> fitter.enable_adaptive_edge_loss()
        >>> fitter.fit_global()  # Gradient boosting active for edge peaks
    """
    self.use_adaptive_edge_loss = True
    
    logging.info("Adaptive edge loss enabled (gradient boosting for edge peaks)")

def disable_adaptive_edge_loss(self):
    """Disable adaptive edge loss."""
    self.use_adaptive_edge_loss = False
    
    logging.info("Adaptive edge loss disabled")

def calculate_peak_visibility(self, pos_x, pos_y, width):
    """
    Calculate what fraction of each peak is visible in the image.
    
    For a Gaussian, ~99.7% of intensity is within 3*sigma of center.
    We check how much of this region overlaps with the image.
    
    Args:
        pos_x: Peak center x positions (tensor or array)
        pos_y: Peak center y positions (tensor or array)
        width: Peak widths (sigma) (tensor or array)
        
    Returns:
        visibility: Fraction of peak visible (0.01 to 1) for each peak
    """
    h, w = self.image.shape
    
    # Define the "effective region" as 3*sigma around center
    radius = 3.0 * width
    
    # Calculate overlap with image bounds for each dimension
    x_min = torch.maximum(pos_x - radius, 0.0)
    x_max = torch.minimum(pos_x + radius, w - 1)
    y_min = torch.maximum(pos_y - radius, 0.0)
    y_max = torch.minimum(pos_y + radius, h - 1)
    
    # Visible width and height
    visible_width = torch.maximum(x_max - x_min, 0.0)
    visible_height = torch.maximum(y_max - y_min, 0.0)
    
    # Total width and height of effective region
    total_width = 2 * radius
    total_height = 2 * radius
    
    # Visibility as fraction of area
    visibility = (visible_width * visible_height) / (total_width * total_height)
    
    # Clamp to [0.01, 1.0] to avoid division by zero and extreme values
    visibility = torch.clamp(visibility, 0.01, 1.0)
    
    return visibility

def calculate_boundary_penalty(self, pos_x, pos_y, width, max_distance=3.0):
    """
    Calculate soft boundary penalty for positions near or outside image edges.
    
    This penalty allows peaks to be outside the image by up to max_distance * width,
    but applies a smooth quadratic penalty for positions beyond that.
    
    Args:
        pos_x: Peak x positions (tensor or array)
        pos_y: Peak y positions (tensor or array)
        width: Peak widths (tensor or array)
        max_distance: Maximum allowed distance outside (in units of sigma). Default 3.0
        
    Returns:
        penalty: Scalar penalty value
    """
    h, w = self.image.shape
    
    # Calculate how far outside the boundary each peak is
    # Negative means inside, positive means outside
    dist_left = -pos_x
    dist_right = pos_x - (w - 1)
    dist_top = -pos_y
    dist_bottom = pos_y - (h - 1)
    
    # Maximum allowed distance for each peak
    allowed = max_distance * width
    
    # Penalty only when exceeding allowed distance
    # Use smooth quadratic penalty
    penalty_left = torch.maximum(dist_left - allowed, 0.0) ** 2
    penalty_right = torch.maximum(dist_right - allowed, 0.0) ** 2
    penalty_top = torch.maximum(dist_top - allowed, 0.0) ** 2
    penalty_bottom = torch.maximum(dist_bottom - allowed, 0.0) ** 2
    
    total_penalty = torch.sum(
        penalty_left + penalty_right + penalty_top + penalty_bottom
    )
    
    return total_penalty

def loss(self, y_true, y_pred, use_adaptive_edge_loss=None):
    """
    Compute the loss value between the image and the prediction.

    Parameters:
    -----------
    y_true : np.ndarray
        The original image tensor (ground truth).
    y_pred : np.ndarray
        The predicted image tensor (model output).
    use_adaptive_edge_loss : bool, optional
        If True, use adaptive gradient boosting for edge peaks.
        If None, uses self.use_adaptive_edge_loss. Default None.

    Returns:
    --------
    float
        The computed loss value.
    """
    # Use instance variable if not explicitly specified
    if use_adaptive_edge_loss is None:
        use_adaptive_edge_loss = getattr(self, 'use_adaptive_edge_loss', False)
    
    diff = y_true - y_pred
    if self._window_t is None or self._window_t.device != diff.device:
        self._window_t = torch.as_tensor(
            self.window, dtype=torch.float32, device=diff.device,
        )
    diff = torch.mul(diff, self._window_t)
    
    # Base MSE loss
    mse = torch.sqrt(torch.mean(torch.square(diff)))
    
    # Optionally use adaptive edge loss for better gradient signal
    if use_adaptive_edge_loss:
        # Use the model currently being optimized (if available)
        model = getattr(self, '_optimization_model', self.model)
        if model is not None:
            # Get current parameters
            params = model.get_params()
            pos_x = params['pos_x']
            pos_y = params['pos_y']
            width = params['width']
            
            # Calculate visibility and apply gradient boost
            visibility = self.calculate_peak_visibility(pos_x, pos_y, width)
            boost_factor = 1.0 / torch.sqrt(visibility)
            avg_boost = torch.mean(boost_factor)
            mse = mse * avg_boost
    
    # Add soft boundary penalty if enabled
    if hasattr(self, 'use_boundary_penalty') and self.use_boundary_penalty:
        # Use the model currently being optimized (if available)
        model = getattr(self, '_optimization_model', self.model)
        if model is not None:
            # Get current parameters
            params = model.get_params()
            pos_x = params['pos_x']
            pos_y = params['pos_y']
            width = params['width']
            
            # Calculate soft boundary penalty
            boundary_penalty = self.calculate_boundary_penalty(
                pos_x, pos_y, width, max_distance=3.0
            )
            
            # Apply penalty with strength factor
            penalty_weight = getattr(self, 'boundary_strength', 0.01)
            penalty_term = penalty_weight * boundary_penalty
            mse = mse + penalty_term
    
    return mse 

def disable_edge_window(self):
    """
    Disable edge dampening window for better edge peak fitting.
    
    The default Butterworth window dampens edge pixels to reduce
    Fourier artifacts, but this makes fitting edge peaks harder.
    Call this method to use uniform weighting across the image.
    
    Example:
        >>> fitter.disable_edge_window()
        >>> fitter.enable_boundary_penalty()
        >>> fitter.fit_global()  # Better edge peak fitting
    """
    self._window = np.ones_like(self.image)
    self._window_t = None  # invalidate cached torch view
    logging.info("Edge window dampening disabled (uniform weighting)")



def _bind(cls) -> None:
    """Attach extracted methods back onto Fitter at class-load time."""
    cls.enable_boundary_penalty = enable_boundary_penalty
    cls.disable_boundary_penalty = disable_boundary_penalty
    cls.enable_adaptive_edge_loss = enable_adaptive_edge_loss
    cls.disable_adaptive_edge_loss = disable_adaptive_edge_loss
    cls.calculate_peak_visibility = calculate_peak_visibility
    cls.calculate_boundary_penalty = calculate_boundary_penalty
    cls.loss = loss
    cls.disable_edge_window = disable_edge_window


__all__ = [
    "enable_boundary_penalty",
    "disable_boundary_penalty",
    "enable_adaptive_edge_loss",
    "disable_adaptive_edge_loss",
    "calculate_peak_visibility",
    "calculate_boundary_penalty",
    "loss",
    "disable_edge_window",
    "_bind",
]
