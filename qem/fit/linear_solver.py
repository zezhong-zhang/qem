"""
Linear solver module for QEM image fitting.
Provides modular functions for breaking down the large linear_estimator method.
"""

import logging
import warnings
from typing import Dict, Optional, Tuple, Union

import numpy as np
import keras
from scipy.optimize import lsq_linear
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import spsolve

from qem.utils.params import safe_convert_to_numpy

from qem.schema.exceptions import ParameterError, DataError, ValidationError


class ParameterValidator:
    """Validates input parameters for linear estimation."""
    
    @staticmethod
    def validate_params(params: Dict) -> Dict:
        """
        Validate and clean input parameters.
        
        Args:
            params: Dictionary of model parameters
            
        Returns:
            Validated parameters dictionary
            
        Raises:
            ParameterError: If parameters are invalid
        """
        if not isinstance(params, dict):
            raise ParameterError("Parameters must be a dictionary")
        
        required_keys = ["pos_x", "pos_y", "height", "width"]
        missing_keys = [key for key in required_keys if key not in params]
        if missing_keys:
            raise ParameterError(
                f"Missing required parameters: {missing_keys}",
                suggestion="Please provide all required parameters: pos_x, pos_y, height, width"
            )
        
        # Validate shapes
        pos_x = params["pos_x"]
        pos_y = params["pos_y"]
        height = params["height"]
        
        if keras.ops.shape(pos_x)[0] != keras.ops.shape(pos_y)[0]:
            raise ParameterError("pos_x and pos_y must have same length")
        
        if keras.ops.shape(pos_x)[0] != keras.ops.shape(height)[0]:
            raise ParameterError("pos_x and height must have same length")
        
        # Check for NaN or inf values
        for key in ["pos_x", "pos_y", "height", "width"]:
            values = safe_convert_to_numpy(params[key])
            if np.any(np.isnan(values)) or np.any(np.isinf(values)):
                raise ParameterError(f"Parameter '{key}' contains NaN or infinite values")
        
        return params


class DesignMatrixBuilder:
    """Builds design matrix for linear estimation."""
    
    def __init__(self, model, nx: int, ny: int):
        self.model = model
        self.nx = nx
        self.ny = ny
    
    def build_local_peaks(self, params: Dict, same_width: bool, atom_types: np.ndarray) -> Tuple:
        """
        Build local peak representations.
        
        Args:
            params: Model parameters
            same_width: Whether to use same width for all peaks
            atom_types: Array of atom type indices
            
        Returns:
            Tuple of (peak_local, global_x, global_y, mask)
        """
        pos_x = params["pos_x"]
        pos_y = params["pos_y"]
        width = params["width"]
        height = params["height"]
        ratio = params.get("ratio", None)
        
        if same_width:
            width = width[atom_types]
            if ratio is not None:
                ratio = ratio[atom_types]
        
        # Create local coordinate system
        window_size = keras.ops.cast(keras.ops.max(width) * 5, dtype="int32")
        x = keras.ops.arange(-window_size, window_size + 1, 1, dtype="float32")
        y = keras.ops.arange(-window_size, window_size + 1, 1, dtype="float32")
        local_x, local_y = keras.ops.meshgrid(x, y, indexing="xy")
        
        # Prepare model input parameters
        input_params = (keras.ops.mod(pos_x, 1), keras.ops.mod(pos_y, 1), height, width)
        if ratio is not None:
            input_params += (ratio,)
        
        # Generate local peaks
        peak_local = self.model.model_fn(
            local_x[..., None], local_y[..., None], *input_params
        )
        
        # Calculate global coordinates
        pos_x_int = keras.ops.floor(pos_x)
        pos_y_int = keras.ops.floor(pos_y)
        
        global_x = keras.ops.expand_dims(local_x, -1) + pos_x_int
        global_y = keras.ops.expand_dims(local_y, -1) + pos_y_int
        
        # Create boundary mask
        mask = (
            (global_x >= 0) & (global_x < self.nx) &
            (global_y >= 0) & (global_y < self.ny)
        )
        
        return peak_local, global_x, global_y, mask
    
    def build_sparse_matrix(self, peak_local, global_x, global_y, mask, 
                          fit_background: bool, num_coordinates: int, x_grid, y_grid) -> coo_matrix:
        """
        Build sparse design matrix from peak data.
        
        Args:
            peak_local: Local peak representations
            global_x, global_y: Global coordinate arrays
            mask: Boundary mask
            fit_background: Whether to fit background
            num_coordinates: Number of atomic coordinates
            x_grid, y_grid: Image coordinate grids
            
        Returns:
            Sparse design matrix
        """
        # Get valid indices
        valid_indices = keras.ops.where(mask)
        
        # Calculate flat indices for data extraction
        shape = keras.ops.shape(peak_local)
        flat_indices = (
            valid_indices[0] * (shape[1] * shape[2])
            + valid_indices[1] * shape[2]
            + valid_indices[2]
        )
        
        # Extract valid data
        data_tensor = keras.ops.take(keras.ops.reshape(peak_local, (-1,)), flat_indices)
        global_x_valid = keras.ops.take(keras.ops.reshape(global_x, (-1,)), flat_indices)
        global_y_valid = keras.ops.take(keras.ops.reshape(global_y, (-1,)), flat_indices)
        
        # Calculate matrix indices
        cols_tensor = valid_indices[2]
        rows_tensor = keras.ops.cast(global_y_valid, "int32") * self.nx + keras.ops.cast(
            global_x_valid, "int32"
        )
        
        # Add background terms if needed
        if fit_background:
            background_rows = keras.ops.reshape(y_grid, (-1,)) * self.nx + keras.ops.reshape(x_grid, (-1,))
            rows_tensor = keras.ops.concatenate([rows_tensor, keras.ops.cast(background_rows, "int32")])
            cols_tensor = keras.ops.concatenate([
                cols_tensor,
                keras.ops.full((self.nx * self.ny,), num_coordinates, dtype="int32")
            ])
            data_tensor = keras.ops.concatenate([data_tensor, keras.ops.ones((self.nx * self.ny,), dtype="float32")])
            shape = (self.nx * self.ny, num_coordinates + 1)
        else:
            shape = (self.nx * self.ny, num_coordinates)
        
        backend = keras.backend.backend()
        if backend == "torch":
            import torch

            def build_sparse_matrix_torch(data, row, col, shape, device="cuda"):
                # PyTorch sparse COO tensor
                indices = torch.stack([row, col])
                values = data
                sparse_mat = torch.sparse_coo_tensor(indices, values, size=shape, device=device)
                return sparse_mat
        
            design_matrix = build_sparse_matrix_torch(data_tensor, rows_tensor, cols_tensor, shape)
        elif backend == "tensorflow":
            import tensorflow as tf
            def build_sparse_matrix_tf(data, row, col, shape):
                # TensorFlow SparseTensor
                indices = tf.stack([row, col], axis=1)
                sparse_mat = tf.sparse.SparseTensor(indices, data, dense_shape=shape)
                return sparse_mat
            
            design_matrix = build_sparse_matrix_tf(data_tensor, rows_tensor, cols_tensor, shape)
        elif backend == "jax":
            import jax
            import jax.numpy as jnp
            from jax.experimental import sparse as jsparse

            def build_sparse_matrix_jax(data, row, col, shape):
                # JAX sparse COO matrix
                mat = jsparse.BCOO((data, jnp.stack([row, col])), shape=shape)
                return mat     
                   
            design_matrix = build_sparse_matrix_jax(data_tensor, rows_tensor, cols_tensor, shape)
        else:
            # Build sparse matrix
            design_matrix = coo_matrix(
                (
                    safe_convert_to_numpy(data_tensor),
                    (safe_convert_to_numpy(rows_tensor), safe_convert_to_numpy(cols_tensor))
                ),
                shape=shape
            )
            
        return design_matrix



class LinearSystemSolver:
    """Solves linear systems with robust error handling."""
    
    @staticmethod
    def solve_system(design_matrix: coo_matrix, target: np.ndarray, 
                    non_negative: bool = False) -> Optional[np.ndarray]:
        """
        Solve linear system with robust error handling.
        
        Args:
            design_matrix: Sparse design matrix
            target: Target vector
            non_negative: Whether to enforce non-negative constraints
            
        Returns:
            Solution vector or None if solving fails
            
        Raises:
            DataError: If system cannot be solved
        """
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            
            # Check for singular matrix explicitly
            AtA = design_matrix.T @ design_matrix
            
            # Use normal equations for regular case
            Atb = design_matrix.T @ target
            solution = keras.ops.lstsq(AtA.cpu().to_dense(), Atb.cpu().to_dense())

            # Check for singular matrix warnings
            if w and any("singular matrix" in str(warning.message).lower() for warning in w):
                raise DataError(
                    "Singular matrix encountered. Peak positions may need refinement."
                )
            
            return solution
                


class SolutionProcessor:
    """Processes and validates linear system solutions."""
    
    @staticmethod
    def validate_solution(solution: np.ndarray) -> bool:
        """
        Validate solution for common issues.
        
        Args:
            solution: Solution array
            
        Returns:
            True if solution is valid, False otherwise
        """
        if solution is None:
            return False
        
        # Check for NaN or infinite values
        if keras.ops.any(keras.ops.isnan(solution)) or keras.ops.any(keras.ops.isinf(solution)):
            logging.warning("Solution contains NaN or infinite values")
            return False
        
        return True
    
    @staticmethod
    def process_height_scaling(height_scale: np.ndarray, 
                             min_scale: float = 0.8, 
                             max_scale: float = 1.2) -> np.ndarray:
        """
        Process and constrain height scaling factors.
        
        Args:
            height_scale: Raw height scaling factors
            min_scale: Minimum allowed scale factor
            max_scale: Maximum allowed scale factor
            
        Returns:
            Processed height scaling factors
        """        
        # Count out-of-bounds values for logging
        too_small = keras.ops.sum(height_scale < min_scale)
        too_large = keras.ops.sum(height_scale > max_scale)
        
        # Apply constraints
        height_scale = keras.ops.clip(height_scale, min_scale, max_scale)
        
        # Log warnings if constraints were applied
        if too_small > 0:
            logging.warning(
                f"Clipped {too_small} height scale values below {min_scale:.2f}. "
                "Consider improving peak initialization."
            )
        
        if too_large > 0:
            logging.warning(
                f"Clipped {too_large} height scale values above {max_scale:.2f}. "
                "Linear estimation may be inaccurate."
            )
        
        # Warn if too many values were clipped
        total_clipped = too_small + too_large
        if total_clipped > len(height_scale) * 0.3:
            logging.warning(
                f"Over 30% of height values were clipped ({total_clipped}/{len(height_scale)}). "
                "Consider refining peak positions or checking model parameters."
            )
        
        return height_scale