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


class MemoryEstimator:
    """Estimates memory usage for linear system operations."""
    
    @staticmethod
    def estimate_matrix_memory(num_rows: int, num_cols: int, nnz: int, dtype_size: int = 4) -> dict:
        """
        Estimate memory usage for sparse and dense matrices.
        
        Args:
            num_rows: Number of matrix rows
            num_cols: Number of matrix columns  
            nnz: Number of non-zero elements
            dtype_size: Size of data type in bytes (4 for float32)
            
        Returns:
            Dictionary with memory estimates in MB
        """
        # Sparse matrix memory (COO format: data + 2 index arrays)
        sparse_memory = nnz * (dtype_size + 2 * 4) / (1024 * 1024)  # 4 bytes for int32 indices
        
        # Dense matrix memory
        dense_memory = num_rows * num_cols * dtype_size / (1024 * 1024)
        
        # AtA matrix memory (for normal equations)
        ata_memory = num_cols * num_cols * dtype_size / (1024 * 1024)
        
        return {
            'sparse_mb': sparse_memory,
            'dense_mb': dense_memory,
            'ata_mb': ata_memory,
            'sparsity': nnz / (num_rows * num_cols) if num_rows * num_cols > 0 else 0
        }
    
    @staticmethod
    def choose_solver_strategy(memory_estimate: dict, available_memory_mb: float = 1000) -> str:
        """
        Choose optimal solver strategy based on memory constraints.
        
        Args:
            memory_estimate: Memory estimate dictionary
            available_memory_mb: Available GPU/system memory in MB
            
        Returns:
            Recommended strategy: 'sparse', 'chunked', or 'iterative'
        """
        if memory_estimate['ata_mb'] > available_memory_mb * 0.8:
            return 'iterative'  # Use iterative solver
        elif memory_estimate['sparse_mb'] > available_memory_mb * 0.5:
            return 'chunked'    # Use chunked processing
        else:
            return 'sparse'     # Use direct sparse solver


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
                          fit_background: bool, num_coordinates: int, x_grid, y_grid, 
                          chunk_size: int = 10000):
        """
        Build sparse design matrix from peak data with memory optimization.
        
        Args:
            peak_local: Local peak representations
            global_x, global_y: Global coordinate arrays
            mask: Boundary mask
            fit_background: Whether to fit background
            num_coordinates: Number of atomic coordinates
            x_grid, y_grid: Image coordinate grids
            chunk_size: Size of chunks for memory-efficient processing
            
        Returns:
            Sparse design matrix
        """        
        # Process in chunks to avoid memory issues
        valid_indices = keras.ops.where(mask)
        total_valid = keras.ops.shape(valid_indices[1])[0]
        
        if total_valid > chunk_size:
            return self._build_sparse_matrix_chunked(
                peak_local, global_x, global_y, mask, fit_background, 
                num_coordinates, x_grid, y_grid, chunk_size
            )
        
        # Original implementation for smaller matrices
        return self._build_sparse_matrix_direct(
            peak_local, global_x, global_y, mask, fit_background,
            num_coordinates, x_grid, y_grid
        )
    
    def _build_sparse_matrix_chunked(self, peak_local, global_x, global_y, mask,
                                   fit_background: bool, num_coordinates: int, 
                                   x_grid, y_grid, chunk_size: int):
        """Build sparse matrix in chunks to reduce memory usage."""
        backend = keras.backend.backend()
        valid_indices = keras.ops.where(mask)
        total_valid = keras.ops.shape(valid_indices[1])[0]
        
        # Collect all chunks
        all_data = []
        all_rows = []
        all_cols = []
        
        for start_idx in range(0, total_valid, chunk_size):
            end_idx = min(start_idx + chunk_size, total_valid)
            
            # Extract chunk indices
            chunk_indices = [
                valid_indices[0][start_idx:end_idx],
                valid_indices[1][start_idx:end_idx], 
                valid_indices[2][start_idx:end_idx]
            ]
            
            # Process chunk
            chunk_data, chunk_rows, chunk_cols = self._process_chunk(
                peak_local, global_x, global_y, chunk_indices
            )
            
            all_data.append(chunk_data)
            all_rows.append(chunk_rows)
            all_cols.append(chunk_cols)
        
        # Concatenate all chunks
        data_tensor = keras.ops.concatenate(all_data)
        rows_tensor = keras.ops.concatenate(all_rows)
        cols_tensor = keras.ops.concatenate(all_cols)
        
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
        
        return self._create_backend_sparse_matrix(data_tensor, rows_tensor, cols_tensor, shape)
    
    def _build_sparse_matrix_direct(self, peak_local, global_x, global_y, mask,
                                  fit_background: bool, num_coordinates: int, x_grid, y_grid):
        """Direct sparse matrix building for smaller matrices."""
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
        
        return self._create_backend_sparse_matrix(data_tensor, rows_tensor, cols_tensor, shape)
    
    def _process_chunk(self, peak_local, global_x, global_y, chunk_indices):
        """Process a single chunk of indices."""
        # Calculate flat indices for this chunk
        shape = keras.ops.shape(peak_local)
        flat_indices = (
            chunk_indices[0] * (shape[1] * shape[2])
            + chunk_indices[1] * shape[2]
            + chunk_indices[2]
        )
        
        # Extract valid data for this chunk
        data_tensor = keras.ops.take(keras.ops.reshape(peak_local, (-1,)), flat_indices)
        global_x_valid = keras.ops.take(keras.ops.reshape(global_x, (-1,)), flat_indices)
        global_y_valid = keras.ops.take(keras.ops.reshape(global_y, (-1,)), flat_indices)
        
        # Calculate matrix indices
        cols_tensor = chunk_indices[2]
        rows_tensor = keras.ops.cast(global_y_valid, "int32") * self.nx + keras.ops.cast(
            global_x_valid, "int32"
        )
        
        return data_tensor, rows_tensor, cols_tensor
    
    def _create_backend_sparse_matrix(self, data_tensor, rows_tensor, cols_tensor, shape):
        """Create sparse matrix in the appropriate backend format."""
        backend = keras.backend.backend()
        
        if backend == "torch":
            import torch
            # PyTorch sparse COO tensor
            indices = torch.stack([rows_tensor, cols_tensor])
            values = data_tensor
            sparse_mat = torch.sparse_coo_tensor(indices, values, size=shape)
            return sparse_mat.coalesce()  # Combine duplicate entries
            
        elif backend == "tensorflow":
            import tensorflow as tf
            # TensorFlow SparseTensor
            indices = tf.stack([rows_tensor, cols_tensor], axis=1)
            sparse_mat = tf.sparse.SparseTensor(indices, data_tensor, dense_shape=shape)
            return tf.sparse.reorder(sparse_mat)  # Ensure canonical ordering
            
        elif backend == "jax":
            import jax.numpy as jnp
            from jax.experimental import sparse as jsparse
            # JAX sparse COO matrix
            mat = jsparse.BCOO((data_tensor, jnp.stack([rows_tensor, cols_tensor])), shape=shape)
            return mat
            
        else:
            # SciPy sparse matrix
            sparse_mat = coo_matrix(
                (
                    safe_convert_to_numpy(data_tensor),
                    (safe_convert_to_numpy(rows_tensor), safe_convert_to_numpy(cols_tensor))
                ),
                shape=shape
            )
            return sparse_mat



class LinearSystemSolver:
    """Solves linear systems with robust error handling."""
    
    @staticmethod
    def solve_system(design_matrix, target: np.ndarray, 
                    non_negative: bool = False) -> Optional[np.ndarray]:
        """
        Solve linear system with robust error handling using sparse operations.
        
        Args:
            design_matrix: Sparse design matrix (backend-specific sparse format)
            target: Target vector
            non_negative: Whether to enforce non-negative constraints
            
        Returns:
            Solution vector or None if solving fails
            
        Raises:
            DataError: If system cannot be solved
        """
        backend = keras.backend.backend()
        
        try:
            if backend == "torch":
                return LinearSystemSolver._solve_torch_sparse(design_matrix, target, non_negative)
            elif backend == "tensorflow":
                return LinearSystemSolver._solve_tf_sparse(design_matrix, target, non_negative)
            elif backend == "jax":
                return LinearSystemSolver._solve_jax_sparse(design_matrix, target, non_negative)
            else:
                # Fallback to scipy sparse
                return LinearSystemSolver._solve_scipy_sparse(design_matrix, target, non_negative)
                
        except Exception as e:
            logging.error(f"Sparse linear solve failed: {str(e)}")
            raise DataError(f"Linear system solving failed: {str(e)}")
    
    @staticmethod
    def _solve_torch_sparse(design_matrix, target, non_negative: bool):
        """Solve using PyTorch sparse operations."""
        import torch
        from torch.sparse import mm
        
        # Convert target to tensor if needed
        if not isinstance(target, torch.Tensor):
            target = torch.tensor(target, device=design_matrix.device, dtype=design_matrix.dtype)
        
        # Use sparse matrix operations to avoid dense conversion
        # A^T @ A @ x = A^T @ b
        AtA = torch.sparse.mm(design_matrix.t(), design_matrix)
        Atb = torch.sparse.mm(design_matrix.t(), target.unsqueeze(1)).squeeze()
        
        # Convert to dense only for the smaller AtA matrix
        AtA_dense = AtA.to_dense()
        
        # Solve using torch.linalg.solve for better numerical stability
        try:
            solution = torch.linalg.solve(AtA_dense, Atb)
        except torch.linalg.LinAlgError:
            # Fallback to least squares if singular
            solution = torch.linalg.lstsq(AtA_dense, Atb).solution
        
        if non_negative:
            solution = torch.clamp(solution, min=0.0)
            
        return solution.detach().cpu().numpy()
    
    @staticmethod
    def _solve_tf_sparse(design_matrix, target, non_negative: bool):
        """Solve using TensorFlow sparse operations."""
        import tensorflow as tf
        
        # Convert target to tensor if needed
        if not isinstance(target, tf.Tensor):
            target = tf.convert_to_tensor(target, dtype=design_matrix.dtype)
        
        # Use sparse matrix operations
        AtA = tf.sparse.sparse_dense_matmul(
            tf.sparse.transpose(design_matrix), 
            tf.sparse.to_dense(design_matrix)
        )
        Atb = tf.sparse.sparse_dense_matmul(
            tf.sparse.transpose(design_matrix), 
            tf.expand_dims(target, 1)
        )
        Atb = tf.squeeze(Atb, axis=1)
        
        # Solve the system
        try:
            solution = tf.linalg.solve(AtA, Atb)
        except tf.errors.InvalidArgumentError:
            # Fallback to least squares
            solution = tf.linalg.lstsq(AtA, Atb)
        
        if non_negative:
            solution = tf.maximum(solution, 0.0)
            
        return solution.numpy()
    
    @staticmethod
    def _solve_jax_sparse(design_matrix, target, non_negative: bool):
        """Solve using JAX sparse operations."""
        import jax.numpy as jnp
        from jax.experimental import sparse as jsparse
        
        # Convert target to jax array if needed
        if not isinstance(target, jnp.ndarray):
            target = jnp.array(target)
        
        # Use sparse operations
        AtA = design_matrix.T @ design_matrix
        Atb = design_matrix.T @ target
        
        # Convert to dense for solving (AtA is much smaller than original matrix)
        AtA_dense = AtA.todense()
        
        # Solve using JAX
        try:
            solution = jnp.linalg.solve(AtA_dense, Atb)
        except jnp.linalg.LinAlgError:
            solution = jnp.linalg.lstsq(AtA_dense, Atb)[0]
        
        if non_negative:
            solution = jnp.maximum(solution, 0.0)
            
        return np.array(solution)
    
    @staticmethod
    def _solve_scipy_sparse(design_matrix: coo_matrix, target: np.ndarray, non_negative: bool):
        """Solve using SciPy sparse operations."""
        from scipy.sparse.linalg import spsolve
        from scipy.sparse import csc_matrix
        
        # Convert to CSC format for efficient operations
        A_csc = design_matrix.tocsc()
        
        # Use sparse normal equations: A^T A x = A^T b
        AtA = A_csc.T @ A_csc
        Atb = A_csc.T @ target
        
        try:
            # Try direct sparse solve first
            solution = spsolve(AtA, Atb)
        except Exception:
            # Fallback to dense solve for the smaller AtA matrix
            AtA_dense = AtA.toarray()
            solution = np.linalg.lstsq(AtA_dense, Atb, rcond=None)[0]
        
        if non_negative:
            solution = np.maximum(solution, 0.0)
            
        return solution
    
    @staticmethod
    def solve_iterative(design_matrix, target: np.ndarray, 
                       max_iter: int = 1000, tol: float = 1e-6) -> Optional[np.ndarray]:
        """
        Solve linear system using iterative methods for very large systems.
        
        Args:
            design_matrix: Sparse design matrix
            target: Target vector
            max_iter: Maximum iterations
            tol: Convergence tolerance
            
        Returns:
            Solution vector or None if solving fails
        """
        backend = keras.backend.backend()
        
        try:
            if backend == "torch":
                return LinearSystemSolver._solve_torch_iterative(design_matrix, target, max_iter, tol)
            elif backend == "tensorflow":
                return LinearSystemSolver._solve_tf_iterative(design_matrix, target, max_iter, tol)
            elif backend == "jax":
                return LinearSystemSolver._solve_jax_iterative(design_matrix, target, max_iter, tol)
            else:
                return LinearSystemSolver._solve_scipy_iterative(design_matrix, target, max_iter, tol)
                
        except Exception as e:
            logging.error(f"Iterative solve failed: {str(e)}")
            return None
    
    @staticmethod
    def _solve_torch_iterative(design_matrix, target, max_iter: int, tol: float):
        """Iterative solve using PyTorch."""
        import torch
        
        # Convert target to tensor if needed
        if not isinstance(target, torch.Tensor):
            target = torch.tensor(target, device=design_matrix.device, dtype=design_matrix.dtype)
        
        # Initialize solution
        x = torch.zeros(design_matrix.shape[1], device=design_matrix.device, dtype=design_matrix.dtype)
        
        # Conjugate gradient method
        r = target - torch.sparse.mm(design_matrix, x.unsqueeze(1)).squeeze()
        p = r.clone()
        rsold = torch.dot(r, r)
        
        for i in range(max_iter):
            Ap = torch.sparse.mm(design_matrix, p.unsqueeze(1)).squeeze()
            alpha = rsold / torch.dot(p, Ap)
            x = x + alpha * p
            r = r - alpha * Ap
            rsnew = torch.dot(r, r)
            
            if torch.sqrt(rsnew) < tol:
                break
                
            beta = rsnew / rsold
            p = r + beta * p
            rsold = rsnew
        
        return x.detach().cpu().numpy()
    
    @staticmethod
    def _solve_scipy_iterative(design_matrix: coo_matrix, target: np.ndarray, 
                              max_iter: int, tol: float):
        """Iterative solve using SciPy."""
        from scipy.sparse.linalg import cg, lsqr
        
        # Convert to CSR for efficient operations
        A_csr = design_matrix.tocsr()
        
        # Try conjugate gradient first (for symmetric positive definite AtA)
        try:
            AtA = A_csr.T @ A_csr
            Atb = A_csr.T @ target
            solution, info = cg(AtA, Atb, maxiter=max_iter, tol=tol)
            
            if info == 0:  # Successful convergence
                return solution
        except Exception:
            pass
        
        # Fallback to LSQR for general least squares
        try:
            solution = lsqr(A_csr, target, iter_lim=max_iter, atol=tol, btol=tol)[0]
            return solution
        except Exception as e:
            logging.error(f"LSQR failed: {str(e)}")
            return None
    
    @staticmethod
    def _solve_tf_iterative(design_matrix, target, max_iter: int, tol: float):
        """Iterative solve using TensorFlow (simplified CG)."""
        import tensorflow as tf
        
        # Convert target to tensor if needed
        if not isinstance(target, tf.Tensor):
            target = tf.convert_to_tensor(target, dtype=design_matrix.dtype)
        
        # Simple gradient descent approach
        x = tf.Variable(tf.zeros([design_matrix.dense_shape[1]], dtype=design_matrix.dtype))
        
        for i in range(max_iter):
            with tf.GradientTape() as tape:
                pred = tf.sparse.sparse_dense_matmul(design_matrix, tf.expand_dims(x, 1))
                pred = tf.squeeze(pred, axis=1)
                loss = tf.reduce_mean(tf.square(pred - target))
            
            gradients = tape.gradient(loss, [x])
            x.assign_sub(0.001 * gradients[0])  # Simple gradient step
            
            if loss < tol:
                break
        
        return x.numpy()
    
    @staticmethod
    def _solve_jax_iterative(design_matrix, target, max_iter: int, tol: float):
        """Iterative solve using JAX."""
        import jax.numpy as jnp
        from jax import jit
        
        # Convert target to jax array if needed
        if not isinstance(target, jnp.ndarray):
            target = jnp.array(target)
        
        # Simple conjugate gradient
        @jit
        def cg_step(x, r, p, rsold):
            Ap = design_matrix @ p
            alpha = rsold / jnp.dot(p, Ap)
            x_new = x + alpha * p
            r_new = r - alpha * Ap
            rsnew = jnp.dot(r_new, r_new)
            beta = rsnew / rsold
            p_new = r_new + beta * p
            return x_new, r_new, p_new, rsnew
        
        # Initialize
        x = jnp.zeros(design_matrix.shape[1])
        r = target - design_matrix @ x
        p = r
        rsold = jnp.dot(r, r)
        
        for i in range(max_iter):
            x, r, p, rsold = cg_step(x, r, p, rsold)
            if jnp.sqrt(rsold) < tol:
                break
        
        return np.array(x)
                


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

        # replace nan with 1
        height_scale = keras.ops.where(keras.ops.isnan(height_scale), keras.ops.ones_like(height_scale), height_scale)
        
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
                f"Over {total_clipped/len(height_scale)*100:.2f}% of height values were clipped ({total_clipped}/{len(height_scale)}). "
                "Consider refining peak positions or checking model parameters."
            )
        
        return height_scale

    def process_background(self, solution, params, init_background, update_threshold=0.2):
        """
        Process and update the background parameter based on the solution.
        Returns the new background and a flag indicating if the update is valid.
        """
        background = max(solution[-1], init_background)
        prev_background = params["background"]
        update_rel = (background - prev_background) / (prev_background + 1e-10)
        if keras.ops.abs(update_rel) > update_threshold * 2:
            # Update too large, skip update
            return prev_background, False
        if keras.ops.abs(update_rel) > update_threshold:
            update_rel_clip = keras.ops.clip(update_rel, -update_threshold, update_threshold)
            background = prev_background * (1 + update_rel_clip)
        return background, True