"""
Linear solver module for QEM image fitting.
Provides memory-efficient sparse linear system solving with automatic strategy selection.
"""

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, Optional, Tuple, Union, Protocol

import numpy as np
import keras
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import spsolve, lsqr, cg

from qem.utils.params import safe_convert_to_numpy, safe_convert_to_tensor
from qem.schema.exceptions import ParameterError, DataError, ValidationError


@dataclass
class MemoryInfo:
    """Memory information for a computing backend."""
    total_mb: float
    allocated_mb: float
    free_mb: float
    backend: str


class BackendSolver(Protocol):
    """Protocol for backend-specific sparse linear solvers."""
    
    def solve_direct(self, A, b: np.ndarray, non_negative: bool = False) -> np.ndarray:
        """Solve using direct method (computes AtA)."""
        ...
    
    def solve_iterative(self, A, b: np.ndarray, non_negative: bool = False, 
                       max_iter: int = 1000, tol: float = 1e-6) -> np.ndarray:
        """Solve using iterative method (avoids AtA computation)."""
        ...


class TorchSolver:
    """PyTorch-specific sparse linear solver with MPS fallback."""
    
    @staticmethod
    def _is_mps_device(device):
        """Check if device is MPS (Apple Silicon) or if MPS is available."""
        import torch
        # Check if device is explicitly MPS or if MPS is available (indicating Apple Silicon)
        return (hasattr(torch.backends, 'mps') and 
                (str(device).startswith('mps') or torch.backends.mps.is_available()))
    
    @staticmethod
    def _convert_to_scipy_matrix(A):
        """Convert PyTorch tensor to scipy sparse matrix."""
        import torch
        from scipy.sparse import coo_matrix
        
        if hasattr(A, 'tocsr'):
            # Already a scipy sparse matrix
            return A
        elif hasattr(A, 'is_sparse') and A.is_sparse:
            # PyTorch sparse tensor
            A_coo = A.coalesce()
            indices = A_coo.indices().cpu().numpy()
            values = A_coo.values().cpu().numpy()
            shape = A_coo.shape
            return coo_matrix((values, (indices[0], indices[1])), shape=shape)
        elif hasattr(A, 'cpu'):
            # PyTorch dense tensor
            return coo_matrix(A.cpu().numpy())
        else:
            # Assume it's already numpy or scipy
            return coo_matrix(A)
    
    @staticmethod
    def solve_direct(A, b: np.ndarray, non_negative: bool = False) -> np.ndarray:
        import torch
        
        # Check for MPS device and fallback to scipy if needed
        device = getattr(A, 'device', torch.device('cpu'))
        if TorchSolver._is_mps_device(device):
            logging.warning("MPS backend detected, falling back to scipy sparse solver for compatibility")
            A_scipy = TorchSolver._convert_to_scipy_matrix(A)
            b_numpy = b.cpu().numpy() if hasattr(b, 'cpu') else np.asarray(b)
            return SciPySolver.solve_direct(A_scipy, b_numpy, non_negative)
        
        try:
            if not isinstance(b, torch.Tensor):
                b = torch.tensor(b, device=A.device, dtype=A.dtype)
            
            AtA = torch.sparse.mm(A.t(), A).to_dense()
            Atb = torch.sparse.mm(A.t(), b.unsqueeze(1)).squeeze()
            
            # Regularization for stability
            reg = 1e-8 * torch.trace(AtA) / AtA.shape[0]
            AtA += reg * torch.eye(AtA.shape[0], device=AtA.device, dtype=AtA.dtype)
            
            try:
                solution = torch.linalg.solve(AtA, Atb)
            except Exception:
                solution = torch.linalg.lstsq(AtA, Atb).solution
            
            if non_negative:
                solution = torch.clamp(solution, min=0.0)
            return solution.detach().cpu().numpy()
            
        except (RuntimeError, torch.cuda.OutOfMemoryError) as e:
            error_msg = str(e)
            
            # Check for MPS-specific errors
            if any(keyword in error_msg for keyword in ["SparseMPS", "_sparse_coo_tensor_with_dims_and_tensors", "aten::addmm", "SparseCsrMPS"]):
                logging.warning(f"PyTorch sparse operation failed on MPS: {e}")
                logging.info("Falling back to scipy sparse solver")
                A_scipy = TorchSolver._convert_to_scipy_matrix(A)
                b_numpy = b.cpu().numpy() if hasattr(b, 'cpu') else np.asarray(b)
                return SciPySolver.solve_direct(A_scipy, b_numpy, non_negative)
            
            # Check for memory errors
            elif any(keyword in error_msg.lower() for keyword in ["out of memory", "cuda out of memory", "mps out of memory", "memory"]):
                logging.warning(f"PyTorch out of memory: {e}")
                logging.info("Falling back to memory-efficient scipy sparse solver")
                A_scipy = TorchSolver._convert_to_scipy_matrix(A)
                b_numpy = b.cpu().numpy() if hasattr(b, 'cpu') else np.asarray(b)
                return SciPySolver.solve_direct(A_scipy, b_numpy, non_negative)
            
            else:
                raise
        except MemoryError as e:
            logging.warning(f"System memory error in PyTorch: {e}")
            logging.info("Falling back to memory-efficient scipy sparse solver")
            A_scipy = TorchSolver._convert_to_scipy_matrix(A)
            b_numpy = b.cpu().numpy() if hasattr(b, 'cpu') else np.asarray(b)
            return SciPySolver.solve_direct(A_scipy, b_numpy, non_negative)
    
    @staticmethod
    def solve_iterative(A, b: np.ndarray, non_negative: bool = False, 
                       max_iter: int = 1000, tol: float = 1e-6) -> np.ndarray:
        import torch
        
        # Check for MPS device and fallback to scipy if needed
        device = getattr(A, 'device', torch.device('cpu'))
        if TorchSolver._is_mps_device(device):
            logging.warning("MPS backend detected, falling back to scipy sparse solver for compatibility")
            A_scipy = TorchSolver._convert_to_scipy_matrix(A)
            b_numpy = b.cpu().numpy() if hasattr(b, 'cpu') else np.asarray(b)
            return SciPySolver.solve_iterative(A_scipy, b_numpy, non_negative, max_iter, tol)
        
        try:
            if not isinstance(b, torch.Tensor):
                b = torch.tensor(b, device=A.device, dtype=A.dtype)
            
            # Conjugate gradient for A^T A x = A^T b
            x = torch.zeros(A.shape[1], device=A.device, dtype=A.dtype)
            r = torch.sparse.mm(A.t(), b.unsqueeze(1)).squeeze()  # A^T b
            p = r.clone()
            rsold = torch.dot(r, r)
            
            for i in range(max_iter):
                Ap = torch.sparse.mm(A, p.unsqueeze(1)).squeeze()
                AtAp = torch.sparse.mm(A.t(), Ap.unsqueeze(1)).squeeze()
                
                alpha = rsold / torch.dot(p, AtAp)
                x += alpha * p
                r -= alpha * AtAp
                rsnew = torch.dot(r, r)
                
                if torch.sqrt(rsnew) < tol:
                    break
                
                p = r + (rsnew / rsold) * p
                rsold = rsnew
            
            if non_negative:
                x = torch.clamp(x, min=0.0)
            return x.detach().cpu().numpy()
            
        except (RuntimeError, torch.cuda.OutOfMemoryError) as e:
            error_msg = str(e)
            
            # Check for MPS-specific errors
            if any(keyword in error_msg for keyword in ["SparseMPS", "_sparse_coo_tensor_with_dims_and_tensors", "aten::addmm", "SparseCsrMPS"]):
                logging.warning(f"PyTorch sparse operation failed on MPS: {e}")
                logging.info("Falling back to scipy sparse solver")
                A_scipy = TorchSolver._convert_to_scipy_matrix(A)
                b_numpy = b.cpu().numpy() if hasattr(b, 'cpu') else np.asarray(b)
                return SciPySolver.solve_iterative(A_scipy, b_numpy, non_negative, max_iter, tol)
            
            # Check for memory errors
            elif any(keyword in error_msg.lower() for keyword in ["out of memory", "cuda out of memory", "mps out of memory", "memory"]):
                logging.warning(f"PyTorch out of memory: {e}")
                logging.info("Falling back to memory-efficient scipy sparse solver")
                A_scipy = TorchSolver._convert_to_scipy_matrix(A)
                b_numpy = b.cpu().numpy() if hasattr(b, 'cpu') else np.asarray(b)
                return SciPySolver.solve_iterative(A_scipy, b_numpy, non_negative, max_iter, tol)
            
            else:
                raise
        except MemoryError as e:
            logging.warning(f"System memory error in PyTorch: {e}")
            logging.info("Fallback to memory-efficient scipy sparse solver")
            A_scipy = TorchSolver._convert_to_scipy_matrix(A)
            b_numpy = b.cpu().numpy() if hasattr(b, 'cpu') else np.asarray(b)
            return SciPySolver.solve_iterative(A_scipy, b_numpy, non_negative, max_iter, tol)


class TensorFlowSolver:
    """TensorFlow-specific sparse linear solver."""
    
    @staticmethod
    def solve_direct(A, b: np.ndarray, non_negative: bool = False) -> np.ndarray:
        import tensorflow as tf
        if not isinstance(b, tf.Tensor):
            b = tf.convert_to_tensor(b, dtype=A.dtype)
        
        AtA = tf.sparse.sparse_dense_matmul(tf.sparse.transpose(A), tf.sparse.to_dense(A))
        Atb = tf.sparse.sparse_dense_matmul(tf.sparse.transpose(A), tf.expand_dims(b, 1))
        Atb = tf.squeeze(Atb, axis=1)
        
        reg = 1e-8 * tf.linalg.trace(AtA) / tf.cast(tf.shape(AtA)[0], AtA.dtype)
        AtA += reg * tf.eye(tf.shape(AtA)[0], dtype=AtA.dtype)
        
        try:
            solution = tf.linalg.solve(AtA, Atb)
        except Exception:
            solution = tf.linalg.lstsq(AtA, Atb)
        
        if non_negative:
            solution = tf.maximum(solution, 0.0)
        return solution.numpy()
    
    @staticmethod
    def solve_iterative(A, b: np.ndarray, non_negative: bool = False, 
                       max_iter: int = 1000, tol: float = 1e-6) -> np.ndarray:
        import tensorflow as tf
        if not isinstance(b, tf.Tensor):
            b = tf.convert_to_tensor(b, dtype=A.dtype)
        
        x = tf.Variable(tf.zeros([A.dense_shape[1]], dtype=A.dtype))
        r = tf.sparse.sparse_dense_matmul(tf.sparse.transpose(A), tf.expand_dims(b, 1))
        r = tf.squeeze(r, axis=1)
        p = tf.Variable(r)
        rsold = tf.reduce_sum(r * r)
        
        for i in range(max_iter):
            Ap = tf.sparse.sparse_dense_matmul(A, tf.expand_dims(p, 1))
            AtAp = tf.sparse.sparse_dense_matmul(tf.sparse.transpose(A), Ap)
            AtAp = tf.squeeze(AtAp, axis=1)
            
            alpha = rsold / tf.reduce_sum(p * AtAp)
            x.assign_add(alpha * p)
            r.assign_sub(alpha * AtAp)
            rsnew = tf.reduce_sum(r * r)
            
            if tf.sqrt(rsnew) < tol:
                break
            
            p.assign(r + (rsnew / rsold) * p)
            rsold = rsnew
        
        if non_negative:
            x.assign(tf.maximum(x, 0.0))
        return x.numpy()


class JAXSolver:
    """JAX-specific sparse linear solver."""
    
    @staticmethod
    def solve_direct(A, b: np.ndarray, non_negative: bool = False) -> np.ndarray:
        import jax.numpy as jnp
        if not isinstance(b, jnp.ndarray):
            b = jnp.array(b)
        
        AtA = (A.T @ A).todense()
        Atb = A.T @ b
        
        reg = 1e-8 * jnp.trace(AtA) / AtA.shape[0]
        AtA += reg * jnp.eye(AtA.shape[0])
        
        try:
            solution = jnp.linalg.solve(AtA, Atb)
        except Exception:
            solution = jnp.linalg.lstsq(AtA, Atb)[0]
        
        if non_negative:
            solution = jnp.maximum(solution, 0.0)
        return np.array(solution)
    
    @staticmethod
    def solve_iterative(A, b: np.ndarray, non_negative: bool = False, 
                       max_iter: int = 1000, tol: float = 1e-6) -> np.ndarray:
        import jax.numpy as jnp
        from jax import jit
        
        if not isinstance(b, jnp.ndarray):
            b = jnp.array(b)
        
        @jit
        def cg_step(x, r, p, rsold):
            Ap = A @ p
            AtAp = A.T @ Ap
            alpha = rsold / jnp.dot(p, AtAp)
            x_new = x + alpha * p
            r_new = r - alpha * AtAp
            rsnew = jnp.dot(r_new, r_new)
            beta = rsnew / rsold
            p_new = r_new + beta * p
            return x_new, r_new, p_new, rsnew
        
        x = jnp.zeros(A.shape[1])
        r = A.T @ b
        p = r
        rsold = jnp.dot(r, r)
        
        for i in range(max_iter):
            x, r, p, rsold = cg_step(x, r, p, rsold)
            if jnp.sqrt(rsold) < tol:
                break
        
        if non_negative:
            x = jnp.maximum(x, 0.0)
        return np.array(x)


class SciPySolver:
    """SciPy-specific sparse linear solver."""
    
    @staticmethod
    def solve_direct(A: coo_matrix, b: np.ndarray, non_negative: bool = False) -> np.ndarray:
        A_csr = A.tocsr()
        AtA = A_csr.T @ A_csr
        Atb = A_csr.T @ b
        
        try:
            solution = spsolve(AtA, Atb)
        except Exception:
            AtA_dense = AtA.toarray()
            solution = np.linalg.lstsq(AtA_dense, Atb, rcond=None)[0]
        
        if non_negative:
            solution = np.maximum(solution, 0.0)
        return solution
    
    @staticmethod
    def solve_iterative(A: coo_matrix, b: np.ndarray, non_negative: bool = False, 
                       max_iter: int = 1000, tol: float = 1e-6) -> np.ndarray:
        A_csr = A.tocsr()
        
        try:
            solution = lsqr(A_csr, b, iter_lim=max_iter, atol=tol, btol=tol)[0]
        except Exception:
            # Fallback to CG on normal equations
            Atb = A_csr.T @ b
            from scipy.sparse.linalg import LinearOperator
            
            def matvec(x):
                return A_csr.T @ (A_csr @ x)
            
            AtA_op = LinearOperator((A_csr.shape[1], A_csr.shape[1]), matvec=matvec)
            solution, _ = cg(AtA_op, Atb, maxiter=max_iter, tol=tol)
        
        if non_negative:
            solution = np.maximum(solution, 0.0)
        return solution


class LinearSystemSolver:
    """
    Memory-aware sparse linear system solver with automatic backend detection.
    
    This class provides a unified interface for solving sparse linear systems
    across different backends (PyTorch, TensorFlow, JAX, SciPy) with automatic
    memory management and strategy selection.
    """
    
    # Backend-specific thresholds for memory usage (fraction of available memory)
    MEMORY_THRESHOLDS = {
        'torch': 0.4,      # 40% of available memory
        'tensorflow': 0.25, # 25% (more conservative)
        'jax': 0.2,        # 20% (most conservative)  
        'numpy': 0.5       # 50% for CPU
    }
    
    # Backend solver registry
    SOLVERS = {
        'torch': TorchSolver,
        'tensorflow': TensorFlowSolver,
        'jax': JAXSolver,
        'numpy': SciPySolver
    }
    
    def __init__(self):
        self.backend = keras.backend.backend()
        self.solver = self.SOLVERS.get(self.backend, SciPySolver)
    
    def get_memory_info(self, device=None) -> MemoryInfo:
        """Get available memory information for the current backend."""
        if self.backend == "torch" and device is not None:
            import torch
            if device.type == 'cuda':
                try:
                    total = torch.cuda.get_device_properties(device).total_memory
                    allocated = torch.cuda.memory_allocated(device)
                    reserved = torch.cuda.memory_reserved(device)
                    return MemoryInfo(
                        total_mb=total / (1024 * 1024),
                        allocated_mb=allocated / (1024 * 1024),
                        free_mb=(total - reserved) / (1024 * 1024),
                        backend=self.backend
                    )
                except Exception:
                    pass
        
        # Conservative fallbacks
        return MemoryInfo(
            total_mb=4000, allocated_mb=1000, free_mb=3000, backend=self.backend
        )
    
    def choose_strategy(self, n_cols: int, device=None) -> tuple[str, MemoryInfo]:
        """
        Choose optimal solving strategy based on problem size and available memory.
        
        Args:
            n_cols: Number of columns in the design matrix
            device: Computing device (for GPU memory detection)
            
        Returns:
            Tuple of (strategy, memory_info) where strategy is 'direct' or 'iterative'
        """
        ata_memory_mb = (n_cols * n_cols * 4) / (1024 * 1024)  # float32
        memory_info = self.get_memory_info(device)
        
        threshold = self.MEMORY_THRESHOLDS.get(self.backend, 0.3)
        
        # Be more conservative if we're close to memory limits
        if memory_info.free_mb < 1000:  # Less than 1GB free
            threshold *= 0.5  # Use only half the normal threshold
            logging.warning(f"Low memory detected ({memory_info.free_mb:.1f}MB free), using conservative threshold")
        
        max_memory = memory_info.free_mb * threshold
        
        # Force iterative solver for very large problems or low memory
        if ata_memory_mb > max_memory or memory_info.free_mb < 500:
            strategy = 'iterative'
        else:
            strategy = 'direct'
        
        logging.info(
            f"AtA memory: {ata_memory_mb:.1f}MB, available: {memory_info.free_mb:.1f}MB, "
            f"max_memory: {max_memory:.1f}MB, using {strategy} solver"
        )
        
        return strategy, memory_info
    
    def solve_system(self, design_matrix, target: np.ndarray, 
                    non_negative: bool = False) -> Optional[np.ndarray]:
        """
        Solve linear system with memory-aware strategy selection.
        
        Args:
            design_matrix: Sparse design matrix (backend-specific format)
            target: Target vector
            non_negative: Whether to enforce non-negative constraints
            
        Returns:
            Solution vector or None if solving fails
            
        Raises:
            DataError: If all solving methods fail
        """
        try:
            # Convert target to numpy if it's a tensor (especially MPS tensors)
            if hasattr(target, 'cpu'):
                target = target.cpu().numpy()
            elif not isinstance(target, np.ndarray):
                target = np.asarray(target)
            
            # Get device info for memory-aware strategy selection
            device = getattr(design_matrix, 'device', None)
            n_cols = design_matrix.shape[1]
            
            # Choose strategy based on memory constraints
            strategy, memory_info = self.choose_strategy(n_cols, device)
            
            # Check if we have a scipy sparse matrix (fallback case)
            if hasattr(design_matrix, 'tocsr'):  # scipy sparse matrix
                logging.info("Using scipy sparse solver for fallback compatibility")
                if strategy == 'direct':
                    solution = SciPySolver.solve_direct(design_matrix, target, non_negative)
                else:
                    solution = SciPySolver.solve_iterative(design_matrix, target, non_negative)
            else:
                # Use backend-specific solver
                if strategy == 'direct':
                    solution = self.solver.solve_direct(design_matrix, target, non_negative)
                else:
                    solution = self.solver.solve_iterative(design_matrix, target, non_negative)
            
            return solution
            
        except Exception as e:
            error_msg = str(e)
            logging.error(f"Linear solve failed: {error_msg}")
            
            # Check if it's a memory-related error
            is_memory_error = any(keyword in error_msg.lower() for keyword in 
                                ["out of memory", "cuda out of memory", "mps out of memory", "memory", "memoryerror"])
            
            if is_memory_error:
                logging.warning("Memory error detected, forcing scipy fallback for memory efficiency")
            
            # Try iterative solver as fallback
            try:
                logging.info("Attempting iterative solver as fallback...")
                
                # Ensure target is numpy array for fallback
                fallback_target = target
                if hasattr(target, 'cpu'):
                    fallback_target = target.cpu().numpy()
                elif not isinstance(target, np.ndarray):
                    fallback_target = np.asarray(target)
                
                if hasattr(design_matrix, 'tocsr'):  # scipy sparse matrix
                    return SciPySolver.solve_iterative(design_matrix, fallback_target, non_negative)
                elif is_memory_error:
                    # Force scipy conversion for memory errors
                    logging.info("Converting to scipy for memory efficiency...")
                    if hasattr(design_matrix, 'coalesce'):  # PyTorch sparse
                        A_coo = design_matrix.coalesce()
                        indices = A_coo.indices().cpu().numpy()
                        values = A_coo.values().cpu().numpy()
                        shape = A_coo.shape
                        scipy_matrix = coo_matrix((values, (indices[0], indices[1])), shape=shape)
                        return SciPySolver.solve_iterative(scipy_matrix, fallback_target, non_negative)
                else:
                    return self.solver.solve_iterative(design_matrix, fallback_target, non_negative)
                    
            except Exception as e2:
                logging.error(f"Fallback solver also failed: {str(e2)}")
                # Final fallback: try scipy if we haven't already
                if not hasattr(design_matrix, 'tocsr'):
                    try:
                        logging.info("Final fallback: attempting scipy conversion...")
                        # Try to convert to scipy sparse matrix
                        if hasattr(design_matrix, 'coalesce'):  # PyTorch sparse
                            A_coo = design_matrix.coalesce()
                            indices = A_coo.indices().cpu().numpy()
                            values = A_coo.values().cpu().numpy()
                            shape = A_coo.shape
                            scipy_matrix = coo_matrix((values, (indices[0], indices[1])), shape=shape)
                            
                            # Ensure target is also on CPU
                            if hasattr(target, 'cpu'):
                                target_numpy = target.cpu().numpy()
                            else:
                                target_numpy = np.asarray(target)
                            
                            return SciPySolver.solve_iterative(scipy_matrix, target_numpy, non_negative)
                    except Exception as e3:
                        logging.error(f"Final fallback also failed: {str(e3)}")
                
                raise DataError(f"All linear system solving methods failed. Original error: {str(e)}")


class DesignMatrixBuilder:
    """
    Builds sparse design matrices for linear estimation.
    
    This class handles the construction of sparse design matrices from peak
    representations, including background terms and backend-specific sparse
    matrix formats.
    """
    
    def __init__(self, model, nx: int, ny: int):
        self.model = model
        self.nx = nx
        self.ny = ny
        self.backend = keras.backend.backend()
    
    def build_local_peaks(self, params: Dict, same_width: bool, atom_types: np.ndarray) -> Tuple:
        """Build local peak representations."""
        pos_x, pos_y = params["pos_x"], params["pos_y"]
        width, height = params["width"], params["height"]
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
        
        # Generate local peaks
        input_params = (keras.ops.mod(pos_x, 1), keras.ops.mod(pos_y, 1), height, width)
        if ratio is not None:
            input_params += (ratio,)
        
        peak_local = self.model.model_fn(local_x[..., None], local_y[..., None], *input_params)
        
        # Calculate global coordinates and mask
        pos_x_int, pos_y_int = keras.ops.floor(pos_x), keras.ops.floor(pos_y)
        global_x = keras.ops.expand_dims(local_x, -1) + pos_x_int
        global_y = keras.ops.expand_dims(local_y, -1) + pos_y_int
        
        mask = ((global_x >= 0) & (global_x < self.nx) & 
                (global_y >= 0) & (global_y < self.ny))
        
        return peak_local, global_x, global_y, mask
    
    def build_sparse_matrix(self, peak_local, global_x, global_y, mask, 
                          fit_background: bool, num_coordinates: int, x_grid, y_grid):
        """Build sparse design matrix from peak data."""
        # Extract valid data
        valid_indices = keras.ops.where(mask)
        shape = keras.ops.shape(peak_local)
        
        flat_indices = (valid_indices[0] * (shape[1] * shape[2]) + 
                       valid_indices[1] * shape[2] + valid_indices[2])
        
        data_tensor = keras.ops.take(keras.ops.reshape(peak_local, (-1,)), flat_indices)
        global_x_valid = keras.ops.take(keras.ops.reshape(global_x, (-1,)), flat_indices)
        global_y_valid = keras.ops.take(keras.ops.reshape(global_y, (-1,)), flat_indices)
        
        # Calculate matrix indices
        cols_tensor = valid_indices[2]
        rows_tensor = (keras.ops.cast(global_y_valid, "int32") * self.nx + 
                      keras.ops.cast(global_x_valid, "int32"))
        
        # Add background terms if needed
        if fit_background:
            bg_rows = keras.ops.reshape(y_grid * self.nx + x_grid, (-1,))
            rows_tensor = keras.ops.concatenate([rows_tensor, keras.ops.cast(bg_rows, "int32")])
            cols_tensor = keras.ops.concatenate([cols_tensor, 
                keras.ops.full((self.nx * self.ny,), num_coordinates, dtype="int32")])
            data_tensor = keras.ops.concatenate([data_tensor, 
                keras.ops.ones((self.nx * self.ny,), dtype="float32")])
            shape = (self.nx * self.ny, num_coordinates + 1)
        else:
            shape = (self.nx * self.ny, num_coordinates)
        
        return self._create_sparse_matrix(data_tensor, rows_tensor, cols_tensor, shape)
    
    def _create_sparse_matrix(self, data_tensor, rows_tensor, cols_tensor, shape):
        """Create sparse matrix in the appropriate backend format."""
        if self.backend == "torch":
            import torch
            
            # Check if we're on MPS device - always fallback to scipy for MPS
            device = getattr(data_tensor, 'device', torch.device('cpu'))
            if hasattr(torch.backends, 'mps') and (str(device).startswith('mps') or torch.backends.mps.is_available()):
                logging.warning("MPS backend detected, falling back to scipy sparse matrix for full compatibility")
                # Always use scipy for MPS to avoid any sparse operation issues
                return coo_matrix((safe_convert_to_numpy(data_tensor),
                                 (safe_convert_to_numpy(rows_tensor), safe_convert_to_numpy(cols_tensor))),
                                shape=shape)
            
            try:
                indices = torch.stack([rows_tensor, cols_tensor])
                return torch.sparse_coo_tensor(indices, data_tensor, size=shape).coalesce()
            except RuntimeError as e:
                error_msg = str(e)
                if any(keyword in error_msg for keyword in ["SparseMPS", "_sparse_coo_tensor_with_dims_and_tensors", "aten::addmm", "SparseCsrMPS"]):
                    logging.warning(f"PyTorch sparse tensor creation failed: {e}")
                    logging.info("Falling back to scipy sparse matrix")
                    # Fallback to scipy sparse matrix
                    return coo_matrix((safe_convert_to_numpy(data_tensor),
                                     (safe_convert_to_numpy(rows_tensor), safe_convert_to_numpy(cols_tensor))),
                                    shape=shape)
                else:
                    raise
                    
        elif self.backend == "tensorflow":
            import tensorflow as tf
            indices = tf.stack([rows_tensor, cols_tensor], axis=1)
            return tf.sparse.reorder(tf.sparse.SparseTensor(indices, data_tensor, dense_shape=shape))
        elif self.backend == "jax":
            import jax.numpy as jnp
            from jax.experimental import sparse as jsparse
            return jsparse.BCOO((data_tensor, jnp.stack([rows_tensor, cols_tensor])), shape=shape)
        else:
            return coo_matrix((safe_convert_to_numpy(data_tensor),
                             (safe_convert_to_numpy(rows_tensor), safe_convert_to_numpy(cols_tensor))),
                            shape=shape)


class ParameterValidator:
    """Validates and processes input parameters for linear estimation."""
    
    @staticmethod
    def validate_params(params: Dict) -> Dict:
        """Validate and clean input parameters."""
        if not isinstance(params, dict):
            raise ParameterError("Parameters must be a dictionary")
        
        required_keys = ["pos_x", "pos_y", "height", "width"]
        missing_keys = [key for key in required_keys if key not in params]
        if missing_keys:
            raise ParameterError(f"Missing required parameters: {missing_keys}")
        
        # Validate shapes and values
        pos_x, pos_y, height = params["pos_x"], params["pos_y"], params["height"]
        
        if keras.ops.shape(pos_x)[0] != keras.ops.shape(pos_y)[0] != keras.ops.shape(height)[0]:
            raise ParameterError("pos_x, pos_y, and height must have same length")
        
        # Check for invalid values
        for key in required_keys:
            values = safe_convert_to_numpy(params[key])
            if np.any(np.isnan(values)) or np.any(np.isinf(values)):
                raise ParameterError(f"Parameter '{key}' contains NaN or infinite values")
        
        return params


class SolutionProcessor:
    """Processes and validates linear system solutions."""
    
    @staticmethod
    def validate_solution(solution: np.ndarray) -> bool:
        """Validate solution for common issues."""
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
        """Process and constrain height scaling factors."""
        # Convert to tensor for processing if it's a numpy array
        if isinstance(height_scale, np.ndarray):
            height_tensor = safe_convert_to_tensor(height_scale)
        else:
            height_tensor = height_scale
        
        # Count out-of-bounds values for logging
        too_small = keras.ops.sum(height_tensor < min_scale)
        too_large = keras.ops.sum(height_tensor > max_scale)

        # Replace nan with 1
        height_tensor = keras.ops.where(keras.ops.isnan(height_tensor), 
                                      keras.ops.ones_like(height_tensor), height_tensor)
        
        # Apply constraints
        height_tensor = keras.ops.clip(height_tensor, min_scale, max_scale)
        
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
                f"Over {total_clipped/len(height_scale)*100:.2f}% of height values were clipped "
                f"({total_clipped}/{len(height_scale)}). "
                "Consider refining peak positions or checking model parameters."
            )
        
        # Convert back to numpy for consistency with original interface
        return safe_convert_to_numpy(height_tensor)

    @staticmethod
    def process_background(solution, params, init_background, update_threshold=0.2):
        """Process and update the background parameter based on the solution."""
        # Extract background value (last element of solution)
        if isinstance(solution, np.ndarray):
            background_val = float(solution[-1])
        else:
            background_val = float(safe_convert_to_numpy(solution[-1]))
        
        background = max(background_val, init_background)
        
        # Get previous background value
        prev_background = params["background"]
        if hasattr(prev_background, 'shape'):  # It's a tensor
            prev_bg_val = float(safe_convert_to_numpy(prev_background))
        else:
            prev_bg_val = float(prev_background)
        
        update_rel = (background - prev_bg_val) / (prev_bg_val + 1e-10)
        
        if abs(update_rel) > update_threshold * 2:
            # Update too large, skip update
            return prev_bg_val, False
        
        if abs(update_rel) > update_threshold:
            update_rel_clip = max(-update_threshold, min(update_threshold, update_rel))
            background = prev_bg_val * (1 + update_rel_clip)
        
        return background, True