"""
Linear solver module for QEM image fitting.

Sparse linear system solving with a simple two-step fallback: try direct,
on memory / singular / linalg error fall back to iterative.
"""

import logging
from typing import Dict, Optional, Tuple, Protocol

import numpy as np
import torch
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import spsolve, lsqr, cg

from qem.utils.params import safe_convert_to_numpy, safe_convert_to_tensor
from qem.utils.backend import release_backend_memory
from qem.utils.config import get_config
from qem.utils.exceptions import ParameterError, DataError


class BackendSolver(Protocol):
    """Protocol for sparse linear solvers."""

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
    def _convert_target_to_numpy(b):
        """Convert target vector to numpy with configured precision."""
        b_numpy = b.cpu().numpy() if hasattr(b, 'cpu') else np.asarray(b)
        return create_linear_solver_array(b_numpy)
    
    @staticmethod
    def _convert_to_scipy_matrix(A):
        """Convert PyTorch tensor to scipy sparse matrix with configured precision."""
        import torch
        from scipy.sparse import coo_matrix
        
        config = get_config()
        target_dtype = config.linear_solver_numpy_dtype
        
        if hasattr(A, 'tocsr'):
            # Already a scipy sparse matrix - ensure correct precision
            if A.dtype != target_dtype:
                return A.astype(target_dtype)
            return A
        elif hasattr(A, 'is_sparse') and A.is_sparse:
            # PyTorch sparse tensor
            A_coo = A.coalesce()
            indices = A_coo.indices().cpu().numpy()
            values = A_coo.values().cpu().numpy().astype(target_dtype)
            shape = A_coo.shape
            return coo_matrix((values, (indices[0], indices[1])), shape=shape)
        elif hasattr(A, 'cpu'):
            # PyTorch dense tensor
            return coo_matrix(A.cpu().numpy().astype(target_dtype))
        else:
            # Assume it's already numpy or scipy
            data = np.asarray(A, dtype=target_dtype)
            return coo_matrix(data)
    
    @staticmethod
    def solve_direct(A, b: np.ndarray, non_negative: bool = False) -> np.ndarray:
        import torch
        
        # Check for MPS device and fallback to scipy if needed
        device = getattr(A, 'device', torch.device('cpu'))
        if TorchSolver._is_mps_device(device):
            logging.warning("MPS backend detected, falling back to scipy sparse solver for compatibility")
            A_scipy = TorchSolver._convert_to_scipy_matrix(A)
            b_numpy = TorchSolver._convert_target_to_numpy(b)
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
                b_numpy = TorchSolver._convert_target_to_numpy(b)
                return SciPySolver.solve_direct(A_scipy, b_numpy, non_negative)
            
            # Check for memory errors
            elif any(keyword in error_msg.lower() for keyword in ["out of memory", "cuda out of memory", "mps out of memory", "memory"]):
                logging.warning(f"PyTorch out of memory: {e}")
                logging.info("Falling back to memory-efficient scipy sparse solver")
                A_scipy = TorchSolver._convert_to_scipy_matrix(A)
                b_numpy = TorchSolver._convert_target_to_numpy(b)
                return SciPySolver.solve_direct(A_scipy, b_numpy, non_negative)
            
            else:
                raise
        except MemoryError as e:
            logging.warning(f"System memory error in PyTorch: {e}")
            logging.info("Falling back to memory-efficient scipy sparse solver")
            A_scipy = TorchSolver._convert_to_scipy_matrix(A)
            b_numpy = TorchSolver._convert_target_to_numpy(b)
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
            b_numpy = TorchSolver._convert_target_to_numpy(b)
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
                b_numpy = TorchSolver._convert_target_to_numpy(b)
                return SciPySolver.solve_iterative(A_scipy, b_numpy, non_negative, max_iter, tol)
            
            # Check for memory errors
            elif any(keyword in error_msg.lower() for keyword in ["out of memory", "cuda out of memory", "mps out of memory", "memory"]):
                logging.warning(f"PyTorch out of memory: {e}")
                logging.info("Falling back to memory-efficient scipy sparse solver")
                A_scipy = TorchSolver._convert_to_scipy_matrix(A)
                b_numpy = TorchSolver._convert_target_to_numpy(b)
                return SciPySolver.solve_iterative(A_scipy, b_numpy, non_negative, max_iter, tol)
            
            else:
                raise
        except MemoryError as e:
            logging.warning(f"System memory error in PyTorch: {e}")
            logging.info("Fallback to memory-efficient scipy sparse solver")
            A_scipy = TorchSolver._convert_to_scipy_matrix(A)
            b_numpy = TorchSolver._convert_target_to_numpy(b)
            return SciPySolver.solve_iterative(A_scipy, b_numpy, non_negative, max_iter, tol)


class SciPySolver:
    """SciPy-specific sparse linear solver."""
    
    @staticmethod
    def solve_direct(A: coo_matrix, b: np.ndarray, non_negative: bool = False) -> np.ndarray:
        # Ensure inputs use configured precision
        config = get_config()
        target_dtype = config.linear_solver_numpy_dtype
        
        # Convert to target precision if needed
        if A.dtype != target_dtype:
            A = A.astype(target_dtype)
        if b.dtype != target_dtype:
            b = b.astype(target_dtype)
        
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
        
        # Ensure output precision
        return solution.astype(target_dtype)
    
    @staticmethod
    def solve_iterative(A: coo_matrix, b: np.ndarray, non_negative: bool = False, 
                       max_iter: int = 1000, tol: float = 1e-6) -> np.ndarray:
        # Ensure inputs use configured precision
        config = get_config()
        target_dtype = config.linear_solver_numpy_dtype
        
        # Convert to target precision if needed
        if A.dtype != target_dtype:
            A = A.astype(target_dtype)
        if b.dtype != target_dtype:
            b = b.astype(target_dtype)
        
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
        
        # Ensure output precision
        return solution.astype(target_dtype)


class LinearSystemSolver:
    """Sparse linear system solver: try direct, fall back to iterative.

    Replaces the previous "memory-aware strategy selection" + nested
    try/except. The cost of always *trying* the direct solver and
    falling back on actual failure is much lower than psutil polling
    every solve.
    """

    def __init__(self):
        self.solver = TorchSolver

    def solve_system(self, design_matrix, target: np.ndarray,
                     non_negative: bool = False) -> Optional[np.ndarray]:
        """Solve ``design_matrix @ x = target``.

        scipy sparse inputs always go through the iterative scipy path.
        For backend tensors: try direct; on memory / singular / linalg
        error fall back to iterative; on a second failure raise DataError.
        """
        # Coerce target to numpy (MPS tensors live on accelerators).
        if hasattr(target, "cpu"):
            target = target.cpu().numpy()
        elif not isinstance(target, np.ndarray):
            target = np.asarray(target)

        # scipy sparse matrices skip the direct path — they have no
        # backend solver attached.
        if hasattr(design_matrix, "tocsr"):
            return SciPySolver.solve_iterative(design_matrix, target, non_negative)

        try:
            return self.solver.solve_direct(design_matrix, target, non_negative)
        except (MemoryError, np.linalg.LinAlgError) as exc:
            logging.info("Direct solver fallback (%s); trying iterative.", exc)
        except Exception as exc:
            err = str(exc).lower()
            if not any(k in err for k in ("singular", "memory", "out of memory")):
                raise
            logging.info("Direct solver hit numerical issue (%s); trying iterative.", exc)

        try:
            return self.solver.solve_iterative(design_matrix, target, non_negative)
        except Exception as exc:
            raise DataError(
                f"Linear solve failed: {exc}",
                technical_details={"matrix_shape": getattr(design_matrix, "shape", None)},
            ) from exc


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
        window_size = (torch.max(width) * 5).to(dtype=torch.int32)
        x = torch.arange(-window_size, window_size + 1, 1, dtype=torch.float32)
        y = torch.arange(-window_size, window_size + 1, 1, dtype=torch.float32)
        local_x, local_y = torch.meshgrid(x, y, indexing="xy")
        
        # Generate local peaks
        input_params = (torch.remainder(pos_x, 1), torch.remainder(pos_y, 1), height, width)
        if ratio is not None:
            input_params += (ratio,)
        
        peak_local = self.model.model_fn(local_x[..., None], local_y[..., None], *input_params)
        
        # Calculate global coordinates and mask
        pos_x_int, pos_y_int = torch.floor(pos_x), torch.floor(pos_y)
        global_x = torch.unsqueeze(local_x, -1) + pos_x_int
        global_y = torch.unsqueeze(local_y, -1) + pos_y_int
        
        mask = ((global_x >= 0) & (global_x < self.nx) & 
                (global_y >= 0) & (global_y < self.ny))
        
        return peak_local, global_x, global_y, mask
    
    def build_sparse_matrix(self, peak_local, global_x, global_y, mask, 
                          fit_background: bool, num_coordinates: int, x_grid, y_grid,
                          background_2d: np.ndarray = None):
        """Build sparse design matrix from peak data with optional 2D background."""
        # Extract valid data
        valid_indices = torch.where(mask)
        shape = tuple(peak_local.shape)
        
        flat_indices = (valid_indices[0] * (shape[1] * shape[2]) + 
                       valid_indices[1] * shape[2] + valid_indices[2])
        
        data_tensor = torch.take(torch.reshape(peak_local, (-1,)), flat_indices)
        global_x_valid = torch.take(torch.reshape(global_x, (-1,)), flat_indices)
        global_y_valid = torch.take(torch.reshape(global_y, (-1,)), flat_indices)
        
        # Calculate matrix indices
        cols_tensor = valid_indices[2]
        rows_tensor = (global_y_valid.to(dtype=torch.int32) * self.nx +
                       global_x_valid.to(dtype=torch.int32))
        
        # Add background terms if needed
        if fit_background:
            bg_rows = torch.reshape(y_grid * self.nx + x_grid, (-1,))
            rows_tensor = torch.cat([rows_tensor, bg_rows.to(dtype=torch.int32)])

            del bg_rows
            release_backend_memory()

            cols_tensor = torch.cat([cols_tensor,
                torch.full((self.nx * self.ny,), num_coordinates, dtype=torch.int32)])
            
            if background_2d is not None:
                # Use 2D background values instead of ones
                bg_data = torch.as_tensor(background_2d.ravel(), dtype=torch.float32)
                data_tensor = torch.cat([data_tensor, bg_data])
            else:
                # Use scalar background (ones)
                data_tensor = torch.cat([data_tensor, 
                    torch.ones((self.nx * self.ny,), dtype=torch.float32)])
            
            shape = (self.nx * self.ny, num_coordinates + 1)

        else:
            shape = (self.nx * self.ny, num_coordinates)
        sparse_matrix = self._create_sparse_matrix(data_tensor, rows_tensor, cols_tensor, shape)
        del rows_tensor
        del cols_tensor
        del data_tensor
        release_backend_memory()
        return sparse_matrix
    
    def _create_sparse_matrix(self, data_tensor, rows_tensor, cols_tensor, shape, device: str = 'cpu'):
        """Build a SciPy COO design matrix from PyTorch tensors.

        We always materialise design matrices on CPU as ``scipy.sparse.coo_matrix``
        because the downstream solvers (``SciPySolver`` and the SciPy fallback in
        ``TorchSolver``) operate on SciPy matrices, and PyTorch sparse tensors on
        MPS/CUDA do not support all the operations we need.
        """
        if device != 'cpu':
            logging.debug(
                "DesignMatrixBuilder: forcing CPU/SciPy sparse matrix (requested device=%s)", device
            )
        return coo_matrix(
            (
                safe_convert_to_numpy(data_tensor),
                (safe_convert_to_numpy(rows_tensor), safe_convert_to_numpy(cols_tensor)),
            ),
            shape=shape,
        )


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

        lengths = {
            tuple(pos_x.shape)[0],
            tuple(pos_y.shape)[0],
            tuple(height.shape)[0],
        }
        if len(lengths) != 1:
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
        
        # Check for NaN or infinite values (input may be numpy or tensor).
        if np.any(np.isnan(np.asarray(solution))) or np.any(np.isinf(np.asarray(solution))):
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
        too_small = torch.sum(height_tensor < min_scale)
        too_large = torch.sum(height_tensor > max_scale)

        # Replace nan with 1
        height_tensor = torch.where(torch.isnan(height_tensor), 
                                      torch.ones_like(height_tensor), height_tensor)
        
        # Apply constraints
        height_tensor = torch.clamp(height_tensor, min_scale, max_scale)
        
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
