"""Linear estimation for image-fitting heights and background.

Three responsibilities:

* :class:`DesignMatrixBuilder` — render the per-atom Gaussian peak
  windows into a sparse design matrix.
* :class:`LinearSystemSolver` — non-negative ridge LS dispatcher.
  The non-negative path uses :func:`qem.fit.sparse_torch.pg_nnls`
  (torch sparse CSR + projected gradient with BB step). The
  unconstrained path uses ``scipy.sparse.linalg.lsqr`` as a thin
  fallback for the rare case when callers actually want it.
* :func:`linear_estimator` — public Fitter method that wires the
  two together and updates the height parameter from the solution.
"""

from __future__ import annotations

import logging
from contextlib import nullcontext
from typing import Any, Dict, Optional

import numpy as np
import torch
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import lsqr

from qem.utils.exceptions import DataError, ParameterError
from qem.utils.tensors import release_memory, to_numpy, to_tensor


# ---------------------------------------------------------------------------
# Design matrix
# ---------------------------------------------------------------------------

class DesignMatrixBuilder:
    """Builds the per-pixel sparse design matrix for height linear estimation.

    Each column corresponds to one atom (plus an optional background
    column at the end). Each non-zero entry is the peak shape value
    at that pixel.
    """

    def __init__(self, model, nx: int, ny: int):
        self.model = model
        self.nx = nx
        self.ny = ny

    def build_local_peaks(self, params: Dict, same_width: bool, atom_types: np.ndarray):
        """Render each atom's peak on its local window.

        Returns ``(peak_local, global_x, global_y, mask)`` where
        ``peak_local`` has shape ``(2W+1, 2W+1, N)`` with W set to
        5σ to keep enough tail for the linear estimator (vs 3σ in the
        Adam loop's ``_sum_local`` — accuracy matters more than speed
        here since this runs once per fit).
        """
        pos_x, pos_y = params["pos_x"], params["pos_y"]
        width, height = params["width"], params["height"]
        ratio = params.get("ratio", None)

        if same_width:
            width = width[atom_types]
            if ratio is not None:
                ratio = ratio[atom_types]

        window_size = (torch.max(width) * 5).to(dtype=torch.int32)
        x = torch.arange(-window_size, window_size + 1, 1, dtype=torch.float32)
        y = torch.arange(-window_size, window_size + 1, 1, dtype=torch.float32)
        local_x, local_y = torch.meshgrid(x, y, indexing="xy")

        peak_args = (torch.remainder(pos_x, 1), torch.remainder(pos_y, 1), height, width)
        if ratio is not None:
            peak_args += (ratio,)
        peak_local = self.model.model_fn(local_x[..., None], local_y[..., None], *peak_args)

        pos_x_int, pos_y_int = torch.floor(pos_x), torch.floor(pos_y)
        global_x = local_x.unsqueeze(-1) + pos_x_int
        global_y = local_y.unsqueeze(-1) + pos_y_int
        mask = (
            (global_x >= 0) & (global_x < self.nx)
            & (global_y >= 0) & (global_y < self.ny)
        )
        return peak_local, global_x, global_y, mask

    def build_sparse_matrix(
        self,
        peak_local,
        global_x,
        global_y,
        mask,
        fit_background: bool,
        num_coordinates: int,
        x_grid,
        y_grid,
        background_2d: Optional[np.ndarray] = None,
    ) -> coo_matrix:
        """Stack ``(peak, x, y, mask)`` into a scipy ``coo_matrix``.

        Always returns a CPU scipy matrix because the downstream solver
        (:func:`qem.fit.sparse_torch.pg_nnls`) works on torch CSR built
        from scipy at call time.
        """
        valid = torch.where(mask)
        shape = tuple(peak_local.shape)
        flat_idx = (
            valid[0] * (shape[1] * shape[2])
            + valid[1] * shape[2]
            + valid[2]
        )

        data = torch.take(peak_local.reshape(-1), flat_idx)
        gx_valid = torch.take(global_x.reshape(-1), flat_idx)
        gy_valid = torch.take(global_y.reshape(-1), flat_idx)

        cols = valid[2].to(dtype=torch.int32)
        rows = (
            gy_valid.to(dtype=torch.int32) * self.nx
            + gx_valid.to(dtype=torch.int32)
        )

        if fit_background:
            dev = cols.device
            bg_rows = (y_grid * self.nx + x_grid).reshape(-1).to(
                dtype=torch.int32, device=dev,
            )
            rows = torch.cat([rows, bg_rows])
            cols = torch.cat([
                cols,
                torch.full((self.nx * self.ny,), num_coordinates,
                           dtype=torch.int32, device=dev),
            ])
            if background_2d is not None:
                bg_data = torch.as_tensor(
                    background_2d.ravel(), dtype=torch.float32, device=dev,
                )
            else:
                bg_data = torch.ones(
                    (self.nx * self.ny,), dtype=torch.float32, device=dev,
                )
            data = torch.cat([data, bg_data])
            shape_out = (self.nx * self.ny, num_coordinates + 1)
        else:
            shape_out = (self.nx * self.ny, num_coordinates)

        sparse = coo_matrix(
            (to_numpy(data), (to_numpy(rows), to_numpy(cols))),
            shape=shape_out,
        )
        del data, rows, cols
        release_memory()
        return sparse


# ---------------------------------------------------------------------------
# Solver
# ---------------------------------------------------------------------------

class LinearSystemSolver:
    """Solve ``A x = b`` (sparse, possibly with bounds and ridge).

    Non-negative path: torch sparse CSR + projected gradient (BB step)
    via :func:`qem.fit.sparse_torch.pg_nnls`. Roughly 5× faster than
    ``scipy.optimize.lsq_linear`` with bounds on the design matrices
    QEM builds.

    Unconstrained path: scipy ``lsqr`` with optional Tikhonov ridge.
    Rarely called — most callers want non-negativity (heights ≥ 0).
    """

    def solve_system(
        self,
        design_matrix: coo_matrix,
        target: np.ndarray,
        non_negative: bool = True,
        ridge: float = 1e-4,
    ) -> np.ndarray:
        if hasattr(target, "cpu"):
            target = target.cpu().numpy()
        elif not isinstance(target, np.ndarray):
            target = np.asarray(target)
        target = target.astype(np.float32)

        if non_negative:
            from qem.fit.sparse_torch import pg_nnls

            return pg_nnls(design_matrix, target, ridge=ridge)

        # Unconstrained ridge LS via scipy lsqr (rare path).
        from scipy.sparse import eye as sp_eye
        from scipy.sparse import vstack as sp_vstack

        A = design_matrix.tocsr().astype(np.float32)
        b = target
        if ridge > 0.0:
            n = A.shape[1]
            A = sp_vstack([A, np.sqrt(ridge) * sp_eye(n, dtype=np.float32)]).tocsr()
            b = np.concatenate([b, np.zeros(n, dtype=np.float32)])
        try:
            return lsqr(A, b)[0].astype(np.float32)
        except Exception as exc:
            raise DataError(f"lsqr failed: {exc}") from exc


# ---------------------------------------------------------------------------
# Validation / solution post-processing
# ---------------------------------------------------------------------------

class ParameterValidator:
    """Validate the parameter dict before linear estimation."""

    @staticmethod
    def validate_params(params: Dict) -> Dict:
        if not isinstance(params, dict):
            raise ParameterError("Parameters must be a dictionary")

        required = ("pos_x", "pos_y", "height", "width")
        missing = [k for k in required if k not in params]
        if missing:
            raise ParameterError(f"Missing required parameters: {missing}")

        lengths = {tuple(params[k].shape)[0] for k in ("pos_x", "pos_y", "height")}
        if len(lengths) != 1:
            raise ParameterError("pos_x, pos_y, and height must have same length")

        for key in required:
            arr = to_numpy(params[key])
            if np.any(np.isnan(arr)) or np.any(np.isinf(arr)):
                raise ParameterError(f"Parameter {key!r} contains NaN or infinite values")
        return params


class SolutionProcessor:
    """Post-process the LS solution back into Fitter parameters."""

    @staticmethod
    def validate_solution(solution: np.ndarray) -> bool:
        if solution is None:
            return False
        a = np.asarray(solution)
        if np.any(np.isnan(a)) or np.any(np.isinf(a)):
            logging.warning("Solution contains NaN or infinite values")
            return False
        return True

    @staticmethod
    def process_height_scaling(
        height_scale: np.ndarray, min_scale: float = 0.5, max_scale: float = 2.0,
    ) -> np.ndarray:
        """Clamp height-scale corrections; replace NaNs with 1.0."""
        h = to_tensor(height_scale) if isinstance(height_scale, np.ndarray) else height_scale
        h = torch.where(torch.isnan(h), torch.ones_like(h), h)
        too_small = int((h < min_scale).sum())
        too_large = int((h > max_scale).sum())
        h = torch.clamp(h, min_scale, max_scale)
        if too_small + too_large > len(height_scale) * 0.3:
            logging.warning(
                "%.0f%% of height scales clipped (%d/%d) — refine peak positions or check init",
                100.0 * (too_small + too_large) / len(height_scale),
                too_small + too_large,
                len(height_scale),
            )
        return to_numpy(h)

    @staticmethod
    def process_background(
        solution, params, init_background, update_threshold: float = 0.2,
    ):
        """Validate the background update; clip large jumps."""
        background = max(float(np.asarray(solution[-1])), init_background)
        prev = params["background"]
        prev_val = float(to_numpy(prev)) if hasattr(prev, "shape") else float(prev)
        rel = (background - prev_val) / (prev_val + 1e-10)
        if abs(rel) > update_threshold * 2:
            return prev_val, False
        if abs(rel) > update_threshold:
            rel = max(-update_threshold, min(update_threshold, rel))
            background = prev_val * (1 + rel)
        return background, True


# ---------------------------------------------------------------------------
# Public Fitter methods (bound onto Fitter via _bind)
# ---------------------------------------------------------------------------

def linear_estimator(
    self,
    params: Optional[Dict] = None,
    *,
    non_negative: bool = True,
    ridge: float = 1e-4,
    best_effort: bool = False,
) -> Dict:
    """Refine peak heights via non-negative ridge least-squares.

    Solves ``min ‖A x − b‖² + λ‖x‖²`` subject to ``x ≥ 0``. Updates
    ``params["height"]`` in place by multiplying by the per-atom scale
    factor (clamped to ``[0.5, 2.0]``).

    Args:
        params: parameter dict (defaults to ``self.params``).
        non_negative: enforce x ≥ 0 (default ``True`` — heights are
            physically non-negative).
        ridge: Tikhonov ridge strength. ``1e-4`` stabilises without
            biasing scales meaningfully on STEM data.
        best_effort: swallow errors and return the input params
            unchanged (used by ``fit_stochastic``'s pre-conditioner).

    Returns:
        Updated parameters dict.
    """
    if params is None:
        if self.params is None:
            self.init_params()
        params = self.params

    op = (
        self.memory_monitor.monitor_operation("linear_estimator")
        if self.memory_monitor
        else nullcontext()
    )

    def _run() -> Dict:
        validated = ParameterValidator.validate_params(params)
        builder = DesignMatrixBuilder(self.model, self.nx, self.ny)
        peak_local, gx, gy, mask = builder.build_local_peaks(
            validated, self.same_width, self.atom_types,
        )
        bg_2d = (
            self.background_estimator.get_background_for_linear_estimation()
            if self.background_estimator.use_2d_background
            else None
        )
        A = builder.build_sparse_matrix(
            peak_local, gx, gy, mask,
            self.fit_background, self.num_coordinates,
            self.x_grid, self.y_grid, bg_2d,
        )
        target = self._prepare_target_vector(validated)
        solution = LinearSystemSolver().solve_system(
            A, target, non_negative=non_negative, ridge=ridge,
        )
        return self._process_solution(solution, validated)

    with op:
        if not best_effort:
            return _run()
        try:
            return _run()
        except Exception as exc:
            logging.warning(
                "linear_estimator failed in best_effort mode; "
                "returning input parameters unchanged: %s", exc,
            )
            return params


def _prepare_target_vector(self, params: Dict) -> np.ndarray:
    """Flatten the image, subtract scalar/2D background if not jointly fit."""
    target = self.image_tensor.ravel()
    if not self.fit_background:
        if self.background_estimator.use_2d_background:
            target = target - self.get_current_background().ravel()
        else:
            bg_key = "background_scale" if "background_scale" in params else "background"
            target = target - params[bg_key]
    return target


def _process_solution(
    self, solution: np.ndarray, params: Dict, update_threshold: float = 0.2,
) -> Dict:
    """Apply the LS solution back onto ``params`` (height + optional background)."""
    proc = SolutionProcessor()
    if not proc.validate_solution(solution):
        logging.warning("Invalid solution obtained, returning original parameters")
        return params

    if self.fit_background:
        if self.background_estimator.use_2d_background:
            bg_scale = float(solution[-1])
            if 0.01 < bg_scale < 100.0:
                self.update_2d_background_scale(bg_scale)
                params["background_scale"] = to_tensor(bg_scale)
                params.pop("background", None)
            else:
                logging.warning(
                    "2D background scale out of bounds: %.3f, keeping current scale", bg_scale,
                )
            height_scale = solution[:-1]
        else:
            background, ok = proc.process_background(
                solution, params, self.init_background, update_threshold,
            )
            if not ok:
                logging.warning(
                    "Background update too large, skipping parameter update with linear estimator",
                )
                return params
            params["background"] = to_tensor(background)
            height_scale = solution[:-1]
    else:
        height_scale = solution

    scale = to_tensor(proc.process_height_scaling(height_scale))
    params["height"] = params["height"] * scale
    self.params = params
    return params


def _bind(cls) -> None:
    """Attach the linear-estimator methods back onto Fitter at load time."""
    cls.linear_estimator = linear_estimator
    cls._prepare_target_vector = _prepare_target_vector
    cls._process_solution = _process_solution


__all__ = [
    "DesignMatrixBuilder",
    "LinearSystemSolver",
    "ParameterValidator",
    "SolutionProcessor",
    "linear_estimator",
    "_prepare_target_vector",
    "_process_solution",
    "_bind",
]
