"""Linear estimation for image-fitting heights and background.

Module-level functions, not classes. The four ``ClassName.staticmethod``
namespaces this file used to have were Java-style packaging — a class
holding only ``@staticmethod`` is a function in disguise. Python modules
are already namespaces.

Pipeline (called from :func:`linear_estimator`):

    build_local_peaks → build_sparse_matrix → solve_system →
    _validate_solution → _process_height_scaling → _process_background

Each stage is a small pure function; the only stateful step is the
``Fitter`` mixin method :func:`linear_estimator`.
"""

from __future__ import annotations

import logging
from contextlib import nullcontext
from typing import Any, Optional

import numpy as np
import torch
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import lsqr

from qem.utils.exceptions import DataError, ParameterError
from qem.utils.tensors import release_memory, to_numpy, to_tensor

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Design matrix
# ---------------------------------------------------------------------------

def build_local_peaks(model, params: dict, same_width: bool, atom_types: np.ndarray):
    """Render each atom's peak on its local window.

    Returns ``(peak_local, global_x, global_y, mask)``. ``peak_local``
    has shape ``(2W+1, 2W+1, N)`` with W = 5σ — wider than the 3σ used
    by the Adam-loop renderer because the linear estimator runs once
    per fit, so accuracy beats speed.
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
    peak_local = model.model_fn(local_x[..., None], local_y[..., None], *peak_args)

    pos_x_int, pos_y_int = torch.floor(pos_x), torch.floor(pos_y)
    global_x = local_x.unsqueeze(-1) + pos_x_int
    global_y = local_y.unsqueeze(-1) + pos_y_int
    return peak_local, global_x, global_y


def build_sparse_matrix(
    peak_local,
    global_x,
    global_y,
    *,
    nx: int,
    ny: int,
    fit_background: bool,
    num_coordinates: int,
    x_grid,
    y_grid,
    background_2d: Optional[np.ndarray] = None,
) -> coo_matrix:
    """Stack ``(peak, x, y)`` into a scipy ``coo_matrix``.

    Always returns a CPU scipy matrix because the downstream solver
    (:func:`qem.fit.sparse_torch.pg_nnls`) builds a torch CSR from it.
    """
    mask = (
        (global_x >= 0) & (global_x < nx)
        & (global_y >= 0) & (global_y < ny)
    )
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
        gy_valid.to(dtype=torch.int32) * nx
        + gx_valid.to(dtype=torch.int32)
    )

    if fit_background:
        dev = cols.device
        bg_rows = (y_grid * nx + x_grid).reshape(-1).to(
            dtype=torch.int32, device=dev,
        )
        rows = torch.cat([rows, bg_rows])
        cols = torch.cat([
            cols,
            torch.full((nx * ny,), num_coordinates,
                       dtype=torch.int32, device=dev),
        ])
        if background_2d is not None:
            bg_data = torch.as_tensor(
                background_2d.ravel(), dtype=torch.float32, device=dev,
            )
        else:
            bg_data = torch.ones(
                (nx * ny,), dtype=torch.float32, device=dev,
            )
        data = torch.cat([data, bg_data])
        shape_out = (nx * ny, num_coordinates + 1)
    else:
        shape_out = (nx * ny, num_coordinates)

    sparse = coo_matrix(
        (to_numpy(data), (to_numpy(rows), to_numpy(cols))),
        shape=shape_out,
    )
    del data, rows, cols
    release_memory()
    return sparse


# ---------------------------------------------------------------------------
# Solve A x = b (sparse, with optional non-negativity / ridge)
# ---------------------------------------------------------------------------

def solve_system(
    design_matrix: coo_matrix,
    target: np.ndarray,
    *,
    non_negative: bool = True,
    ridge: float = 1e-4,
) -> np.ndarray:
    """Solve ``A x = b`` (or its non-negative variant).

    * ``non_negative=True``: torch sparse CSR + projected gradient (BB
      step) via :func:`qem.fit.sparse_torch.pg_nnls`. ~5× faster than
      ``scipy.optimize.lsq_linear`` on the design matrices QEM builds.
    * ``non_negative=False``: scipy ``lsqr`` with optional Tikhonov
      ridge. Rarely needed — heights are physically non-negative.
    """
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

def validate_params(params: dict) -> dict:
    """Sanity-check the parameter dict before linear estimation.

    Raises :class:`ParameterError` on missing keys, length mismatch, or
    NaN/Inf values; otherwise returns ``params`` unchanged.
    """
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


def validate_solution(solution: np.ndarray) -> bool:
    """Reject NaN/Inf solutions."""
    if solution is None:
        return False
    a = np.asarray(solution)
    if np.any(np.isnan(a)) or np.any(np.isinf(a)):
        log.warning("Solution contains NaN or infinite values")
        return False
    return True


def process_height_scaling(
    height_scale: np.ndarray,
    *,
    min_scale: float = 0.05,
    max_scale: float = 20.0,
) -> np.ndarray:
    """Clamp height-scale corrections; replace NaNs with 1.0.

    Bounds are deliberately wide. The scale is multiplicative on
    ``init_params``'s ``image[y, x] - background`` height guess; edge
    atoms in inhomogeneous samples (nanoparticle on substrate)
    legitimately need >2× downward correction. A tight ``[0.5, 2.0]``
    clamp silently truncated those, leaving edge SCS biased high by
    ~40%. The >30%-clipped warning still fires for genuinely runaway
    solutions.
    """
    h = to_tensor(height_scale) if isinstance(height_scale, np.ndarray) else height_scale
    h = torch.where(torch.isnan(h), torch.ones_like(h), h)
    too_small = int((h < min_scale).sum())
    too_large = int((h > max_scale).sum())
    h = torch.clamp(h, min_scale, max_scale)
    n = len(height_scale)
    if too_small + too_large > n * 0.3:
        log.warning(
            "%.0f%% of height scales clipped (%d/%d) — refine peak positions or check init",
            100.0 * (too_small + too_large) / n, too_small + too_large, n,
        )
    return to_numpy(h)


def process_background(
    solution, params, init_background, *, update_threshold: float = 0.2,
):
    """Validate the scalar background update; clip large jumps.

    Returns ``(background, ok)`` — ``ok=False`` means the proposed
    update was beyond ``2·update_threshold`` and should be rejected.
    """
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
# Public Fitter methods (mixed in via FitterSolverMixin below)
# ---------------------------------------------------------------------------

def linear_estimator(
    self,
    params: Optional[dict] = None,
    *,
    non_negative: bool = True,
    ridge: float = 1e-4,
    best_effort: bool = False,
) -> dict:
    """Refine peak heights via non-negative ridge least-squares.

    Solves ``min ‖A x − b‖² + λ‖x‖²`` subject to ``x ≥ 0``. Updates
    ``params["height"]`` in place with the per-atom scale factor
    (clamped to ``[0.05, 20]``).

    Args:
        params: parameter dict (defaults to ``self.params``).
        non_negative: enforce x ≥ 0 (default True).
        ridge: Tikhonov ridge strength.
        best_effort: swallow errors and return the input params
            unchanged (used by ``fit_stochastic``'s pre-conditioner).
    """
    if params is None:
        if self.params is None:
            self.init_params()
        params = self.params

    op = (
        self.memory_monitor.monitor_operation("linear_estimator")
        if self.memory_monitor else nullcontext()
    )

    def _run() -> dict:
        validated = validate_params(params)
        peak_local, gx, gy = build_local_peaks(
            self.model, validated, self.same_width, self.atom_types,
        )
        bg_2d = (
            self.background_estimator.get_background_for_linear_estimation()
            if self.background_estimator.use_2d_background else None
        )
        A = build_sparse_matrix(
            peak_local, gx, gy,
            nx=self.nx, ny=self.ny,
            fit_background=self.fit_background,
            num_coordinates=self.num_coordinates,
            x_grid=self.x_grid, y_grid=self.y_grid,
            background_2d=bg_2d,
        )
        target = self._prepare_target_vector(validated)
        solution = solve_system(A, target, non_negative=non_negative, ridge=ridge)
        return self._process_solution(solution, validated)

    with op:
        if not best_effort:
            return _run()
        try:
            return _run()
        except Exception as exc:
            log.warning(
                "linear_estimator failed in best_effort mode; "
                "returning input parameters unchanged: %s", exc,
            )
            return params


def _prepare_target_vector(self, params: dict) -> np.ndarray:
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
    self, solution: np.ndarray, params: dict, update_threshold: float = 0.2,
) -> dict:
    """Apply the LS solution back onto ``params`` (height + optional bg)."""
    if not validate_solution(solution):
        log.warning("Invalid solution obtained, returning original parameters")
        return params

    if self.fit_background:
        if self.background_estimator.use_2d_background:
            bg_scale = float(solution[-1])
            if 0.01 < bg_scale < 100.0:
                self.update_2d_background_scale(bg_scale)
                params["background_scale"] = to_tensor(bg_scale)
                params.pop("background", None)
            else:
                log.warning(
                    "2D background scale out of bounds: %.3f, keeping current scale",
                    bg_scale,
                )
            height_scale = solution[:-1]
        else:
            background, ok = process_background(
                solution, params, self.init_background,
                update_threshold=update_threshold,
            )
            if not ok:
                log.warning(
                    "Background update too large, skipping parameter update with linear estimator",
                )
                return params
            params["background"] = to_tensor(background)
            height_scale = solution[:-1]
    else:
        height_scale = solution

    scale = to_tensor(process_height_scaling(height_scale))
    params["height"] = params["height"] * scale
    self.params = params
    return params


class FitterSolverMixin:
    """NNLS linear estimator (heights + background) for :class:`Fitter`."""

    linear_estimator = linear_estimator
    _prepare_target_vector = _prepare_target_vector
    _process_solution = _process_solution


__all__ = [
    "FitterSolverMixin",
    # Pipeline functions:
    "build_local_peaks",
    "build_sparse_matrix",
    "solve_system",
    "validate_params",
    "validate_solution",
    "process_height_scaling",
    "process_background",
    "linear_estimator",
]
