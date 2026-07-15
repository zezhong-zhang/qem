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

import numpy as np
import torch
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import lsqr

from qem.utils.exceptions import DataError, ParameterError
from qem.utils.tensors import release_memory, to_numpy, to_tensor

log = logging.getLogger(__name__)

# Local render window half-width, in units of the peak width (σ). 5σ
# captures ~100% of a Gaussian's mass while keeping the window compact.
PEAK_WINDOW_SIGMA_EXTENT = 5


# ---------------------------------------------------------------------------
# Design matrix
# ---------------------------------------------------------------------------

def build_local_peaks(model, params: dict, same_width: bool, atom_types):
    """Render each atom's peak on its local window with **unit amplitude**.

    Returns ``(peak_local, global_x, global_y)``. ``peak_local`` has
    shape ``(2W+1, 2W+1, N)`` with W = 5σ — wider than the 3σ used
    by the Adam-loop renderer because the linear estimator runs once
    per fit, so accuracy beats speed.

    Rendering with ``height = 1.0`` (rather than the current per-atom
    height) is critical for the LS solver: the LS variable then *is*
    the new height, not a multiplicative scale on the existing height.
    With the scale-factor formulation, when the current heights are
    small (e.g., per-atom VarPro produced near-zero η on a few atoms
    after position update), the design-matrix atom columns are tiny
    relative to the bg column of ones — LS dumps everything into bg
    and proposes ``bg ≈ image.mean()``, which then trips the bg
    validator and the entire LE update is rejected. Observed on
    fivefold: 60/60 inner LE calls rejected, heights and bg frozen
    after the first varpro round.
    """
    pos_x, pos_y = params["pos_x"], params["pos_y"]
    width = params["width"]
    ratio = params.get("ratio")

    # All tensors share pos_x's device — params can come from any
    # accelerator (MPS / CUDA / CPU). atom_types is sometimes a numpy
    # array; promote and migrate if so.
    device = pos_x.device
    if same_width:
        atom_types_t = atom_types
        if not torch.is_tensor(atom_types_t):
            atom_types_t = torch.as_tensor(atom_types_t, dtype=torch.int64)
        atom_types_t = atom_types_t.to(device=device, dtype=torch.int64)
        width = width.to(device)[atom_types_t]
        if ratio is not None:
            ratio = ratio.to(device)[atom_types_t]
    else:
        width = width.to(device)
        if ratio is not None:
            ratio = ratio.to(device)
    # Unit amplitude: shape factor stays purely geometric (depends on
    # width, position; not on height).
    unit_height = torch.ones_like(pos_x)

    window_size = (torch.max(width) * PEAK_WINDOW_SIGMA_EXTENT).to(dtype=torch.int32)
    x = torch.arange(-window_size, window_size + 1, 1, dtype=torch.float32, device=device)
    y = torch.arange(-window_size, window_size + 1, 1, dtype=torch.float32, device=device)
    local_x, local_y = torch.meshgrid(x, y, indexing="xy")

    peak_args = (torch.remainder(pos_x, 1), torch.remainder(pos_y, 1), unit_height, width)
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
    background_2d: np.ndarray | None = None,
) -> coo_matrix:
    """Stack ``(peak, x, y)`` into a scipy ``coo_matrix``.

    Always returns a CPU scipy matrix because the downstream solver
    (:func:`qem.fit.sparse_torch.pg_nnls`) builds a torch CSR from it.
    Inputs may live on any device (CPU/MPS/CUDA); we migrate to CPU at
    entry. This is cheap (one shot per fit) and sidesteps two MPS
    issues at once: ``torch.take`` is not implemented on MPS, and the
    eventual ``to_numpy()`` would force the migration anyway.
    """
    peak_local = peak_local.detach().cpu()
    global_x = global_x.detach().cpu()
    global_y = global_y.detach().cpu()
    if torch.is_tensor(x_grid):
        x_grid = x_grid.detach().cpu()
    if torch.is_tensor(y_grid):
        y_grid = y_grid.detach().cpu()

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

    # Plain advanced indexing rather than torch.take — same semantics,
    # supported on every backend.
    data = peak_local.reshape(-1)[flat_idx]
    gx_valid = global_x.reshape(-1)[flat_idx]
    gy_valid = global_y.reshape(-1)[flat_idx]

    cols = valid[2].to(dtype=torch.int32)
    rows = (
        gy_valid.to(dtype=torch.int32) * nx
        + gx_valid.to(dtype=torch.int32)
    )

    if fit_background:
        bg_rows = (y_grid * nx + x_grid).reshape(-1).to(dtype=torch.int32)
        rows = torch.cat([rows, bg_rows])
        cols = torch.cat([
            cols,
            torch.full((nx * ny,), num_coordinates, dtype=torch.int32),
        ])
        if background_2d is not None:
            bg_data = torch.as_tensor(
                background_2d.ravel(), dtype=torch.float32,
            )
        else:
            bg_data = torch.ones((nx * ny,), dtype=torch.float32)
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
    except (RuntimeError, ValueError, np.linalg.LinAlgError) as exc:
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
    prev_heights: np.ndarray | None = None,
    min_scale: float = 0.05,
    max_scale: float = 20.0,
) -> np.ndarray:
    """Sanity-check the LS height solution.

    Now that ``build_local_peaks`` renders peaks with **unit
    amplitude** (see comment there), ``height_scale`` IS the new
    per-atom height in image-intensity units, not a multiplicative
    scale on the prior. The validator:

    * replaces NaN/Inf with the prior height (so a degenerate atom
      keeps its old value rather than crashing),
    * clamps negative values to 0 (NNLS upstream should already
      ensure non-negativity, but be defensive).

    The legacy ``min_scale`` / ``max_scale`` arguments are kept for
    callsite compatibility but no longer mean the same thing —
    relative bounds make no sense once the LS variable is the height
    itself rather than a scale on it.
    """
    h = to_tensor(height_scale) if isinstance(height_scale, np.ndarray) else height_scale
    if prev_heights is not None:
        prev = to_tensor(prev_heights).to(h)
        h = torch.where(torch.isnan(h) | torch.isinf(h), prev, h)
    else:
        h = torch.where(torch.isnan(h) | torch.isinf(h), torch.zeros_like(h), h)
    h = torch.clamp(h, min=0.0)
    return to_numpy(h)


def process_background(
    solution, params, init_background, *, image_std: float | None = None,
    update_threshold: float = 0.4,
):
    """Validate the scalar background update.

    Returns ``(background, ok)``. The LS solves jointly for ``[heights,
    bg]`` — clamping bg without re-solving heights breaks the invariant
    and corrupts the fit. So this validator is binary: accept the full
    LS update or reject it (keeping previous heights and bg).

    The acceptance test handles two regimes that defeat a single rule:

    * **Near-zero background** (e.g., Au_rod_0_2016: image ∈ [0, 0.3],
      prev_bg ≈ 0.003): fall back to an *absolute* threshold scaled
      by ``image_std``. A relative threshold rejects every LE call
      because rel = ``Δbg / 0.003`` is huge for any non-trivial change.
    * **Otherwise**: relative threshold ``|Δbg / bg_prev| ≤
      2·update_threshold``. Catches model-misspecification overshoots
      (fivefold: prev=3514, LS=6992, rel=99% — reject).

    Negative or non-finite bg → reject. Lower-clamped to
    ``init_background``.
    """
    proposed = float(np.asarray(solution[-1]))
    prev = params["background"]
    prev_val = float(to_numpy(prev)) if hasattr(prev, "shape") else float(prev)
    if not np.isfinite(proposed) or proposed < 0.0:
        return prev_val, False

    # Pick the regime by comparing prev to image scale. ``image_std``
    # is the natural unit; without it we can't differentiate "0.003 is
    # near-zero" from "3514 is significant", so fall through to the
    # relative test (legacy behaviour).
    near_zero_threshold = 0.05 * image_std if image_std is not None else 0.0
    if prev_val < near_zero_threshold:
        # Absolute test: |Δbg| as a fraction of image std.
        abs_change = abs(proposed - prev_val)
        if abs_change > 0.5 * image_std:
            return prev_val, False
    else:
        # Relative test (the original criterion).
        rel = (proposed - prev_val) / (prev_val + 1e-30)
        if abs(rel) > 2 * update_threshold:
            return prev_val, False

    if proposed < init_background:
        proposed = float(init_background)
    return proposed, True


# ---------------------------------------------------------------------------
# Public Fitter methods (mixed in via FitterSolverMixin below)
# ---------------------------------------------------------------------------

def linear_estimator(
    self,
    params: dict | None = None,
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
        # Conservative set covering the linear-estimation path: QEM solver
        # failures (DataError/ParameterError), bad shapes/values/indices,
        # and LinAlgError from the LS solve.
        except (
            DataError, ParameterError, RuntimeError, ValueError, IndexError,
            np.linalg.LinAlgError,
        ) as exc:
            log.warning(
                "linear_estimator failed in best_effort mode (%s); "
                "returning input parameters unchanged: %s",
                type(exc).__name__, exc,
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

    # Anchor every new tensor we install to the device the existing
    # height parameter lives on. ``to_tensor`` returns CPU tensors by
    # default; without this dance the output dict is a CPU/accelerator
    # mongrel and downstream ops (``height * scale``, ``image_tensor -
    # params['background']``) crash on MPS / CUDA.
    height = params["height"]
    device = height.device if torch.is_tensor(height) else torch.device("cpu")
    dtype = height.dtype if torch.is_tensor(height) else torch.float32

    if self.fit_background:
        if self.background_estimator.use_2d_background:
            bg_scale = float(solution[-1])
            if 0.01 < bg_scale < 100.0:
                self.update_2d_background_scale(bg_scale)
                params["background_scale"] = to_tensor(bg_scale).to(device=device, dtype=dtype)
                params.pop("background", None)
            else:
                log.warning(
                    "2D background scale out of bounds: %.3f, keeping current scale",
                    bg_scale,
                )
            height_scale = solution[:-1]
        else:
            # Pass image std so process_background can switch from
            # relative to absolute test when prev_bg is near-zero
            # (e.g. Au_rod_0 with image ∈ [0, 0.3] and bg ≈ 0.003).
            image_std = float(np.asarray(self.image).std())
            background, ok = process_background(
                solution, params, self.init_background,
                image_std=image_std,
            )
            if not ok:
                log.warning(
                    "Background update too large, skipping parameter update with linear estimator",
                )
                return params
            params["background"] = to_tensor(background).to(device=device, dtype=dtype)
            height_scale = solution[:-1]
    else:
        height_scale = solution

    # Peaks were rendered with unit amplitude in build_local_peaks, so
    # the LS solution `height_scale` IS the new heights — not a
    # multiplicative scale on the prior heights. Sanity-clamp via
    # ``process_height_scaling`` (NaN→prior, negative→0).
    new_heights = to_tensor(
        process_height_scaling(height_scale, prev_heights=to_numpy(height))
    ).to(device=device, dtype=dtype)
    params["height"] = new_heights
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
