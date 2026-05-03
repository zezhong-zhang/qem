"""Sparse linear-estimation helpers for QEM image fitting.

The classes in this module keep the high-level image-fitting code focused on
workflow orchestration:

* :class:`ParameterValidator` checks model parameter shape and finiteness.
* :class:`DesignMatrixBuilder` creates the sparse peak/background design matrix.
* :class:`LinearSystemSolver` solves constrained or unconstrained systems.
* :class:`SolutionProcessor` validates and bounds height-scale updates.

The public class and method names intentionally match the previous
implementation so existing callers continue to work.
"""

from __future__ import annotations

from dataclasses import dataclass
import logging
import warnings
from typing import Any, Mapping, MutableMapping

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import lsq_linear
from scipy.sparse import coo_matrix, issparse, spmatrix
from scipy.sparse.linalg import MatrixRankWarning, spsolve

from qem.exceptions import DataError, ParameterError
from qem.utils import safe_convert_to_numpy

logger = logging.getLogger(__name__)

Params = MutableMapping[str, Any]
Array = NDArray[np.float64]


def _keras_ops():
    """Import Keras ops lazily so sparse solver utilities remain lightweight."""
    from keras import ops

    return ops


@dataclass(frozen=True)
class LinearSolverConfig:
    """Configuration for :class:`LinearSystemSolver`.

    Parameters
    ----------
    singular_tolerance
        Minimum reciprocal condition number accepted for dense, small systems.
    dense_rank_check_limit
        Maximum number of unknowns for the pre-solve dense rank check. Larger
        systems rely on sparse solver warnings and finite-solution validation to
        avoid expensive densification.
    """

    singular_tolerance: float = 1e-12
    dense_rank_check_limit: int = 512


class ParameterValidator:
    """Validate input parameters for linear estimation."""

    REQUIRED_KEYS = ("pos_x", "pos_y", "height", "width")

    @staticmethod
    def validate_params(params: Mapping[str, Any]) -> Params:
        """Validate and return model parameters.

        Parameters
        ----------
        params
            Mapping with at least ``pos_x``, ``pos_y``, ``height`` and
            ``width`` entries. Values may be Keras tensors or NumPy-compatible
            arrays.

        Returns
        -------
        MutableMapping[str, Any]
            The original parameter mapping, after validation.

        Raises
        ------
        ParameterError
            If required keys are absent, array lengths differ, or values contain
            NaN/Inf.
        """
        if not isinstance(params, MutableMapping):
            raise ParameterError("Parameters must be a dictionary")

        missing_keys = [key for key in ParameterValidator.REQUIRED_KEYS if key not in params]
        if missing_keys:
            raise ParameterError(
                f"Missing required parameters: {missing_keys}",
                suggestion="Please provide all required parameters: pos_x, pos_y, height, width",
            )

        lengths = {
            key: np.asarray(safe_convert_to_numpy(params[key])).shape[0]
            for key in ("pos_x", "pos_y", "height")
        }
        if lengths["pos_x"] != lengths["pos_y"]:
            raise ParameterError("pos_x and pos_y must have same length")
        if lengths["pos_x"] != lengths["height"]:
            raise ParameterError("pos_x and height must have same length")

        for key in ParameterValidator.REQUIRED_KEYS:
            values = np.asarray(safe_convert_to_numpy(params[key]))
            if not np.all(np.isfinite(values)):
                raise ParameterError(f"Parameter '{key}' contains NaN or infinite values")

        return params


class DesignMatrixBuilder:
    """Build sparse design matrices for linear image estimation."""

    def __init__(self, model: Any, nx: int, ny: int) -> None:
        """Create a design-matrix builder.

        Parameters
        ----------
        model
            Model object exposing ``model_fn``.
        nx, ny
            Image width and height in pixels.
        """
        self.model = model
        self.nx = int(nx)
        self.ny = int(ny)

    def build_local_peaks(
        self,
        params: Mapping[str, Any],
        same_width: bool,
        atom_types: NDArray[np.integer],
    ) -> tuple[Any, Any, Any, Any]:
        """Build local peak patches and their image-coordinate masks.

        Parameters
        ----------
        params
            Model parameter mapping.
        same_width
            Whether widths are stored per atom type rather than per coordinate.
        atom_types
            Atom-type index for each coordinate.

        Returns
        -------
        tuple
            ``(peak_local, global_x, global_y, mask)`` tensors.
        """
        ops = _keras_ops()
        pos_x = params["pos_x"]
        pos_y = params["pos_y"]
        width = params["width"]
        height = params["height"]
        ratio = params.get("ratio")

        if same_width:
            width = width[atom_types]
            if ratio is not None:
                ratio = ratio[atom_types]

        window_size = ops.cast(ops.max(width) * 5, dtype="int32")
        x = ops.arange(-window_size, window_size + 1, 1, dtype="float32")
        y = ops.arange(-window_size, window_size + 1, 1, dtype="float32")
        local_x, local_y = ops.meshgrid(x, y, indexing="xy")

        input_params = (ops.mod(pos_x, 1), ops.mod(pos_y, 1), height, width)
        if ratio is not None:
            input_params += (ratio,)

        peak_local = self.model.model_fn(local_x[..., None], local_y[..., None], *input_params)

        pos_x_int = ops.floor(pos_x)
        pos_y_int = ops.floor(pos_y)
        global_x = ops.expand_dims(local_x, -1) + pos_x_int
        global_y = ops.expand_dims(local_y, -1) + pos_y_int

        mask = (
            (global_x >= 0)
            & (global_x < self.nx)
            & (global_y >= 0)
            & (global_y < self.ny)
        )
        return peak_local, global_x, global_y, mask

    def build_sparse_matrix(
        self,
        peak_local: Any,
        global_x: Any,
        global_y: Any,
        mask: Any,
        fit_background: bool,
        num_coordinates: int,
        x_grid: Any,
        y_grid: Any,
    ) -> coo_matrix:
        """Build a sparse design matrix from local peak patches.

        Parameters
        ----------
        peak_local
            Local peak intensity tensors.
        global_x, global_y
            Pixel-coordinate tensors for each local peak value.
        mask
            Boolean tensor indicating which peak samples are inside the image.
        fit_background
            Whether to append a constant background column.
        num_coordinates
            Number of atomic coordinates.
        x_grid, y_grid
            Full image coordinate grids.

        Returns
        -------
        scipy.sparse.coo_matrix
            Sparse matrix with one row per image pixel and one column per peak,
            plus an optional background column.
        """
        ops = _keras_ops()
        valid_indices = ops.where(mask)
        shape = ops.shape(peak_local)
        flat_indices = (
            valid_indices[0] * (shape[1] * shape[2])
            + valid_indices[1] * shape[2]
            + valid_indices[2]
        )

        data_tensor = ops.take(ops.reshape(peak_local, (-1,)), flat_indices)
        global_x_valid = ops.take(ops.reshape(global_x, (-1,)), flat_indices)
        global_y_valid = ops.take(ops.reshape(global_y, (-1,)), flat_indices)

        rows_tensor = ops.cast(global_y_valid, "int32") * self.nx + ops.cast(
            global_x_valid, "int32"
        )
        cols_tensor = valid_indices[2]
        matrix_shape = (self.nx * self.ny, int(num_coordinates))

        if fit_background:
            pixel_count = self.nx * self.ny
            background_rows = ops.reshape(y_grid, (-1,)) * self.nx + ops.reshape(x_grid, (-1,))
            rows_tensor = ops.concatenate([rows_tensor, ops.cast(background_rows, "int32")])
            cols_tensor = ops.concatenate(
                [cols_tensor, ops.full((pixel_count,), num_coordinates, dtype="int32")]
            )
            data_tensor = ops.concatenate(
                [data_tensor, ops.ones((pixel_count,), dtype="float32")]
            )
            matrix_shape = (pixel_count, int(num_coordinates) + 1)

        return coo_matrix(
            (
                safe_convert_to_numpy(data_tensor),
                (safe_convert_to_numpy(rows_tensor), safe_convert_to_numpy(cols_tensor)),
            ),
            shape=matrix_shape,
        )


class LinearSystemSolver:
    """Solve sparse linear systems with consistent QEM errors."""

    def __init__(self, config: LinearSolverConfig | None = None) -> None:
        self.config = config or LinearSolverConfig()

    @staticmethod
    def solve_system(
        design_matrix: spmatrix,
        target: NDArray[np.floating],
        non_negative: bool = False,
        config: LinearSolverConfig | None = None,
    ) -> Array:
        """Solve a linear system.

        Parameters
        ----------
        design_matrix
            Sparse design matrix.
        target
            Flattened target image values.
        non_negative
            If ``True``, solve a bounded least-squares problem with
            non-negative coefficients.
        config
            Optional solver configuration.

        Returns
        -------
        numpy.ndarray
            Solution vector.

        Raises
        ------
        DataError
            If the system is singular, ill-conditioned, or the solver fails.
        """
        cfg = config or LinearSolverConfig()
        if design_matrix.shape == (0, 0):
            return np.asarray([], dtype=float)
        if not issparse(design_matrix):
            design_matrix = coo_matrix(design_matrix)

        matrix = design_matrix.tocsr()
        target_array = np.asarray(target, dtype=float)
        if matrix.shape[0] != target_array.size:
            raise DataError(
                "target length does not match design matrix row count",
                technical_details={"matrix_shape": matrix.shape, "target_shape": target_array.shape},
            )

        try:
            if non_negative:
                result = lsq_linear(matrix, target_array, bounds=(0, np.inf))
                if not result.success:
                    raise DataError(f"Non-negative solver failed: {result.message}")
                solution = result.x
            else:
                normal_matrix = matrix.T @ matrix
                _raise_if_small_system_is_rank_deficient(normal_matrix, cfg)
                rhs = matrix.T @ target_array
                with warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter("always", MatrixRankWarning)
                    solution = spsolve(normal_matrix, rhs)
                if any(issubclass(item.category, MatrixRankWarning) for item in caught):
                    raise DataError("singular matrix: system is underdetermined or ill-conditioned")

            solution = np.asarray(solution, dtype=float)
            if not np.all(np.isfinite(solution)):
                raise DataError("singular matrix: system is underdetermined or ill-conditioned")
            return solution

        except DataError:
            raise
        except np.linalg.LinAlgError as exc:
            raise DataError(f"Linear algebra error: {exc}") from exc
        except Exception as exc:
            if "singular" in str(exc).lower():
                raise DataError("singular matrix: system is underdetermined or ill-conditioned") from exc
            raise DataError(f"System solving failed: {exc}") from exc


def _raise_if_small_system_is_rank_deficient(
    normal_matrix: spmatrix,
    config: LinearSolverConfig,
) -> None:
    """Run a dense rank check only when it is cheap enough to be practical."""
    rows, cols = normal_matrix.shape
    if rows != cols or rows > config.dense_rank_check_limit:
        return

    dense = normal_matrix.toarray()
    if dense.size == 0:
        return

    rank = np.linalg.matrix_rank(dense, tol=config.singular_tolerance)
    if rank < rows:
        raise DataError("singular matrix: system is underdetermined or ill-conditioned")


class SolutionProcessor:
    """Validate and constrain linear system solutions."""

    @staticmethod
    def validate_solution(solution: NDArray[np.floating] | None) -> bool:
        """Return whether a solution vector is finite.

        Parameters
        ----------
        solution
            Solution array returned by :class:`LinearSystemSolver`.

        Returns
        -------
        bool
            ``True`` when the solution is present and finite.
        """
        if solution is None:
            return False

        if not np.all(np.isfinite(solution)):
            logger.warning("Solution contains NaN or infinite values")
            return False

        return True

    @staticmethod
    def process_height_scaling(
        height_scale: NDArray[np.floating],
        min_scale: float = 0.1,
        max_scale: float = 10.0,
    ) -> NDArray[np.floating]:
        """Clip height scaling factors to accepted bounds.

        Parameters
        ----------
        height_scale
            Raw height scaling factors.
        min_scale
            Minimum allowed scale factor.
        max_scale
            Maximum allowed scale factor.

        Returns
        -------
        numpy.ndarray
            Clipped height scaling factors.
        """
        height_scale = np.asarray(height_scale, dtype=float)
        too_small = int(np.count_nonzero(height_scale < min_scale))
        too_large = int(np.count_nonzero(height_scale > max_scale))

        clipped = np.clip(height_scale, min_scale, max_scale)

        if too_small:
            logger.warning(
                "Clipped %d height scale values below %.2f. Consider improving peak initialization.",
                too_small,
                min_scale,
            )
        if too_large:
            logger.warning(
                "Clipped %d height scale values above %.2f. Linear estimation may be inaccurate.",
                too_large,
                max_scale,
            )

        total_clipped = too_small + too_large
        if height_scale.size and total_clipped > height_scale.size * 0.3:
            logger.warning(
                "Over 30%% of height values were clipped (%d/%d). "
                "Consider refining peak positions or checking model parameters.",
                total_clipped,
                height_scale.size,
            )

        return clipped
