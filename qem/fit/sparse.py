"""Sparse linear-system helpers.

Pure numpy/scipy.  Used by :mod:`qem.fusion` and the linear-estimation
core; the heavier image-fitting design-matrix builder lives next door
in :mod:`qem.fit.solver`.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import lsq_linear
from scipy.sparse import coo_matrix, issparse, spmatrix
from scipy.sparse.linalg import MatrixRankWarning, spsolve

from qem.utils.exceptions import DataError


Array = NDArray[np.float64]


@dataclass(frozen=True)
class SolverConfig:
    """Tunables for :class:`SparseSolver`."""

    singular_tolerance: float = 1e-8
    dense_rank_check_limit: int = 1024


class SparseSolver:
    """Solve sparse normal equations with consistent QEM error semantics."""

    @staticmethod
    def solve(
        design_matrix: spmatrix,
        target: NDArray[np.floating],
        *,
        non_negative: bool = False,
        config: SolverConfig | None = None,
    ) -> Array:
        """Solve ``design_matrix @ x = target`` (least-squares for tall systems).

        Raises :class:`qem.utils.exceptions.DataError` for singular / ill-conditioned
        systems and shape mismatches; preserves all numerical exceptions
        from scipy.
        """
        cfg = config or SolverConfig()
        if design_matrix.shape == (0, 0):
            return np.asarray([], dtype=float)
        if not issparse(design_matrix):
            design_matrix = coo_matrix(design_matrix)

        matrix = design_matrix.tocsr()
        target_array = np.asarray(target, dtype=float)
        if matrix.shape[0] != target_array.size:
            raise DataError(
                "target length does not match design matrix row count",
                technical_details={
                    "matrix_shape": matrix.shape,
                    "target_shape": target_array.shape,
                },
            )

        try:
            if non_negative:
                result = lsq_linear(matrix, target_array, bounds=(0, np.inf))
                if not result.success:
                    raise DataError(f"Non-negative solver failed: {result.message}")
                solution = result.x
            else:
                normal_matrix = matrix.T @ matrix
                _check_rank(normal_matrix, cfg)
                rhs = matrix.T @ target_array
                with warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter("always", MatrixRankWarning)
                    solution = spsolve(normal_matrix, rhs)
                if any(issubclass(w.category, MatrixRankWarning) for w in caught):
                    raise DataError(
                        "singular matrix: system is underdetermined or ill-conditioned"
                    )

            solution = np.asarray(solution, dtype=float)
            if not np.all(np.isfinite(solution)):
                raise DataError(
                    "singular matrix: system is underdetermined or ill-conditioned"
                )
            return solution

        except DataError:
            raise
        except np.linalg.LinAlgError as exc:
            raise DataError(f"Linear algebra error: {exc}") from exc
        except Exception as exc:
            if "singular" in str(exc).lower():
                raise DataError(
                    "singular matrix: system is underdetermined or ill-conditioned"
                ) from exc
            raise DataError(f"System solving failed: {exc}") from exc


def clip_height_scale(
    scale: NDArray[np.floating],
    *,
    min_scale: float = 0.1,
    max_scale: float = 10.0,
) -> Array:
    """Clip a per-atom height-scale array to ``[min_scale, max_scale]``.

    Returns a copy; never mutates the caller's array.
    """
    return np.clip(np.asarray(scale, dtype=float), min_scale, max_scale)


def _check_rank(normal_matrix: spmatrix, config: SolverConfig) -> None:
    """Cheap dense rank check for small normal matrices only."""
    rows, cols = normal_matrix.shape
    if rows != cols or rows > config.dense_rank_check_limit:
        return
    dense = normal_matrix.toarray()
    if dense.size == 0:
        return
    if np.linalg.matrix_rank(dense, tol=config.singular_tolerance) < rows:
        raise DataError(
            "singular matrix: system is underdetermined or ill-conditioned"
        )


__all__ = ["SparseSolver", "SolverConfig", "clip_height_scale"]
