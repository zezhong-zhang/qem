"""Core sparse-solver tests (no fitting infrastructure)."""

from __future__ import annotations

import numpy as np
import pytest
from scipy.sparse import coo_matrix

from qem.utils.exceptions import DataError
from qem.fit.sparse import SparseSolver, clip_height_scale


def test_sparse_solve_success() -> None:
    matrix = coo_matrix(np.eye(2, dtype=float))
    target = np.array([1.0, 2.0])
    solution = SparseSolver.solve(matrix, target)
    np.testing.assert_allclose(solution, target)


def test_sparse_solve_rejects_singular_system() -> None:
    matrix = coo_matrix(np.array([[1.0, 1.0], [1.0, 1.0]]))
    target = np.array([1.0, 1.0])
    with pytest.raises(DataError, match="singular matrix"):
        SparseSolver.solve(matrix, target)


def test_sparse_solve_validates_target_length() -> None:
    matrix = coo_matrix(np.eye(2, dtype=float))
    with pytest.raises(DataError, match="target length"):
        SparseSolver.solve(matrix, np.array([1.0]))


def test_clip_height_scale_returns_clipped_copy() -> None:
    scale = np.array([0.05, 2.0, 12.0])
    clipped = clip_height_scale(scale, min_scale=0.1, max_scale=10.0)
    np.testing.assert_allclose(clipped, np.array([0.1, 2.0, 10.0]))
    np.testing.assert_allclose(scale, np.array([0.05, 2.0, 12.0]))


