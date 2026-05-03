"""Core tests for sparse linear solving without importing Keras."""

from __future__ import annotations

import numpy as np
import pytest
from scipy.sparse import coo_matrix

from qem.backend_utils import detect_available_backends
from qem.exceptions import DataError
from qem.linear_solver import LinearSystemSolver, SolutionProcessor


def test_sparse_solver_success() -> None:
    """Solve a simple full-rank sparse system."""
    matrix = coo_matrix(np.eye(2, dtype=float))
    target = np.array([1.0, 2.0])

    solution = LinearSystemSolver.solve_system(matrix, target)

    np.testing.assert_allclose(solution, target)


def test_sparse_solver_rejects_singular_system() -> None:
    """Raise a QEM data error for rank-deficient normal equations."""
    matrix = coo_matrix(np.array([[1.0, 1.0], [1.0, 1.0]]))
    target = np.array([1.0, 1.0])

    with pytest.raises(DataError, match="singular matrix"):
        LinearSystemSolver.solve_system(matrix, target)


def test_sparse_solver_validates_target_length() -> None:
    """Reject targets that cannot align with matrix rows."""
    matrix = coo_matrix(np.eye(2, dtype=float))

    with pytest.raises(DataError, match="target length"):
        LinearSystemSolver.solve_system(matrix, np.array([1.0]))


def test_process_height_scaling_returns_clipped_copy() -> None:
    """Clip out-of-range scale factors without mutating caller data."""
    scale = np.array([0.05, 2.0, 12.0])

    clipped = SolutionProcessor.process_height_scaling(scale, min_scale=0.1, max_scale=10.0)

    np.testing.assert_allclose(clipped, np.array([0.1, 2.0, 10.0]))
    np.testing.assert_allclose(scale, np.array([0.05, 2.0, 12.0]))


def test_backend_detection_handles_broken_optional_backends() -> None:
    """Backend detection should return a list instead of raising on import errors."""
    assert isinstance(detect_available_backends(), list)
