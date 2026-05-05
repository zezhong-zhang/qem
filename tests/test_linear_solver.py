"""Tests for qem.fit.solver — the function-based linear estimator pipeline.

(Was a class-namespaced API in the past — DesignMatrixBuilder /
LinearSystemSolver / ParameterValidator / SolutionProcessor — collapsed
to module-level functions for the same reason Linus collapses
@staticmethod-only classes: a class with no state is a function.)
"""

import numpy as np
import pytest
import torch
from scipy.sparse import coo_matrix

from qem.fit.model import GaussianModel
from qem.fit.solver import (
    build_local_peaks,
    build_sparse_matrix,
    process_height_scaling,
    solve_system,
    validate_params,
    validate_solution,
)
from qem.utils.exceptions import ParameterError


# ---------------------------------------------------------------------------
# validate_params
# ---------------------------------------------------------------------------

class TestValidateParams:
    def test_valid_params(self):
        params = {
            "pos_x": torch.as_tensor([1.0, 2.0, 3.0]),
            "pos_y": torch.as_tensor([1.0, 2.0, 3.0]),
            "height": torch.as_tensor([1.0, 1.0, 1.0]),
            "width": torch.as_tensor([1.0, 1.0, 1.0]),
        }
        validated = validate_params(params)
        assert len(validated["pos_x"]) == 3

    def test_invalid_params_type(self):
        with pytest.raises(ParameterError, match="Parameters must be a dictionary"):
            validate_params("invalid")  # type: ignore[arg-type]

    def test_missing_required_keys(self):
        params = {"pos_x": torch.as_tensor([1.0, 2.0])}
        with pytest.raises(ParameterError, match="Missing required parameters"):
            validate_params(params)

    def test_mismatched_lengths(self):
        params = {
            "pos_x": torch.as_tensor([1.0, 2.0]),
            "pos_y": torch.as_tensor([1.0]),
            "height": torch.as_tensor([1.0, 1.0]),
            "width": torch.as_tensor([1.0, 1.0]),
        }
        with pytest.raises(
            ParameterError, match="pos_x, pos_y, and height must have same length",
        ):
            validate_params(params)

    def test_height_length_mismatch_caught(self):
        # Regression: pos_x == pos_y but height differs — chained comparison
        # used to silently accept this; the validator must reject it.
        params = {
            "pos_x": torch.as_tensor([1.0, 2.0]),
            "pos_y": torch.as_tensor([1.0, 2.0]),
            "height": torch.as_tensor([1.0]),
            "width": torch.as_tensor([1.0, 1.0]),
        }
        with pytest.raises(
            ParameterError, match="pos_x, pos_y, and height must have same length",
        ):
            validate_params(params)

    def test_nan_values(self):
        params = {
            "pos_x": torch.as_tensor([1.0, float("nan")]),
            "pos_y": torch.as_tensor([1.0, 2.0]),
            "height": torch.as_tensor([1.0, 1.0]),
            "width": torch.as_tensor([1.0, 1.0]),
        }
        with pytest.raises(ParameterError, match="contains NaN or infinite values"):
            validate_params(params)


# ---------------------------------------------------------------------------
# build_local_peaks / build_sparse_matrix
# ---------------------------------------------------------------------------

class TestBuildDesignMatrix:
    def setup_method(self):
        self.model = GaussianModel(dx=1.0)
        self.nx, self.ny = 50, 50

    def test_build_local_peaks(self):
        params = {
            "pos_x": torch.as_tensor([10.0, 20.0]),
            "pos_y": torch.as_tensor([15.0, 25.0]),
            "height": torch.as_tensor([1.0, 1.0]),
            "width": torch.as_tensor([2.0, 2.0]),
        }
        atom_types = np.array([0, 0])
        peak_local, gx, gy = build_local_peaks(
            self.model, params, same_width=True, atom_types=atom_types,
        )
        assert peak_local is not None
        assert gx is not None
        assert gy is not None

    def test_build_sparse_matrix(self):
        peak_local = torch.ones((5, 5, 2))
        gx = torch.ones((5, 5, 2)) * 10
        gy = torch.ones((5, 5, 2)) * 10
        x_grid = torch.ones((50, 50))
        y_grid = torch.ones((50, 50))
        design_matrix = build_sparse_matrix(
            peak_local, gx, gy,
            nx=self.nx, ny=self.ny,
            fit_background=True, num_coordinates=2,
            x_grid=x_grid, y_grid=y_grid,
        )
        assert isinstance(design_matrix, coo_matrix)
        assert design_matrix.shape[0] == 50 * 50
        assert design_matrix.shape[1] == 3  # 2 peaks + 1 background


# ---------------------------------------------------------------------------
# solve_system
# ---------------------------------------------------------------------------

class TestSolveSystem:
    def test_solve_system_success(self):
        # Default non-negative path → PG-NNLS.
        A = coo_matrix(np.array([[1, 0], [0, 1]], dtype=np.float32))
        b = np.array([1, 2], dtype=np.float32)
        solution = solve_system(A, b)
        assert solution is not None
        assert len(solution) == 2
        np.testing.assert_array_almost_equal(solution, [1, 2], decimal=2)

    def test_solve_system_non_negative(self):
        A = coo_matrix(np.array([[1, 0], [0, 1]], dtype=np.float32))
        b = np.array([1, 2], dtype=np.float32)
        solution = solve_system(A, b, non_negative=True)
        assert solution is not None
        assert np.all(solution >= 0)

    def test_solve_singular_matrix(self):
        A = coo_matrix(np.array([[1, 1], [1, 1]], dtype=np.float32))
        b = np.array([1, 2], dtype=np.float32)
        solution = solve_system(A, b)
        if solution is not None:
            assert isinstance(solution, np.ndarray)
            assert len(solution) == 2

    def test_solve_empty_system(self):
        A = coo_matrix((0, 0), dtype=np.float32)
        b = np.array([], dtype=np.float32)
        solution = solve_system(A, b)
        assert solution is not None
        assert len(solution) == 0


# ---------------------------------------------------------------------------
# validate_solution / process_height_scaling
# ---------------------------------------------------------------------------

class TestSolutionPostProcess:
    def test_validate_solution_valid(self):
        assert validate_solution(np.array([1.0, 2.0, 3.0])) is True

    def test_validate_solution_none(self):
        assert validate_solution(None) is False

    def test_validate_solution_nan(self):
        assert validate_solution(np.array([1.0, float("nan"), 3.0])) is False

    def test_validate_solution_inf(self):
        assert validate_solution(np.array([1.0, float("inf"), 3.0])) is False

    def test_process_height_scaling(self):
        h = np.array([0.05, 1.5, 5.0, 15.0], dtype=np.float32)
        out = process_height_scaling(h, min_scale=0.1, max_scale=10.0)
        np.testing.assert_array_almost_equal(out, [0.1, 1.5, 5.0, 10.0])

    def test_process_height_scaling_no_clipping(self):
        h = np.array([0.9, 1.0, 1.1], dtype=np.float32)
        out = process_height_scaling(h)
        np.testing.assert_array_almost_equal(out, h)


# ---------------------------------------------------------------------------
# Integration: full pipeline
# ---------------------------------------------------------------------------

class TestIntegration:
    def test_full_linear_estimation_workflow(self):
        model = GaussianModel(dx=1.0)
        nx, ny = 10, 10
        params = {
            "pos_x": torch.as_tensor([5.0]),
            "pos_y": torch.as_tensor([5.0]),
            "height": torch.as_tensor([10.0]),
            "width": torch.as_tensor([2.0]),
        }
        peak_local, gx, gy = build_local_peaks(
            model, params, same_width=True, atom_types=np.array([0]),
        )
        x_grid = torch.ones((ny, nx))
        y_grid = torch.ones((ny, nx))
        A = build_sparse_matrix(
            peak_local, gx, gy,
            nx=nx, ny=ny,
            fit_background=False, num_coordinates=1,
            x_grid=x_grid, y_grid=y_grid,
        )
        target = np.ones(nx * ny, dtype=np.float32) * 5.0
        solution = solve_system(A, target)
        assert solution is not None
        assert len(solution) == 1
        out = process_height_scaling(solution)
        assert out is not None
