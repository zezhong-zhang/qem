"""
Tests for the linear_solver module.
"""

import numpy as np
import pytest
from scipy.sparse import coo_matrix
from qem.utils import torch_compat as keras

from qem.schema.exceptions import ParameterError, DataError, ValidationError
from qem.fit.linear_solver import (
    ParameterValidator,
    DesignMatrixBuilder,
    LinearSystemSolver,
    SolutionProcessor,
)
from qem.fit.model import GaussianModel
from qem.utils.config import get_config, create_linear_solver_array


class TestParameterValidator:
    """Test parameter validation functionality."""

    def test_valid_params(self):
        """Test validation with valid parameters."""
        params = {
            "pos_x": keras.ops.convert_to_tensor([1.0, 2.0, 3.0]),
            "pos_y": keras.ops.convert_to_tensor([1.0, 2.0, 3.0]),
            "height": keras.ops.convert_to_tensor([1.0, 1.0, 1.0]),
            "width": keras.ops.convert_to_tensor([1.0, 1.0, 1.0]),
        }

        validated = ParameterValidator.validate_params(params)
        assert validated is not None
        assert len(validated["pos_x"]) == 3

    def test_invalid_params_type(self):
        """Test validation with non-dict input."""
        with pytest.raises(ParameterError, match="Parameters must be a dictionary"):
            ParameterValidator.validate_params("invalid")

    def test_missing_required_keys(self):
        """Test validation with missing required keys."""
        params = {"pos_x": [1.0, 2.0]}  # Missing pos_y, height, width

        with pytest.raises(ParameterError, match="Missing required parameters"):
            ParameterValidator.validate_params(params)

    def test_mismatched_lengths(self):
        """Test validation with mismatched array lengths."""
        params = {
            "pos_x": keras.ops.convert_to_tensor([1.0, 2.0]),
            "pos_y": keras.ops.convert_to_tensor([1.0]),  # Different length
            "height": keras.ops.convert_to_tensor([1.0, 1.0]),
            "width": keras.ops.convert_to_tensor([1.0, 1.0]),
        }

        with pytest.raises(
            ParameterError, match="pos_x, pos_y, and height must have same length"
        ):
            ParameterValidator.validate_params(params)

    def test_height_length_mismatch_caught(self):
        """Regression: pos_x == pos_y but height differs — chained comparison
        used to silently accept this; the validator must reject it."""
        params = {
            "pos_x": keras.ops.convert_to_tensor([1.0, 2.0]),
            "pos_y": keras.ops.convert_to_tensor([1.0, 2.0]),
            "height": keras.ops.convert_to_tensor([1.0]),  # Different length
            "width": keras.ops.convert_to_tensor([1.0, 1.0]),
        }

        with pytest.raises(
            ParameterError, match="pos_x, pos_y, and height must have same length"
        ):
            ParameterValidator.validate_params(params)

    def test_nan_values(self):
        """Test validation with NaN values."""
        params = {
            "pos_x": keras.ops.convert_to_tensor([1.0, float("nan")]),
            "pos_y": keras.ops.convert_to_tensor([1.0, 2.0]),
            "height": keras.ops.convert_to_tensor([1.0, 1.0]),
            "width": keras.ops.convert_to_tensor([1.0, 1.0]),
        }

        with pytest.raises(ParameterError, match="contains NaN or infinite values"):
            ParameterValidator.validate_params(params)


class TestDesignMatrixBuilder:
    """Test design matrix building functionality."""

    def setup_method(self):
        """Set up test fixtures."""
        self.model = GaussianModel(dx=1.0)
        self.builder = DesignMatrixBuilder(self.model, nx=50, ny=50)

    def test_build_local_peaks(self):
        """Test building local peaks."""
        params = {
            "pos_x": keras.ops.convert_to_tensor([10.0, 20.0]),
            "pos_y": keras.ops.convert_to_tensor([15.0, 25.0]),
            "height": keras.ops.convert_to_tensor([1.0, 1.0]),
            "width": keras.ops.convert_to_tensor([2.0, 2.0]),
        }
        atom_types = np.array([0, 0])

        peak_local, global_x, global_y, mask = self.builder.build_local_peaks(
            params, same_width=True, atom_types=atom_types
        )

        assert peak_local is not None
        assert global_x is not None
        assert global_y is not None
        assert mask is not None
        assert mask.ndim >= 2

    def test_build_sparse_matrix(self):
        """Test building sparse design matrix."""
        # Create simple test data
        peak_local = keras.ops.ones((5, 5, 2))  # 2 peaks
        global_x = keras.ops.ones((5, 5, 2)) * 10
        global_y = keras.ops.ones((5, 5, 2)) * 10
        mask = keras.ops.ones((5, 5, 2), dtype=bool)

        x_grid = keras.ops.ones((50, 50))
        y_grid = keras.ops.ones((50, 50))

        design_matrix = self.builder.build_sparse_matrix(
            peak_local,
            global_x,
            global_y,
            mask,
            fit_background=True,
            num_coordinates=2,
            x_grid=x_grid,
            y_grid=y_grid,
        )

        assert isinstance(design_matrix, coo_matrix)
        assert design_matrix.shape[0] == 50 * 50
        assert design_matrix.shape[1] == 3  # 2 peaks + 1 background


class TestLinearSystemSolver:
    """Test linear system solving functionality."""

    def test_solve_system_success(self):
        """Test successful system solving."""
        # Create a simple test system
        A = coo_matrix(np.array([[1, 0], [0, 1]], dtype=np.float32))
        b = np.array([1, 2], dtype=np.float32)

        solver = LinearSystemSolver()
        solution = solver.solve_system(A, b)
        assert solution is not None
        assert len(solution) == 2
        np.testing.assert_array_almost_equal(solution, [1, 2])

    def test_solve_system_non_negative(self):
        """Test non-negative constraint solving."""
        A = coo_matrix(np.array([[1, 0], [0, 1]], dtype=np.float32))
        b = np.array([1, 2], dtype=np.float32)

        solver = LinearSystemSolver()
        solution = solver.solve_system(A, b, non_negative=True)
        assert solution is not None
        assert np.all(solution >= 0)

    def test_solve_singular_matrix(self):
        """Test handling of singular matrices."""
        # Create a singular matrix
        A = coo_matrix(np.array([[1, 1], [1, 1]], dtype=np.float32))
        b = np.array([1, 2], dtype=np.float32)  # Inconsistent system

        solver = LinearSystemSolver()
        # The solver might handle this gracefully, so let's just check it doesn't crash
        solution = solver.solve_system(A, b)
        # If it returns a solution, it should be a valid array
        if solution is not None:
            assert isinstance(solution, np.ndarray)
            assert len(solution) == 2

    def test_solve_empty_system(self):
        """Test handling of empty system."""
        A = coo_matrix((0, 0), dtype=np.float32)
        b = np.array([], dtype=np.float32)

        solver = LinearSystemSolver()
        solution = solver.solve_system(A, b)
        assert solution is not None
        assert len(solution) == 0


class TestSolutionProcessor:
    """Test solution processing functionality."""

    def test_validate_solution_valid(self):
        """Test validation of valid solution."""
        solution = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        assert SolutionProcessor.validate_solution(solution) is True

    def test_validate_solution_none(self):
        """Test validation with None solution."""
        assert SolutionProcessor.validate_solution(None) is False

    def test_validate_solution_nan(self):
        """Test validation with NaN values."""
        solution = np.array([1.0, float("nan"), 3.0], dtype=np.float32)
        assert SolutionProcessor.validate_solution(solution) is False

    def test_validate_solution_inf(self):
        """Test validation with infinite values."""
        solution = np.array([1.0, float("inf"), 3.0], dtype=np.float32)
        assert SolutionProcessor.validate_solution(solution) is False

    def test_process_height_scaling(self):
        """Test height scaling processing."""
        height_scale = np.array([0.05, 1.5, 5.0, 15.0], dtype=np.float32)
        processed = SolutionProcessor.process_height_scaling(
            height_scale, min_scale=0.1, max_scale=10.0
        )

        expected = np.array([0.1, 1.5, 5.0, 10.0], dtype=np.float32)
        np.testing.assert_array_almost_equal(processed, expected)

    def test_process_height_scaling_no_clipping(self):
        """Test height scaling with no clipping needed."""
        height_scale = np.array([0.9, 1.0, 1.1], dtype=np.float32)  # Values within default range
        processed = SolutionProcessor.process_height_scaling(height_scale)

        np.testing.assert_array_almost_equal(processed, height_scale)


class TestIntegration:
    """Integration tests for the linear solver module."""

    def test_full_linear_estimation_workflow(self):
        """Test complete linear estimation workflow."""
        # Create a simple test case
        model = GaussianModel(dx=1.0)
        nx, ny = 10, 10
        builder = DesignMatrixBuilder(model, nx, ny)

        params = {
            "pos_x": keras.ops.convert_to_tensor([5.0]),
            "pos_y": keras.ops.convert_to_tensor([5.0]),
            "height": keras.ops.convert_to_tensor([10.0]),
            "width": keras.ops.convert_to_tensor([2.0]),
        }

        # Build design matrix
        peak_local, global_x, global_y, mask = builder.build_local_peaks(
            params, same_width=True, atom_types=np.array([0])
        )

        x_grid = keras.ops.ones((ny, nx))
        y_grid = keras.ops.ones((ny, nx))

        design_matrix = builder.build_sparse_matrix(
            peak_local,
            global_x,
            global_y,
            mask,
            fit_background=False,
            num_coordinates=1,
            x_grid=x_grid,
            y_grid=y_grid,
        )

        # Create target vector
        target = np.ones(nx * ny, dtype=np.float32) * 5.0

        # Solve system
        solver = LinearSystemSolver()
        solution = solver.solve_system(design_matrix, target)
        assert solution is not None
        assert len(solution) == 1  # Single peak

        # Process solution
        processed = SolutionProcessor.process_height_scaling(solution)
        assert processed is not None
