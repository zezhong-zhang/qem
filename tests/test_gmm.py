#!/usr/bin/env python3
"""
Comprehensive pytest test suite for Gaussian Mixture Model implementation.
"""

import pytest
import numpy as np
import matplotlib.pyplot as plt
import sys
import os
from unittest.mock import patch, MagicMock

# Add the local qem module to path
sys.path.insert(0, '.')

# Import the modules we want to test
from qem.analysis.gaussian_mixture_model import GaussianMixtureModel, GaussianComponents, safe_ln

class TestGaussianMixtureModel:
    """Test cases for GaussianMixtureModel class."""
    
    @pytest.fixture
    def synthetic_data(self):
        """Create synthetic test data with known components."""
        np.random.seed(42)
        # Three well-separated Gaussian components
        data1 = np.random.normal(2, 0.5, 100)
        data2 = np.random.normal(6, 0.8, 150) 
        data3 = np.random.normal(10, 0.6, 80)
        
        return np.concatenate([data1, data2, data3]).reshape(-1, 1)
    
    @pytest.fixture
    def simple_data(self):
        """Create simple 2-component data."""
        np.random.seed(123)
        data1 = np.random.normal(1, 0.3, 50)
        data2 = np.random.normal(4, 0.4, 70)
        return np.concatenate([data1, data2]).reshape(-1, 1)
    
    def test_gmm_initialization(self, synthetic_data):
        """Test GMM initialization."""
        gmm = GaussianMixtureModel(synthetic_data)
        
        assert gmm.cross_sections is not None
        assert gmm.cross_sections.shape == synthetic_data.shape
        assert gmm.fit_result is None
        assert gmm.electron_dose is None
    
    def test_gmm_initialization_with_electron_dose(self, synthetic_data):
        """Test GMM initialization with electron dose."""
        electron_dose = 1000.0
        gmm = GaussianMixtureModel(synthetic_data, electron_dose=electron_dose)
        
        assert gmm.electron_dose == electron_dose
    
    def test_initialize_fitting_conditions(self, synthetic_data):
        """Test initialization of fitting conditions."""
        gmm = GaussianMixtureModel(synthetic_data)
        
        gmm.initialize_fitting_conditions(
            num_components=5,
            data_channels=None,
            optimization_metric="icl",
            scoring_methods=["icl", "aic", "bic"],
            initialization_method="middle",
            convergence_tolerance=1e-3,
            max_iterations=100,
            initial_weights=None,
            initial_means=None,
            initial_widths=None,
            step_sizes=None,
            constraints=[]
        )
        
        assert len(gmm.component_range) == 5
        assert gmm.scoring_methods == ["icl", "aic", "bic", "nllh"]  # nllh should be added
        assert gmm.convergence_tolerance == 1e-3
        assert gmm.max_iterations == 100
        assert gmm.optimization_metric == "icl"
    
    def test_fit_gaussian_mixture_model_basic(self, simple_data):
        """Test basic GMM fitting."""
        gmm = GaussianMixtureModel(simple_data)
        
        # Fit with small number of components for faster testing
        gmm.fit_gaussian_mixture_model(
            num_components=3,
            scoring_methods=["icl", "aic"],
            max_iterations=50,  # Reduced for testing
            use_first_local_minimum=True
        )
        
        # Check that fitting completed
        assert gmm.fit_result is not None
        assert hasattr(gmm, 'recommended_components')
        assert gmm.recommended_components >= 1
        assert gmm.recommended_components <= 3
        
        # Check that results have expected structure
        assert len(gmm.fit_result.weight) == 3
        assert len(gmm.fit_result.mean) == 3
        assert len(gmm.fit_result.width) == 3
        assert "icl" in gmm.fit_result.score
        assert "aic" in gmm.fit_result.score
    
    def test_find_first_local_minimum(self, synthetic_data):
        """Test first local minimum detection."""
        gmm = GaussianMixtureModel(synthetic_data)
        
        # Test with synthetic scores that have a clear local minimum
        scores = [10.0, 8.0, 6.0, 7.0, 8.5, 9.0]  # Local min at index 2 (3 components)
        result = gmm._find_first_local_minimum(scores)
        assert result == 3  # 1-indexed
        
        # Test with no local minimum (should return global minimum)
        scores = [10.0, 8.0, 6.0, 4.0, 2.0, 1.0]  # Monotonically decreasing
        result = gmm._find_first_local_minimum(scores)
        assert result == 6  # Global minimum
    
    def test_get_optimal_components(self, simple_data):
        """Test optimal component selection methods."""
        gmm = GaussianMixtureModel(simple_data)
        gmm.fit_gaussian_mixture_model(num_components=3, max_iterations=20)
        
        # Test recommendation method
        rec_components = gmm.get_optimal_components("recommendation")
        assert isinstance(rec_components, (int, np.integer))
        assert 1 <= rec_components <= 3
        
        # Test global minimum method
        global_min = gmm.get_optimal_components("global_min")
        assert isinstance(global_min, (int, np.integer))
        assert 1 <= global_min <= 3
        
        # Test invalid method
        with pytest.raises(ValueError):
            gmm.get_optimal_components("invalid_method")
    
    def test_get_optimal_components_no_fit(self, simple_data):
        """Test that get_optimal_components raises error when no fit is available."""
        gmm = GaussianMixtureModel(simple_data)
        
        with pytest.raises(ValueError, match="Please run fit_gaussian_mixture_model first"):
            gmm.get_optimal_components("recommendation")
    
    def test_initialize_weights(self, synthetic_data):
        """Test weight initialization."""
        gmm = GaussianMixtureModel(synthetic_data)
        gmm.initialize_fitting_conditions(
            num_components=3, data_channels=None, optimization_metric="icl",
            scoring_methods=["icl"], initialization_method="middle",
            convergence_tolerance=1e-3, max_iterations=100,
            initial_weights=None, initial_means=None, initial_widths=None,
            step_sizes=None, constraints=[]
        )
        
        weights = gmm._initialize_weights(3)
        assert len(weights) == 3
        assert np.allclose(weights.sum(), 1.0)  # Weights should sum to 1
        assert all(w > 0 for w in weights)  # All weights should be positive
    
    def test_initialize_means_different_methods(self, synthetic_data):
        """Test different mean initialization methods."""
        gmm = GaussianMixtureModel(synthetic_data)
        gmm.initialize_fitting_conditions(
            num_components=3, data_channels=None, optimization_metric="icl",
            scoring_methods=["icl"], initialization_method="equionce",
            convergence_tolerance=1e-3, max_iterations=100,
            initial_weights=None, initial_means=None, initial_widths=None,
            step_sizes=None, constraints=[]
        )
        
        # Test equionce method
        means = gmm._initialize_means("equionce", np.array([[2.0]]), 3)
        assert len(means) == 1  # Returns list with one initialization
        assert means[0].shape == (3, 1)
        
        # Test middle method
        means = gmm._initialize_means("middle", np.array([[2.0], [4.0]]), 3)
        assert len(means) == 3  # Returns multiple initializations
    
    def test_plotting_methods_exist(self, simple_data):
        """Test that plotting methods exist and can be called without errors."""
        gmm = GaussianMixtureModel(simple_data)
        gmm.fit_gaussian_mixture_model(num_components=2, max_iterations=10)
        
        # Test that plotting method exists
        assert hasattr(gmm, 'plot_interactive_gmm_selection')
        
        # Test non-interactive plotting (should not raise errors)
        with patch('matplotlib.pyplot.show'):
            with patch('matplotlib.pyplot.subplots') as mock_subplots:
                # Mock the subplots return
                mock_fig = MagicMock()
                mock_ax1 = MagicMock()
                mock_ax2 = MagicMock()
                mock_subplots.return_value = (mock_fig, (mock_ax1, mock_ax2))
                
                # Mock hist method
                mock_ax1.hist.return_value = (None, None, None)
                
                try:
                    result = gmm.plot_interactive_gmm_selection(
                        simple_data, "test_element", 
                        save_results=False, interactive_selection=False
                    )
                    assert isinstance(result, (int, np.integer))
                except Exception as e:
                    # If there are matplotlib/widget issues, that's expected in testing
                    assert "widget" in str(e).lower() or "backend" in str(e).lower()


class TestGaussianComponents:
    """Test cases for GaussianComponents class."""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample data for GaussianComponents testing."""
        np.random.seed(42)
        return np.random.normal(0, 1, (100, 1))
    
    def test_gaussian_components_init(self, sample_data):
        """Test GaussianComponents initialization."""
        weights = np.array([0.6, 0.4])
        means = np.array([[1.0], [3.0]])
        variances = np.array([[1.0], [2.0]])
        
        gc = GaussianComponents(weights, means, variances, sample_data)
        
        assert gc.weights.shape == (2,)
        assert gc.means.shape == (2, 1)
        assert gc.variances.shape == (2, 1)
        assert gc.num_components == 2
        assert gc.num_data_points == 100
        assert gc.num_dimensions == 1
        assert not gc.has_failed
    
    def test_gaussian_components_responsibilities(self, sample_data):
        """Test responsibility calculation and setting."""
        weights = np.array([0.5, 0.5])
        means = np.array([[1.0], [3.0]])
        variances = np.array([[1.0], [1.0]])
        
        gc = GaussianComponents(weights, means, variances, sample_data)
        
        # Check that responsibilities were calculated
        assert gc.responsibilities is not None
        assert gc.responsibilities.shape == (2, 100)
        
        # Responsibilities should sum to 1 for each data point
        resp_sums = gc.responsibilities.sum(axis=0)
        assert np.allclose(resp_sums, 1.0, rtol=1e-10)
    
    def test_maximization_step(self, sample_data):
        """Test maximization step."""
        weights = np.array([0.5, 0.5])
        means = np.array([[1.0], [3.0]])
        variances = np.array([[1.0], [1.0]])
        
        gc = GaussianComponents(weights, means, variances, sample_data)
        
        # Store original values
        orig_weights = gc.weights.copy()
        orig_means = gc.means.copy()
        orig_variances = gc.variances.copy()
        
        # Run maximization step
        gc.maximization_step([1.0, [1.0], [1.0]], [])
        
        # Parameters should have been updated
        assert not np.array_equal(gc.weights, orig_weights)
        assert not np.array_equal(gc.means, orig_means)
        assert not np.array_equal(gc.variances, orig_variances)


class TestUtilityFunctions:
    """Test utility functions."""
    
    def test_safe_ln(self):
        """Test safe natural logarithm function."""
        # Test normal values
        x = np.array([1.0, 2.0, np.e])
        result = safe_ln(x)
        expected = np.log([1.0, 2.0, np.e])
        assert np.allclose(result, expected)
        
        # Test with very small values (should be clipped)
        x = np.array([1e-400, 1.0, 1e-300])
        result = safe_ln(x)
        # Should not raise warnings or errors
        assert len(result) == 3
        assert not np.any(np.isnan(result))
        assert not np.any(np.isinf(result))


class TestIntegrationTests:
    """Integration tests that test the full workflow."""
    
    def test_full_gmm_workflow(self):
        """Test the complete GMM workflow from data to results."""
        # Create synthetic data with known structure
        np.random.seed(42)
        data1 = np.random.normal(2, 0.5, 80)
        data2 = np.random.normal(6, 0.7, 120)
        test_data = np.concatenate([data1, data2]).reshape(-1, 1)
        
        # Initialize and fit GMM
        gmm = GaussianMixtureModel(test_data)
        gmm.fit_gaussian_mixture_model(
            num_components=4,  # Test with more components than true (2)
            scoring_methods=["icl", "aic", "bic"],
            initialization_method="middle",
            max_iterations=100,
            use_first_local_minimum=True
        )
        
        # Check that fitting completed successfully
        assert gmm.fit_result is not None
        assert hasattr(gmm, 'recommended_components')
        
        # Check that all scoring methods were calculated
        assert "icl" in gmm.fit_result.score
        assert "aic" in gmm.fit_result.score
        assert "bic" in gmm.fit_result.score
        assert "nllh" in gmm.fit_result.score
        
        # Check that we can get optimal components
        optimal = gmm.get_optimal_components("recommendation")
        assert isinstance(optimal, (int, np.integer))
        assert 1 <= optimal <= 4
        
        # Check that we can access component parameters
        component_idx = optimal - 1
        weights = gmm.fit_result.weight[component_idx]
        means = gmm.fit_result.mean[component_idx]
        widths = gmm.fit_result.width[component_idx]
        
        assert len(weights) == optimal
        assert means.shape == (optimal, 1)
        assert widths.shape == (optimal, 1)
        
        # Check that we can assign data points to components
        assignments = gmm.fit_result.idxComponentOfScs(component_idx)
        assert len(assignments) == len(test_data)
        assert assignments.min() >= 0
        assert assignments.max() < optimal


# Test configuration and fixtures
@pytest.fixture(autouse=True)
def setup_matplotlib():
    """Set up matplotlib for testing."""
    # Use non-interactive backend for testing
    plt.switch_backend('Agg')


def test_import_successful():
    """Test that all required modules can be imported."""
    from qem.analysis.gaussian_mixture_model import GaussianMixtureModel, GaussianComponents, safe_ln
    assert GaussianMixtureModel is not None
    assert GaussianComponents is not None
    assert safe_ln is not None


if __name__ == "__main__":
    # Allow running tests directly
    pytest.main([__file__, "-v"])