import pytest
import numpy as np
import matplotlib.pyplot as plt
from unittest.mock import patch, MagicMock
from qem.crystal_analyzer import CrystalAnalyzer

# Mocking external dependencies
plt.imshow = MagicMock()
plt.scatter = MagicMock()
plt.legend = MagicMock()
plt.show = MagicMock()
get_unique_colors = MagicMock(return_value=iter(['#000000', '#FFFFFF']))

@pytest.mark.parametrize("atom_types, elements, peak_positions, expected_calls, test_id", [
    # Happy path tests
    (np.array([1, 1, 2]), {1: 'H', 2: 'He'}, np.array([[0, 0], [1, 1], [2, 2]]), 2, "happy_path_basic"),
    (np.array([1, 2, 2, 3]), {1: 'H', 2: 'He', 3: 'Li'}, np.array([[0, 0], [1, 1], [2, 2], [3, 3]]), 3, "happy_path_multiple_elements"),
    
    # Edge cases
    (np.array([]), {}, np.array([]), 0, "edge_case_no_atoms"),
    (np.array([1]), {1: 'H'}, np.array([[0, 0]]), 1, "edge_case_single_atom"),
    
    # Error cases
    # Assuming the function should handle or avoid errors internally, no explicit error cases are designed here.
    # This is because the function does not have explicit error handling and relies on correct inputs.
])
def test_plot(atom_types, elements, peak_positions, expected_calls, test_id):
    # Arrange
    analyzer = CrystalAnalyzer()
    analyzer.image = np.zeros((10, 10))  # Mocking an image
    analyzer.atom_types = atom_types
    analyzer.elements = elements
    analyzer.peak_positions = peak_positions

    # Act
    with patch('matplotlib.pyplot.scatter') as mock_scatter:
        analyzer.plot()

    # Assert
    assert mock_scatter.call_count == expected_calls, f"Test Failed: {test_id}"
