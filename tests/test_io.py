"""
Tests for the IO module, specifically for legacy StatSTEM file reading.
"""

import numpy as np
import pytest
import os
from pathlib import Path

from qem.io import read_statstem


class TestReadLegacyInputStatSTEM:
    """Test the read_statstem function for legacy StatSTEM .mat file support."""
    
    def test_file_not_found(self):
        """Test that FileNotFoundError is raised for non-existent files."""
        with pytest.raises(FileNotFoundError, match="nonexistent.mat not found"):
            read_statstem("nonexistent.mat")
    
    def test_read_real_benchmark_files(self):
        """Test reading actual StatSTEM .mat files from benchmark directory."""
        benchmark_dir = Path("data/benchmark")
        
        # Test with multiple real .mat files
        test_files = [
            "output_fivefoldsymmetry2015_121416_2.mat",
            "StatSTEM_200C_no_gas_1_1_conv_1_rotated.mat",
            "StatSTEM_300C_H2_1_1_conv_1_rotated.mat",
            "StatSTEM_Aurod_0_2016.mat"
        ]
        
        for filename in test_files:
            file_path = benchmark_dir / filename
            if file_path.exists():
                result = read_statstem(str(file_path))
                
                # Verify basic structure
                assert isinstance(result, dict)
                assert len(result) > 0
                
                # Verify presence of input data (common in StatSTEM files)
                if 'input' in result:
                    assert isinstance(result['input'], dict)
                    # Check for common input fields
                    if 'obs' in result['input']:
                        obs = result['input']['obs']
                        assert isinstance(obs, np.ndarray)
                        assert obs.ndim == 2  # Should be 2D image
                    
                    if 'dx' in result['input']:
                        dx = result['input']['dx']
                        assert isinstance(dx, (int, float, np.number))
                
                # Verify presence of output data
                if 'output' in result:
                    assert isinstance(result['output'], dict)
                    
                    # Check for coordinate data
                    coord_keys = ['coordinates', 'BetaX', 'BetaY']
                    has_coords = any(key in result['output'] for key in coord_keys)
                    if has_coords:
                        # Get coordinate data
                        if 'coordinates' in result['output']:
                            coords = result['output']['coordinates']
                        elif 'BetaX' in result['output'] and 'BetaY' in result['output']:
                            coords = np.column_stack([
                                result['output']['BetaX'],
                                result['output']['BetaY']
                            ])
                        else:
                            coords = None
                        
                        if coords is not None:
                            assert isinstance(coords, np.ndarray)
                            assert coords.ndim == 2
                            assert coords.shape[1] == 2  # Should be Nx2
    
    def test_real_file_structure_validation(self):
        """Test structure validation with real StatSTEM files."""
        benchmark_dir = Path("data/benchmark")
        
        # Get first available .mat file
        mat_files = list(benchmark_dir.glob("*.mat"))
        if not mat_files:
            pytest.skip("No .mat files found in benchmark directory")
        
        test_file = mat_files[0]
        result = read_statstem(str(test_file))
        
        # Validate structure
        assert isinstance(result, dict)
        
        # Check for private keys being filtered out
        private_keys = [key for key in result.keys() if key.startswith('_')]
        assert len(private_keys) == 0, f"Found private keys: {private_keys}"
        
        # Validate data types
        for key, value in result.items():
            if key in ['input', 'output']:
                assert isinstance(value, dict), f"Expected dict for {key}, got {type(value)}"
                
                # Check nested structure
                for sub_key, sub_value in value.items():
                    assert isinstance(sub_value, (np.ndarray, int, float, str, dict, list)), \
                        f"Unexpected type for {key}.{sub_key}: {type(sub_value)}"
    
    def test_benchmark_integration_real_data(self):
        """Test integration with benchmark workflow using real data."""
        benchmark_dir = Path("data/benchmark")
        
        # Find a suitable test file
        test_files = list(benchmark_dir.glob("*.mat"))
        if not test_files:
            pytest.skip("No .mat files found in benchmark directory")
        
        test_file = test_files[0]
        result = read_statstem(str(test_file))
        
        # Verify structure expected by Benchmark class
        assert isinstance(result, dict)
        
        # Check for essential fields used in benchmark.py
        has_input = 'input' in result
        has_output = 'output' in result
        
        if has_input:
            input_data = result['input']
            assert isinstance(input_data, dict)
            
            # Check for image data
            if 'obs' in input_data:
                obs = input_data['obs']
                assert isinstance(obs, np.ndarray)
                assert obs.ndim == 2
                assert obs.size > 0
            
            # Check for pixel size
            if 'dx' in input_data:
                dx = input_data['dx']
                assert isinstance(dx, (int, float, np.number))
                assert dx > 0
        
        if has_output:
            output_data = result['output']
            assert isinstance(output_data, dict)
            
            # Check for coordinate data
            coord_found = False
            if 'coordinates' in output_data:
                coords = output_data['coordinates']
                if isinstance(coords, np.ndarray) and coords.ndim == 2:
                    coord_found = True
            elif 'BetaX' in output_data and 'BetaY' in output_data:
                beta_x = output_data['BetaX']
                beta_y = output_data['BetaY']
                if (isinstance(beta_x, np.ndarray) and isinstance(beta_y, np.ndarray) and
                    len(beta_x) == len(beta_y)):
                    coord_found = True
            
            # At least some coordinate-like data should be present
            assert coord_found or len(output_data) > 0, "No coordinate data found in output"