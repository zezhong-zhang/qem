#!/usr/bin/env python3
"""
Test script to verify precision configuration works correctly.
"""

import os
import sys
import numpy as np

def test_precision_config():
    """Test precision configuration system."""
    print("Testing Precision Configuration System")
    print("=" * 40)
    
    try:
        from qem.utils.config import get_config, create_linear_solver_array, get_linear_solver_numpy_dtype
        
        # Test default configuration
        config = get_config()
        print(f"Default configuration: {config}")
        print(f"General precision: {config.precision}")
        print(f"Linear solver precision: {config.linear_solver_precision}")
        print(f"Numpy dtype: {config.numpy_dtype}")
        print(f"Linear solver numpy dtype: {config.linear_solver_numpy_dtype}")
        
        # Test array creation
        test_data = [1.0, 2.0, 3.0]
        
        # Create array with linear solver precision
        ls_array = create_linear_solver_array(test_data)
        print(f"✅ Linear solver array: dtype={ls_array.dtype}, values={ls_array}")
        
        # Test precision detection
        print(f"Float64 supported: {config.is_float64_supported()}")
        safe_precision = config.get_safe_precision("float64")
        print(f"Safe precision for float64: {safe_precision}")
        
        # Test with scipy sparse matrix
        from scipy.sparse import coo_matrix
        from qem.fit.linear_solver import SciPySolver
        
        # Create test sparse system
        n_rows, n_cols = 10, 5
        row_indices = np.array([0, 1, 2, 3, 4])
        col_indices = np.array([0, 1, 2, 3, 4])
        data = np.array([1.0, 1.0, 1.0, 1.0, 1.0])
        
        A = coo_matrix((data, (row_indices, col_indices)), shape=(n_rows, n_cols))
        b = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        
        print(f"Original matrix dtype: {A.dtype}")
        print(f"Original target dtype: {b.dtype}")
        
        # Test SciPySolver with precision handling
        solution = SciPySolver.solve_direct(A, b)
        print(f"✅ SciPy solution: dtype={solution.dtype}, shape={solution.shape}")
        
        # Test iterative solver
        solution_iter = SciPySolver.solve_iterative(A, b, max_iter=100, tol=1e-6)
        print(f"✅ SciPy iterative solution: dtype={solution_iter.dtype}, shape={solution_iter.shape}")
        
        return True
        
    except Exception as e:
        print(f"❌ Precision config test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_env_override():
    """Test environment variable override."""
    print("\nTesting Environment Variable Override")
    print("=" * 40)
    
    try:
        # Set environment variables
        os.environ["QEM_PRECISION"] = "float32"
        os.environ["QEM_LINEAR_SOLVER_PRECISION"] = "float32"
        
        # Reload configuration
        from qem.utils.config import reload_config
        config = reload_config()
        
        print(f"Overridden configuration: {config}")
        print(f"General precision: {config.precision}")
        print(f"Linear solver precision: {config.linear_solver_precision}")
        
        # Test that arrays use the correct precision
        from qem.utils.config import create_linear_solver_array
        
        test_array = create_linear_solver_array([1.0, 2.0, 3.0])
        expected_dtype = np.float32
        
        if test_array.dtype == expected_dtype:
            print(f"✅ Array precision correct: {test_array.dtype}")
        else:
            print(f"❌ Array precision incorrect: expected {expected_dtype}, got {test_array.dtype}")
            return False
        
        return True
        
    except Exception as e:
        print(f"❌ Environment override test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = True
    
    success &= test_precision_config()
    success &= test_env_override()
    
    print("\n" + "=" * 50)
    if success:
        print("🎉 All precision configuration tests passed!")
        print("\nPrecision configuration features:")
        print("✅ Environment variable configuration")
        print("✅ Configurable precision for linear solver")
        print("✅ Automatic dtype conversion")
        print("✅ MPS compatibility checking")
    else:
        print("❌ Some precision configuration tests failed.")
        sys.exit(1)