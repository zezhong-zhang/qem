# QEM Refactoring Report

## Completed

- Refactored `qem/linear_solver.py` around explicit responsibilities:
  - lazy Keras ops import for design-matrix paths
  - typed parameter mappings and NumPy array returns
  - `LinearSolverConfig` dataclass for sparse solver thresholds
  - sparse-safe solver path that avoids unconditional dense determinant checks
  - consistent `DataError` handling for singular, invalid, or mismatched systems
  - module logger usage instead of root `logging.warning` calls
- Hardened `qem/backend_utils.py`:
  - typed backend configuration helpers
  - centralized backend preference order
  - robust optional-backend detection that catches broken installs, not only missing modules
  - logger-based diagnostics instead of `print`
- Added `tests/test_linear_solver_core.py` to cover solver behavior without requiring Keras import.

## Verification

- `python -m py_compile qem/backend_utils.py qem/linear_solver.py`
- `pytest tests/test_linear_solver_core.py -q` passes: 5 tests.

## Remaining Blocker

The local environment cannot collect the existing Keras-backed tests. `pytest
tests/test_linear_solver.py tests/test_utils.py -q` fails before repository code
runs because importing Keras raises:

- `ModuleNotFoundError: No module named 'tensorflow'` with the default backend
- `AttributeError: module 'numpy.dtypes' has no attribute 'StringDType'` when Keras imports the installed JAX package

The refactored backend detection now excludes that broken JAX backend, but the
installed Keras package still imports JAX internally on the Torch path. Full
suite verification needs a consistent Keras backend environment, for example a
working TensorFlow install or compatible Keras/JAX/NumPy versions.
