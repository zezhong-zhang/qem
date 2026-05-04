"""Precision configuration tests."""

import os

import numpy as np
import pytest
from scipy.sparse import coo_matrix


def test_precision_config_defaults():
    """Default configuration must expose float32/float64 dtypes and arrays."""
    from qem.utils.config import (
        create_linear_solver_array,
        get_config,
        get_linear_solver_numpy_dtype,
    )

    config = get_config()
    assert config.precision in {"float32", "float64"}
    assert config.linear_solver_precision in {"float32", "float64"}
    assert config.numpy_dtype in {np.float32, np.float64}
    assert get_linear_solver_numpy_dtype() == config.linear_solver_numpy_dtype

    ls_array = create_linear_solver_array([1.0, 2.0, 3.0])
    assert ls_array.dtype == config.linear_solver_numpy_dtype


def test_safe_precision_falls_back_to_float32_when_float64_unsupported():
    from qem.utils.config import get_config

    config = get_config()
    if config.is_float64_supported():
        assert config.get_safe_precision("float64") == "float64"
    else:
        assert config.get_safe_precision("float64") == "float32"


@pytest.fixture
def env_precision_override(monkeypatch):
    monkeypatch.setenv("QEM_PRECISION", "float32")
    monkeypatch.setenv("QEM_LINEAR_SOLVER_PRECISION", "float32")
    from qem.utils.config import reload_config

    return reload_config()


def test_env_override_applies_to_arrays(env_precision_override):
    from qem.utils.config import create_linear_solver_array, reload_config

    config = env_precision_override
    assert config.precision == "float32"
    assert config.linear_solver_precision == "float32"

    test_array = create_linear_solver_array([1.0, 2.0, 3.0])
    assert test_array.dtype == np.float32

    # Restore default config to keep other tests independent of env state.
    for key in ("QEM_PRECISION", "QEM_LINEAR_SOLVER_PRECISION"):
        os.environ.pop(key, None)
    reload_config()
