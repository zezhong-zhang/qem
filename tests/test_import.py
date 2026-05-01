"""Headless / no-backend import regression tests.

The package promises that ``import qem`` succeeds in any environment, even
when no Keras backend (torch / jax / tensorflow) is installed. These tests
exercise the auto-selection logic in ``qem/__init__.py``.
"""
from __future__ import annotations

import os
import subprocess
import sys
import textwrap


def _last_line(stdout: str) -> str:
    """Return the last non-empty line of stdout (skips import-time warnings)."""
    return [line for line in stdout.splitlines() if line.strip()][-1].strip()


def _run(env_overrides: dict[str, str | None], script: str) -> subprocess.CompletedProcess:
    env = os.environ.copy()
    for key, value in env_overrides.items():
        if value is None:
            env.pop(key, None)
        else:
            env[key] = value
    return subprocess.run(
        [sys.executable, "-c", textwrap.dedent(script)],
        capture_output=True,
        text=True,
        env=env,
    )


def test_import_qem_without_keras_backend_env():
    """Bare ``import qem`` must succeed when KERAS_BACKEND is unset."""
    result = _run(
        {"KERAS_BACKEND": None},
        """
        import qem  # noqa: F401
        import keras
        print(keras.backend.backend())
        """,
    )
    assert result.returncode == 0, result.stderr
    assert _last_line(result.stdout) in {"torch", "jax", "tensorflow", "numpy"}


def test_import_qem_falls_back_to_numpy_backend():
    """When no accelerated backend is importable, fall back to numpy."""
    result = _run(
        {"KERAS_BACKEND": None},
        """
        import importlib.util
        _real = importlib.util.find_spec
        def _stub(name, *a, **k):
            if name in ('torch', 'jax', 'tensorflow'):
                return None
            return _real(name, *a, **k)
        importlib.util.find_spec = _stub

        import qem  # noqa: F401
        import keras
        print(keras.backend.backend())
        """,
    )
    assert result.returncode == 0, result.stderr
    assert _last_line(result.stdout) == "numpy"


def test_user_keras_backend_is_respected():
    """A user-set ``KERAS_BACKEND`` must not be overridden."""
    result = _run(
        {"KERAS_BACKEND": "numpy"},
        """
        import qem  # noqa: F401
        import keras
        print(keras.backend.backend())
        """,
    )
    assert result.returncode == 0, result.stderr
    assert _last_line(result.stdout) == "numpy"
