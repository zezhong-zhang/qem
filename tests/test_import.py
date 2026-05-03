"""Import regression tests for the PyTorch-native package."""
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


def test_import_qem_uses_pytorch_backend():
    """Bare ``import qem`` must succeed and expose the torch backend."""
    result = _run(
        {"QEM_BACKEND": None},
        """
        import qem  # noqa: F401
        from qem.utils.backend import get_best_backend
        print(get_best_backend())
        """,
    )
    assert result.returncode == 0, result.stderr
    assert _last_line(result.stdout) == "torch"


def test_import_qem_reports_missing_torch():
    """When PyTorch is not importable, backend selection reports the dependency."""
    result = _run(
        {"QEM_BACKEND": None},
        """
        import importlib.util
        _real = importlib.util.find_spec
        def _stub(name, *a, **k):
            if name == 'torch':
                return None
            return _real(name, *a, **k)
        importlib.util.find_spec = _stub

        from qem.utils.backend import detect_available_backends
        print(detect_available_backends())
        """,
    )
    assert result.returncode == 0, result.stderr
    assert _last_line(result.stdout) == "[]"


def test_legacy_keras_backend_env_is_ignored():
    """A legacy KERAS_BACKEND setting no longer affects QEM."""
    result = _run(
        {"KERAS_BACKEND": "numpy"},
        """
        import qem  # noqa: F401
        from qem.utils.backend import get_best_backend
        print(get_best_backend())
        """,
    )
    assert result.returncode == 0, result.stderr
    assert _last_line(result.stdout) == "torch"
