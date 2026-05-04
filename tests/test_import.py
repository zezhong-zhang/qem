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


def test_import_qem_uses_pytorch():
    """Bare ``import qem`` must succeed and pick a torch device."""
    result = _run(
        {"QEM_DEVICE": None},
        """
        import qem  # noqa: F401
        from qem.utils.tensors import best_device
        device = best_device()
        # Any of cpu / cuda / mps is fine — must be a torch device.
        assert device.type in ("cpu", "cuda", "mps"), device
        print("torch")
        """,
    )
    assert result.returncode == 0, result.stderr
    assert _last_line(result.stdout) == "torch"


def test_qem_device_env_override():
    """QEM_DEVICE=cpu forces best_device() to return CPU."""
    result = _run(
        {"QEM_DEVICE": "cpu"},
        """
        from qem.utils.tensors import best_device
        print(best_device().type)
        """,
    )
    assert result.returncode == 0, result.stderr
    assert _last_line(result.stdout) == "cpu"


def test_legacy_keras_backend_env_is_ignored():
    """A legacy KERAS_BACKEND setting no longer affects QEM."""
    result = _run(
        {"KERAS_BACKEND": "numpy"},
        """
        import qem  # noqa: F401
        from qem.utils.tensors import best_device
        # Just exercise the import path — KERAS_BACKEND should be ignored entirely.
        print(best_device().type in ("cpu", "cuda", "mps"))
        """,
    )
    assert result.returncode == 0, result.stderr
    assert _last_line(result.stdout) == "True"
