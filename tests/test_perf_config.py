"""Tests for HYP-58 Phase 1 perf knobs: TF32 setup, perf flags, maybe_compile."""

import pytest


def _reload_config():
    """Force a fresh PrecisionConfig pickup of current env vars."""
    from qem.utils import config as cfg_mod
    return cfg_mod.reload_config()


def test_precision_config_defaults_when_env_unset(monkeypatch):
    monkeypatch.delenv("QEM_TF32", raising=False)
    monkeypatch.delenv("QEM_COMPILE", raising=False)
    cfg = _reload_config()
    assert cfg.enable_tf32 is True
    assert cfg.enable_compile is False


@pytest.mark.parametrize("value,expected", [
    ("0", False),
    ("false", False),
    ("False", False),
    ("1", True),
    ("true", True),
    ("anything-else", True),
])
def test_qem_tf32_env_parsing(monkeypatch, value, expected):
    monkeypatch.setenv("QEM_TF32", value)
    cfg = _reload_config()
    assert cfg.enable_tf32 is expected


@pytest.mark.parametrize("value,expected", [
    ("0", False),
    ("false", False),
    ("1", True),
    ("yes", True),
])
def test_qem_compile_env_parsing(monkeypatch, value, expected):
    monkeypatch.setenv("QEM_COMPILE", value)
    cfg = _reload_config()
    assert cfg.enable_compile is expected


def test_setup_torch_runtime_returns_status_dict():
    """setup_torch_runtime is always callable and returns a status dict.

    On non-CUDA / non-torch envs it must be a quiet no-op (no exceptions).
    """
    from qem.utils.backend import setup_torch_runtime

    info = setup_torch_runtime()
    assert isinstance(info, dict)
    assert "backend" in info
    assert "cuda" in info
    assert "tf32" in info
    # tf32 only flips True if CUDA + Ampere+ + flag enabled
    if not info["cuda"]:
        assert info["tf32"] is False


def test_setup_torch_runtime_disable_flag(monkeypatch):
    """enable_tf32=False short-circuits even on CUDA-capable hardware."""
    from qem.utils.backend import setup_torch_runtime

    info = setup_torch_runtime(enable_tf32=False)
    assert info["tf32"] is False


def test_setup_torch_runtime_idempotent():
    """Calling setup multiple times produces consistent state."""
    from qem.utils.backend import setup_torch_runtime

    a = setup_torch_runtime()
    b = setup_torch_runtime()
    assert a == b


def test_torch_inference_context_returns_context_manager():
    """torch_inference_context() always returns something usable in `with`."""
    from qem.utils.backend import torch_inference_context

    ctx = torch_inference_context()
    with ctx:
        pass  # Must not raise on any backend.


def test_maybe_compile_passthrough_when_disabled(monkeypatch):
    """With QEM_COMPILE=0 (default), maybe_compile returns the function unchanged."""
    monkeypatch.setenv("QEM_COMPILE", "0")
    _reload_config()
    from qem.utils.config import maybe_compile

    def hot(x):
        return x * 2

    wrapped = maybe_compile(hot)
    assert wrapped is hot


def test_maybe_compile_enabled_but_no_torch_or_cuda(monkeypatch):
    """When opt-in but the conditions aren't met, fall back to identity."""
    monkeypatch.setenv("QEM_COMPILE", "1")
    _reload_config()
    from qem.utils.config import maybe_compile

    def hot(x):
        return x + 1

    wrapped = maybe_compile(hot)
    # On torch+CUDA this would be a torch._dynamo.eval_frame.OptimizedModule;
    # everywhere else we expect the original callable. Either way, it must
    # remain callable and produce the same result on a plain Python int.
    assert callable(wrapped)
    assert wrapped(2) == 3
