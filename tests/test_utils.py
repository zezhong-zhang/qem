"""Tests for tensor / numpy interop helpers in qem.utils.tensors."""
import numpy as np
import pytest
import torch

from qem.utils.tensors import _resolve_dtype, clone_params, to_numpy, to_tensor


def test_to_numpy_passthrough_for_ndarray():
    np_array = np.array([1.0, 2.0, 3.0])
    result = to_numpy(np_array)
    assert isinstance(result, np.ndarray)
    np.testing.assert_array_equal(result, np_array)


def test_to_numpy_from_tensor():
    tensor = torch.as_tensor([1.0, 2.0, 3.0], dtype=torch.float32)
    result = to_numpy(tensor)
    assert isinstance(result, np.ndarray)
    np.testing.assert_array_almost_equal(result, [1.0, 2.0, 3.0])


def test_to_tensor_from_ndarray():
    np_array = np.array([1.0, 2.0, 3.0])
    np.testing.assert_array_almost_equal(to_numpy(to_tensor(np_array)), np_array)


def test_to_tensor_dtype_override():
    np_array = np.array([1.0, 2.0, 3.0])
    np.testing.assert_array_equal(
        to_numpy(to_tensor(np_array, dtype=torch.int32)), [1, 2, 3]
    )


def test_clone_params_deep_copies_tensors_and_metadata():
    original = {
        "pos_x": torch.as_tensor([1.0, 2.0], dtype=torch.float32),
        "pos_y": torch.as_tensor([3.0, 4.0], dtype=torch.float32),
        "height": torch.as_tensor([0.5, 0.8], dtype=torch.float32),
        "width": torch.as_tensor([1.0, 1.2], dtype=torch.float32),
        "background": torch.as_tensor(0.1, dtype=torch.float32),
        "metadata": {"test": "value"},
    }

    copied = clone_params(original)
    assert set(copied) == set(original)

    for key in ("pos_x", "pos_y", "height", "width", "background"):
        np.testing.assert_array_almost_equal(
            to_numpy(copied[key]), to_numpy(original[key])
        )
        assert copied[key] is not original[key]

    assert copied["metadata"] == original["metadata"]
    assert copied["metadata"] is not original["metadata"]


@pytest.mark.parametrize(
    "dtype_input, expected_torch_dtype_attr",
    [
        ("bool", "bool"),
        ("int64", "int64"),
        ("float64", "float64"),
    ],
)
def test_tensors_resolve_dtype(dtype_input, expected_torch_dtype_attr):
    expected = getattr(torch, expected_torch_dtype_attr)
    assert _resolve_dtype(dtype_input) is expected


def test_gradient_handling():
    """to_numpy detaches grad tensors silently."""
    torch_tensor = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
    result = to_numpy(torch_tensor)
    assert isinstance(result, np.ndarray)
    np.testing.assert_array_almost_equal(result, [1.0, 2.0, 3.0])
