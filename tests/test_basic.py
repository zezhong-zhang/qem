"""Basic tests for core functionality."""
import numpy as np
import torch

from qem.fit.model import GaussianModel
from qem.fit.fitter import Fitter
from qem.utils.tensors import to_numpy, to_tensor


def test_safe_conversions():
    """Round-trip np.ndarray -> torch.Tensor -> np.ndarray preserves values."""
    np_array = np.array([1.0, 2.0, 3.0])
    tensor = to_tensor(np_array)
    np.testing.assert_array_almost_equal(to_numpy(tensor), np_array)


def test_gaussian_model_basic():
    """Test basic Gaussian model functionality."""
    size = 20
    x_grid = torch.arange(size, dtype=torch.float32)
    y_grid = torch.arange(size, dtype=torch.float32)
    x_grid, y_grid = torch.meshgrid(x_grid, y_grid, indexing="xy")

    params = {
        "pos_x": torch.as_tensor([10.0], dtype=torch.float32),
        "pos_y": torch.as_tensor([10.0], dtype=torch.float32),
        "height": torch.as_tensor([1.0], dtype=torch.float32),
        "width": torch.as_tensor([2.0], dtype=torch.float32),
        "background": torch.as_tensor(0.1, dtype=torch.float32),
    }

    model = GaussianModel(dx=1.0)
    model.set_params(params)
    model.build()

    result_np = to_numpy(model.sum(x_grid, y_grid, local=False))
    assert result_np.shape == (size, size)
    assert result_np.max() > 0.1
    assert result_np.min() >= 0.1

    volumes_np = to_numpy(model.volume(params))
    assert len(volumes_np) == 1
    assert volumes_np[0] > 0


def test_image_fitting_basic():
    """Test basic Fitter functionality."""
    size = 20
    image = np.zeros((size, size))
    center = size // 2
    for i in range(size):
        for j in range(size):
            r2 = (i - center) ** 2 + (j - center) ** 2
            image[i, j] = np.exp(-r2 / (2 * 2 ** 2)) + 0.1

    fitter = Fitter(image=image, dx=1.0, model_type="gaussian")
    fitter.coordinates = np.array([[center, center]], dtype=float)
    params = fitter.init_params(atom_size=2.0)

    assert params is not None
    assert {"pos_x", "pos_y", "height", "width"} <= set(params)
    assert fitter.num_coordinates == 1

    prediction_np = to_numpy(fitter.predict(local=False))
    assert prediction_np.shape == image.shape


def test_pytorch_specific():
    """Sanity-check torch tensor → numpy round-trip on a grad tensor."""
    tensor = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
    np.testing.assert_array_almost_equal(to_numpy(tensor), [1.0, 2.0, 3.0])
