#!/usr/bin/env python3
"""Smoke test for the model API."""

import torch

from qem.fit.model import GaussianModel
from qem.utils.tensors import to_numpy


def test_model_api_smoke():
    x = torch.arange(10, dtype=torch.float32)
    y = torch.arange(10, dtype=torch.float32)
    xg, yg = torch.meshgrid(x, y, indexing="xy")

    params = {
        "pos_x": torch.as_tensor([5.0], dtype=torch.float32),
        "pos_y": torch.as_tensor([5.0], dtype=torch.float32),
        "height": torch.as_tensor([1.0], dtype=torch.float32),
        "width": torch.as_tensor([2.0], dtype=torch.float32),
        "background": torch.as_tensor(0.1, dtype=torch.float32),
    }

    model = GaussianModel(dx=1.0)
    model.set_params(params)
    model.build()

    result_np = to_numpy(model.sum(xg, yg, local=False))
    assert result_np.shape == (10, 10)
    assert result_np.max() > 0.1
    assert result_np.min() >= 0.1

    volumes_np = to_numpy(model.volume(params))
    assert len(volumes_np) == 1
