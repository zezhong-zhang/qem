"""Image-fitting peak models — Gaussian / Lorentzian / Voigt.

Pure PyTorch.  Each model is an :class:`torch.nn.Module` whose
parameters (``pos_x``, ``pos_y``, ``height``, ``width``, ``background``,
optional ``ratio``) are :class:`torch.nn.Parameter` instances created
lazily via :meth:`set_params`.  Forward pass evaluates the model on a
real-space grid; ``sum`` does the same with an option for a JIT-friendly
local windowing.
"""

from __future__ import annotations

from abc import abstractmethod
from typing import Any, Mapping

import numpy as np
import torch
from numba import jit as njit
from torch import nn

from qem.utils.tensors import to_numpy, to_tensor


def _as_param(value: Any, *, requires_grad: bool = True) -> nn.Parameter:
    """Wrap a value as a trainable ``nn.Parameter`` of dtype float32."""
    tensor = to_tensor(value, dtype="float32").detach().clone()
    return nn.Parameter(tensor, requires_grad=requires_grad)


class ImageModel(nn.Module):
    """Base class for parametric peak-shape image models."""

    def __init__(self, dx: float = 1.0):
        super().__init__()
        self.dx = float(dx)
        self.input_params: dict[str, Any] | None = None
        self.built: bool = False
        # Cache the local-window meshgrid keyed by (window_size, dtype, device).
        # See _sum_local — re-allocating these every forward pass is the
        # single biggest hot-loop allocation in fit_global.
        self._window_cache: dict[tuple, tuple[torch.Tensor, torch.Tensor]] = {}

    # ------------------------------------------------------------------
    # parameter management
    # ------------------------------------------------------------------

    def set_params(self, params: Mapping[str, Any]) -> None:
        """Stash an initial parameter set; build lazily on first use."""
        self.input_params = {k: to_tensor(v) for k, v in params.items()}
        if self.built:
            self.update_params(self.input_params)

    def update_params(self, params: Mapping[str, Any]) -> None:
        """Update existing :class:`nn.Parameter` values in-place."""
        config_keys = {"same_width", "atom_types"}
        with torch.no_grad():
            for key, value in params.items():
                if key in config_keys:
                    continue
                if not hasattr(self, key):
                    raise ValueError(f"Parameter {key!r} does not exist on the model.")
                target: nn.Parameter = getattr(self, key)
                target.copy_(to_tensor(value, dtype=target.dtype).to(target.device))

    def build(self, input_shape: tuple[int, ...] | None = None) -> None:
        """Create :class:`nn.Parameter` instances from ``input_params``."""
        if self.input_params is None:
            raise ValueError("set_params() must be called before build().")
        if self.built:
            return
        ip = self.input_params
        self.pos_x = _as_param(ip["pos_x"])
        self.pos_y = _as_param(ip["pos_y"])
        self.height = _as_param(ip["height"])
        self.width = _as_param(ip["width"])
        self.background = _as_param(ip["background"])
        self.built = True

    def get_params(self) -> dict[str, Any]:
        """Snapshot the current parameter values as detached tensors."""
        if not self.built:
            raise RuntimeError("Model has not been built yet.")
        ip = self.input_params or {}
        out: dict[str, Any] = {
            "pos_x": self.pos_x.detach(),
            "pos_y": self.pos_y.detach(),
            "height": self.height.detach(),
            "width": self.width.detach(),
            "background": self.background.detach(),
            "same_width": ip.get("same_width", False),
        }
        if "atom_types" in ip:
            out["atom_types"] = to_tensor(ip["atom_types"], dtype="int32")
        else:
            out["atom_types"] = torch.zeros_like(self.height, dtype=torch.int32)
        if hasattr(self, "ratio"):
            out["ratio"] = self.ratio.detach()
        return out

    # ------------------------------------------------------------------
    # forward + sum
    # ------------------------------------------------------------------

    def forward(self, inputs: tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        x_grid, y_grid = inputs
        return self.sum(x_grid, y_grid)

    @abstractmethod
    def model_fn(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        pos_x: torch.Tensor,
        pos_y: torch.Tensor,
        height: torch.Tensor,
        width: torch.Tensor,
        *args: torch.Tensor,
    ) -> torch.Tensor:
        """Compute peak intensity at (x, y).  Subclasses provide this."""

    @abstractmethod
    def volume(self, params: Mapping[str, Any]) -> np.ndarray:
        """Per-peak volume (height × area, in dx-scaled units)."""

    def sum(
        self,
        x_grid: Any,
        y_grid: Any,
        local: bool = True,
    ) -> torch.Tensor:
        """Render all peaks plus background onto the supplied grid."""
        if self.input_params is None:
            raise RuntimeError("set_params()/build() must run before sum().")

        x_grid = to_tensor(x_grid, dtype="float32")
        y_grid = to_tensor(y_grid, dtype="float32")

        has_batch = x_grid.dim() > 2
        if has_batch:
            x_grid = x_grid.squeeze(0)
            y_grid = y_grid.squeeze(0)

        # Per-atom width / ratio (broadcast across atom_types when same_width).
        width = self.width
        ratio = self.ratio if hasattr(self, "ratio") else None
        if self.input_params.get("same_width", False):
            atom_types = to_tensor(self.input_params["atom_types"], dtype="int64")
            width = width[atom_types]
            if ratio is not None:
                ratio = ratio[atom_types]
        extra = (width, ratio) if ratio is not None else (width,)

        if not local:
            peaks = self.model_fn(
                x_grid[..., None], y_grid[..., None],
                self.pos_x, self.pos_y,
                self.height, *extra,
            )
            result = peaks.sum(dim=-1) + self.background
        else:
            result = self._sum_local(x_grid, y_grid, extra)

        return result.unsqueeze(0) if has_batch else result

    def _sum_local(
        self,
        x_grid: torch.Tensor,
        y_grid: torch.Tensor,
        extra: tuple[torch.Tensor, ...],
    ) -> torch.Tensor:
        """Memory-efficient local-window peak rendering with scatter-add.

        The (2W+1)² meshgrid is cached per (window_size, dtype, device) —
        only atom positions vary between fit-loop iterations, so we keep
        the static window grid and skip the per-call allocation.
        """
        assert self.input_params is not None
        # width is a Parameter on the right device — no numpy round-trip.
        max_width = float(self.width.detach().max().item())
        window_size = int(max_width * 4)
        cache_key = (window_size, x_grid.dtype, x_grid.device)
        cached = self._window_cache.get(cache_key)
        if cached is None:
            coords = torch.arange(
                -window_size, window_size + 1,
                dtype=x_grid.dtype, device=x_grid.device,
            )
            local_x, local_y = torch.meshgrid(coords, coords, indexing="xy")
            self._window_cache[cache_key] = (local_x, local_y)
        else:
            local_x, local_y = cached

        peak_args = (
            torch.remainder(self.pos_x, 1.0),
            torch.remainder(self.pos_y, 1.0),
            self.height, *extra,
        )
        local_peaks = self.model_fn(local_x[..., None], local_y[..., None], *peak_args)

        pos_x_int = torch.floor(self.pos_x)
        pos_y_int = torch.floor(self.pos_y)
        global_x = local_x.unsqueeze(-1) + pos_x_int
        global_y = local_y.unsqueeze(-1) + pos_y_int

        h, w = x_grid.shape
        in_bounds = (global_x >= 0) & (global_x < w) & (global_y >= 0) & (global_y < h)
        # Element-wise mask multiply skips the torch.zeros_like allocation
        # that the previous torch.where(in_bounds, peaks, zeros) needed.
        masked_peaks = local_peaks * in_bounds.to(local_peaks.dtype)

        global_x_safe = torch.clamp(global_x, 0, w - 1).to(torch.int64)
        global_y_safe = torch.clamp(global_y, 0, h - 1).to(torch.int64)
        flat_indices = global_y_safe.reshape(-1) * w + global_x_safe.reshape(-1)

        canvas = torch.zeros_like(x_grid).reshape(-1)
        canvas = canvas.scatter_add(0, flat_indices, masked_peaks.reshape(-1))
        return canvas.reshape(x_grid.shape) + self.background


class GaussianModel(ImageModel):
    """Isotropic Gaussian peak."""

    def volume(self, params: Mapping[str, Any]) -> np.ndarray:
        height = to_numpy(params["height"])
        width = to_numpy(params["width"])
        return height * 2 * np.pi * width**2 * self.dx**2

    def model_fn(self, x, y, pos_x, pos_y, height, width, *args):
        return height * torch.exp(
            -((x - pos_x) ** 2 + (y - pos_y) ** 2) / (2 * width**2)
        )


class LorentzianModel(ImageModel):
    """Isotropic Lorentzian peak."""

    def volume(self, params: Mapping[str, Any]) -> np.ndarray:
        height = to_numpy(params["height"])
        width = to_numpy(params["width"])
        return height * np.pi * width**2 * self.dx**2

    def model_fn(self, x, y, pos_x, pos_y, height, width, *args):
        return height / (1 + ((x - pos_x) ** 2 + (y - pos_y) ** 2) / width**2)


class VoigtModel(ImageModel):
    """Voigt = ratio·Gaussian + (1−ratio)·Lorentzian."""

    def __init__(self, dx: float = 1.0):
        super().__init__(dx)
        self.ratio: nn.Parameter | None = None  # type: ignore[assignment]

    def build(self, input_shape: tuple[int, ...] | None = None) -> None:
        if self.input_params is None:
            raise ValueError("set_params() must be called before build().")
        if self.built:
            return
        self.ratio = _as_param(self.input_params["ratio"])
        super().build(input_shape)

    def get_params(self) -> dict[str, Any]:
        params = super().get_params()
        params["ratio"] = self.ratio.detach()  # type: ignore[union-attr]
        return params

    def volume(self, params: Mapping[str, Any]) -> np.ndarray:
        height = to_numpy(params["height"])
        width = to_numpy(params["width"])
        ratio = to_numpy(params["ratio"])
        gaussian = height * 2 * np.pi * width**2 * self.dx**2
        lorentzian = height * np.pi * width**2 * self.dx**2
        return ratio * gaussian + (1 - ratio) * lorentzian

    def model_fn(self, x, y, pos_x, pos_y, height, width, ratio):
        sigma = width
        gamma = width / torch.sqrt(torch.tensor(2.0 * np.log(2.0)))
        r2 = (x - pos_x) ** 2 + (y - pos_y) ** 2
        gaussian = torch.exp(-r2 / (2 * sigma**2))
        lorentzian = gamma**3 / torch.pow(r2 + gamma**2, 1.5)
        return height * (ratio * gaussian + (1 - ratio) * lorentzian)


class GaussianKernel:
    """Convolutional Gaussian filter (separable kernel + 2D conv)."""

    def gaussian_kernel(self, sigma: float) -> torch.Tensor:
        size = int(4 * sigma + 0.5) * 2 + 1
        x = torch.arange(-(size // 2), (size // 2) + 1, dtype=torch.float32)
        x_grid, y_grid = torch.meshgrid(x, x, indexing="xy")
        kernel = torch.exp(-(x_grid**2 + y_grid**2) / (2 * sigma**2))
        return kernel / kernel.sum()

    def gaussian_filter(self, image: Any, sigma: float) -> torch.Tensor:
        image_t = to_tensor(image, dtype="float32")
        kernel = self.gaussian_kernel(sigma)
        # NCHW for conv2d: image [1,1,H,W], kernel [1,1,kh,kw].
        image_t = image_t.unsqueeze(0).unsqueeze(0)
        kernel = kernel.unsqueeze(0).unsqueeze(0)
        pad = kernel.shape[-1] // 2
        return torch.nn.functional.conv2d(image_t, kernel, padding=pad).squeeze()


@njit
def gaussian_2d_single(xy, pos_x, pos_y, height, width, background):
    """Numba-jitted 2D Gaussian (used by point-potential fits)."""
    x_grid, y_grid = xy
    return (
        height
        * np.exp(
            -((x_grid[:, :, None] - pos_x) ** 2 + (y_grid[:, :, None] - pos_y) ** 2)
            / (2 * width**2)
        ) + background
    ).ravel()
