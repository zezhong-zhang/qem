"""Small PyTorch-native compatibility surface for the former Keras call sites."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F


_NAME_TO_TORCH_DTYPE: dict[str, torch.dtype] = {
    "float32": torch.float32,
    "float64": torch.float64,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
    "int32": torch.int32,
    "int64": torch.int64,
    "int16": torch.int16,
    "int8": torch.int8,
    "uint8": torch.uint8,
    "bool": torch.bool,
}

_PYTHON_TO_TORCH_DTYPE: dict[type, torch.dtype] = {
    bool: torch.bool,
    int: torch.int64,
    float: torch.float32,
}


def _dtype(dtype: Any) -> torch.dtype | None:
    """Translate a dtype identifier into a ``torch.dtype``.

    Accepts ``None``, ``torch.dtype`` instances, dtype strings (``"bool"``,
    ``"float32"``, ...), Python builtins (``bool``, ``int``, ``float``), and
    NumPy dtype objects / scalar types (``np.bool_``, ``np.float32``, ...).
    Returns ``None`` only when the input itself is ``None``.
    """
    if dtype is None or isinstance(dtype, torch.dtype):
        return dtype
    if isinstance(dtype, type) and dtype in _PYTHON_TO_TORCH_DTYPE:
        return _PYTHON_TO_TORCH_DTYPE[dtype]
    try:
        canonical = np.dtype(dtype).name  # handles np.bool_, np.float32, np.dtype('bool'), ...
    except TypeError:
        canonical = str(dtype)
    if canonical in _NAME_TO_TORCH_DTYPE:
        return _NAME_TO_TORCH_DTYPE[canonical]
    raise TypeError(f"Unsupported dtype for PyTorch backend: {dtype!r}")


def _as_tensor(value: Any, dtype: Any = None) -> torch.Tensor:
    target_dtype = _dtype(dtype)
    if isinstance(value, nn.Parameter):
        value = value.detach()
    if isinstance(value, torch.Tensor):
        value = value.detach() if isinstance(value, nn.Parameter) else value
        value = value.cpu()
        return value.to(dtype=target_dtype) if target_dtype is not None else value
    return torch.as_tensor(value, dtype=target_dtype)


class _Backend:
    @staticmethod
    def backend() -> str:
        return "torch"

    @staticmethod
    def clear_session() -> None:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


class _Constant:
    def __init__(self, value: Any):
        self.value = value


class _Ops:
    @staticmethod
    def convert_to_tensor(value: Any, dtype: Any = None) -> torch.Tensor:
        return _as_tensor(value, dtype)

    @staticmethod
    def convert_to_numpy(value: Any) -> np.ndarray:
        if isinstance(value, nn.Parameter):
            value = value.detach()
        if isinstance(value, torch.Tensor):
            return value.detach().cpu().numpy()
        return np.asarray(value)

    @staticmethod
    def arange(*args: Any, dtype: Any = None) -> torch.Tensor:
        normalized = [int(a.item()) if isinstance(a, torch.Tensor) and a.ndim == 0 else a for a in args]
        return torch.arange(*normalized, dtype=_dtype(dtype) or torch.float32)

    @staticmethod
    def meshgrid(*tensors: torch.Tensor, indexing: str = "xy") -> tuple[torch.Tensor, ...]:
        return torch.meshgrid(*tensors, indexing=indexing)

    @staticmethod
    def conv(image: torch.Tensor, kernel: torch.Tensor, padding: str = "same") -> torch.Tensor:
        # Former Keras call sites use NHWC/HWIO tensors. Convert to NCHW/OIHW.
        image_nchw = image.permute(0, 3, 1, 2)
        kernel_oihw = kernel.permute(3, 2, 0, 1).to(device=image_nchw.device, dtype=image_nchw.dtype)
        if padding == "same":
            pad_y = kernel_oihw.shape[-2] // 2
            pad_x = kernel_oihw.shape[-1] // 2
            image_nchw = F.pad(image_nchw, (pad_x, pad_x, pad_y, pad_y), mode="constant", value=0)
            padding = 0
        result = F.conv2d(image_nchw, kernel_oihw, padding=padding)
        return result.permute(0, 2, 3, 1)

    @staticmethod
    def scatter_update(tensor: torch.Tensor, indices: torch.Tensor, values: torch.Tensor) -> torch.Tensor:
        result = tensor.clone()
        flat_indices = indices.reshape(-1).long()
        result.reshape(-1)[flat_indices] = _as_tensor(values, tensor.dtype).reshape(-1)
        return result

    def __getattr__(self, name: str) -> Any:
        mapping = {
            "abs": lambda x: torch.abs(_as_tensor(x)),
            "any": lambda x: torch.any(_as_tensor(x)),
            "cast": lambda x, dtype: _as_tensor(x, dtype),
            "clip": torch.clamp,
            "concatenate": torch.cat,
            "expand_dims": torch.unsqueeze,
            "exp": lambda x: torch.exp(_as_tensor(x)),
            "floor": lambda x: torch.floor(_as_tensor(x)),
            "full": lambda shape, fill_value, dtype=None: torch.full(tuple(shape), fill_value, dtype=_dtype(dtype) or torch.float32),
            "isinf": lambda x: torch.isinf(_as_tensor(x)),
            "isnan": lambda x: torch.isnan(_as_tensor(x)),
            "log": lambda x: torch.log(_as_tensor(x)),
            "maximum": lambda x, y: torch.maximum(_as_tensor(x), _as_tensor(y)),
            "minimum": lambda x, y: torch.minimum(_as_tensor(x), _as_tensor(y)),
            "max": lambda x: torch.max(_as_tensor(x)),
            "mean": lambda x: torch.mean(_as_tensor(x)),
            "mod": torch.remainder,
            "multiply": torch.mul,
            "ones": lambda shape, dtype=None: torch.ones(tuple(shape), dtype=_dtype(dtype) or torch.float32),
            "ones_like": torch.ones_like,
            "power": lambda x, exponent: torch.pow(_as_tensor(x), exponent),
            "reshape": torch.reshape,
            "shape": lambda x: tuple(x.shape),
            "sqrt": lambda x: torch.sqrt(_as_tensor(x)),
            "square": lambda x: torch.square(_as_tensor(x)),
            "squeeze": torch.squeeze,
            "stack": torch.stack,
            "std": torch.std,
            "stop_gradient": lambda x: x.detach() if isinstance(x, torch.Tensor) else x,
            "sum": lambda x, axis=None, **kwargs: torch.sum(_as_tensor(x), dim=axis, **kwargs),
            "take": lambda x, indices: torch.take(x, indices.long()),
            "where": torch.where,
            "zeros": lambda shape, dtype=None: torch.zeros(tuple(shape), dtype=_dtype(dtype) or torch.float32),
            "zeros_like": lambda x, dtype=None: torch.zeros_like(_as_tensor(x), dtype=_dtype(dtype) or None),
        }
        if name in mapping:
            return mapping[name]
        raise AttributeError(name)


def _assign_parameter(parameter: nn.Parameter, value: Any) -> nn.Parameter:
    with torch.no_grad():
        parameter.copy_(_as_tensor(value, parameter.dtype).to(parameter.device))
    return parameter


if not hasattr(nn.Parameter, "assign"):
    nn.Parameter.assign = _assign_parameter  # type: ignore[attr-defined]


class Model(nn.Module):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__()
        self.built = False
        self._compiled_optimizer = None
        self._compiled_loss = None

    def add_weight(
        self,
        shape: tuple[int, ...],
        initializer: _Constant,
        name: str,
        trainable: bool = True,
    ) -> nn.Parameter:
        value = _as_tensor(initializer.value, "float32").clone().detach()
        if tuple(value.shape) != tuple(shape):
            value = torch.full(shape, float(value), dtype=torch.float32)
        parameter = nn.Parameter(value, requires_grad=trainable)
        if hasattr(self, name) and name not in self._parameters:
            delattr(self, name)
        self.register_parameter(name, parameter)
        return parameter

    def build(self, input_shape: Any = None) -> None:
        self.built = True

    def forward(self, inputs: Any) -> torch.Tensor:
        return self.call(inputs)

    def compile(self, optimizer: Any, loss: Any) -> None:
        self._compiled_optimizer = optimizer.make(self.parameters())
        self._compiled_loss = loss

    def fit(self, x: Any, y: torch.Tensor, epochs: int, verbose: bool = True, callbacks: Any = None, batch_size: int | None = None) -> None:
        if self._compiled_optimizer is None or self._compiled_loss is None:
            raise RuntimeError("Model must be compiled before fit().")
        for epoch in range(epochs):
            self._compiled_optimizer.zero_grad()
            y_pred = self(x)
            loss = self._compiled_loss(y, y_pred)
            loss.backward()
            self._compiled_optimizer.step()
            if verbose and (epoch == 0 or (epoch + 1) % 100 == 0):
                print(f"Epoch {epoch + 1}/{epochs} - loss: {float(loss.detach()):.6f}")


class _OptimizerFactory:
    optimizer_cls: type[torch.optim.Optimizer]

    def __init__(self, learning_rate: float = 0.001, **kwargs: Any):
        self.learning_rate = learning_rate
        self.kwargs = kwargs

    def make(self, parameters: Any) -> torch.optim.Optimizer:
        return self.optimizer_cls(parameters, lr=self.learning_rate, **self.kwargs)


class _Adam(_OptimizerFactory):
    optimizer_cls = torch.optim.Adam


class _AdamW(_OptimizerFactory):
    optimizer_cls = torch.optim.AdamW


class _SGD(_OptimizerFactory):
    optimizer_cls = torch.optim.SGD


class _Callback:
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        pass


ops = _Ops()
backend = _Backend()
initializers = SimpleNamespace(Constant=_Constant)
optimizers = SimpleNamespace(Adam=_Adam, AdamW=_AdamW, SGD=_SGD)
callbacks = SimpleNamespace(EarlyStopping=_Callback, ReduceLROnPlateau=_Callback, LambdaCallback=_Callback)
