"""Route B joint least-squares fusion solver."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Mapping, Optional

import numpy as np

from qem.exceptions import DataError, ParameterError
from qem.periodic_table import chemical_symbols

from .dataset import MultiModalDataset


@dataclass
class FusionResult:
    """Result returned by :class:`JointLeastSquaresRoute`."""

    concentrations: np.ndarray
    elements: List[str]
    cost_history: Dict[str, List[float]]
    metadata: Dict[str, object] = field(default_factory=dict)

    def as_maps(self) -> Dict[str, np.ndarray]:
        return {
            element: self.concentrations[..., index]
            for index, element in enumerate(self.elements)
        }


def _normalise_signal(data: np.ndarray) -> np.ndarray:
    data = np.asarray(data, dtype=np.float64)
    scale = np.nanmax(np.abs(data))
    if not np.isfinite(scale) or scale == 0:
        return np.zeros_like(data, dtype=np.float64)
    return data / scale


def _periodic_weights(elements: Iterable[str]) -> np.ndarray:
    lookup = {symbol: index for index, symbol in enumerate(chemical_symbols)}
    weights = []
    for element in elements:
        if element not in lookup:
            raise ParameterError(f"unknown element symbol: {element}", "elements")
        weights.append(float(lookup[element]))
    return np.asarray(weights, dtype=np.float64)


def _resize_spectrum_image(data: np.ndarray, target_shape) -> np.ndarray:
    if data.shape[:2] == target_shape:
        return data
    try:
        from skimage.transform import resize
    except ImportError as exc:
        raise ImportError("scikit-image is required to resize spectrum images") from exc
    return resize(
        data,
        target_shape + (data.shape[-1],),
        preserve_range=True,
        anti_aliasing=True,
    ).astype(np.float64)


def _tv_gradient(concentrations: np.ndarray, epsilon: float = 1e-8) -> np.ndarray:
    grad = np.zeros_like(concentrations)
    dx = np.diff(concentrations, axis=1)
    dy = np.diff(concentrations, axis=0)
    wx = dx / np.sqrt(dx * dx + epsilon)
    wy = dy / np.sqrt(dy * dy + epsilon)
    grad[:, :-1, :] -= wx
    grad[:, 1:, :] += wx
    grad[:-1, :, :] -= wy
    grad[1:, :, :] += wy
    return grad


class JointLeastSquaresRoute:
    """
    Projected-gradient Route B ADF-EDX-EELS joint least-squares optimizer.

    The optimized variable is a non-negative concentration image with shape
    ``(height, width, elements)``.
    """

    def __init__(
        self,
        elements: List[str],
        gamma: float = 1.6,
        lambda_adf: float = 1.0,
        lambda_edx: float = 1.0,
        lambda_eels: float = 1.0,
        lambda_tv: float = 0.0,
        step_size: float = 0.05,
        max_iter: int = 100,
        tolerance: float = 1e-6,
        adf_weights: Optional[Iterable[float]] = None,
        normalize_inputs: bool = True,
        resize_modalities: bool = True,
    ) -> None:
        if gamma <= 1.0:
            raise ParameterError("gamma must be greater than 1", "gamma")
        if step_size <= 0:
            raise ParameterError("step_size must be positive", "step_size")
        if max_iter < 1:
            raise ParameterError("max_iter must be at least 1", "max_iter")

        self.elements = [str(element) for element in elements]
        self.gamma = float(gamma)
        self.lambda_adf = float(lambda_adf)
        self.lambda_edx = float(lambda_edx)
        self.lambda_eels = float(lambda_eels)
        self.lambda_tv = float(lambda_tv)
        self.step_size = float(step_size)
        self.max_iter = int(max_iter)
        self.tolerance = float(tolerance)
        self.adf_weights = (
            np.asarray(list(adf_weights), dtype=np.float64)
            if adf_weights is not None
            else _periodic_weights(self.elements)
        )
        self.normalize_inputs = bool(normalize_inputs)
        self.resize_modalities = bool(resize_modalities)
        self.result_: Optional[FusionResult] = None

        if self.adf_weights.shape != (len(self.elements),):
            raise ParameterError("adf_weights length must match elements", "adf_weights")

    def fit(
        self,
        dataset: MultiModalDataset,
        initial: Optional[np.ndarray] = None,
    ) -> "JointLeastSquaresRoute":
        self._validate_dataset(dataset)
        adf = _normalise_signal(dataset.adf) if self.normalize_inputs else dataset.adf.copy()

        edx = dataset.edx
        eels = dataset.eels
        if self.resize_modalities:
            if edx is not None:
                edx = _resize_spectrum_image(edx, dataset.spatial_shape)
            if eels is not None:
                eels = _resize_spectrum_image(eels, dataset.spatial_shape)
        elif (edx is not None and edx.shape[:2] != dataset.spatial_shape) or (
            eels is not None and eels.shape[:2] != dataset.spatial_shape
        ):
            raise DataError("all modalities must have the same spatial shape")

        edx = _normalise_signal(edx) if edx is not None and self.normalize_inputs else edx
        eels = _normalise_signal(eels) if eels is not None and self.normalize_inputs else eels
        edx_ref = self._normalise_reference(dataset.edx_reference)
        eels_ref = self._normalise_reference(dataset.eels_reference)

        x = self._initial_concentrations(dataset, edx, eels, edx_ref, eels_ref, initial)
        costs = {"total": [], "adf": [], "edx": [], "eels": [], "tv": []}

        previous_total = np.inf
        for iteration in range(self.max_iter):
            grad = np.zeros_like(x)

            adf_pred = self._adf_forward(x)
            adf_residual = adf_pred - adf
            grad += self.lambda_adf * self._adf_gradient(x, adf_residual)
            adf_cost = 0.5 * self.lambda_adf * float(np.mean(adf_residual ** 2))

            edx_cost = 0.0
            if edx is not None and edx_ref is not None:
                edx_pred = np.tensordot(x, edx_ref.T, axes=([-1], [0]))
                edx_residual = edx_pred - edx
                grad += self.lambda_edx * np.tensordot(edx_residual, edx_ref, axes=([-1], [0]))
                edx_cost = 0.5 * self.lambda_edx * float(np.mean(edx_residual ** 2))

            eels_cost = 0.0
            if eels is not None and eels_ref is not None:
                eels_pred = np.tensordot(x, eels_ref.T, axes=([-1], [0]))
                eels_residual = eels_pred - eels
                grad += self.lambda_eels * np.tensordot(eels_residual, eels_ref, axes=([-1], [0]))
                eels_cost = 0.5 * self.lambda_eels * float(np.mean(eels_residual ** 2))

            tv_cost = 0.0
            if self.lambda_tv:
                grad += self.lambda_tv * _tv_gradient(x)
                tv_cost = self.lambda_tv * self._tv_value(x)

            grad /= max(1.0, float(x.shape[-1]))
            x = np.maximum(x - self.step_size * grad, 0.0)

            total = adf_cost + edx_cost + eels_cost + tv_cost
            costs["total"].append(total)
            costs["adf"].append(adf_cost)
            costs["edx"].append(edx_cost)
            costs["eels"].append(eels_cost)
            costs["tv"].append(tv_cost)

            if abs(previous_total - total) <= self.tolerance * max(1.0, previous_total):
                break
            previous_total = total

        self.result_ = FusionResult(
            concentrations=x,
            elements=self.elements.copy(),
            cost_history=costs,
            metadata={
                "iterations": iteration + 1,
                "gamma": self.gamma,
                "lambda_adf": self.lambda_adf,
                "lambda_edx": self.lambda_edx,
                "lambda_eels": self.lambda_eels,
                "lambda_tv": self.lambda_tv,
            },
        )
        return self

    def get_results(self) -> FusionResult:
        if self.result_ is None:
            raise RuntimeError("fit() must be called before get_results()")
        return self.result_

    def _validate_dataset(self, dataset: MultiModalDataset) -> None:
        if dataset.elements != self.elements:
            raise ParameterError("dataset elements must match solver elements", "elements")
        if dataset.edx is not None and dataset.edx_reference is None:
            raise DataError("edx_reference is required when edx data is present")
        if dataset.eels is not None and dataset.eels_reference is None:
            raise DataError("eels_reference is required when eels data is present")

    def _normalise_reference(self, reference: Optional[np.ndarray]) -> Optional[np.ndarray]:
        if reference is None:
            return None
        ref = np.asarray(reference, dtype=np.float64)
        if ref.shape[1] != len(self.elements):
            raise DataError("reference column count must match elements", data_shape=ref.shape)
        scale = np.linalg.norm(ref, axis=0)
        scale[scale == 0] = 1.0
        return ref / scale

    def _initial_concentrations(
        self,
        dataset: MultiModalDataset,
        edx: Optional[np.ndarray],
        eels: Optional[np.ndarray],
        edx_ref: Optional[np.ndarray],
        eels_ref: Optional[np.ndarray],
        initial: Optional[np.ndarray],
    ) -> np.ndarray:
        if initial is not None:
            x = np.asarray(initial, dtype=np.float64)
            if x.shape != dataset.spatial_shape + (len(self.elements),):
                raise DataError("initial has incompatible shape", data_shape=x.shape)
            return np.maximum(x, 0.0)

        refs = []
        data = []
        if edx is not None and edx_ref is not None:
            refs.append(edx_ref)
            data.append(edx.reshape(-1, edx.shape[-1]))
        if eels is not None and eels_ref is not None:
            refs.append(eels_ref)
            data.append(eels.reshape(-1, eels.shape[-1]))

        if refs:
            design = np.vstack(refs)
            observations = np.hstack(data)
            pinv = np.linalg.pinv(design)
            x = observations.dot(pinv.T)
            return np.maximum(x.reshape(dataset.spatial_shape + (len(self.elements),)), 0.0)

        adf = _normalise_signal(dataset.adf)
        weights = self.adf_weights / np.max(self.adf_weights)
        guess = np.repeat(adf[..., None], len(self.elements), axis=-1)
        return np.maximum(guess / np.maximum(weights, 1e-12), 0.0)

    def _adf_forward(self, concentrations: np.ndarray) -> np.ndarray:
        weights = self.adf_weights / np.max(self.adf_weights)
        return np.tensordot(
            np.power(np.maximum(concentrations, 0.0), self.gamma),
            weights,
            axes=([-1], [0]),
        )

    def _adf_gradient(self, concentrations: np.ndarray, residual: np.ndarray) -> np.ndarray:
        weights = self.adf_weights / np.max(self.adf_weights)
        base = np.power(np.maximum(concentrations, 1e-12), self.gamma - 1.0)
        return self.gamma * residual[..., None] * base * weights

    def _tv_value(self, concentrations: np.ndarray) -> float:
        dx = np.diff(concentrations, axis=1)
        dy = np.diff(concentrations, axis=0)
        return float(np.mean(np.abs(dx)) + np.mean(np.abs(dy)))
