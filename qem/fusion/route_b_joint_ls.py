"""Route B joint least-squares fusion solver."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Mapping, Optional

import numpy as np
import torch
from torch import nn

from qem.fit._loop import fit_loop, make_optimizer
from qem.utils.exceptions import DataError, ParameterError
from qem.utils.elements import chemical_symbols
from qem.utils.tensors import best_device

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

        x_init = self._initial_concentrations(dataset, edx, eels, edx_ref, eels_ref, initial)

        device = best_device()
        dtype = torch.float32

        adf_t = torch.as_tensor(adf, dtype=dtype, device=device)
        edx_t = torch.as_tensor(edx, dtype=dtype, device=device) if edx is not None else None
        eels_t = torch.as_tensor(eels, dtype=dtype, device=device) if eels is not None else None
        edx_ref_t = (
            torch.as_tensor(edx_ref, dtype=dtype, device=device)
            if edx_ref is not None and edx is not None else None
        )
        eels_ref_t = (
            torch.as_tensor(eels_ref, dtype=dtype, device=device)
            if eels_ref is not None and eels is not None else None
        )
        weights_norm = self.adf_weights / np.max(self.adf_weights)
        weights_t = torch.as_tensor(weights_norm, dtype=dtype, device=device)

        n_elements = max(1, len(self.elements))
        scale = 1.0 / float(n_elements)

        costs = {"total": [], "adf": [], "edx": [], "eels": [], "tv": []}

        class _ConcentrationModel(nn.Module):
            def __init__(self, x0: torch.Tensor) -> None:
                super().__init__()
                self.x = nn.Parameter(x0)

            def forward(self, _inputs):  # noqa: ARG002
                return self.x

        gamma = self.gamma
        lambda_adf = self.lambda_adf
        lambda_edx = self.lambda_edx
        lambda_eels = self.lambda_eels
        lambda_tv = self.lambda_tv

        def joint_loss(_target, x_param: torch.Tensor) -> torch.Tensor:
            # Gradient is scaled by 1/n_elements to mirror the historic
            # `grad /= max(1.0, n_elements)` from the numpy loop.
            x_pos = torch.clamp(x_param, min=0.0)
            adf_pred = torch.tensordot(
                torch.pow(x_pos, gamma), weights_t, dims=([-1], [0])
            )
            adf_residual = adf_pred - adf_t
            adf_cost = 0.5 * lambda_adf * torch.mean(adf_residual ** 2)
            cost = adf_cost
            edx_cost = torch.zeros((), dtype=dtype, device=device)
            eels_cost = torch.zeros((), dtype=dtype, device=device)
            tv_cost = torch.zeros((), dtype=dtype, device=device)

            if edx_t is not None and edx_ref_t is not None:
                edx_pred = torch.tensordot(x_param, edx_ref_t.T, dims=([-1], [0]))
                edx_cost = 0.5 * lambda_edx * torch.mean((edx_pred - edx_t) ** 2)
                cost = cost + edx_cost
            if eels_t is not None and eels_ref_t is not None:
                eels_pred = torch.tensordot(x_param, eels_ref_t.T, dims=([-1], [0]))
                eels_cost = 0.5 * lambda_eels * torch.mean((eels_pred - eels_t) ** 2)
                cost = cost + eels_cost
            if lambda_tv:
                dx = x_param[:, 1:, :] - x_param[:, :-1, :]
                dy = x_param[1:, :, :] - x_param[:-1, :, :]
                tv_cost = lambda_tv * (torch.mean(torch.abs(dx)) + torch.mean(torch.abs(dy)))
                cost = cost + tv_cost

            costs["adf"].append(float(adf_cost.detach()))
            costs["edx"].append(float(edx_cost.detach()))
            costs["eels"].append(float(eels_cost.detach()))
            costs["tv"].append(float(tv_cost.detach()))
            costs["total"].append(float(cost.detach()))
            return cost * scale

        model = _ConcentrationModel(
            torch.as_tensor(x_init, dtype=dtype, device=device)
        )

        # Project to the non-negative cone after each optimizer step
        # (PGD: take the step, then project).
        def _project_nonneg(m: nn.Module) -> None:
            m.x.clamp_(min=0.0)  # type: ignore[union-attr]

        # SGD mirrors the historic projected-gradient loop (fixed step,
        # no momentum). Adam at the user-supplied LR — typically ~0.5 —
        # would wildly overshoot since its per-step update magnitude is
        # O(lr) regardless of the gradient scale.
        optimizer = make_optimizer("sgd", model.parameters(), self.step_size)
        result = fit_loop(
            model=model,
            inputs=None,
            target=None,  # type: ignore[arg-type]  # loss_fn ignores target
            loss_fn=joint_loss,
            optimizer=optimizer,
            epochs=self.max_iter,
            tol=self.tolerance,
            patience=self.max_iter,
            lr_patience=10,
            lr_factor=0.5,
            min_lr=1e-6,
            snapshot_every=max(self.max_iter, 1),
            post_step=_project_nonneg,
            verbose=False,
        )

        x = torch.clamp(model.x.detach(), min=0.0).cpu().numpy().astype(np.float64)

        self.result_ = FusionResult(
            concentrations=x,
            elements=self.elements.copy(),
            cost_history=costs,
            metadata={
                "iterations": result.epochs_run,
                "gamma": self.gamma,
                "lambda_adf": self.lambda_adf,
                "lambda_edx": self.lambda_edx,
                "lambda_eels": self.lambda_eels,
                "lambda_tv": self.lambda_tv,
                "device": str(device),
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

