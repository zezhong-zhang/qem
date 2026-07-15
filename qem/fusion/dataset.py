"""Dataset containers and loaders for pre-aligned ADF-EDX-EELS data."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import scipy.io as sio

from qem.utils.exceptions import DataError, ParameterError


def _as_float_array(data: np.ndarray, name: str) -> np.ndarray:
    arr = np.asarray(data, dtype=np.float64)
    if arr.size == 0:
        raise DataError(f"{name} cannot be empty", data_shape=arr.shape)
    if not np.all(np.isfinite(arr)):
        raise DataError(f"{name} contains NaN or infinite values", data_shape=arr.shape)
    return arr


def _axis_values(signal, axis_index: int = -1) -> np.ndarray | None:
    try:
        axis = signal.axes_manager.signal_axes[axis_index]
        return axis.axis.astype(np.float64)
    except Exception:
        return None


def _load_hspy(path: Path, lazy: bool = False) -> tuple[np.ndarray, np.ndarray | None, object]:
    try:
        import hyperspy.api as hs
    except ImportError as exc:
        raise ImportError("HyperSpy is required to load .hspy files") from exc

    signal = hs.load(str(path), lazy=lazy)
    data = signal.data
    if lazy and hasattr(data, "compute"):
        data = data.compute()
    return np.asarray(data), _axis_values(signal), signal


def _normalize_columns(matrix: np.ndarray) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=np.float64)
    scale = np.linalg.norm(matrix, axis=0)
    scale[scale == 0] = 1.0
    return matrix / scale


def _gaussian_reference_from_calibration(
    calibration_path: Path,
    elements: Sequence[str],
    energy_axis: np.ndarray | None,
    channels: int,
) -> np.ndarray:
    mat = sio.loadmat(str(calibration_path), simplify_cells=True)
    comp_dict = mat.get("comp_dict", {})
    if energy_axis is None:
        energy_axis = np.arange(channels, dtype=np.float64)
    else:
        energy_axis = np.asarray(energy_axis, dtype=np.float64)

    references = np.zeros((energy_axis.size, len(elements)), dtype=np.float64)
    for col, element in enumerate(elements):
        line_dict = comp_dict.get(element, {})
        if not isinstance(line_dict, Mapping):
            continue
        for params in line_dict.values():
            if not isinstance(params, Mapping):
                continue
            amplitude = float(params.get("A", 0.0))
            centre = float(params.get("centre", 0.0))
            sigma = max(float(params.get("sigma", 1.0)), 1e-12)
            references[:, col] += amplitude * np.exp(
                -0.5 * ((energy_axis - centre) / sigma) ** 2
            )

    return _normalize_columns(references)


def _eels_reference_from_map(
    reference_path: Path,
    elements: Sequence[str],
    channels: int,
) -> np.ndarray:
    mat = sio.loadmat(str(reference_path), simplify_cells=True)
    references = np.zeros((channels, len(elements)), dtype=np.float64)
    for col, element in enumerate(elements):
        value = float(mat.get(element, 0.0))
        references[:, col] = value
    return _normalize_columns(references)


@dataclass
class MultiModalDataset:
    """
    Container for pre-aligned ADF, EDX, and EELS signals.

    Parameters
    ----------
    adf:
        2D annular dark-field image.
    edx:
        Optional 3D EDX spectrum image in ``(height, width, channels)`` order.
    eels:
        Optional 3D EELS spectrum image in ``(height, width, channels)`` order.
    elements:
        Element symbols corresponding to concentration channels.
    edx_reference, eels_reference:
        Forward spectra with shape ``(channels, elements)``.
    """

    adf: np.ndarray
    edx: np.ndarray | None = None
    eels: np.ndarray | None = None
    elements: list[str] = field(default_factory=list)
    edx_reference: np.ndarray | None = None
    eels_reference: np.ndarray | None = None
    edx_energy: np.ndarray | None = None
    eels_energy: np.ndarray | None = None
    metadata: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.adf = _as_float_array(self.adf, "adf")
        if self.adf.ndim != 2:
            raise DataError("adf must be a 2D image", data_shape=self.adf.shape)

        if self.edx is not None:
            self.edx = _as_float_array(self.edx, "edx")
            if self.edx.ndim != 3:
                raise DataError("edx must be a 3D spectrum image", data_shape=self.edx.shape)
        if self.eels is not None:
            self.eels = _as_float_array(self.eels, "eels")
            if self.eels.ndim != 3:
                raise DataError("eels must be a 3D spectrum image", data_shape=self.eels.shape)

        if not self.elements:
            raise ParameterError("elements must contain at least one element symbol", "elements")
        self.elements = [str(element) for element in self.elements]

        if self.edx_reference is not None:
            self.edx_reference = _as_float_array(self.edx_reference, "edx_reference")
            if self.edx_reference.shape[1] != len(self.elements):
                raise DataError("edx_reference column count must match elements")
        if self.eels_reference is not None:
            self.eels_reference = _as_float_array(self.eels_reference, "eels_reference")
            if self.eels_reference.shape[1] != len(self.elements):
                raise DataError("eels_reference column count must match elements")

    @property
    def spatial_shape(self) -> tuple[int, int]:
        return tuple(self.adf.shape)

    @classmethod
    def from_chemical_maps(
        cls,
        adf: np.ndarray,
        chemical_maps: Mapping[str, np.ndarray],
        elements: Sequence[str] | None = None,
    ) -> MultiModalDataset:
        """
        Build a dataset from tutorial-style elemental maps and ADF.

        The vendor ``multi_modal`` tutorial starts from already-quantified EDX
        maps rather than raw spectra. This constructor represents those maps as
        an EDX spectrum image with one channel per element and an identity EDX
        measurement matrix, so the same joint optimizer can be used.
        """

        if elements is None:
            elements = list(chemical_maps.keys())
        maps = []
        for element in elements:
            if element not in chemical_maps:
                raise ParameterError(f"missing chemical map for {element}", "chemical_maps")
            maps.append(np.asarray(chemical_maps[element], dtype=np.float64))

        edx = np.stack(maps, axis=-1)
        edx_reference = np.eye(len(elements), dtype=np.float64)
        return cls(
            adf=adf,
            edx=edx,
            elements=list(elements),
            edx_reference=edx_reference,
            metadata={"source": "chemical_maps"},
        )

    @classmethod
    def from_hspy(
        cls,
        directory: str,
        elements: Sequence[str],
        adf_file: str = "adf_aligned.hspy",
        edx_file: str = "edx_aligned.hspy",
        eels_high_loss_file: str = "eels_hl_aligned_bin.hspy",
        eels_low_loss_file: str | None = None,
        edx_calibration: str | None = "edx_calibration.mat",
        eels_reference: str | None = "eels_element_maps.mat",
        adf_reducer: str = "mean",
        lazy: bool = False,
    ) -> MultiModalDataset:
        """
        Load pre-aligned HSPY files and optional calibration references.

        ADF stacks are reduced to a single image with ``mean`` or by selecting
        the first frame. EDX and EELS are kept as spectrum images in
        ``(height, width, channels)`` order as reported by HyperSpy.
        """

        base = Path(directory).expanduser()
        adf, _, _ = _load_hspy(base / adf_file, lazy=lazy)
        if adf.ndim == 3:
            if adf_reducer == "mean":
                adf = np.mean(adf, axis=0)
            elif adf_reducer == "first":
                adf = adf[0]
            else:
                raise ParameterError("adf_reducer must be 'mean' or 'first'", "adf_reducer")

        edx, edx_energy, _ = _load_hspy(base / edx_file, lazy=lazy)
        eels, eels_energy, _ = _load_hspy(base / eels_high_loss_file, lazy=lazy)
        if eels_low_loss_file:
            _, low_loss_energy, _ = _load_hspy(base / eels_low_loss_file, lazy=lazy)
            eels_energy = eels_energy if eels_energy is not None else low_loss_energy

        edx_ref = None
        if edx_calibration:
            edx_ref = _gaussian_reference_from_calibration(
                base / edx_calibration, elements, edx_energy, edx.shape[-1]
            )

        eels_ref = None
        if eels_reference:
            eels_ref = _eels_reference_from_map(base / eels_reference, elements, eels.shape[-1])

        return cls(
            adf=adf,
            edx=edx,
            eels=eels,
            elements=list(elements),
            edx_reference=edx_ref,
            eels_reference=eels_ref,
            edx_energy=edx_energy,
            eels_energy=eels_energy,
            metadata={"source_directory": str(base)},
        )

    @classmethod
    def synthetic(
        cls,
        concentrations: np.ndarray,
        elements: Sequence[str],
        edx_reference: np.ndarray | None = None,
        eels_reference: np.ndarray | None = None,
        adf_weights: Iterable[float] | None = None,
        gamma: float = 1.6,
        noise: float = 0.0,
        random_state: int | None = None,
    ) -> MultiModalDataset:
        """Create a synthetic dataset for examples and tests."""

        conc = _as_float_array(concentrations, "concentrations")
        if conc.ndim != 3 or conc.shape[-1] != len(elements):
            raise DataError("concentrations must have shape (height, width, elements)")

        if adf_weights is None:
            from qem.utils.elements import chemical_symbols

            lookup = {symbol: index for index, symbol in enumerate(chemical_symbols)}
            weights = np.array([lookup.get(element, 1) for element in elements], dtype=np.float64)
        else:
            weights = np.asarray(list(adf_weights), dtype=np.float64)

        rng = np.random.default_rng(random_state)
        adf = np.tensordot(np.power(np.maximum(conc, 0.0), gamma), weights, axes=([-1], [0]))
        if noise:
            adf = adf + noise * rng.standard_normal(adf.shape)

        edx = None
        if edx_reference is not None:
            edx_reference = np.asarray(edx_reference, dtype=np.float64)
            edx = np.tensordot(conc, edx_reference.T, axes=([-1], [0]))
            if noise:
                edx = edx + noise * rng.standard_normal(edx.shape)

        eels = None
        if eels_reference is not None:
            eels_reference = np.asarray(eels_reference, dtype=np.float64)
            eels = np.tensordot(conc, eels_reference.T, axes=([-1], [0]))
            if noise:
                eels = eels + noise * rng.standard_normal(eels.shape)

        return cls(
            adf=adf,
            edx=edx,
            eels=eels,
            elements=list(elements),
            edx_reference=edx_reference,
            eels_reference=eels_reference,
            metadata={"synthetic": True},
        )
