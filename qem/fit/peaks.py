"""Peak finding / refinement / dedup helpers — extracted from
qem.fit.fitter (Linus #9). Bound back onto Fitter via _bind(Fitter)
from qem.fit.fitter so existing fitter.find_peaks(...) call sites
keep working.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.ndimage import gaussian_filter
from skimage.feature import peak_local_max
from tqdm import tqdm

from qem.fit.refine import calculate_center_of_mass
from qem.fit.voronoi import voronoi_point_record
from qem.utils.tensors import to_tensor
from qem.viz.geometry import remove_close_coordinates as _geom_remove_close_coordinates
from qem.viz.select import InteractivePlot

if TYPE_CHECKING:
    from qem.fit.fitter import Fitter  # noqa: F401


def import_coordinates(self, coordinates: np.ndarray):
    self.coordinates = coordinates[:, :2]

def find_peaks(
    self,
    min_distance: int = 10,
    threshold_rel: float = 0.2,
    threshold_abs=None,
    exclude_border: bool = False,
    plot: bool = True,
    region_index: int = 0,
    sigma: float = 5,
):
    """
    Find peaks (atomic columns) in a region of the image by local maxima.

    Args:
        min_distance (int, optional): Minimum distance between peaks, in pixels.
            Defaults to 10.
        threshold_rel (float, optional): Relative intensity threshold in [0, 1].
            Defaults to 0.2.
        threshold_abs (float, optional): Absolute intensity threshold; overrides
            ``threshold_rel`` when set. Defaults to None.
        exclude_border (bool, optional): Whether to exclude the image border.
            Defaults to False.
        plot (bool, optional): Whether to open the interactive peak editor.
            Defaults to True.
        region_index (int, optional): Region to search; must be in ``self.regions``.
            Defaults to 0.
        sigma (float, optional): Gaussian pre-smoothing sigma, in pixels.
            Defaults to 5.

    Returns:
        np.array: The (N, 2) array of [x, y] peak coordinates in pixels.
    """
    if region_index not in self.regions.keys:
        raise ValueError(
            f"region_index {region_index} not in regions {list(self.regions.keys)}"
        )
    region_map = self.regions.region_map == region_index
    image_filtered = gaussian_filter(self.image, sigma)
    peaks_locations = peak_local_max(
        image_filtered * region_map,
        min_distance=min_distance,
        threshold_rel=threshold_rel,
        threshold_abs=threshold_abs,
        exclude_border=exclude_border,
    )
    if self.coordinates.size > 0:
        column_mask = self.region_column_labels == region_index
        coordinates = np.delete(self.coordinates, np.where(column_mask), axis=0)
        coordinates = np.vstack(
            [coordinates, peaks_locations[:, [1, 0]].astype(float)]
        )
        self.coordinates = coordinates
        atom_types = np.delete(self.atom_types, np.where(column_mask), axis=0)
        atom_types = np.append(
            atom_types, np.zeros(peaks_locations.shape[0], dtype=int)
        )
        self.atom_types = atom_types
    else:
        self.coordinates = peaks_locations[:, [1, 0]].astype(float)
        self.atom_types = np.zeros(peaks_locations.shape[0], dtype=int)
    self._assert_columns_aligned()
    if plot:
        self.add_or_remove_peaks(min_distance=min_distance, image=self.image)
    return self.coordinates

def get_nearest_peak_distance(self, peak_position: np.ndarray):
    """
    Get the distance of the nearest peak for each peak.

    Args:
        peak_positions (np.array): The positions of the peaks.
        threshold (int, optional): The threshold distance. Defaults to 10.

    Returns:
        np.array: The distances of the nearest peaks.
    """
    other_peaks = np.delete(
        self.coordinates, np.where(self.coordinates == peak_position), axis=0
    )
    distances = np.linalg.norm(other_peaks - peak_position, axis=1).min()
    return distances

def refine_center_of_mass(self, params=None, plot=False):
    # Refine center of mass for each Voronoi cell
    pre_coordinates = self.coordinates.copy()
    current_coordinates = self.coordinates.copy()
    converged = False

    if params is None and hasattr(self, "params") and len(self.params) > 0:
        params = self.params
    elif params is None:
        params = self.init_params()
    while not converged:
        # Generate Voronoi cell map
        coords = np.stack([pre_coordinates[:, 1], pre_coordinates[:, 0]])  # (y, x)
        max_radius = params["width"].max() * 5
        point_record = voronoi_point_record(self.image, coords, max_radius)

        # In refine_center_of_mass, replace the for-loop with:
        with ThreadPoolExecutor() as executor:
            futures = [
                executor.submit(self._refine_one_center, i, point_record, plot)
                for i in range(self.num_coordinates)
            ]
            for future in tqdm(
                as_completed(futures),
                total=self.num_coordinates,
                desc="Refining center of mass",
            ):
                result, i = future.result()
                if result is not None:
                    current_coordinates[i] = result

        converged = np.abs(current_coordinates - pre_coordinates).mean() < 0.5
        pre_coordinates = current_coordinates.copy()
    params["pos_x"] = current_coordinates[:, 0]
    params["pos_y"] = current_coordinates[:, 1]
    self.params = params
    self.coordinates = current_coordinates
    return params

def _refine_one_center(self, i: int, point_record: np.ndarray, plot: bool = False):
    mask = point_record == (i + 1)
    if not np.any(mask):
        return None, i

    cell_img = self.image * mask
    ys, xs = np.where(mask)
    y0, y1 = ys.min(), ys.max() + 1
    x0, x1 = xs.min(), xs.max() + 1
    cropped_img = cell_img[y0:y1, x0:x1]
    cropped_mask = mask[y0:y1, x0:x1]

    # Subtract local min (only over masked region)
    local_min = cropped_img[cropped_mask].min()
    cropped_img = cropped_img - local_min
    cropped_img[~cropped_mask] = 0

    # Normalize for center of mass
    if cropped_img[cropped_mask].max() > 0:
        norm_img = (cropped_img - cropped_img[cropped_mask].min()) / (
            cropped_img[cropped_mask].max() - cropped_img[cropped_mask].min()
        )
    else:
        norm_img = cropped_img
    norm_img[~cropped_mask] = 0

    # Compute center of mass in the cropped region
    local_y, local_x = calculate_center_of_mass(norm_img)
    if not (isinstance(local_x, float) and isinstance(local_y, float)):
        raise TypeError(
            "calculate_center_of_mass must return float coordinates, got "
            f"{type(local_x).__name__}, {type(local_y).__name__}"
        )
    result = np.array(
        [
            x0 + local_x,
            y0 + local_y,
        ],
        dtype=float,
    )

    if plot:
        plt.clf()
        plt.imshow(norm_img, cmap="gray")
        plt.scatter(local_x, local_y, color="red", s=2, label="refined")
        plt.legend()
        plt.pause(1.0)
    return result, i

def refine_local_max(
    self,
    plot=False,
    min_distance=10,
    threshold_rel=0.3,
    threshold_abs=None,
    exclude_border=True,
):
    windows_size = min_distance * 2
    peak_total = np.array([], dtype=int).reshape(0, 2)
    for coordinate_index in range(self.num_coordinates):
        x, y = self.coordinates[coordinate_index]
        top = max(int(x) - windows_size, 0)
        bottom = min(int(x) + windows_size + 1, self.nx)
        left = max(int(y) - windows_size, 0)
        right = min(int(y) + windows_size + 1, self.ny)
        # calculate the mask for distance < r
        region = self.image[left:right, top:bottom]
        peaks_locations = peak_local_max(
            region,
            min_distance=int(min_distance / 4),
            threshold_rel=threshold_rel,
            threshold_abs=threshold_abs,
            exclude_border=exclude_border,
        )
        peaks_locations = peaks_locations[:, [1, 0]].astype(int)
        if peaks_locations.shape[0] > 0:
            peak_total = np.append(
                peak_total,
                peaks_locations
                + np.array([int(x) - windows_size, int(y) - windows_size]),
                axis=0,
            )
        if plot:
            plt.clf()
            plt.subplot(1, 2, 1)
            plt.imshow(self.image, cmap="gray")
            plt.scatter(
                self.coordinates[:, 0],
                self.coordinates[:, 1],
                color="blue",
                s=1,
            )
            plt.scatter(x, y, color="red", s=2)
            plt.subplot(1, 2, 2)
            plt.imshow(region, cmap="gray")
            plt.scatter(
                x % 1 + windows_size, y % 1 + windows_size, color="red", s=2
            )
            if peaks_locations.shape[0] > 0:
                plt.scatter(
                    peaks_locations[:, 0],
                    peaks_locations[:, 1],
                    color="green",
                    s=2,
                )
            plt.show()
            plt.pause(1.0)
    self.coordinates = np.unique(peak_total, axis=0)
    # self.coordinates = self.refine_duplicate_peaks()
    return self.coordinates

def remove_close_coordinates(self, threshold: int = 10):
    """
    Remove coordinates that are too close to each other, considering periodic boundary conditions (PBC) if enabled.
    Also removes the corresponding atom types from self.atom_types.

    Args:
        threshold (int): Minimum allowed distance between coordinates. Defaults to 10.

    Returns:
        np.ndarray: The filtered coordinates.
    """
    if self.pbc:
        # Remove close coordinates in the original box
        coords, atom_types, _ = _geom_remove_close_coordinates(self.coordinates.copy(), self.atom_types.copy(), threshold)

        # Identify coordinates near the boundary
        mask_boundary = (
            (coords[:, 0] < threshold)
            | (coords[:, 0] > self.nx - threshold)
            | (coords[:, 1] < threshold)
            | (coords[:, 1] > self.ny - threshold)
        )
        coords_boundary = coords[mask_boundary]
        atom_types_boundary = atom_types[mask_boundary]

        # Generate periodic images of boundary coordinates
        shifts = np.array([
            [i * self.nx, j * self.ny]
            for i, j in [(1, 0), (0, 1), (1, 1), (-1, 0), (0, -1), (-1, -1), (1, -1), (-1, 1)]
        ])

        # Check if any periodic image is too close to the original boundary coordinates
        to_remove = set()
        for shift in shifts:
            shifted_coords = coords_boundary + shift
            for i, coord in enumerate(coords_boundary):
                distances = np.linalg.norm(shifted_coords - coord, axis=1)
                if (distances < threshold).any():
                    to_remove.add(i)

        # Remove overlapping boundary coordinates and corresponding atom types
        coords_boundary_filtered = np.delete(coords_boundary, list(to_remove), axis=0)
        atom_types_boundary_filtered = np.delete(atom_types_boundary, list(to_remove), axis=0)

        # Combine non-boundary and filtered boundary coordinates and atom types
        self.coordinates = np.vstack([coords[~mask_boundary], coords_boundary_filtered])
        self.atom_types = np.concatenate([atom_types[~mask_boundary], atom_types_boundary_filtered])
    else:
        self.coordinates, self.atom_types,_ = _geom_remove_close_coordinates(self.coordinates, self.atom_types, threshold)

    return self.coordinates, self.atom_types

def add_or_remove_peaks(self, min_distance: int = 2, image=None):
    if image is None:
        image = self.image
    peaks_locations = self.coordinates
    interactive_plot = InteractivePlot(
        image=image,
        peaks_locations=peaks_locations,
        atom_types=self.atom_types,
        tolerance=min_distance,
    )
    interactive_plot.add_or_remove(tolerance=min_distance)
    peaks_locations = [interactive_plot.pos_x, interactive_plot.pos_y]
    peaks_locations = np.array(peaks_locations).T.astype(float)
    self.coordinates = peaks_locations
    self.atom_types = interactive_plot.atom_types
    return peaks_locations

def remove_peaks_outside_image(self):
    coordinates = self.coordinates
    mask = (
        (coordinates[:, 0] >= 0)
        & (coordinates[:, 0] < self.nx)
        & (coordinates[:, 1] >= 0)
        & (coordinates[:, 1] < self.ny)
    )
    self.coordinates = coordinates[mask]
    return self.coordinates

# loss function and model prediction



# ---------------------------------------------------------------------------
# Sub-pixel parabolic peak refinement (used as the upstream warmup before
# any joint fit; see qem.fit.pipeline.fit_pipeline).
# ---------------------------------------------------------------------------

def subpixel_parabolic_refine(
    image: np.ndarray,
    coords_px: np.ndarray,
    *,
    search_window: int = 0,
    max_shift: float = 0.5,
) -> np.ndarray:
    """Refine peak positions to sub-pixel accuracy via parabolic fit.

    For each input coordinate, parabolic-fit the 3×3 around the rounded
    integer position and solve ∇f = 0 for the sub-pixel offset. ±0.05 px
    typical accuracy on STEM peaks.

    The fit is only accepted when the 3×3 patch's quadratic form is
    actually a maximum — both 2nd derivatives negative AND positive
    Hessian determinant. On low-contrast or noisy patches the fit can
    yield a saddle or a minimum; in that case we keep the input
    coordinate rather than displace it. This is critical: without the
    check, ~25% of atoms on low-contrast images get pushed up to
    ``max_shift`` pixels in essentially random directions, destroying
    the lattice geometry.

    Args:
        image: 2D image (H, W). Float-valued.
        coords_px: ``(N, 2)`` array of ``[x, y]`` in pixels (may be float).
        search_window: optional integer-pixel local-max search radius
            *before* the parabolic fit. Default ``0`` (no search — anchor
            on ``round(coord)``). Use a positive value only when input
            coordinates are noisier than 0.5 px; with already-good
            sub-pixel inputs (e.g. from a prior fit), a non-zero search
            window can snap to a neighbouring integer pixel and bias
            the result by up to ``search_window`` pixels.
        max_shift: cap on |Δ| per axis (in pixels). Default 0.5 (the
            largest interpolation that still lies inside the 3×3 patch).
            Tighten to 0.3 if the input coords are already known to be
            sub-pixel-accurate.

    Returns:
        ``(N, 2)`` array of refined ``[x, y]`` in pixels.
    """
    image = np.asarray(image, dtype=np.float64)
    H, W = image.shape
    N = coords_px.shape[0]
    out = coords_px.astype(np.float64).copy()
    sw = max(0, int(search_window))

    for k in range(N):
        x0, y0 = float(coords_px[k, 0]), float(coords_px[k, 1])
        ix = int(round(x0))
        iy = int(round(y0))
        if sw > 0:
            x_lo = max(ix - sw, 1)
            x_hi = min(ix + sw + 1, W - 1)
            y_lo = max(iy - sw, 1)
            y_hi = min(iy + sw + 1, H - 1)
            if x_hi <= x_lo or y_hi <= y_lo:
                continue
            block = image[y_lo:y_hi, x_lo:x_hi]
            my, mx = np.unravel_index(int(np.argmax(block)), block.shape)
            ix_max = mx + x_lo
            iy_max = my + y_lo
        else:
            ix_max, iy_max = ix, iy
        if ix_max < 1 or ix_max >= W - 1 or iy_max < 1 or iy_max >= H - 1:
            out[k] = [float(ix_max), float(iy_max)]
            continue

        f = image[iy_max - 1 : iy_max + 2, ix_max - 1 : ix_max + 2]
        # f(dx, dy) = f00 + a·dx + b·dy + c·dx² + d·dy² + e·dx·dy
        a = (f[1, 2] - f[1, 0]) * 0.5
        b = (f[2, 1] - f[0, 1]) * 0.5
        c = (f[1, 2] + f[1, 0] - 2.0 * f[1, 1]) * 0.5
        d = (f[2, 1] + f[0, 1] - 2.0 * f[1, 1]) * 0.5
        e = (f[2, 2] - f[2, 0] - f[0, 2] + f[0, 0]) * 0.25
        det = (2.0 * c) * (2.0 * d) - e * e
        # Accept only if the patch is a genuine maximum:
        #   * Hessian H = [[2c, e], [e, 2d]] negative definite ⇔
        #     2c < 0  AND  4cd − e² > 0
        #   * Equivalently: c < 0, d < 0, det > 0.
        # Reject saddles, minima, and degenerate (det ≈ 0) patches.
        # On low-contrast / noisy data this is the difference between
        # 25% of atoms moved randomly and a clean refinement.
        if c >= 0.0 or d >= 0.0 or det <= 1e-12:
            out[k] = [float(ix_max), float(iy_max)]
            continue
        dx = (-a * 2.0 * d + b * e) / det
        dy = (-b * 2.0 * c + a * e) / det
        # Cap to keep us inside the 3×3 patch where the parabolic
        # approximation is valid (and to limit damage from a bad fit
        # we somehow still accepted).
        dx = max(-max_shift, min(max_shift, dx))
        dy = max(-max_shift, min(max_shift, dy))
        out[k, 0] = float(ix_max) + dx
        out[k, 1] = float(iy_max) + dy
    return out


def refine_peaks_subpixel(self, *, search_window: int = 2) -> np.ndarray:
    """Refine ``self.coordinates`` via parabolic sub-pixel fit.

    Convenience wrapper around :func:`subpixel_parabolic_refine` that
    operates on ``self.image`` and ``self.coordinates``. Updates the
    fitter in place and returns the new ``(N, 2)`` coords.
    """
    refined = subpixel_parabolic_refine(
        self.image, self.coordinates, search_window=search_window,
    )
    self.coordinates = refined
    if hasattr(self, "params") and self.params:
        with torch.inference_mode():
            self.params["pos_x"] = to_tensor(refined[:, 0], dtype="float32")
            self.params["pos_y"] = to_tensor(refined[:, 1], dtype="float32")
    return refined


class FitterPeaksMixin:
    """Peak-detection / refinement / curation API for :class:`Fitter`.

    Class-level method *bindings* (not redefinitions) — the bodies live
    as module-level functions above. This pattern replaces the legacy
    ``_bind(cls)`` monkey-patch with a statically-declared mixin so
    type-checkers see the methods and ``super()`` works in subclasses.
    Same runtime semantics; zero overhead.
    """

    import_coordinates = import_coordinates
    find_peaks = find_peaks
    get_nearest_peak_distance = get_nearest_peak_distance
    refine_center_of_mass = refine_center_of_mass
    _refine_one_center = _refine_one_center
    refine_local_max = refine_local_max
    remove_close_coordinates = remove_close_coordinates
    add_or_remove_peaks = add_or_remove_peaks
    remove_peaks_outside_image = remove_peaks_outside_image
    refine_peaks_subpixel = refine_peaks_subpixel


__all__ = [
    "FitterPeaksMixin",
    "subpixel_parabolic_refine",
    # Free-function-style names also exported for backwards compatibility:
    "import_coordinates",
    "find_peaks",
    "get_nearest_peak_distance",
    "refine_center_of_mass",
    "refine_local_max",
    "remove_close_coordinates",
    "add_or_remove_peaks",
    "remove_peaks_outside_image",
    "refine_peaks_subpixel",
]
