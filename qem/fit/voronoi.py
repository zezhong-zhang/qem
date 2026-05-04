from __future__ import annotations

from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numba as nb
import numpy as np
import torch
from matplotlib_scalebar.scalebar import ScaleBar
from skimage.segmentation import watershed
from tqdm import tqdm as progressbar
from scipy.spatial import cKDTree

from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import nullcontext

from scipy.optimize import curve_fit
from tqdm import tqdm

from qem.fit.model import gaussian_2d_single
from qem.utils.tensors import clone_params, to_numpy, to_tensor

if TYPE_CHECKING:
    from qem.fit.fitter import Fitter  # noqa: F401


def voronoi_integrate(
    image,
    points_x,
    points_y,
    method="Voronoi",
    max_radius="Auto",
    show_progressbar=True,
    remove_edge_cells=False,
    edge_pixels=1,
    pbc=False,
):
    """Given a spectrum image a set of points and a maximum outer radius,
    this function integrates around each point in an image, using either
    Voronoi cell or watershed segmentation methods.

    Parameters
    ----------
    image : 2D, 3D or 4D array-like
        Assuming 2D, 3D or 4D dataset where the spatial dimensions are 2D and
        any remaining dimensions are spectral.
    point_x, point_y : list
        Detailed list of the x and y coordinates of each point of
        interest within the image.
    method : string
        'Voronoi' or 'Watershed'
    max_radius : {'Auto'} int
        A maximum outer radius for each Voronoi Cell.
        If a pixel exceeds this radius it will not be included in the cell.
        This allows analysis of a surface and particles.
        If 'max_radius' is left as 'Auto' then it will be set to the largest
        dimension in the image.
    remove_edge_cells : bool
        Determine whether to replace the cells touching the signal edge with
        np.nan values, which makes automatic contrast estimation easier later
    edge_pixels : int
        Only used if remove_edge_cells is True. Determines the number of
        pixels from the border to remove.
    show_progressbar : bool, optional
        Default True

    Returns
    -------
    integrated_intensity : NumPy array
        An array where dimension 0 is the same length as points, and subsequent
        subsequent dimension are energy dimensions.
    intensity_record : HyperSpy signal, same size as s
        Each pixel/voxel in a particular segment or region has the value of the
        integration, value.
    point_record : 2D numpy array, same size as image
        Image showing where each integration region is, pixels equating to
        point 0 (integrated_intensity[0]) all have value 0, all pixels
        equating to integrated_intensity[1] all have value 1 etc.

    Note
    ----
    Works in principle with 3D and 4D data sets but will quickly hit a
    memory error with large sizes.

    """
    if len(image.shape) < 2:
        raise ValueError("image must have at least 2 dimensions")
    intensity_record = np.zeros_like(image, dtype=float)
    integrated_intensity = np.zeros(image.shape[:-2])
    integrated_intensity = np.stack(
        [integrated_intensity for i in range(len(points_x))]
    )

    points = np.array((points_y, points_x))
    # Setting max_radius to the width of the image, if none is set.
    if method == "Voronoi":
        if max_radius == "Auto":
            max_radius = max(image.shape[-2:])
        elif max_radius <= 0:
            raise ValueError("max_radius must be higher than 0.")
        point_record = voronoi_point_record(image, points, max_radius, pbc=pbc)

    elif method == "Watershed":
        if len(image.shape) > 2:
            raise ValueError(
                "Currently Watershed method is only implemented for 2D data."
            )
        points_map = _make_mask(image.T, points[0], points[1])
        point_record = watershed(-image, points_map.T)

    else:
        raise NotImplementedError("Oops! You have asked for an unimplemented method.")
    point_record -= 1

    def process_point(point_index):
        return point_index, get_integrated_intensity(point_record, image, point_index)

    with ThreadPoolExecutor() as executor:
        futures = [
            executor.submit(process_point, point_index)
            for point_index in range(points.shape[1])
        ]
        for future in tqdm(
            as_completed(futures),
            total=points.shape[1],
            desc="Integrating",
            disable=not show_progressbar,
        ):
            point_index, intensity = future.result()
            integrated_intensity[point_index] = intensity

    for i, j in progressbar(
        np.ndindex(image.shape[-2:]),
        desc="Building intensity map",
        total=np.prod(image.shape[-2:]),
        leave=False,
    ):

        point_index = point_record[i, j]
        if point_index == -1:
            intensity_record[..., i, j] = np.nan
        else:
            summed = integrated_intensity[point_index]
            intensity_record[..., i, j] = summed

    if remove_edge_cells:
        remove_integrated_edge_cells(
            integrated_intensity,
            intensity_record,
            point_record,
            edge_pixels=edge_pixels,
            use_nans=True,
            inplace=True,
        )
    return integrated_intensity, intensity_record, point_record

def voronoi_point_record(image, points, max_radius, pbc=False, box=None):
    """
    Fast Voronoi cell assignment using cKDTree.

    Parameters
    ----------
    image : 2D array or shape tuple
        The image (or its shape) to create the Voronoi map for.
    points : array-like, shape (2, N)
        Coordinates of the points (y, x).
    max_radius : float
        Maximum radius for Voronoi cell assignment.
    pbc : bool, optional
        Whether to use periodic boundary conditions.
    box : tuple or None
        Box size for PBC, e.g., (height, width). Required if pbc=True.

    Returns
    -------
    point_record : 2D numpy array
        Voronoi array where equal values belong to the same Voronoi cell.
    """
    shape = image.shape if hasattr(image, "shape") else image
    points = np.asarray(points)
    if points.shape[0] != 2:
        raise ValueError("points should have shape (2, N)")
    points_xy = np.column_stack((points[0], points[1]))

    # Setup KDTree (with PBC if requested)
    if pbc:
        if box is None:
            box = shape
        tree = cKDTree(points_xy, boxsize=box)
    else:
        tree = cKDTree(points_xy)

    grid_y, grid_x = np.indices(shape)
    grid_points = np.column_stack((grid_y.ravel(), grid_x.ravel()))

    dist, idx = tree.query(grid_points, distance_upper_bound=max_radius)
    idx[dist >= max_radius] = -1  # Mark as outside any cell

    point_record = idx.reshape(shape) + 1  # To match calculate_point_record (0 = background)
    return point_record

def calculate_point_record(image, points, max_radius, pbc=False):
    """
    Creates a Voronoi array where equal values belong to
    the same Voronoi cell
    
    This is slow version from atomap

    Parameters
    ----------
    point_record : 2D zero array of same shape as the image to be mapped
    points: Array like of shape (2,N)
    max_radius: Integer, max radius of each Voronoi Cell

    Returns
    -------
    point_record : Voronoi array where equal values belong to
    the same Voronoi cell
    """
    point_record = np.zeros(image.shape[-2:], dtype=int)
    for i, j in progressbar(
        np.ndindex(point_record.shape),
        desc="Calculating Voronoi",
        total=np.prod(point_record.shape),
        leave=False,
    ):
        min_index, dist_min = find_smallest_distance(
            i, j, points, image_shape=image.shape, pbc=pbc
        )
        if dist_min >= max_radius:
            point_record[i][j] = 0
        else:
            point_record[i][j] = min_index + 1
    return point_record

def get_integrated_intensity(point_record, image, point_index, include_edge_cells=True):
    """
    Using a Voronoi point_record array, integrate a (minimum 2D)
    image array at each pixel

    Parameters
    ----------
    point_record : 2D zero array of same shape as the image to be mapped
    image : The ndarray to integrate the voronoi cells on
    point_index: Array like of shape (2,N)

    Returns
    -------
    integrated_record : Voronoi array where equal values belong to
    the same Voronoi cell
    """
    currentMask = point_record == point_index
    currentFeature = currentMask * image
    integrated_record = np.sum(currentFeature, axis=(-1, -2))
    return integrated_record


@nb.jit()
def find_smallest_distance(i, j, points, image_shape=None, pbc=False):
    """
    Finds the smallest distance between coordinates (i, j)
    and a list of coordinates.

    Parameters
    ----------
    i : Integer
    j : Integer
    points : array like of shape (2,N)

    Returns
    -------
    distMin  : Minimum distance
    minIndex : Index of minimum distance in points

    Example
    -------
    >>> import numpy as np
    >>> points = np.random.random((2, 10000))
    >>> i, j = 0.5, 0.5
    >>> smallest_distance = find_smallest_distance(i, j, points)

    """
    if pbc:
        height, width = image_shape
        distance_log = np.inf * np.ones(points.shape[1])
        for k in range(points.shape[1]):
            dx = min(abs(points[0, k] - i), width - abs(points[0, k] - i))
            dy = min(abs(points[1, k] - j), height - abs(points[1, k] - j))
            distance_log[k] = (dx**2 + dy**2) ** 0.5
    else:
        distance_log = (
            (points[0] - float(i)) ** 2 + (points[1] - float(j)) ** 2
        ) ** 0.5
    minIndex = np.argmin(distance_log)
    distMin = distance_log[minIndex]
    return minIndex, distMin


def remove_integrated_edge_cells(
    i_points, i_record, p_record, edge_pixels=1, use_nans=True, inplace=False
):
    """Removes any cells that touch within a number of pixels of
    the image border.

    Note on using use_nans: If this is used on a dataset with more than
    two dimensions, the resulting HyperSpy i_record signal might be needed to
    be viewed with i_record.plot(navigator='slider'), since HyperSpy may throw
    an error when plotting a dataset with only NaNs present.

    Parameters
    ----------
    i_points : NumPy array
        The output of the Atomap integrate function or method
    i_record : HyperSpy signal
        The output of the Atomap integrate function or method
    p_record : HyperSpy signal
        The output of the Atomap integrate function or method

    Returns
    -------
    i_points : NumPy array
        Modified list of integrated intensities with either np.nan or 0
        on the removed values, which preserves the atom index.
    i_record : HyperSpy signal
        Modified integrated intensity record, with either np.nan or 0
        on the removed values, which preserves the atom index
    p_record : HyperSpy signal, same size as image
        Modified points record, where removed areas have value = -1.

    Example
    -------

    >>> s = am.dummy_data.get_fantasite()
    >>> points_x, points_y = am.get_atom_positions(s).T
    >>> i, ir, pr = am.integrate(
    ...    s,
    ...    points_x,
    ...    points_y,
    ...    method='Voronoi',
    ...    remove_edge_cells=False)
    >>> from atomap.tools import remove_integrated_edge_cells
    >>> i2, ir2, pr2 = remove_integrated_edge_cells(
    ...    i, ir, pr, edge_pixels=5, use_nans=True)

    """
    if not inplace:
        i_points = i_points.copy()
        i_record = i_record.deepcopy()
        p_record = p_record.deepcopy()

    border = _border_elems(p_record.data, edge_pixels)
    border_indices = np.array(list(set(border)))
    indices = np.in1d(p_record.data, border_indices)
    indices = indices.reshape(p_record.data.shape)
    i_points[border_indices] = np.nan if use_nans else 0
    i_record.data[..., indices] = np.nan if use_nans else 0
    p_record.data[indices] = -1

    if not inplace:
        return i_points, i_record, p_record


def _make_mask(image, points_x, points_y):
    """
    Create points_map for the watershed integration
    function
    """
    mask = np.zeros(image.shape[-2:])
    indices = np.round(np.array([points_y, points_x])).astype(int)
    values = np.arange(len(points_x))
    mask[tuple(indices)] = values
    return mask


def _border_elems(image, pixels=1):
    """
    Return the values of the edges along the border of the image, with
    border width `pixels`.

    Example
    -------
    >>> import numpy as np
    >>> a = np.array([
    ...     [1,1,1],
    ...     [2,5,3],
    ...     [4,4,4]])
    >>> b = _border_elems(a, pixels=1)

    """
    arr = np.ones_like(image, dtype=bool)
    arr[pixels: -1 - (pixels - 1), pixels: -1 - (pixels - 1)] = False
    return image[arr]


def fit_voronoi(
    self,
    params: dict = None,  # initial params, optional
    max_radius: int = None,  # optional, for Voronoi cell size
    tol: float = 1e-3,
    border: int = 0,  # optional, exclude border pixels
):
    """
    Fit a Gaussian model to each Voronoi cell defined by the current coordinates.
    Each cell is fit independently and in parallel.
    The local minimum is subtracted from each cell before fitting.
    """
    if params is None:
        if self.params is not None:
            if "pos_x" in self.params and "pos_y" in self.params:
                params = self.params
            else:
                params = self.init_params()
        else:
            params = self.init_params()

    pos_x = params["pos_x"]
    pos_y = params["pos_y"]
    coords = torch.stack([pos_y, pos_x])
    num_coordinates = coords.shape[1]

    # Generate Voronoi cell map
    if max_radius is None:
        max_radius = params["width"].max() * 3

    image = to_numpy(self.image)
    max_radius = to_numpy(max_radius)
    coords = to_numpy(coords)

    point_record = voronoi_point_record(image, coords, max_radius)

    # Prepare per-cell fitting function
    def fit_cell(index, params):
        mask = point_record == index + 1
        if not np.any(mask):
            return None  # No pixels in this cell

        cell_img = image * mask
        # Crop to bounding box for efficiency
        ys, xs = np.where(mask)
        y0, y1 = ys.min(), ys.max() + 1
        x0, x1 = xs.min(), xs.max() + 1
        cropped_img = cell_img[y0:y1, x0:x1]
        cropped_mask = mask[y0:y1, x0:x1]

        # Subtract local min (only over masked region)
        local_min = cropped_img[cropped_mask].min()
        cropped_img = cropped_img - local_min
        cropped_img[~cropped_mask] = 0

        # Prepare grid for fitting
        x_c, y_c = torch.meshgrid(
            torch.arange(x0, x1), torch.arange(y0, y1), indexing="xy"
        )
        x_c = to_numpy(x_c)
        y_c = to_numpy(y_c)

        # Prepare initial params for this cell
        local_param = {}
        local_param["pos_x"] = [params["pos_x"][index]]
        local_param["pos_y"] = [params["pos_y"][index]]
        local_param["height"] = (
            params["height"][index] + params["background"] - local_min
        )
        local_param["width"] = params["width"]
        local_param["background"] = [0.0]
        self.fit_background = False

        atoms_selected = np.zeros(self.num_coordinates, dtype=bool)
        atoms_selected[index] = True

        p0 = [
            local_param["pos_x"][0],
            local_param["pos_y"][0],
            local_param["height"],
            local_param["width"][self.atom_types[index]],
            local_param["background"][0],
        ]
        if border > 0 and (
            pos_x.min() < border
            or pos_x.max() > self.nx - border
            or pos_y.min() < border
            or pos_y.max() > self.ny - border
        ):
            popt = p0
        else:
            try:
                popt, _ = curve_fit(  # pylint: disable=unbalanced-tuple-unpacking
                    gaussian_2d_single,
                    (x_c, y_c),
                    cropped_img.ravel(),
                    p0=p0,
                    maxfev=2000,
                )
            except Exception as _:
                popt = p0  # fallback if fit fails

        # if popt[0] < 0 or popt[1] < 0:
        #     popt = p0
        # if popt[0] > self.image.shape[0] or popt[1] > self.image.shape[1]:
        #     popt = p0

        optimized_param = {
            "pos_x": popt[0],
            "pos_y": popt[1],
            "height": popt[2],
            "width": popt[3],
            "background": popt[4],
        }
        return optimized_param, index

    converged = False
    pre_params = clone_params(self.params)
    current_params = clone_params(self.params)


    operation_context = (
        self.memory_monitor.monitor_operation("fit_voronoi") 
        if self.memory_monitor else nullcontext()
    )
    
    with operation_context:
        while not converged:
            with ThreadPoolExecutor() as executor:
                futures = [
                    executor.submit(fit_cell, i, current_params)
                    for i in range(num_coordinates)
                ]
                # Collect all updates first
                pos_x_updates = {}
                pos_y_updates = {}

                for future in tqdm(
                    as_completed(futures), total=num_coordinates, desc="Fitting cells"
                ):
                    result = future.result()
                    if result is None:
                        continue
                    optimized_param, index = result
                    pos_x_updates[index] = optimized_param["pos_x"]
                    pos_y_updates[index] = optimized_param["pos_y"]

                # Apply updates by creating new tensors (avoid in-place operations)
                if pos_x_updates:
                    pos_x_array = to_numpy(current_params["pos_x"]).copy()
                    pos_y_array = to_numpy(current_params["pos_y"]).copy()

                    for index, value in pos_x_updates.items():
                        pos_x_array[index] = value
                    for index, value in pos_y_updates.items():
                        pos_y_array[index] = value

                    current_params["pos_x"] = to_tensor(
                        pos_x_array, dtype=torch.float32
                    )
                    current_params["pos_y"] = to_tensor(
                        pos_y_array, dtype=torch.float32
                    )
            converged = self.convergence(current_params, pre_params, tol)
            pre_params = clone_params(current_params)
    self.params = current_params
    # self.model = self.predict(self.params, self.x_grid, self.y_grid)
    return self.params

def voronoi_integration(self, max_radius: float = None, plot=False,save=False):
    """
    Compute the Voronoi integration of the atomic columns.

    Returns:
        np.array: The Voronoi integration of the atomic columns.
    """
    if self.params is None:
        raise ValueError("Please initialize the parameters first.")
    if self.fit_background:
        image = (self.image - to_numpy(self.params["background"]))
    else:
        image = (self.image - self.init_background)
    pos_x = self.params["pos_x"]
    pos_y = self.params["pos_y"]
    pos_x = to_numpy(pos_x)
    pos_y = to_numpy(pos_y)
    if max_radius is None:
        max_radius = self.params["width"].max() * 5
        max_radius = to_numpy(max_radius)
    integrated_intensity, intensity_record, point_record = voronoi_integrate(
        image, pos_x, pos_y, max_radius=max_radius, pbc=self.pbc
    )
    integrated_intensity = integrated_intensity * self.dx**2
    intensity_record = intensity_record * self.dx**2
    self._voronoi_volume = integrated_intensity
    self._voronoi_map = intensity_record
    self._voronoi_cell = point_record
    if plot:
        plt.imshow(intensity_record, cmap="viridis")
        plt.colorbar(label="Voronoi Integrated Intensity")
    if save:
        plt.savefig("Voronoi Integrated Intensity.png", dpi=300)
        plt.savefig("Voronoi Integrated Intensity.svg")

    return integrated_intensity, intensity_record, point_record

# parameters updates and convergence



def _bind(cls) -> None:
    """Attach extracted methods back onto Fitter at class-load time."""
    cls.fit_voronoi = fit_voronoi
    cls.voronoi_integration = voronoi_integration


__all__ = [
    "fit_voronoi",
    "voronoi_integration",
    "_bind",
]

