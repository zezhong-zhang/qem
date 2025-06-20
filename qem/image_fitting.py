import copy
import logging
import warnings

import jax
import matplotlib.pyplot as plt
import numpy as np
import optax
from ase import Atoms
from ase.visualize import view
from hyperspy._signals.signal2d import Signal2D
from jax import numpy as jnp
from jax import value_and_grad
from jax.example_libraries import optimizers
from jax.scipy.optimize import minimize
from jaxlib.xla_extension import XlaRuntimeError
from jaxopt import OptaxSolver
from matplotlib.path import Path
from matplotlib_scalebar.scalebar import ScaleBar
from scipy.ndimage import gaussian_filter
from scipy.optimize import lsq_linear
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import spsolve
from skimage.feature.peak import peak_local_max
from tqdm import tqdm

from qem.crystal_analyzer import CrystalAnalyzer
from qem.gui_classes import GetAtomSelection, GetRegionSelection, InteractivePlot
from qem.model import (
    add_peak_at_positions,
    butterworth_window,
    gaussian_2d_numba,
    gaussian_2d_single,
    gaussian_sum_parallel,
    lorentzian_2d_numba,
    lorentzian_sum_parallel,
    mask_grads,
    voigt_2d_numba,
    voigt_sum_parallel,
    get_static_window_size,
    create_gaussian_sum_local
)
from qem.refine import calculate_center_of_mass
from qem.utils import get_random_indices_in_batches, remove_close_coordinates
from qem.voronoi import voronoi_integrate, calculate_point_record, fast_voronoi_point_record
from qem.region import Region, Regions

from scipy.optimize import curve_fit
from concurrent.futures import ThreadPoolExecutor, as_completed

logging.basicConfig(level=logging.INFO)


class ImageModelFitting:
    def __init__(
        self,
        image: np.ndarray,
        dx: float = 1.0,
        units: str = "A",
        elements: list[str] = None,  # type: ignore
        model_type: str = "gaussian",
        same_width: bool = True,
        pbc: bool = False,
        fit_background: bool = True,
        gpu_memory_limit: bool = True,
    ):
        """
        Initialize the Fitting class.

        Args:
            image (np.array): The input image as a numpy array.
            dx (float, optional): The size of each pixel. Defaults to 1.
            units (str, optional): The units of the image. Defaults to "A".
            elements (list[str], optional): The elements in the image. Defaults to None. If None, the elements are ["A", "B", "C"].
            model_type (str, optional): Type of model to use. Defaults to "gaussian".
            same_width (bool, optional): Whether to use same width for all peaks. Defaults to True.
            pbc (bool, optional): Whether to use periodic boundary conditions. Defaults to False.
            fit_background (bool, optional): Whether to fit background. Defaults to True.
            gpu_memory_limit (bool, optional): Whether to use memory-efficient GPU computation. Defaults to True.
        """
        if elements is None:
            elements = ["A", "B", "C"]

        if len(image.shape) == 2:
            self.ny, self.nx = image.shape

        self.device = "cuda"
        self.image = image.astype(np.float32)
        self.model = np.zeros(image.shape)
        self.regions = Regions(image)

        units_dict = {"A": 1, "nm": 10, "pm": 0.01, "um": 1e4}
        assert units in units_dict.keys(), "The units should be in A, nm, pm or um."
        scale_factor = units_dict[units]

        self.dx = dx * scale_factor
        self.units = "A"
        self._atom_types = np.array([])
        logging.info(
            f"Elements: {elements}, the order is used for the atom types. Please initiate the correct elements in your system."
        )
        self.elements = elements
        self.atoms_selected = np.array([])
        self._coordinates = np.array([])
        self.coordinates_history = dict()
        self.coordinates_state = 0
        
        self.local_shape = self.image.shape # the shape of the image

        # Initialize model parameters
        self.model_type = model_type
        self.same_width = same_width
        self.pbc = pbc
        self.fit_background = fit_background
        self.init_background = 0.0
        self.gpu_memory_limit = gpu_memory_limit
        
        # Initialize coordinate system
        x = np.arange(self.nx)
        y = np.arange(self.ny)
        self.X, self.Y = np.meshgrid(x, y, indexing="xy")
        self.converged = False
        self.params = dict()
        
        # Create JIT-compiled function for this image size if using GPU memory limit
        if gpu_memory_limit:
            self._gaussian_sum_local = create_gaussian_sum_local(self.ny, self.nx)

    # init peaks and parameters
    def guess_radius(self):
        """
        Estimate the density of atomic columns in an image.

        Parameters:
        id (int): Identifier for a specific image or set of coordinates.

        Returns:
        tuple: density, influence_map, background_region
        """
        num_coordinates = self.coordinates.shape[0]
        if num_coordinates == 0:
            raise ValueError("No coordinates found for the given id.")

        rate, rate_max, n_filled, n = 1, 1, 0, 0
        nx, ny = self.image.shape

        while rate > 0.5 * rate_max:
            influence_map = np.zeros((nx, ny))
            for i in range(num_coordinates):
                i_l = np.maximum(self.coordinates[i, 0] - n, 0).astype(np.int64)
                i_r = np.minimum(self.coordinates[i, 0] + n, self.nx).astype(np.int64)
                i_u = np.maximum(self.coordinates[i, 1] - n, 0).astype(np.int64)
                i_d = np.minimum(self.coordinates[i, 1] + n, self.ny).astype(np.int64)
                influence_map[i_l: i_r + 1, i_u: i_d + 1] = 1
            if n == 0:
                rate = (np.sum(influence_map) - n_filled) / num_coordinates
            else:
                rate = (np.sum(influence_map) - n_filled) / (8 * n) / num_coordinates
            n_filled = np.sum(influence_map)
            rate_max = max(rate_max, rate)
            n += 1

        # Scaled factors
        n1 = int(np.round((n - 1) * 10))
        n2 = int(np.round((n - 1) * 1))

        influence_map = np.zeros((nx, ny))
        direct_influence_map = np.zeros((nx, ny))

        for i in range(num_coordinates):
            # Calculate the indices for the larger area (influence_map)
            i_l = np.maximum(self.coordinates[i, 0] - n1, 0).astype(np.int64)
            i_r = np.minimum(self.coordinates[i, 0] + n1, nx).astype(np.int64)
            i_u = np.maximum(self.coordinates[i, 1] - n1, 0).astype(np.int64)
            i_d = np.minimum(self.coordinates[i, 1] + n1, ny).astype(np.int64)
            influence_map[i_l: i_r + 1, i_u: i_d + 1] = 1

            # Calculate the indices for the smaller area (direct_influence_map)
            i_l = np.maximum(self.coordinates[i, 0] - n2, 0).astype(np.int64)
            i_r = np.minimum(self.coordinates[i, 0] + n2, nx).astype(np.int64)
            i_u = np.maximum(self.coordinates[i, 1] - n2, 0).astype(np.int64)
            i_d = np.minimum(self.coordinates[i, 1] + n2, ny).astype(np.int64)
            direct_influence_map[i_l: i_r + 1, i_u: i_d + 1] = 1

        radius = (np.sum(direct_influence_map) / num_coordinates) ** (1 / 2) / np.pi

        background_region = influence_map - direct_influence_map
        return radius, direct_influence_map, background_region

    def init_params(
        self,
        atom_size: float = 0.7,
        guess_radius: bool = False,
        init_background: float = 0.0,
    ):
        self.coordinates_history = dict()
        self.coordinates_state = 0
        if guess_radius:
            width = self.guess_radius()[0]
        else:
            width = atom_size / self.dx
        if self.pbc:
            mask = (self.coordinates[:, 0] < self.nx - 1) & (
                self.coordinates[:, 1] < self.ny - 1
            )
            self.coordinates = self.coordinates[mask]
            if len(self.atom_types) != self.num_coordinates:
                self.atom_types = self.atom_types[mask]

        pos_x = copy.deepcopy(self.coordinates[:, 0]).astype(float)
        pos_y = copy.deepcopy(self.coordinates[:, 1]).astype(float)
        if self.fit_background:
            init_background = self.image.min()
        else:
            self.init_background = init_background
        height = (
            self.image[pos_y.astype(int), pos_x.astype(int)].ravel() - init_background
        )
        # get the lowest 20% of the intensity as the background
        if self.same_width:
            width = np.tile(width, self.num_atom_types).astype(float)
            ratio = np.tile(0.9, self.num_atom_types).astype(float)
        else:
            width = np.tile(width, self.num_coordinates).astype(float)
            ratio = np.tile(0.9, self.num_coordinates).astype(float)
        if self.model_type == "gaussian":
            # Initialize the parameters
            params = {
                "pos_x": pos_x,  # x position
                "pos_y": pos_y,  # y position
                "height": height,  # height
                "sigma": width,  # width
            }
        elif self.model_type == "voigt":
            # Initialize the parameters
            params = {
                "pos_x": pos_x,  # x position
                "pos_y": pos_y,  # y position
                "height": height,  # height
                "sigma": width,  # width
                "gamma": width / np.sqrt(2 * np.log(2)),  # width
                "ratio": ratio,  # ratio
            }
        elif self.model_type == "lorentzian":
            params = {
                "pos_x": pos_x,  # x position
                "pos_y": pos_y,  # y position
                "height": height,  # height
                "gamma": width / np.sqrt(2 * np.log(2)),  # width
            }
        else:
            raise ValueError(f"Invalid model type: {self.model_type}")
        if self.fit_background:
            params["background"] = init_background

        self.params = params
        return params

    # find atomic columns
    def import_coordinates(self, coordinates: np.ndarray):
        self.coordinates = coordinates

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
        Find the peaks in the image.

        Args:
            atom_size (float, optional): The size of the atomic columns. Defaults to 1.
            threshold_rel (float, optional): The relative threshold. Defaults to 0.2.
            exclude_border (bool, optional): Whether to exclude the border. Defaults to False.
            image (np.array, optional): The input image. Defaults to None.

        Returns:
            np.array: The coordinates of the peaks.
        """
        assert (
            region_index in self.regions.keys
        ), "The region index is not in the regions."
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
        if plot:
            self.add_or_remove_peaks(min_distance=min_distance, image=self.image)
        return self.coordinates

    def total_lattice(self, region_index: int = None):
        return self.regions.lattice(region_index)

    def view_3d(self, region_index: int = None):
        self.regions.view_3d(region_index)

    def map_lattice(
        self,
        cif_file: str,
        elements: list[str] = None,
        reciprocal: bool = False,
        region_index: int = 0,
        sigma: float = 0.8,
    ):
        """
        Map the atomic columns in the CIF file to the peaks found in the image.

        Args:
            cif_file (str): The path to the CIF file.
            elements (list[str]): The elements in the CIF file.
            unit_cell (Atoms, optional): The unit cell of the crystal. Defaults to None.
            reciprocal (bool, optional): Whether to use reciprocal space. Defaults to False.
            region_index (int, optional): The index of the region. Defaults to 0.
            sigma (float, optional): The sigma of the Gaussian filter. Defaults to 0.8.

        Returns:
            AtomicColumns: The atomic columns mapped from the CIF file.
        """
        column_mask = self.region_column_labels == region_index
        region_mask = self.regions.region_map == region_index

        if elements is None:
            elements = self.elements

        crystal_analyzer = CrystalAnalyzer(
            image=self.image,
            dx=self.dx,
            peak_positions=self.coordinates[column_mask],
            atom_types=self.atom_types[column_mask],
            elements=elements,
            units=self.units,
            region_mask=region_mask,
        )
        # if unit_cell is not None:
        #     crystal_analyzer.unit_cell = unit_cell
        if cif_file is not None:
            crystal_analyzer.read_cif(cif_file)
        atomic_column_list = crystal_analyzer.get_atomic_columns(reciprocal=reciprocal, sigma=sigma)
        # remove the self.coordinates in the column mask and append the new coordinates find in the atomic_column_list
        coordinates = np.delete(self.coordinates, np.where(column_mask), axis=0)
        coordinates = np.vstack([coordinates, atomic_column_list.positions_pixel])
        self.coordinates = coordinates
        atom_types = np.delete(self.atom_types, np.where(column_mask), axis=0)
        atom_types = np.append(atom_types, atomic_column_list.atom_types)
        self.atom_types = atom_types
        crystal_analyzer.plot_unitcell()
        self.regions[region_index].analyzer = crystal_analyzer
        self.regions[region_index].columns = atomic_column_list
        return atomic_column_list


    def select_atoms(self, invert_selection: bool = False):
        atom_select = GetAtomSelection(
            image=self.image,
            atom_positions=self.coordinates,
            invert_selection=invert_selection,
        )
        while plt.fignum_exists(atom_select.fig.number):  # type: ignore
            plt.pause(0.1)
        peak_positions_selected = np.array(atom_select.atom_positions_selected)
        selection_mask = atom_select.selection_mask

        if peak_positions_selected.shape[0] == 0:
            logging.info("No atoms selected.")
            return None
        else:
            logging.info(
                f"Selected {peak_positions_selected.shape[0]} atoms out of {self.num_coordinates} atoms."
            )

            self.atom_types = self.atom_types[selection_mask]
            self.coordinates = peak_positions_selected
        return selection_mask

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
            norm_img = (cropped_img - cropped_img[cropped_mask].min()) / (cropped_img[cropped_mask].max() - cropped_img[cropped_mask].min())
        else:
            norm_img = cropped_img
        norm_img[~cropped_mask] = 0

        # Compute center of mass in the cropped region
        local_y, local_x = calculate_center_of_mass(norm_img)
        assert isinstance(local_x, float), "local_x is not a float"
        assert isinstance(local_y, float), "local_y is not a float"
        result = np.array([
            x0 + local_x,
            y0 + local_y,
        ], dtype=float)

        if plot:
            plt.clf()
            plt.imshow(norm_img, cmap="gray")
            plt.scatter(local_x, local_y, color="red", s=2, label="refined")
            plt.legend()
            plt.pause(1.0)
        return result, i

    def refine_center_of_mass(self, params = None, plot=False):
        # Refine center of mass for each Voronoi cell
        pre_coordinates = self.coordinates.copy()
        current_coordinates = self.coordinates.copy()
        converged = False

        if params is None and hasattr(self, 'params') and self.params is not None:
            params = self.params
        elif params is None:
            params = self.init_params()
        while not converged:
            # Generate Voronoi cell map
            coords = np.stack([pre_coordinates[:, 1], pre_coordinates[:, 0]])  # (y, x)
            max_radius = params["sigma"].max() * 5
            point_record = fast_voronoi_point_record(self.image, coords, max_radius)

           

            # In refine_center_of_mass, replace the for-loop with:
            with ThreadPoolExecutor() as executor:
                futures = [
                    executor.submit(self._refine_one_center, i, point_record, plot)
                    for i in range(self.num_coordinates)
                ]
                for future in tqdm(as_completed(futures), total=self.num_coordinates, desc="Refining center of mass"):
                    result, i = future.result()
                    if result is not None:
                        current_coordinates[i] = result

            converged = np.abs(current_coordinates - pre_coordinates).mean() < 1
            pre_coordinates = current_coordinates.copy()
        params["pos_x"] = current_coordinates[:, 0]
        params["pos_y"] = current_coordinates[:, 1]
        self.params = params
        self.coordinates = current_coordinates
        return params

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
        if self.pbc:
            coords, _ = remove_close_coordinates(self.coordinates.copy(), threshold)
            # find the coords near the boundary
            mask_boundary = (
                (coords[:, 0] < threshold)
                | (coords[:, 0] > self.nx - threshold)
                | (coords[:, 1] < threshold)
                | (coords[:, 1] > self.ny - threshold)
            )
            # genearate the boundary coords under the pbc
            coords_boundary = coords[mask_boundary]
            # identify the coords in the coords_boundary that are close to the coords_boundary_pbc
            coords_boundary_pbc = coords_boundary.copy()
            for i, j in [(1, 0), (0, 1), (1, 1), (-1, 0), (0, -1), (-1, -1), (1, -1), (-1, 1)]:
                coords_boundary_shifted = coords_boundary + np.array(
                    [i * self.nx, j * self.ny]
                )
                for row in coords_boundary:
                    too_close = (
                        np.linalg.norm(coords_boundary_shifted - row, axis=1)
                        < threshold
                    ).any()
                    # same_type = coords_boundary_shifted[too_close,2] == row[2]
                    if too_close:
                        # find the index of the row in the coords_boundary_pbc
                        idx = np.where((coords_boundary_pbc == row).all(axis=1))[0]
                        # dump the row if it is too close to the boundary
                        coords_boundary_pbc = np.delete(
                            coords_boundary_pbc, idx, axis=0
                        )
            # now combine the coords that are not close to the boundary with the coords_boundary_pbc
            coords_final = np.vstack([coords[~mask_boundary], coords_boundary_pbc])
            self.coordinates = coords_final
        else:
            self.coordinates, _ = remove_close_coordinates(self.coordinates, threshold)
        return self.coordinates

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

    def loss(self, params: dict, image: np.ndarray, X: np.ndarray, Y: np.ndarray):
        # Compute the sum of the Gaussians
        prediction = self.predict(params, X, Y)
        diff = image - prediction
        if not self.skip_window:
            diff = diff * self.window
        # dammping the difference near the edge
        mse = jnp.sqrt(jnp.mean(diff**2))
        L1 = jnp.mean(jnp.abs(diff))
        return mse + L1

    def residual(self, params: dict, image: np.ndarray, X: np.ndarray, Y: np.ndarray):
        # Compute the sum of the Gaussians
        prediction = self.predict(params, X, Y)
        diff = prediction - image
        return diff

    def apply_pbc(self, prediction, prediction_func, params, X, Y):
        """
        Apply periodic boundary conditions to the prediction.

        Parameters:
        -----------
        prediction_func : function
            The prediction function to use.
        params : dict
            Dictionary containing the parameters for the prediction.
        X : np.ndarray
            The x-coordinates for the prediction.
        Y : np.ndarray
            The y-coordinates for the prediction.
        nx : int
            Periodic boundary condition parameter in the x-direction.
        ny : int
            Periodic boundary condition parameter in the y-direction.
        pbc : bool
            Flag indicating whether to apply periodic boundary conditions.

        Returns:
        --------
        np.ndarray
            The prediction with periodic boundary conditions applied.
        """
        for i, j in [
            (1, 0),
            (0, 1),
            (-1, 0),
            (0, -1),
            (1, 1),
            (-1, -1),
            (1, -1),
            (-1, 1),
        ]:
            prediction += prediction_func(
                X,
                Y,
                params["pos_x"] + i * self.nx,
                params["pos_y"] + j * self.ny,
                params["height"],
                params.get("sigma"),
                params.get("gamma"),
                params.get("ratio"),
                0,
            )
        return prediction

    def predict(self, params: dict = None, X: np.ndarray = None, Y: np.ndarray = None):
        if params is None:
            params = self.params
        if X is None or Y is None:
            X = self.X
            Y = self.Y
        background = params.get("background", self.init_background)
        pos_x = jnp.asarray(params["pos_x"])
        pos_y = jnp.asarray(params["pos_y"])
        height = jnp.asarray(params["height"])
        sigma = params.get("sigma")
        gamma = params.get("gamma")
        ratio = params.get("ratio")
        
        if pos_x.size < self.num_coordinates:
            mask = self.atoms_selected
        else:
            mask = np.ones(self.num_coordinates, dtype=bool)

        if self.same_width:
            if self.model_type in {"gaussian", "voigt"} and sigma is not None:
                sigma = jnp.asarray(sigma[self.atom_types[mask]])
            if self.model_type in {"voigt", "lorentzian"} and gamma is not None:
                gamma = jnp.asarray(gamma[self.atom_types[mask]])
            if self.model_type == "voigt" and ratio is not None:
                ratio = jnp.asarray(ratio[self.atom_types[mask]])

        if self.model_type == "gaussian":
            if self.gpu_memory_limit:
                # Calculate static window size based on maximum sigma
                window_size = get_static_window_size(jnp.max(sigma))
                prediction = self._gaussian_sum_local(
                    pos_x,
                    pos_y,
                    height,
                    sigma,
                    background,
                    window_size
                )
            else:
                prediction = gaussian_sum_parallel(
                    X, Y, pos_x, pos_y, height, sigma, background
                )
        elif self.model_type == "voigt":
            prediction = voigt_sum_parallel(
                X, Y, pos_x, pos_y, height, sigma, gamma, ratio, background
            )
        elif self.model_type == "lorentzian":
            prediction = lorentzian_sum_parallel(
                X, Y, pos_x, pos_y, height, gamma, background
            )
        else:
            raise ValueError("The model type is not valid.")

        if self.pbc:
            if self.gpu_memory_limit:
                # Reuse the same window size for periodic images
                window_size = get_static_window_size(jnp.max(sigma))
                for i, j in [(1, 0), (0, 1), (-1, 0), (0, -1), (1, 1), (-1, -1), (1, -1), (-1, 1)]:
                    prediction += self._gaussian_sum_local(
                        pos_x + i * self.nx,
                        pos_y + j * self.ny,
                        height,
                        sigma,
                        0,  # No background for periodic images
                        window_size
                    )
            else:
                prediction = self.apply_pbc(prediction, gaussian_sum_parallel, params, X, Y)

        return prediction

    # fitting

    def linear_estimator(self, params: dict = None, non_negative=False):
        if params is None:
            if self.params is None:
                self.init_params()
            params = self.params
        # create the design matrix as array of gaussian peaks + background
        pos_x = params["pos_x"]
        pos_y = params["pos_y"]
        if self.model_type in {"gaussian", "voigt"}:
            sigma = params["sigma"]
            if self.same_width:
                sigma = sigma[self.atom_types]
            width = sigma.mean()
        elif self.model_type in {"voigt", "lorentzian"}:
            gamma = params["gamma"]
            if self.same_width:
                gamma = gamma[self.atom_types]
            width = gamma.mean()
        if self.model_type == "voigt":
            ratio = params["ratio"]
            if self.same_width:
                ratio = ratio[self.atom_types]
        else:
            raise ValueError("The model type is not valid.")
        height = params["height"]
        if (height < 0).any():
            logging.warning(
                "The height has negative values, the linear estimator is not valid, I will make it to zero but be careful with the results."
            )
            height[height < 0] = 0

        rows = []
        cols = []
        data = []
        window_size = int(width * 5)
        x = np.arange(-window_size, window_size + 1, 1)
        y = np.arange(-window_size, window_size + 1, 1)
        local_X, local_Y = np.meshgrid(x, y, indexing="xy")
        if self.model_type == "gaussian":
            peak_local = gaussian_2d_numba(
                local_X, local_Y, pos_x % 1, pos_y % 1, height, sigma
            )
        elif self.model_type == "lorentzian":
            peak_local = lorentzian_2d_numba(
                local_X, local_Y, pos_x % 1, pos_y % 1, height, gamma
            )
        elif self.model_type == "voigt":
            peak_local = voigt_2d_numba(
                local_X, local_Y, pos_x % 1, pos_y % 1, height, sigma, gamma, ratio
            )
        else:
            raise ValueError("The model type is not valid.")

        for atomic_column_index in range(self.num_coordinates):
            global_X = local_X + pos_x[atomic_column_index].astype(int)
            global_Y = local_Y + pos_y[atomic_column_index].astype(int)
            mask = (
                (global_X >= 0)
                & (global_X < self.nx)
                & (global_Y >= 0)
                & (global_Y < self.ny)
            )
            flat_index = global_Y[mask].flatten() * self.nx + global_X[mask].flatten()
            rows.extend(flat_index)
            cols.extend(np.tile(atomic_column_index, flat_index.shape[0]))
            data.extend(peak_local[:, :, atomic_column_index][mask].ravel())
        if self.fit_background:
            rows.extend(self.Y.flatten() * self.nx + self.X.flatten())
            cols.extend(np.tile(self.num_coordinates, self.nx * self.ny))
            data.extend(np.ones(self.nx * self.ny))
            design_matrix = coo_matrix(
                (data, (rows, cols)),
                shape=(self.nx * self.ny, self.num_coordinates + 1),
            )
        else:
            design_matrix = coo_matrix(
                (data, (rows, cols)), shape=(self.nx * self.ny, self.num_coordinates)
            )
        # create the target as the image
        b = self.image.ravel()
        if not self.fit_background:
            b = b - self.init_background
        # solve the linear equation
        try:
            # Attempt to solve the linear system
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                if non_negative:
                    design_matrix_csr = design_matrix.tocsr()
                    result = lsq_linear(design_matrix_csr, b, bounds=(0, np.inf))
                    solution = result.x
                else:
                    solution = spsolve(
                        design_matrix.T @ design_matrix, design_matrix.T @ b
                    )
                    # Check if any of the caught warnings are related to a singular matrix
                    if w and any(
                        "singular matrix" in str(warning.message) for warning in w
                    ):
                        logging.warning(
                            "Warning: Singular matrix encountered. Please refine the peak positions better before linear estimation. The parameters are not updated."
                        )
                        return params

        except np.linalg.LinAlgError as e:
            if "Singular matrix" in str(e):
                logging.warning("Error: Singular matrix encountered.")
            else:
                raise

        # solution = cg(design_matrix.T @ design_matrix, design_matrix.T @ b)[0]
        # update the background and height

        if self.fit_background:
            params["background"] = (
                solution[-1] if solution[-1] > 0 else self.init_background
            )
            height_scale = solution[:-1]
        else:
            height_scale = solution
        if np.NaN in height_scale:
            logging.warning(
                "The height_scale has NaN, the linear estimator is not valid, parameters are not updated"
            )
            return params
        if (height_scale > 2).any():
            logging.warning(
                "The height_scale has values larger than 2, the linear estimator is probably not accurate. I will limit it to 2 but be careful with the results."
            )
            height_scale[height_scale > 2] = 2
        if (height_scale < 0.5).any():
            logging.warning(
                "The height_scale has values smaller than 0.5, the linear estimator is probably not accurate. I will limit it to 0.5 but be careful with the results."
            )
            height_scale[height_scale < 0.5] = 0.5
        params["height"] = height_scale * params["height"]
        mask_negative_height = params["height"] < 0
        if mask_negative_height.any():
            logging.warning(
                "The height has negative values, the linear estimator is not valid. I will make it to zero but be careful with the results."
            )
        params["height"][mask_negative_height] = 0
        self.params = params
        return params

    def optimize(
        self,
        image: np.ndarray,
        params: dict,
        X: np.ndarray,
        Y: np.ndarray,
        maxiter: int = 1000,
        tol: float = 1e-4,
        step_size: float = 0.01,
        verbose: bool = False,
    ):
        opt = optax.adam(learning_rate=step_size)
        solver = OptaxSolver(
            opt=opt, fun=self.loss, maxiter=maxiter, tol=tol, verbose=verbose
        )
        res = solver.run(params, image=image, X=X, Y=Y)
        params = res[0]
        return params

    def gradient_descent(
        self,
        image: np.ndarray,
        params: dict,
        X: np.ndarray,
        Y: np.ndarray,
        keys_to_mask: list[str],
        step_size: float = 0.001,
        maxiter: int = 10000,
        tol: float = 1e-4,
    ):
        opt_init, opt_update, get_params = optimizers.adam(
            step_size=step_size, b1=0.9, b2=0.999
        )
        opt_state = opt_init(params)

        def step(step_index, opt_state, params, image, X, Y, keys_to_mask=None):
            if keys_to_mask is None:
                keys_to_mask = []
            loss, grads = value_and_grad(self.loss)(params, image, X, Y)
            masked_grads = mask_grads(grads, keys_to_mask)
            opt_state = opt_update(step_index, masked_grads, opt_state)
            return loss, opt_state

        # Initialize the loss
        loss = np.inf
        loss_list = []
        # Loop over the number of iterations
        for i in range(maxiter):
            # Update the parameters
            new_params = get_params(opt_state)
            loss_new, opt_state = step(
                i, opt_state, new_params, image, X, Y, keys_to_mask
            )

            # Check if the loss has converged
            if np.abs(loss - loss_new) < tol * loss:
                break
            # Update the loss
            loss = loss_new
            loss_list.append(loss)

            # Print the loss every 10 iterations
            if i % 10 == 0:
                logging.info(f"Iteration {i}: loss = {loss:.6f}")
        plt.plot(loss_list, "o-")
        plt.xlabel("Iteration")
        plt.ylabel("Loss")
        # Update the model
        self.params = new_params
        return new_params

    def minimize(
        self,
        params: dict = None,  # type: ignore
        tol: float = 1e-4,
    ):
        if params is None:
            params = self.params if self.params is not None else self.init_params()

        def objective_fn(params_array, param_shapes, param_keys, image, X, Y):
            # Convert 1D array back to dictionary of parameters
            params_dict = {}
            start = 0
            for key, shape in zip(param_keys, param_shapes):
                size = np.prod(shape)
                end = start + size
                params_dict[key] = params_array[start:end].reshape(shape)
                start = end
            return self.loss(params_dict, image=image, X=X, Y=Y)

        # Flatten params into a 1D array and get shapes
        param_keys = sorted(params.keys())
        param_shapes = [params[key].shape for key in param_keys]
        params_flat = np.concatenate([params[key].ravel() for key in param_keys])

        # Define the method for minimize
        method = (
            "BFGS"  # Example method, you can choose others like 'CG', 'L-BFGS-B', etc.
        )

        # Perform the optimization
        res = minimize(
            fun=objective_fn,
            x0=params_flat,  # type: ignore
            args=(param_shapes, param_keys, self.image, self.X, self.Y),
            method=method,
            tol=tol,
        )

        # Unflatten the parameters back into the dictionary form
        optimized_params = {}
        start = 0
        for key, shape in zip(param_keys, param_shapes):
            size = np.prod(shape)
            end = start + size
            optimized_params[key] = res.x[start:end].reshape(shape)
            start = end
        # params = self.same_width_on_atom_type(optimized_params)
        self.params = optimized_params
        self.model = self.predict(optimized_params, self.X, self.Y)
        return optimized_params

    def fit_global(
        self,
        params: dict = None,  # type: ignore
        maxiter: int = 1000,
        tol: float = 1e-3,
        step_size: float = 0.01,
        verbose: bool = False,
    ):
        if params is None:
            params = self.params if self.params is not None else self.init_params()
        self.fit_local = False
        params = self.optimize(
            self.image, params, self.X, self.Y, maxiter, tol, step_size, verbose
        )
        # params = self.same_width_on_atom_type(params)
        self.params = params
        self.model = self.predict(params, self.X, self.Y)
        return params

    def fit_random_batch(
        self,
        params: dict = None,  # type: ignore
        num_epoch: int = 5,
        batch_size: int = 500,
        maxiter: int = 50,
        tol: float = 1e-3,
        step_size: float = 1e-2,
        verbose: bool = False,
        plot: bool = False,
    ):
        if params is None:
            params = self.params if self.params is not None else self.init_params()

        self.fit_local = False
        self.converged = False
        params = self.linear_estimator(params)
        while self.converged is False and num_epoch > 0:
            # params = self.linear_estimator(params)
            pre_params = copy.deepcopy(params)
            num_epoch -= 1
            random_batches = get_random_indices_in_batches(
                self.num_coordinates, batch_size
            )
            image = self.image
            X = self.X
            Y = self.Y

            for index in tqdm(random_batches, desc="Fitting random batch"):
                mask = np.zeros(self.num_coordinates, dtype=bool)
                mask[index] = True
                self.atoms_selected = mask
                select_params = self.select_params(params, mask)
                try:
                    if not self.gpu_memory_limit:
                        global_prediction = self.predict(params, X, Y)
                        local_prediction = self.predict(select_params, X, Y)
                    else:
                        raise XlaRuntimeError(
                            "GPU memory limit exceeded, using the fallback of local prediction."
                        )  # Explicitly raise an exception to use the fallback
                except XlaRuntimeError:
                    self.gpu_memory_limit = True
                    global_prediction = self.predict(params, X, Y)
                    local_prediction = self.predict(select_params, X, Y)
                local_residual = global_prediction - local_prediction
                local_target = image - local_residual
                select_params = self.optimize(
                    local_target, select_params, X, Y, maxiter, tol, step_size, verbose
                )
                select_params = self.project_params(select_params)
                params = self.update_from_local_params(params, select_params, mask)
                if plot:
                    plt.subplots(1, 3, figsize=(15, 5))
                    plt.subplot(1, 3, 1)
                    plt.imshow(image, cmap="gray")
                    # plt.scatter(
                    #     params["pos_x"], params["pos_y"], color="b", s=1
                    # )
                    plt.scatter(
                        params["pos_x"][index], params["pos_y"][index], color="r", s=1
                    )
                    plt.gca().set_aspect("equal", adjustable="box")
                    plt.subplot(1, 3, 2)
                    plt.imshow(global_prediction, cmap="gray")
                    plt.scatter(
                        select_params["pos_x"], select_params["pos_y"], color="b", s=1
                    )
                    plt.gca().set_aspect("equal", adjustable="box")
                    plt.subplot(1, 3, 3)
                    plt.imshow(image - global_prediction, cmap="gray")
                    plt.gca().set_aspect("equal", adjustable="box")
                    plt.show()
                    plt.pause(1.0)
        self.converged = self.convergence(params, pre_params, tol)
        params = self.linear_estimator(params)
        self.params = params
        self.model = self.predict(params, self.X, self.Y)
        return params

    def fit_region(
        self,
        params: dict,
        fitting_region: list,
        update_region: list,
        maxiter: int = 1000,
        tol: float = 1e-4,
        step_size: float = 0.01,
        plot: bool = False,
        verbose: bool = False,
    ):
        if params is None:
            params = self.params if self.params is not None else self.init_params()
        self.fit_local = True
        left, right, top, bottom = fitting_region
        center_left, center_right, center_top, center_bottom = update_region
        pos_x = params["pos_x"]
        pos_y = params["pos_y"]
        # get the region of the image based on the patch_size and buffer_size
        # the buffer_size is on both sides of the patch

        image_region = self.image[top:bottom, left:right]
        self.local_shape = image_region.shape

        # get the region of the coordinates

        mask_region = (
            (pos_x > left) & (pos_x < right) & (pos_y > top) & (pos_y < bottom)
        )
        mask_center = (
            (pos_x > center_left)
            & (pos_x < center_right)
            & (pos_y > center_top)
            & (pos_y < center_bottom)
        )
        region_indices = np.where(mask_region)[0]
        center_indices = np.where(mask_center)[0]
        if len(center_indices) == 0:
            return params, None
        index_center_in_region = np.isin(region_indices, center_indices).squeeze()
        # index_center_in_region = find_element_indices(region_indices, center_indices)

        # get the buffer atoms as the difference between the region_atoms and the central_atoms
        local_X, local_Y = (
            self.X[top:bottom, left:right],
            self.Y[top:bottom, left:right],
        )
        if mask_center.sum() == 0:
            return params, None
        local_params = self.select_params(params, mask_region)
        self.atoms_selected = mask_region
        global_prediction = self.predict(params, self.X, self.Y)
        local_prediction = self.predict(local_params, local_X, local_Y)
        local_residual = global_prediction[top:bottom, left:right] - local_prediction
        local_target = image_region - local_residual
        local_params = self.optimize(
            local_target,
            local_params,
            local_X,
            local_Y,
            maxiter,
            tol,
            step_size,
            verbose,
        )
        params = self.update_from_local_params(
            params, local_params, mask_center, index_center_in_region
        )
        # params = self.update_from_local_params(params, local_params, mask_region)
        if plot:
            plt.subplots(1, 3, figsize=(15, 5))
            plt.subplot(1, 3, 1)
            plt.imshow(image_region, cmap="gray")
            # plt.imshow(image_region, cmap="gray")
            plt.scatter(
                pos_x[mask_region] - left,
                pos_y[mask_region] - top,
                color="b",
                s=1,
            )
            plt.scatter(
                pos_x[mask_center] - left,
                pos_y[mask_center] - top,
                color="r",
                s=1,
            )
            # get equal aspect ratio
            plt.gca().set_aspect("equal", adjustable="box")
            plt.gca().invert_yaxis()
            plt.subplot(1, 3, 2)

            plt.imshow(local_prediction, cmap="gray")
            # mask = (local_params["pos_x"] > left) & (local_params["pos_x"] < right) & (local_params["pos_y"] > top) & (local_params["pos_y"] < bottom)
            plt.scatter(
                pos_x[mask_region] - left,
                pos_y[mask_region] - top,
                color="b",
                s=1,
            )
            plt.scatter(
                pos_x[mask_center] - left,
                pos_y[mask_center] - top,
                color="r",
                s=1,
            )
            # plt.scatter(local_params["pos_x"][mask] - left, local_params["pos_y"][mask] - top, color='red',s = 1)
            plt.gca().set_aspect("equal", adjustable="box")
            plt.subplot(1, 3, 3)
            plt.imshow(local_target - local_prediction, cmap="gray")
            # plt.scatter(local_params["pos_x"][mask] - left, local_params["pos_y"][mask] - top, color='red',s = 1)
            plt.gca().set_aspect("equal", adjustable="box")

            plt.gca().invert_yaxis()
            plt.show()
        return params, local_params

    def fit_patch(
        self,
        params: dict,
        step_size: float = 0.01,
        maxiter: int = 1000,
        tol: float = 1e-4,
        patch_size: int = 100,
        buffer_size: int = 0,
        stride_size: int = 100,
        plot: bool = False,
        verbose: bool = False,
        mode: str = "sequential",
        num_random_patches: int = 10,
    ):
        if params is None:
            params = self.params if self.params is not None else self.init_params()
        self.fit_local = True
        if buffer_size is None:
            width, _, _ = (
                self.guess_radius()
            )  # Assuming width is a property of your class
            buffer_size = 5 * int(width)
        if stride_size is None:
            stride_size = int(patch_size / 2)
        # create a sqaure patch with patch_size
        half_patch = int(patch_size / 2)
        if mode == "sequential":
            x_i = np.arange(half_patch, self.nx, stride_size)
            y_i = np.arange(half_patch, self.ny, stride_size)
            ii, jj = np.meshgrid(x_i, y_i, indexing="xy")
            ii = ii.ravel()
            jj = jj.ravel()
        elif mode == "random":
            ii = np.random.randint(
                half_patch, max(self.nx - half_patch, patch_size), num_random_patches
            )
            jj = np.random.randint(
                half_patch, max(self.ny - half_patch, patch_size), num_random_patches
            )
        params = self.linear_estimator(params)
        for index in tqdm(range(len(ii))):
            i, j = ii[index], jj[index]
            left = max(i - half_patch - buffer_size, 0)
            right = min(i + half_patch + buffer_size, self.nx)
            top = max(j - half_patch - buffer_size, 0)
            bottom = min(j + half_patch + buffer_size, self.ny)

            if left < buffer_size:
                left = 0
            if right > self.nx - buffer_size:
                right = self.nx
            if top < buffer_size:
                top = 0
            if bottom > self.ny - buffer_size:
                bottom = self.ny

            center_left = left + buffer_size if left > 0 else 0
            center_right = right - buffer_size if right < self.nx else self.nx
            center_top = top + buffer_size if top > 0 else 0
            center_bottom = bottom - buffer_size if bottom < self.ny else self.ny
            if verbose:
                logging.info(
                    f"left = {left}, right = {right}, top = {top}, bottom = {bottom}"
                )
                logging.info(
                    f"center_left = {center_left}, center_right = {center_right}, center_top = {center_top}, center_bottom = {center_bottom}"
                )
            params, local_params = self.fit_region(
                params,
                [left, right, top, bottom],
                [center_left, center_right, center_top, center_bottom],
                maxiter,
                tol,
                step_size,
                plot,
                verbose,
            )

        # have a linear estimator of the background and height of the gaussian peaks
        # self.same_width_on_atom_type(params)
        params = self.linear_estimator(params)
        self.params = params
        self.model = self.predict(params, self.X, self.Y)
        return params

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
                if 'pos_x' in self.params and 'pos_y' in self.params and self.params["pos_x"].size > 0:
                    params = self.params
                else:
                    params = self.init_params()
            else:
                params = self.init_params()

        pos_x = params["pos_x"]
        pos_y = params["pos_y"]
        coords = np.stack([pos_y, pos_x])
        self.skip_window = True

        # Generate Voronoi cell map
        if max_radius is None:
            max_radius = self.params["sigma"].max() * 3
        point_record = fast_voronoi_point_record(self.image,coords, max_radius)

        # Prepare per-cell fitting function
        def fit_cell(index, params):
            mask = point_record == index + 1
            if not np.any(mask):
                return None  # No pixels in this cell

            cell_img = self.image * mask
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
            Xc, Yc = np.meshgrid(np.arange(x0, x1), np.arange(y0, y1), indexing="xy")

            # Prepare initial params for this cell
            local_param = {}
            local_param['pos_x'] = np.array([params['pos_x'][index]])
            local_param['pos_y'] = np.array([params['pos_y'][index]])
            local_param['height'] = params['height'][index] - local_min
            local_param['sigma'] = params['sigma']
            local_param['background'] = np.array([0.0])
            self.fit_background = False

            atoms_selected = np.zeros(self.num_coordinates, dtype=bool)
            atoms_selected[index] = True
            self.atoms_selected = atoms_selected

            p0 = [
                local_param['pos_x'][0],
                local_param['pos_y'][0],
                local_param['height'],
                local_param['sigma'][self.atom_types[index]],
                local_param['background'][0],
            ]
            if border > 0:
                if pos_x.min() < border or pos_x.max() > self.nx - border or pos_y.min() < border or pos_y.max() > self.ny - border:
                    popt = p0                    
            else:
                try:
                    popt, _ = curve_fit(
                        gaussian_2d_single,
                        (Xc, Yc),
                        cropped_img.ravel(),
                        p0=p0,
                        maxfev=2000
                    )
                except Exception as e:
                    popt = p0  # fallback if fit fails

            if popt[0] < 0 or popt[1] < 0:
                popt = p0
            if popt[0] > Xc.shape[1] or popt[1] > Yc.shape[0]:
                popt = p0

            optimized_param = {
                'pos_x': np.array([popt[0]]),
                'pos_y': np.array([popt[1]]),
                'height': popt[2],
                'sigma': popt[3],
                'background': np.array([popt[4]])
            }
            return optimized_param, index

        # Parallel execution (using jax.vmap or plain Python for now)
        converged = False
        
        pre_params = copy.deepcopy(self.params)
        current_params = copy.deepcopy(self.params)
        while not converged:
            with ThreadPoolExecutor() as executor:
                futures = [executor.submit(fit_cell, i, current_params) for i in range(pos_x.size)]
                for future in tqdm(as_completed(futures), total=pos_x.size, desc="Fitting cells"):
                    optimized_param, index = future.result()
                    current_params['pos_x'][index] = optimized_param['pos_x'][0]
                    current_params['pos_y'][index] = optimized_param['pos_y'][0]
            converged = self.convergence(current_params, pre_params, tol)
            pre_params = current_params
        self.params = current_params
        # self.model = self.predict(self.params, self.X, self.Y)
        return self.params

    def voronoi_integration(self, plot=False):
        """
        Compute the Voronoi integration of the atomic columns.

        Returns:
            np.array: The Voronoi integration of the atomic columns.
        """
        if self.params is None:
            raise ValueError("Please initialize the parameters first.")
        if self.fit_background:
            s = Signal2D(self.image - self.params["background"])
        else:
            s = Signal2D(self.image - self.init_background)
        pos_x = self.params["pos_x"]
        pos_y = self.params["pos_y"]
        try:
            max_radius = self.params["sigma"].max() * 5
        except KeyError:
            max_radius = self.params["gamma"].max() * 5
        integrated_intensity, intensity_record, point_record = voronoi_integrate(
            s, pos_x, pos_y, max_radius=max_radius, pbc=self.pbc
        )
        integrated_intensity = integrated_intensity * self.dx**2
        intensity_record = intensity_record * self.dx**2
        self._voronoi_volume = integrated_intensity
        self._voronoi_map = intensity_record
        self._voronoi_cell = point_record
        if plot:
            intensity_record.plot(cmap="viridis")
        return integrated_intensity, intensity_record, point_record

    # parameters updates and convergence
    def convergence(self, params: dict, pre_params: dict, tol: float = 1e-2):
        """
        Checks if the parameters have converged within a specified tolerance.

        This function iterates over each parameter in `params` and its corresponding
        value in `pre_params` to determine if the change (update) is within a specified
        tolerance level, `tol`. For position parameters ('pos_x', 'pos_y'), it checks if
        the absolute update exceeds 1. For other parameters ('height', 'sigma', 'gamma',
        'ratio', 'background'), it checks if the relative update exceeds `tol`.

        Parameters:
            params (dict): Current values of the parameters.
            pre_params (dict): Previous values of the parameters.
            tol (float, optional): Tolerance level for convergence. Default is 1e-2.

        Returns:
            bool: True if all parameters have converged within the tolerance, False otherwise.
        """
        # Loop through current parameters and their previous values
        for key, value in params.items():
            if key not in pre_params:
                continue  # Skip keys that are not in pre_params

            # Calculate the update difference
            update = np.abs(value - pre_params[key])

            # Check convergence based on parameter type
            if key in ["pos_x", "pos_y"]:
                max_update = update.max()
                logging.info(f"Convergence rate for {key} = {max_update}")
                if max_update > 1:
                    logging.info("Convergence not reached")
                    return False
            else:
                # Avoid division by zero and calculate relative update
                value_with_offset = value + 1e-10
                rate = np.abs(update / value_with_offset).mean()
                logging.info(f"Convergence rate for {key} = {rate}")
                if rate > tol:
                    logging.info("Convergence not reached")
                    return False

        logging.info("Convergence reached")
        return True

    def select_params(self, params: dict, mask: np.ndarray):
        select_params = {}
        if self.fit_background:
            select_params["background"] = params["background"]
        if self.same_width:
            if self.model_type in {"gaussian", "voigt"}:
                select_params["sigma"] = params["sigma"]
            if self.model_type in {"voigt", "lorentzian"}:
                select_params["gamma"] = params["gamma"]
            if self.model_type == "voigt":
                select_params["ratio"] = params["ratio"]
            for key in ["pos_x", "pos_y", "height"]:
                select_params[key] = params[key][mask]
        else:
            for key, value in params.items():
                if key != "background":
                    select_params[key] = value[mask]
        return select_params

    def update_from_local_params(
        self, params: dict, local_params: dict, mask: np.ndarray, mask_local=None
    ):
        for key, value in local_params.items():
            value = np.array(value)
            shared_value_list = ["background"]
            if self.same_width:
                shared_value_list += ["sigma", "gamma", "ratio"]
            if key not in shared_value_list:
                if mask_local is None:
                    params[key][mask] = value
                else:
                    params[key][mask] = value[mask_local]
            else:
                weight = mask.sum() / self.num_coordinates
                update = value - params[key]
                params[key] += update * weight
                # params[key] = value
        return params

    def same_width_on_atom_type(self, params: dict):
        if self.same_width:
            unique_types = np.unique(self.atom_types)
            for atom_type in unique_types:
                mask = self.atom_types == atom_type
                mask = mask.squeeze()
                for key, value in params.items():
                    is_jax_traced = isinstance(params[key], jax.numpy.ndarray)
                    if key in ["sigma", "gamma", "ratio"]:
                        if is_jax_traced:
                            # Calculate the mean only for the masked values using JAX
                            mean_value = jnp.mean(value[mask])
                            # Use JAX's indexing to update the values
                            params[key] = value.at[mask].set(mean_value)
                        else:
                            # For non-JAX arrays, we can proceed with NumPy operations
                            mean_value = np.mean(value[mask])
                            params[key][mask] = mean_value
        return params

    def project_params(self, params: dict):
        for key, value in params.items():
            if key == "pos_x":
                params[key] = jnp.clip(value, 0, self.nx - 1)
            elif key == "pos_y":
                params[key] = jnp.clip(value, 0, self.ny - 1)
            elif key == "height":
                params[key] = jnp.clip(value, 0, np.sum(self.image))
            elif key == "sigma":
                params[key] = jnp.clip(value, 1, min(self.nx, self.ny) / 2)
            elif key == "gamma":
                params[key] = jnp.clip(value, 1, min(self.nx, self.ny) / 2)
            elif key == "ratio":
                params[key] = jnp.clip(value, 0, 1)
            elif key == "background":
                params[key] = jnp.clip(value, 0, np.max(self.image))
        return params

    def update_coordinates(self):
        # check the refined coorinates is different from the current coordinates
        refined_coordinates = np.stack([self.params["pos_x"], self.params["pos_y"]], axis=1)
        if np.allclose(refined_coordinates, self.coordinates):
            logging.info("The coordinates have converged.")
            return self.coordinates
        else:
            # create & save the initial coordinates
            self.coordinates_history[self.coordinates_state] = self.coordinates.copy()
            # update the coordinates from the params refinement
            self.coordinates = np.stack([self.params["pos_x"], self.params["pos_y"]], axis=1)
            self.coordinates_state += 1
            logging.info(f"The coordinates have been updated. Current state: {self.coordinates_state}")
        return self.coordinates

    def update_region_analyzers(self):
        for index, region in self.regions.items:
            region.analyzer.peak_positions = self.coordinates[self.region_column_labels == index]
            region.analyzer.atom_types = self.atom_types[self.region_column_labels == index]
            logging.info(f"Updated region {index} coordinates for crystal analyzer.")

    # plot functions
    def calibrate(self, cif_file: str = None, a: float = None, b: float = None, region_index: int = 0, unit_cell: list = None):
        """
        Calibrate the pixel size based on the FFT of the lattice.
        """
        if self.coordinates.size == 0:
            logging.warning("No coordinates found. Please run find_peaks first.")
            self.find_peaks()
        column_mask = self.region_column_labels == region_index
        region_mask = self.region_map == region_index
        crystal_analyzer = CrystalAnalyzer(
            image=self.image,
            dx=self.dx,
            peak_positions=self.coordinates[column_mask],
            atom_types=self.atom_types[column_mask],
            elements=self.elements,
            units=self.units,
            region_mask=region_mask,
        )
        if unit_cell is not None:
            crystal_analyzer.unit_cell = unit_cell
        if cif_file is not None:
            crystal_analyzer.read_cif(cif_file)

        a = a if a is not None else np.linalg.norm(crystal_analyzer.unit_cell.cell[0])
        b = b if b is not None else np.linalg.norm(crystal_analyzer.unit_cell.cell[1])
        _, vec_a_pixel, vec_b_pixel = crystal_analyzer.select_lattice_vectors(reciprocal=True)
        dx_a = a / np.linalg.norm(vec_a_pixel)
        dx_b = b / np.linalg.norm(vec_b_pixel)
        self.dx = (dx_a + dx_b) / 2
        logging.info(f"Calibrated pixel size: {self.dx} {self.units}")

    def plot(self, vmin=None, vmax=None):
        if vmin is None:
            # get the bottom 5% of the image
            vmin = np.percentile(self.image, 5)

        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        im = plt.imshow(self.image, cmap="gray", vmin=vmin, vmax=vmax)
        plt.axis("off")
        scalebar = self.scalebar
        plt.gca().add_artist(scalebar)
        plt.colorbar(im, fraction=0.046, pad=0.04)
        plt.tight_layout()
        plt.gca().add_artist(scalebar)
        plt.gca().set_aspect("equal", adjustable="box")
        plt.title("Image")

        plt.subplot(1, 2, 2)
        plt.hist(self.image.ravel(), bins=256)
        plt.xlabel("Intensity")
        plt.ylabel("Counts")
        plt.title("Intensity Histogram")
        plt.tight_layout()

    def plot_coordinates(self, s=1):
        """
        Plot the coordinates of the atomic columns.

        Args:
            color (str, optional): The color of the atomic columns. Defaults to "red".
            s (int, optional): The size of the atomic columns. Defaults to 1.
        """
        plt.figure()
        plt.imshow(self.image, cmap="gray")
        for atom_type in np.unique(self.atom_types):
            mask = self.atom_types == atom_type
            elements = self.elements[atom_type]
            plt.scatter(
                self.coordinates[mask][:, 0],
                self.coordinates[mask][:, 1],
                s=s,
                label=elements,
            )
        plt.legend()

    def plot_fitting(self):
        plt.figure(figsize=(15, 5))
        vmin = self.image.min()
        vmax = self.image.max()
        plt.subplot(1, 3, 1)
        im = plt.imshow(self.image, cmap="gray", vmin=vmin, vmax=vmax)
        plt.colorbar(im, fraction=0.046, pad=0.04)
        plt.gca().set_aspect("equal", adjustable="box")
        plt.title("Original Image")
        plt.tight_layout()
        plt.subplot(1, 3, 2)
        im = plt.imshow(self.model, cmap="gray", vmin=vmin, vmax=vmax)
        plt.gca().set_aspect("equal", adjustable="box")
        plt.colorbar(im, fraction=0.046, pad=0.04)
        plt.title("Model")
        plt.tight_layout()
        plt.subplot(1, 3, 3)
        im = plt.imshow(self.image - self.model, cmap="gray")
        plt.colorbar(im, fraction=0.046, pad=0.04)
        plt.gca().set_aspect("equal", adjustable="box")
        plt.title("Residual")
        plt.tight_layout()

    def plot_scs(self, layout="horizontal", per_element=False, s=1, save=False, has_units=True, half: str = None, figsize=(10, 5)):
        assert layout in {
            "horizontal",
            "vertical",
        }, "Layout should be horizontal or vertical"
        if layout == "horizontal":
            row, col = 1, 2
            if per_element:
                col += len(np.unique(self.atom_types)) - 1
        else:
            row, col = 2, 1
            if per_element:
                row += len(np.unique(self.atom_types)) - 1
        plt.figure(figsize=figsize)
        plt.subplot(row, col, 1)
        plt.imshow(self.image, cmap="gray")
        for atom_type in np.unique(self.atom_types):
            mask = self.atom_types == atom_type
            element = self.elements[int(atom_type)]
            if half is not None:
                if half == "top":
                    mask = mask & (self.coordinates[:, 1] < self.ny / 2)
                elif half == "bottom":
                    mask = mask & (self.coordinates[:, 1] > self.ny / 2)
                elif half == "left":
                    mask = mask & (self.coordinates[:, 0] < self.nx / 2)
                elif half == "right":
                    mask = mask & (self.coordinates[:, 0] > self.nx / 2)
            plt.scatter(
                self.coordinates[mask, 0],
                self.coordinates[mask, 1],
                s=s,
                label=element,
            )
        plt.legend(loc="upper right")
        plt.gca().set_aspect("equal", adjustable="box")
        plt.axis("off")
        scalebar = self.scalebar
        plt.gca().add_artist(scalebar)
        plt.title("Image")
        plt.tight_layout()

        # plot the scs
        pos_x = self.params["pos_x"] * self.dx
        pos_y = self.params["pos_y"] * self.dx
        if per_element:
            plt_idx = 1
            col = len(np.unique(self.atom_types)) + 1
            for atom_type in np.unique(self.atom_types):
                plt_idx += 1
                plt.subplot(row, col, plt_idx)
                mask = self.atom_types == atom_type
                element = self.elements[atom_type]
                im = plt.scatter(
                    pos_x[mask],
                    pos_y[mask],
                    c=self.volume[mask],
                    s=s,
                    label=element,
                )
                cbar = plt.colorbar(im, fraction=0.046, pad=0.04)
                plt.gca().set_aspect("equal", adjustable="box")
                # plt.axis("off")
                plt.xlim(0, self.nx * self.dx)
                plt.ylim(0, self.ny * self.dx)
                plt.xlabel(r"X ($\AA$)")
                plt.ylabel(r"Y ($\AA$)")
                plt.title(f"{element}")
                plt.gca().invert_yaxis()
                # add title to the colorbar
                if atom_type == self.atom_types.max():
                    if has_units:
                        cbar.set_label(r"SCS ($\AA^2$)")
                    else:
                        cbar.set_label("Integrated intensities")
                plt.tight_layout()
        else:
            plt.subplot(row, col, 2)
            im = plt.scatter(pos_x, pos_y, c=self.volume, s=2)
            cbar = plt.colorbar(im, fraction=0.046, pad=0.04)
            # plt.axis("off")
            plt.xlim(0, self.nx * self.dx)
            plt.ylim(0, self.ny * self.dx)
            plt.xlabel(r"X ($\AA$)")
            plt.ylabel(r"Y ($\AA$)")
            plt.gca().invert_yaxis()
            plt.gca().set_aspect("equal", adjustable="box")
            if has_units:
                cbar.set_label(r"SCS ($\AA^2$)")
            else:
                cbar.set_label("Integrated intensities")
            plt.tight_layout()
        if save:
            plt.savefig("scs.svg")
            plt.savefig("scs.png", dpi=300)

    def plot_scs_voronoi(self, layout="horizontal", s=1, per_element=False, save=False, has_units=True, half: str = None, figsize=(10, 5)):
        assert self.voronoi_volume is not None, "Please run the voronoi analysis first"
        if per_element:
            row, col = 1, 2
            col += len(np.unique(self.atom_types)) - 1
            plt.figure(figsize=figsize)
            plt.subplot(row, col, 1)
            plt.imshow(self.image, cmap="gray")
            for atom_type in np.unique(self.atom_types):
                mask = self.atom_types == atom_type
                element = self.elements[atom_type]
                if half is not None:
                    if half == "top":
                        mask = mask & (self.coordinates[:, 1] < self.ny / 2)
                    elif half == "bottom":
                        mask = mask & (self.coordinates[:, 1] > self.ny / 2)
                    elif half == "left":
                        mask = mask & (self.coordinates[:, 0] < self.nx / 2)
                    elif half == "right":
                        mask = mask & (self.coordinates[:, 0] > self.nx / 2)
                plt.scatter(
                    self.coordinates[mask, 0],
                    self.coordinates[mask, 1],
                    s=1,
                    label=element,
                )
            plt.legend(loc="upper right")
            plt.gca().add_artist(self.scalebar)
            plot_idx = 2
            for atom_type in np.unique(self.atom_types):
                mask = self.atom_types == atom_type
                plt.subplot(row, col, plot_idx)
                element = self.elements[atom_type]
                pos_x = self.params["pos_x"][mask] * self.dx
                pos_y = self.params["pos_y"][mask] * self.dx
                im = plt.scatter(
                    pos_x, pos_y, c=self.voronoi_volume[mask], s=s, label=element
                )
                plt.gca().set_aspect("equal", adjustable="box")
                cbar = plt.colorbar(im, fraction=0.046, pad=0.04)
                # plt.axis("off")
                plt.xlim(0, self.nx * self.dx)
                plt.ylim(0, self.ny * self.dx)
                plt.xlabel(r"X ($\AA$)")
                plt.ylabel(r"Y ($\AA$)")
                plt.gca().invert_yaxis()
                plt.title(f"{element}")
                if atom_type == self.atom_types.max():
                    if has_units:
                        cbar.set_label(r"Voronoi SCS ($\AA^2$)")
                    else:
                        cbar.set_label("Voronoi integrated intensities")
                plot_idx += 1
        else:
            row, col = (1, 2) if layout == "horizontal" else (2, 1)
            plt.figure()
            plt.subplot(row, col, 1)
            plt.imshow(self.image, cmap="gray")
            for atom_type in np.unique(self.atom_types):
                mask = self.atom_types == atom_type
                element = self.elements[atom_type]
                plt.scatter(
                    self.coordinates[mask, 0],
                    self.coordinates[mask, 1],
                    s=1,
                    label=element,
                )
            plt.legend()
            plt.gca().set_aspect("equal", adjustable="box")
            plt.title("Image")
            plt.subplot(row, col, 2)
            pos_x = self.params["pos_x"] * self.dx
            pos_y = self.params["pos_y"] * self.dx
            im = plt.scatter(pos_x, pos_y, c=self.voronoi_volume, s=s)
            # make aspect ratio equal
            plt.gca().invert_yaxis()
            plt.gca().set_aspect("equal", adjustable="box")
            plt.colorbar(im, fraction=0.046, pad=0.04)
            if has_units:
                plt.title(r"Voronoi scs ($\AA^2$)")
            else:
                plt.title("Voronoi integrated intensities")
        plt.tight_layout()

        if save:
            plt.savefig("voronoi_scs.svg")
            plt.savefig("voronoi_scs.png", dpi=300)

    def plot_scs_histogram(self, save=False, has_units=True):
        plt.figure()
        for atom_type in np.unique(self.atom_types):
            mask = self.atom_types == atom_type
            element = self.elements[atom_type]
            plt.hist(self.volume[mask], bins=100, alpha=0.5, label=element)
        plt.legend()
        if has_units:
            plt.xlabel(r"Refined SCS ($\AA^2$)")
        else:
            plt.xlabel("Integrated intensities")
        plt.ylabel("Frequency")
        plt.title("Histogram of QEM refined SCS")
        if save:
            plt.savefig("scs_histogram.svg")
            plt.savefig("scs_histogram.png", dpi=300)

    def plot_region(self):
        plt.figure()
        # cmap = color_iter('Set3', self.num_regions)
        # cmap = plt.get_cmap("tab10", self.num_regions)
        plt.imshow(self.image, cmap="gray")
        plt.imshow(self.region_map, alpha=0.5)
        scalebar = self.scalebar
        plt.gca().add_artist(scalebar)
        plt.axis("off")
        cbar = plt.colorbar()
        cbar.set_ticks(np.arange(self.num_regions))  # type: ignore
        plt.title("Region Map")

    # Properties

    @property
    def window(self):
        """
        Returns the window used for fitting.

        If `fit_local` is True, a Butterworth window is created with the shape of `local_shape`.
        If `fit_local` is False, a Butterworth window is created with the shape of `image.shape`.

        Returns:
            numpy.ndarray: The Butterworth window used for fitting.
        """
        if self.fit_local:
            return butterworth_window(self.local_shape, 0.5, 10)
        else:
            return butterworth_window(self.image.shape, 0.5, 10)

    @property
    def volume(self):
        params = self.params.copy()
        if self.same_width:
            if self.model_type in {"gaussian", "voigt"}:
                params["sigma"] = params["sigma"][self.atom_types]
            elif self.model_type in {"lorentzian"}:
                params["gamma"] = params["gamma"][self.atom_types]
            elif self.model_type == "voigt":
                params["ratio"] = params["ratio"][self.atom_types]
        volume = np.zeros(self.num_coordinates)
        if self.model_type == "gaussian":
            volume = params["height"] * params["sigma"] ** 2 * np.pi * 2 * self.dx**2
        elif self.model_type == "lorentzian":
            volume = params["height"] * params["gamma"] ** 2 * np.pi * self.dx**2
        elif self.model_type == "voigt":
            gaussian_contrib = (
                params["height"] * params["sigma"] ** 2 * 2 * np.pi * self.dx**2
            )
            lorentzian_contrib = (
                params["height"] * params["gamma"] ** 2 * np.pi * self.dx**2
            )
            volume = gaussian_contrib * params["ratio"] + lorentzian_contrib * (
                1 - params["ratio"]
            )
        return volume

    @property
    def voronoi_volume(self):
        if not hasattr(self, "_voronoi_volume") or self._voronoi_volume is None:
            self.voronoi_integration()
        return self._voronoi_volume

    @property
    def num_coordinates(self):
        return self.coordinates.shape[0]

    @property
    def atom_types(self):
        if len(self._atom_types) == 0 or self._atom_types is None:
            self._atom_types = np.zeros(self.num_coordinates, dtype=int)
        return self._atom_types

    @atom_types.setter
    def atom_types(self, atom_types: np.ndarray):
        self._atom_types = atom_types

    @property
    def region_column_labels(self):
        return self.regions.region_map[
            self.coordinates[:, 1].astype(int), self.coordinates[:, 0].astype(int)
        ]

    @property
    def region_map(self):
        return self.regions.region_map

    @property
    def num_atom_types(self):
        return len(np.unique(self.atom_types))

    @property
    def coordinates(self):
        return self._coordinates

    @coordinates.setter
    def coordinates(self, coordinates: np.ndarray):
        self._coordinates = coordinates

    @property
    def scalebar(self):
        scalebar = ScaleBar(
            self.dx,
            units=self.units,
            location="lower right",
            length_fraction=0.2,
            font_properties={"size": 20},
        )
        return scalebar

    @property
    def num_regions(self):
        return self.regions.num_regions
