# Standard library imports
import copy
import logging
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

# Third-party library imports
import keras
import matplotlib.pyplot as plt
import numpy as np
from hyperspy.signals import Signal2D
from keras import ops
from matplotlib_scalebar.scalebar import ScaleBar
from numpy.typing import NDArray
from scipy.optimize import curve_fit, lsq_linear
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import spsolve
from skimage.feature import peak_local_max
from scipy.ndimage import gaussian_filter
from tqdm import tqdm

# Application-specific imports
from qem.crystal_analyzer import CrystalAnalyzer
from qem.gui_classes import (
    GetAtomSelection,
    GetRegionSelection,
    InteractivePlot,
)
from qem.model import (
    GaussianKernel,
    GaussianModel,
    LorentzianModel,
    VoigtModel,
    gaussian_2d_single,
)
from qem.processing import butterworth_window
from qem.refine import calculate_center_of_mass
from qem.region import Regions
from qem.utils import (
    get_random_indices_in_batches,
    remove_close_coordinates,
    safe_convert_to_numpy,
    safe_convert_to_tensor,
    safe_deepcopy_params,
)
from qem.voronoi import voronoi_integrate, voronoi_point_record


logging.basicConfig(level=logging.INFO)


class ImageFitting:
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
            elements (list[str], optional): The elements in the image. Defaults to None. If None, the elements are ["Sr", "Ti", "O"].
            model_type (str, optional): Type of model to use. Defaults to "gaussian".
            same_width (bool, optional): Whether to use same width for all peaks. Defaults to True.
            pbc (bool, optional): Whether to use periodic boundary conditions. Defaults to False.
            fit_background (bool, optional): Whether to fit background. Defaults to True.
            gpu_memory_limit (bool, optional): Whether to use memory-efficient GPU computation. Defaults to False.
        """
        if elements is None:
            elements = ["A", "B", "C"]

        # Store instance variables
        self.image = image
        self.dx = float(dx)
        self.units = units
        self.elements = elements
        self.model_type = model_type.lower()
        self.same_width = same_width
        self.pbc = pbc
        self.fit_background = fit_background
        self.gpu_memory_limit = gpu_memory_limit
        self.backend = keras.backend.backend()

        # Create model instance based on type
        self.model = self._create_model()

        # Create Gaussian kernel for filtering
        self.kernel = GaussianKernel()
        self._window = None

        # Initialize missing attributes
        self._atom_types = np.array([])
        self._coordinates = np.array([])
        self.coordinates_history = dict()
        self.coordinates_state = 0
        self.init_background = 0.0
        self.prediction = np.zeros_like(self.image)

        # Initialize other attributes
        self.params = {}
        self.converged = False
        self.ny, self.nx = image.shape
        self.regions = Regions(image=image)
        self.initialize_grid()

    def initialize_grid(self):
        """Initialize the coordinate grids for the model."""
        self.image_tensor = ops.convert_to_tensor(self.image, dtype="float32")
        x = ops.arange(self.nx, dtype="float32")
        y = ops.arange(self.ny, dtype="float32")
        x_grid, y_grid = ops.meshgrid(x, y)
        self.x_grid = ops.convert_to_tensor(x_grid, dtype="float32")
        self.y_grid = ops.convert_to_tensor(y_grid, dtype="float32")

        # Set the grids on the model
        self.model.set_grid(self.x_grid, self.y_grid)

    def _create_model(self):
        """Create a new model instance based on the model type."""
        if self.model_type == "gaussian":
            model = GaussianModel(dx=float(self.dx))
        elif self.model_type == "lorentzian":
            model = LorentzianModel(dx=float(self.dx))
        elif self.model_type == "voigt":
            model = VoigtModel(dx=float(self.dx))
        else:
            raise ValueError(f"Model type {self.model_type} not supported.")
        return model

    def predict(self, local: bool = True):
        """Predict the image based on the model's current parameters.

        Args:
            local (bool, optional): If True, calculate peaks locally. Defaults to False.

        Returns:
            array: Predicted image
        """
        prediction = self.model.sum(local=local)

        # Handle periodic boundary conditions by rolling the image
        if self.pbc:
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
                # Temporarily set shifted grids for periodic boundary conditions
                original_x_grid, original_y_grid = self.model.x_grid, self.model.y_grid
                self.model.set_grid(
                    self.x_grid + i * self.nx, self.y_grid + j * self.ny
                )
                prediction += self.model.sum(local=local)
                # Restore original grids
                self.model.set_grid(original_x_grid, original_y_grid)
        return prediction

    # Properties

    @property
    def window(self):
        """
        Returns the window used for fitting.

        Returns:
            numpy.ndarray: A Butterworth-style window used for fitting.
        """
        if self._window is None:
            window = butterworth_window(self.image.shape, 0.5, 10)
            self._window = window
        return self._window

    @property
    def volume(self):
        """Calculate the volume of each peak in the model.

        Returns:
            numpy.ndarray: Array of volumes for each peak.
        """
        if not self.params:
            raise ValueError("Parameters not initialized. Call init_params first.")

        # Update the model's pixel size
        self.model.dx = self.dx

        # Create parameters dict for volume calculation
        params = self.params.copy()
        volume = self.model.volume(params)
        return safe_convert_to_numpy(volume)

    @property
    def scalebar(self):
        scalebar = ScaleBar(
            self.dx,
            units="A",
            location="lower right",
            length_fraction=0.2,
            font_properties={"size": 20},
        )
        return scalebar

    # voronoi integration
    def voronoi_integration(self, max_radius: float = None, plot=False):
        """
        Compute the Voronoi integration of the atomic columns.

        Returns:
            np.array: The Voronoi integration of the atomic columns.
        """
        if self.params is None:
            raise ValueError("Please initialize the parameters first.")
        if self.fit_background:
            s = Signal2D(self.image - safe_convert_to_numpy(self.params["background"]))
        else:
            s = Signal2D(self.image - self.init_background)
        pos_x = self.params["pos_x"]
        pos_y = self.params["pos_y"]
        pos_x = safe_convert_to_numpy(pos_x)
        pos_y = safe_convert_to_numpy(pos_y)
        if max_radius is None:
            max_radius = self.params["width"].max() * 5
            max_radius = safe_convert_to_numpy(max_radius)
        integrated_intensity, intensity_record, point_record = voronoi_integrate(
            s, pos_x, pos_y, max_radius=max_radius, pbc=self.pbc
        )
        integrated_intensity = integrated_intensity * self.dx**2
        intensity_record = intensity_record * self.dx**2
        self._voronoi_volume = integrated_intensity
        self._voronoi_map = intensity_record
        self._voronoi_cell = point_record
        if plot:
            plt.imshow(intensity_record, cmap="viridis")
            plt.colorbar(label="Voronoi Integrated Intensity")
        return integrated_intensity, intensity_record, point_record

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
                i_l = np.maximum(self.coordinates[i, 0] - n, 0).astype(np.int32)
                i_r = np.minimum(self.coordinates[i, 0] + n, self.nx).astype(np.int32)
                i_u = np.maximum(self.coordinates[i, 1] - n, 0).astype(np.int32)
                i_d = np.minimum(self.coordinates[i, 1] + n, self.ny).astype(np.int32)
                influence_map[i_l : i_r + 1, i_u : i_d + 1] = 1
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
            i_l = np.maximum(self.coordinates[i, 0] - n1, 0).astype(np.int32)
            i_r = np.minimum(self.coordinates[i, 0] + n1, nx).astype(np.int32)
            i_u = np.maximum(self.coordinates[i, 1] - n1, 0).astype(np.int32)
            i_d = np.minimum(self.coordinates[i, 1] + n1, ny).astype(np.int32)
            influence_map[i_l : i_r + 1, i_u : i_d + 1] = 1

            # Calculate the indices for the smaller area (direct_influence_map)
            i_l = np.maximum(self.coordinates[i, 0] - n2, 0).astype(np.int32)
            i_r = np.minimum(self.coordinates[i, 0] + n2, nx).astype(np.int32)
            i_u = np.maximum(self.coordinates[i, 1] - n2, 0).astype(np.int32)
            i_d = np.minimum(self.coordinates[i, 1] + n2, ny).astype(np.int32)
            direct_influence_map[i_l : i_r + 1, i_u : i_d + 1] = 1

        radius = (np.sum(direct_influence_map) / num_coordinates) ** (1 / 2) / np.pi

        background_region = influence_map - direct_influence_map
        return radius, direct_influence_map, background_region

    def init_params(
        self,
        atom_size: float = 0.7,
        guess_radius: bool = False,
        init_background: float = 0.0,
    ):
        """Initialize model parameters based on the current model type and settings.

        Args:
            atom_size (float, optional): Base size for atomic peaks. Defaults to 0.7.
            guess_radius (bool, optional): Whether to estimate peak width from data. Defaults to False.
            init_background (float, optional): Initial background value. Defaults to 0.0.
        """
        self.coordinates_history = dict()
        self.coordinates_state = 0

        # Get width parameter
        if guess_radius:
            width = self.guess_radius()[0]
        else:
            width = atom_size / self.dx

        # Handle periodic boundary conditions
        if self.pbc:
            mask = (self.coordinates[:, 0] < self.nx - 1) & (
                self.coordinates[:, 1] < self.ny - 1
            )
            self.coordinates = self.coordinates[mask]

        # Initialize position and height parameters
        pos_x = copy.deepcopy(self.coordinates[:, 0]).astype(float)
        pos_y = copy.deepcopy(self.coordinates[:, 1]).astype(float)
        pos_x = np.clip(pos_x, 0, self.image.shape[0] - 1)
        pos_y = np.clip(pos_y, 0, self.image.shape[1] - 1)

        # Initialize background
        if self.fit_background:
            init_background = self.image.min()
        else:
            self.init_background = init_background

        # Set background in model
        self.model.background = init_background

        # Initialize heights from image values
        height = (
            self.image[pos_y.astype(int), pos_x.astype(int)].ravel() - init_background
        )
        height[height < 0] = 0  # Ensure non-negative heights

        # Initialize width parameters based on model type
        if self.same_width:
            width = np.tile(width, 1).astype(float)
        else:
            width = np.tile(width, self.num_coordinates).astype(float)

        # Create parameter dictionary based on model type
        params = {
            "pos_x": pos_x,
            "pos_y": pos_y,
            "height": height,
        }

        # Add model-specific parameters
        params["width"] = width

        if isinstance(self.model, VoigtModel):
            if self.same_width:
                ratio = 0.9  # Scalar for same_width case
            else:
                ratio = np.tile(0.9, self.num_coordinates).astype(float)
            params["ratio"] = ratio

        if self.fit_background:
            params["background"] = init_background

        for key in params.keys():
            params[key] = ops.convert_to_tensor(params[key], dtype="float32")

        self.params = params
        self.model.set_params(self.params)
        # Build the model with the correct input shape (grid shapes)
        self.model.build()
        return params

    # find atomic columns
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
        # find the column within the region_index
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
            units="A",
            region_mask=region_mask,
        )
        # if unit_cell is not None:
        #     crystal_analyzer.unit_cell = unit_cell
        if cif_file is not None:
            crystal_analyzer.read_cif(cif_file)
        atomic_column_list = crystal_analyzer.get_atomic_columns(
            reciprocal=reciprocal, sigma=sigma
        )
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

    def assign_region_label(
        self, region_index: int = 0, invert_selection: bool = False
    ):
        atom_select = GetRegionSelection(
            image=self.image,
            invert_selection=invert_selection,
            region_map=self.regions.region_map,
        )
        try:
            atom_select.poly.verts = self.regions[region_index].path.vertices  # type: ignore
            atom_select.path = self.regions[region_index].path
        except KeyError:
            pass
        while plt.fignum_exists(atom_select.fig.number):  # type: ignore
            plt.pause(0.1)

        region_mask = atom_select.get_region_mask()
        self.regions.region_map[region_mask] = region_index
        try:
            self.region_path_dict[region_index] = atom_select.path
        except AttributeError:
            pass
        logging.info(
            f"Assigned label {region_index} with {region_mask.sum()} pixels to the region map."
        )

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
        assert isinstance(local_x, float), "local_x is not a float"
        assert isinstance(local_y, float), "local_y is not a float"
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
            for i, j in [
                (1, 0),
                (0, 1),
                (1, 1),
                (-1, 0),
                (0, -1),
                (-1, -1),
                (1, -1),
                (-1, 1),
            ]:
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

    def loss_val(self, params: dict):
        prediction = self.predict(params)
        diff = self.image_tensor - prediction
        diff = ops.multiply(diff, self.window)
        # damping the difference near the edge
        mse = ops.sqrt(ops.mean(ops.square(diff)))
        l1 = ops.mean(ops.abs(diff))
        return mse + l1

    def loss(self, y_true, y_pred):
        """
        Compute the loss value between the image and the prediction.

        Parameters:
        -----------
        y_true : np.ndarray
            The original image tensor (ground truth).
        y_pred : np.ndarray
            The predicted image tensor (model output).

        Returns:
        --------
        float
            The computed loss value.
        """
        diff = y_true - y_pred
        if keras.backend.backend() == "torch":
            window = keras.ops.convert_to_tensor(self.window, dtype="float32")
        else:
            window = self.window
        diff = ops.multiply(diff, window)
        mse = ops.sqrt(ops.mean(ops.square(diff)))
        l1 = ops.mean(ops.abs(diff))
        return mse + l1

    def residual(self, params: dict):
        # Compute the sum of the Gaussians
        prediction = self.predict(params)
        diff = self.image_tensor - prediction
        diff = safe_convert_to_numpy(diff)
        return diff

    # fitting
    def linear_estimator(self, params: dict = None, non_negative=False):
        if params is None:
            if self.params is None:
                self.init_params()
            params = self.params
        # create the design matrix as array of gaussian peaks + background
        pos_x = params["pos_x"]
        pos_y = params["pos_y"]
        width = params["width"]
        height = params["height"]
        ratio = params.get("ratio", None)
        if self.same_width:
            width = width[self.atom_types]
            if ratio is not None:
                ratio = ratio[self.atom_types]
        # if (height < 0).any():
        #     logging.warning(
        #         "The height has negative values, the linear estimator is not valid, I will make it to zero but be careful with the results."
        #     )
        #     height[height < 0] = 0

        # Vectorized implementation using keras.ops
        window_size = ops.cast(ops.max(width) * 5, dtype="int32")
        x = ops.arange(-window_size, window_size + 1, 1, dtype="float32")
        y = ops.arange(-window_size, window_size + 1, 1, dtype="float32")
        local_x, local_y = ops.meshgrid(x, y, indexing="xy")

        input_params = (ops.mod(pos_x, 1), ops.mod(pos_y, 1), height, width)
        if ratio is not None:
            input_params += (ratio,)

        peak_local = self.model.model_fn(
            local_x[..., None], local_y[..., None], *input_params
        )

        pos_x_int = ops.floor(pos_x)
        pos_y_int = ops.floor(pos_y)

        global_x = ops.expand_dims(local_x, -1) + pos_x_int
        global_y = ops.expand_dims(local_y, -1) + pos_y_int

        mask = (
            (global_x >= 0)
            & (global_x < self.nx)
            & (global_y >= 0)
            & (global_y < self.ny)
        )

        # Get the indices of valid elements where the mask is True.
        valid_indices = ops.where(mask)

        # Flatten the tensors and calculate 1D indices to replicate gather_nd,
        # ensuring compatibility with Keras 2 and other backends.
        shape = ops.shape(peak_local)
        # ops.where returns a tuple of index tensors, one for each dimension.
        # We access them with tuple indexing, not array slicing.
        flat_indices = (
            valid_indices[0] * (shape[1] * shape[2])
            + valid_indices[1] * shape[2]
            + valid_indices[2]
        )

        # Gather the valid data from the flattened tensors using the 1D indices.
        data_tensor = ops.take(ops.reshape(peak_local, (-1,)), flat_indices)
        global_x_valid = ops.take(ops.reshape(global_x, (-1,)), flat_indices)
        global_y_valid = ops.take(ops.reshape(global_y, (-1,)), flat_indices)

        # The column indices correspond to the third dimension of the original tensors.
        cols_tensor = valid_indices[2]

        rows_tensor = ops.cast(global_y_valid, "int32") * self.nx + ops.cast(
            global_x_valid, "int32"
        )
        if self.fit_background:
            background_rows = ops.reshape(self.y_grid, (-1,)) * self.nx + ops.reshape(
                self.x_grid, (-1,)
            )
            rows_tensor = ops.concatenate(
                [rows_tensor, ops.cast(background_rows, "int32")]
            )
            cols_tensor = ops.concatenate(
                [
                    cols_tensor,
                    ops.full((self.nx * self.ny,), self.num_coordinates, dtype="int32"),
                ]
            )
            data_tensor = ops.concatenate(
                [data_tensor, ops.ones((self.nx * self.ny,), dtype="float32")]
            )
            shape = (self.nx * self.ny, self.num_coordinates + 1)
        else:
            shape = (self.nx * self.ny, self.num_coordinates)

        design_matrix = coo_matrix(
            (
                safe_convert_to_numpy(data_tensor),
                (
                    safe_convert_to_numpy(rows_tensor),
                    safe_convert_to_numpy(cols_tensor),
                ),
            ),
            shape=shape,
        )

        # create the target as the image
        b = safe_convert_to_numpy(self.image_tensor).ravel()
        if not self.fit_background:
            b = b - safe_convert_to_numpy(self.init_background)
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

        # update the background and height

        if self.fit_background:
            background = solution[-1] if solution[-1] > 0 else self.init_background
            params["background"] = ops.convert_to_tensor(background)
            height_scale = solution[:-1]
        else:
            height_scale = solution
        if np.nan in height_scale:
            logging.warning(
                "The height_scale has NaN, the linear estimator is not valid, parameters are not updated"
            )
            return params
        if (height_scale > 5).any():
            logging.warning(
                "The height_scale has values larger than 5, the linear estimator is probably not accurate. I will limit it to 5 but be careful with the results."
            )
            height_scale[height_scale > 5] = 5
        if (height_scale < 0.1).any():
            logging.warning(
                "The height_scale has values smaller than 0.1, the linear estimator is probably not accurate. I will limit it to 0.1 but be careful with the results."
            )
            height_scale[height_scale < 0.1] = 0.1
        params["height"] = ops.convert_to_tensor(height_scale) * ops.convert_to_tensor(
            params["height"]
        )
        self.params = params
        return params

    def optimize(
        self,
        image_tensor: np.ndarray,
        params: dict,
        maxiter: int = 1000,
        tol: float = 1e-4,
        step_size: float = 0.01,
        verbose: bool = False,
        batch_size: int = 1024,
    ) -> dict[str, NDArray[Any]]:
        if image_tensor is None:
            image_tensor = self.image_tensor
        self.model.set_params(params)

        # Build the model with the correct input shape (grid shapes)
        self.model.build()

        # Backend-specific input preparation
        image_tensor_batch = ops.expand_dims(
            image_tensor, 0
        )  # Always need batch dimension for target

        if self.backend == "torch":
            # PyTorch needs batch dimensions for inputs
            x_grid_batch = ops.expand_dims(self.x_grid, 0)
            y_grid_batch = ops.expand_dims(self.y_grid, 0)
            model_inputs = [x_grid_batch, y_grid_batch]
        else:
            # JAX and TensorFlow: create dummy inputs with correct batch size
            # The model uses grids set via set_grid(), so we just need dummy data with batch dimension
            dummy_input = ops.zeros((1, 1))  # Minimal dummy input with batch dimension
            model_inputs = dummy_input
        
        # JAX-specific model building: call the model once to ensure it's built
        if self.backend == "jax":
            try:
                # Force model building by calling it without arguments (model uses set_grid)
                dummy_output = self.model()
                if verbose:
                    print(
                        f"JAX model built successfully, output shape: {dummy_output.shape}"
                    )
            except Exception as e:
                if verbose:
                    print(f"JAX model building warning: {e}")
                # Continue anyway, let fit() handle it

        self.model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=step_size), loss=self.loss
        )

        early_stopping = keras.callbacks.EarlyStopping(
            monitor="loss",
            min_delta=tol,
            patience=100,
            verbose=verbose,
            restore_best_weights=True,
        )

        self.model.fit(
            x=model_inputs,
            y=image_tensor_batch,
            epochs=maxiter,
            verbose=verbose,
            callbacks=[early_stopping],
            batch_size=1,  # Set to 1 since we have only one sample (the full image)
        )
        optimized_params = self.model.get_params()
        return optimized_params

    def fit_global(
        self,
        params: dict = None,  # type: ignore
        maxiter: int = 1000,
        tol: float = 1e-3,
        step_size: float = 0.01,
        local: bool = True,
        verbose: bool = False,
    ):
        if params is None:
            params = self.params if self.params is not None else self.init_params()
        params = self.optimize(
            image_tensor=self.image_tensor,
            params=params,
            maxiter=maxiter,
            tol=tol,
            step_size=step_size,
            verbose=verbose,
        )
        self.params = params
        self.prediction = safe_convert_to_numpy(self.predict(local=local))

        return params

    def _optimize_local_batch(
        self, select_params, local_target, maxiter, tol, step_size, verbose=False
    ):
        """
        Optimize a local batch of parameters using a temporary model.

        Args:
            select_params: Dictionary of selected parameters for the batch
            local_target: Target image for the local optimization
            maxiter: Maximum number of iterations
            tol: Tolerance for early stopping
            step_size: Learning rate
            verbose: Whether to show verbose output

        Returns:
            Dictionary of optimized parameters
        """
        # Create a temporary model for local optimization
        temp_model = self._create_model()
        temp_model.set_grid(self.x_grid, self.y_grid)
        temp_model.set_params(select_params)

        # Build with the correct input shape (grid shapes)
        temp_model.build()

        # Backend-specific input preparation for local optimization
        local_target_batch = ops.expand_dims(
            local_target, 0
        )  # Always need batch dimension for target

        if self.backend == "torch":
            # PyTorch needs batch dimensions for inputs
            x_grid_batch = ops.expand_dims(self.x_grid, 0)
            y_grid_batch = ops.expand_dims(self.y_grid, 0)
            model_inputs = [x_grid_batch, y_grid_batch]
        else:
            # JAX and TensorFlow: create dummy inputs with correct batch size
            # The model uses grids set via set_grid(), so we just need dummy data with batch dimension
            dummy_input = ops.zeros((1, 1))  # Minimal dummy input with batch dimension
            model_inputs = dummy_input
        
        # JAX-specific model building: call the model once to ensure it's built
        if self.backend == "jax":
            try:
                # Force model building by calling it without arguments (model uses set_grid)
                dummy_output = temp_model()
                if verbose:
                    print(
                        f"JAX temp model built successfully, output shape: {dummy_output.shape}"
                    )
            except Exception as e:
                if verbose:
                    print(f"JAX temp model building warning: {e}")
                # Continue anyway, let fit() handle it

        # Compile the temporary model
        temp_model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=step_size), loss=self.loss
        )

        # Set up early stopping
        early_stopping = keras.callbacks.EarlyStopping(
            monitor="loss",
            min_delta=tol,
            patience=100,
            verbose=verbose,
            restore_best_weights=True,
        )

        # Train the temporary model
        temp_model.fit(
            x=model_inputs,
            y=local_target_batch,
            epochs=maxiter,
            verbose=0 if not verbose else 1,
            callbacks=[early_stopping],
            batch_size=1,
        )

        # Get optimized parameters
        optimized_params = temp_model.get_params()

        # Clean up the temporary model
        del temp_model

        return optimized_params

    def fit_stochastic(
        self,
        params: dict = None,  # initial params, optional
        num_epoch: int = 5,
        batch_size: int = 500,
        maxiter: int = 50,
        tol: float = 1e-3,
        step_size: float = 1e-2,
        verbose: bool = False,
        local: bool = True,
        plot: bool = False,
    ):
        if params is None:
            params = self.params if self.params is not None else self.init_params()

        self.converged = False
        while self.converged is False and num_epoch > 0:
            params = self.linear_estimator(params)
            pre_params = safe_deepcopy_params(params)
            num_epoch -= 1
            random_batches = get_random_indices_in_batches(
                self.num_coordinates, batch_size
            )
            image = self.image_tensor

            for index in tqdm(random_batches, desc="Fitting random batch"):
                current_params = safe_deepcopy_params(params)
                atoms_selected = np.zeros(self.num_coordinates, dtype=bool)
                atoms_selected[index] = True
                select_params = self.select_params(params, atoms_selected)

                # Calculate global prediction with full parameters
                self.model.set_params(params)
                global_prediction = self.predict(local=local)

                # Create a temporary model for local prediction calculation
                temp_model = self._create_model()
                temp_model.set_grid(self.x_grid, self.y_grid)
                temp_model.set_params(select_params)
                # Build with the correct input shape (grid shapes)
                temp_model.build()

                # Calculate local prediction using the temporary model
                local_prediction = temp_model.sum(local=local)
                local_residual = global_prediction - local_prediction
                local_target = ops.stop_gradient(image - local_residual)

                # Clean up the temporary model used for prediction
                del temp_model

                # Optimize the local batch using the dedicated function
                optimized_select_params = self._optimize_local_batch(
                    select_params, local_target, maxiter, tol, step_size, verbose
                )
                clipped_optimized_select_params = self.clip_params(
                    optimized_select_params
                )
                params = self.update_from_local_params(
                    current_params, clipped_optimized_select_params, atoms_selected
                )
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
        self.prediction = safe_convert_to_numpy(self.predict(params))

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
                if "pos_x" in self.params and "pos_y" in self.params:
                    params = self.params
                else:
                    params = self.init_params()
            else:
                params = self.init_params()

        pos_x = params["pos_x"]
        pos_y = params["pos_y"]
        coords = ops.stack([pos_y, pos_x])
        num_coordinates = coords.shape[1]

        # Generate Voronoi cell map
        if max_radius is None:
            max_radius = params["width"].max() * 3

        image = safe_convert_to_numpy(self.image)
        max_radius = safe_convert_to_numpy(max_radius)
        coords = safe_convert_to_numpy(coords)

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
            x_c, y_c = ops.meshgrid(
                ops.arange(x0, x1), ops.arange(y0, y1), indexing="xy"
            )
            x_c = safe_convert_to_numpy(x_c)
            y_c = safe_convert_to_numpy(y_c)

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

        # Parallel execution (using jax.vmap or plain Python for now)
        converged = False
        pre_params = safe_deepcopy_params(self.params)
        current_params = safe_deepcopy_params(self.params)
        if keras.backend.backend() == "jax":
            # conver the params to numpy array
            current_params = {
                key: safe_convert_to_numpy(value)
                for key, value in current_params.items()
            }
            pre_params = {
                key: safe_convert_to_numpy(value) for key, value in pre_params.items()
            }
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
                    pos_x_array = safe_convert_to_numpy(current_params["pos_x"]).copy()
                    pos_y_array = safe_convert_to_numpy(current_params["pos_y"]).copy()

                    for index, value in pos_x_updates.items():
                        pos_x_array[index] = value
                    for index, value in pos_y_updates.items():
                        pos_y_array[index] = value

                    current_params["pos_x"] = safe_convert_to_tensor(
                        pos_x_array, dtype="float32"
                    )
                    current_params["pos_y"] = safe_convert_to_tensor(
                        pos_y_array, dtype="float32"
                    )
            converged = self.convergence(current_params, pre_params, tol)
            pre_params = safe_deepcopy_params(current_params)
        self.params = current_params
        # self.model = self.predict(self.params, self.x_grid, self.y_grid)
        return self.params

    # parameters updates and convergence
    def convergence(self, params: dict, pre_params: dict, tol: float = 1e-2):
        """
        Checks if the parameters have converged within a specified tolerance.

        This function iterates over each parameter in `params` and its corresponding
        value in `pre_params` to determine if the change (update) is within a specified
        tolerance level, `tol`. For position parameters ('pos_x', 'pos_y'), it checks if
        the absolute update exceeds 1. For other parameters ('height', 'width', 'ratio', 'background'), it checks if the relative update exceeds `tol`.

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
            update = ops.abs(value - pre_params[key])

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
                rate = ops.abs(update / value_with_offset).mean()
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
            if "width" in params:
                select_params["width"] = params["width"]
            if "ratio" in params:
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
            # Always convert to tensor (JAX or NumPy) to avoid deleted array refs
            value = ops.convert_to_tensor(value)
            shared_value_list = ["background"]
            if self.same_width:
                shared_value_list += ["width", "ratio"]
            if key not in shared_value_list:
                if keras.backend.backend() == "jax":
                    if mask_local is None:
                        # Use .at[mask].set(value) for JAX, but assign the result!
                        params[key] = params[key].at[mask].set(value)
                    else:
                        params[key] = params[key].at[mask].set(value[mask_local])
                elif keras.backend.backend() == "tensorflow":
                    if mask_local is None:
                        params[key] = ops.scatter_update(params[key], mask, value)
                    else:
                        params[key] = ops.scatter_update(
                            params[key], mask, value[mask_local]
                        )
                elif keras.backend.backend() == "torch":
                    if mask_local is None:
                        params[key] = params[key].scatter(0, ops.where(mask)[0], value)
                    else:
                        params[key] = params[key].scatter(
                            0, ops.where(mask)[0], value[mask_local]
                        )
            else:
                weight = mask.sum() / self.num_coordinates
                update = value - params[key]
                params[key] = params[key] + update * weight
        return params

    def clip_params(self, params: dict):
        for key, value in params.items():
            if key == "pos_x":
                params[key] = ops.clip(value, 0, self.nx - 1)
            elif key == "pos_y":
                params[key] = ops.clip(value, 0, self.ny - 1)
            elif key == "height":
                params[key] = ops.clip(value, 0, self.image.max())
            elif key == "width":
                params[key] = ops.clip(value, 1, min(self.nx, self.ny) / 2)
            elif key == "ratio":
                params[key] = ops.clip(value, 0, 1)
            elif key == "background":
                params[key] = ops.clip(value, 0, np.max(self.image))
        return params

    def update_coordinates(self):
        # check the refined coorinates is different from the current coordinates
        refined_coordinates = np.stack(
            [self.params["pos_x"], self.params["pos_y"]], axis=1
        )
        if np.allclose(refined_coordinates, self.coordinates):
            logging.info("The coordinates have converged.")
            return self.coordinates
        else:
            # create & save the initial coordinates
            self.coordinates_history[self.coordinates_state] = self.coordinates.copy()
            # update the coordinates from the params refinement
            self.coordinates = np.stack(
                [self.params["pos_x"], self.params["pos_y"]], axis=1
            )
            self.coordinates_state += 1
            logging.info(
                f"The coordinates have been updated. Current state: {self.coordinates_state}"
            )
        return self.coordinates

    def update_region_analyzers(self):
        for index, region in self.regions.items:
            region.analyzer.peak_positions = self.coordinates[
                self.region_column_labels == index
            ]
            region.analyzer.atom_types = self.atom_types[
                self.region_column_labels == index
            ]
            logging.info(f"Updated region {index} coordinates for crystal analyzer.")

    # plot functions
    def calibrate(
        self,
        cif_file: str = None,
        a: float = None,
        b: float = None,
        region_index: int = 0,
        unit_cell: list = None,
    ):
        """
        Calibrate the pixel size based on the FFT of the lattice.
        """
        if self.coordinates.size == 0:
            logging.warning("No coordinates found. Please run find_peaks first.")
            self.find_peaks()
        column_mask = self.region_column_labels == region_index
        region_mask = self.regions.region_map == region_index
        crystal_analyzer = CrystalAnalyzer(
            image=self.image,
            dx=self.dx,
            peak_positions=self.coordinates[column_mask],
            atom_types=self.atom_types[column_mask],
            elements=self.elements,
            units="A",
            region_mask=region_mask,
        )
        if unit_cell is not None:
            crystal_analyzer.unit_cell = unit_cell
        if cif_file is not None:
            crystal_analyzer.read_cif(cif_file)

        a = a if a is not None else np.linalg.norm(crystal_analyzer.unit_cell.cell[0])
        b = b if b is not None else np.linalg.norm(crystal_analyzer.unit_cell.cell[1])
        _, vec_a_pixel, vec_b_pixel = crystal_analyzer.select_lattice_vectors(
            reciprocal=True
        )
        dx_a = a / np.linalg.norm(vec_a_pixel)
        dx_b = b / np.linalg.norm(vec_b_pixel)
        self.dx = (dx_a + dx_b) / 2
        logging.info(f"Calibrated pixel size: {self.dx} A")

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
        im = plt.imshow(self.prediction, cmap="gray", vmin=vmin, vmax=vmax)
        plt.gca().set_aspect("equal", adjustable="box")
        plt.colorbar(im, fraction=0.046, pad=0.04)
        plt.title("Model")
        plt.tight_layout()
        plt.subplot(1, 3, 3)
        im = plt.imshow(self.image - self.prediction, cmap="gray")
        plt.colorbar(im, fraction=0.046, pad=0.04)
        plt.gca().set_aspect("equal", adjustable="box")
        plt.title("Residual")
        plt.tight_layout()

    def plot_scs(
        self,
        layout="horizontal",
        per_element=False,
        s=1,
        save=False,
        has_units=True,
        half: str = None,
        figsize=(10, 5),
    ):
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
        pos_x = safe_convert_to_numpy(pos_x)
        pos_y = safe_convert_to_numpy(pos_y)
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
                plt.xlabel(r"X (A)")
                plt.ylabel(r"Y (A)")
                plt.gca().invert_yaxis()
                plt.title(f"{element}")
                if atom_type == self.atom_types.max():
                    if has_units:
                        cbar.set_label(r"SCS (A^2)")
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
            plt.xlabel(r"X (A)")
            plt.ylabel(r"Y (A)")
            plt.gca().invert_yaxis()
            plt.gca().set_aspect("equal", adjustable="box")
            if has_units:
                cbar.set_label(r"SCS (A^2)")
            else:
                cbar.set_label("Integrated intensities")
            plt.tight_layout()
        if save:
            plt.savefig("scs.svg")
            plt.savefig("scs.png", dpi=300)

    def plot_scs_voronoi(
        self,
        layout="horizontal",
        s=1,
        per_element=False,
        save=False,
        has_units=True,
        half: str = None,
        figsize=(10, 5),
    ):
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
                pos_x = safe_convert_to_numpy(pos_x)
                pos_y = safe_convert_to_numpy(pos_y) 
                im = plt.scatter(
                    pos_x, pos_y, c=self.voronoi_volume[mask], s=s, label=element
                )
                plt.gca().set_aspect("equal", adjustable="box")
                cbar = plt.colorbar(im, fraction=0.046, pad=0.04)
                # plt.axis("off")
                plt.xlim(0, self.nx * self.dx)
                plt.ylim(0, self.ny * self.dx)
                plt.xlabel(r"X (A)")
                plt.ylabel(r"Y (A)")
                plt.gca().invert_yaxis()
                plt.title(f"{element}")
                if atom_type == self.atom_types.max():
                    if has_units:
                        cbar.set_label(r"Voronoi SCS (A^2)")
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
                plt.title(r"Voronoi scs (A^2)")
            else:
                plt.title("Voronoi integrated intensities")
        plt.tight_layout()

        if save:
            plt.savefig("voronoi_scs.svg")
            plt.savefig("voronoi_scs.png", dpi=300)

    def estimate_atom_counts_with_gmm(
        self,
        max_components: int = 5,
        scoring_method: str = "icl",
        initialization_method: str = "middle",
        plot_results: bool = True,
        per_element: bool = True,
        save_results: bool = False,
        interactive_selection: bool = True,
        use_first_local_minimum: bool = True,
    ):
        """Estimate atom counts using Gaussian Mixture Model on cross-section histograms.
        
        This method applies GMM to the refined cross-section histogram to statistically
        determine the number of atoms in each atomic column based on scattering cross-sections.
        
        Args:
            max_components: Maximum number of Gaussian components to test
            scoring_method: Information criterion for model selection ('icl', 'aic', 'bic')
            initialization_method: Method for initializing GMM means
            plot_results: Whether to plot the GMM fitting results
            per_element: Whether to fit GMM separately for each element type
            save_results: Whether to save plots and results
            interactive_selection: Whether to allow interactive component selection
            use_first_local_minimum: Whether to use first local minimum instead of global
            
        Returns:
            dict: Dictionary containing GMM results and atom count estimates
        """
        if not hasattr(self, 'params') or self.params is None:
            raise ValueError("Please run fitting first to obtain refined cross-sections")
        
        from qem.gaussian_mixture_model import GaussianMixtureModel
        
        # Get refined cross-sections (volumes)
        cross_sections = self.volume.reshape(-1, 1)  # Reshape for GMM input
        
        gmm_results = {}
        atom_count_estimates = {}
        
        if per_element:
            # Fit GMM separately for each element type
            for atom_type in np.unique(self.atom_types):
                element_name = self.elements[atom_type]
                mask = self.atom_types == atom_type
                element_cross_sections = cross_sections[mask]
                
                if len(element_cross_sections) < 10:  # Skip if too few data points
                    logging.warning(f"Skipping GMM for {element_name}: insufficient data points")
                    continue
                
                # Initialize and fit GMM
                gmm = GaussianMixtureModel(element_cross_sections)
                gmm.fit_gaussian_mixture_model(
                    num_components=max_components,
                    scoring_methods=[scoring_method, "nllh"],
                    initialization_method=initialization_method,
                    use_first_local_minimum=use_first_local_minimum,
                )
                
                # Plot results and allow component selection
                if plot_results:
                    selected_components = gmm.plot_interactive_gmm_selection(
                        element_cross_sections, element_name, 
                        save_results, interactive_selection
                    )
                else:
                    # Use recommendation if no plotting
                    selected_components = gmm.get_optimal_components("recommendation")
                
                # Get component parameters using user-selected components
                component_idx = selected_components - 1
                weights = gmm.fit_result.weight[component_idx]
                means = gmm.fit_result.mean[component_idx]
                widths = gmm.fit_result.width[component_idx]
                
                # Estimate atom counts based on component means
                # Assume components correspond to different atom counts (1, 2, 3, etc.)
                sorted_indices = np.argsort(means.flatten())
                atom_counts = np.arange(1, len(sorted_indices) + 1)
                
                # Assign atom counts to each atomic column
                column_assignments = gmm.fit_result.idxComponentOfScs(component_idx)
                estimated_counts = atom_counts[sorted_indices][column_assignments]
                
                gmm_results[element_name] = {
                    'gmm_model': gmm,
                    'selected_components': selected_components,  # Store user selection
                    'recommended_components': gmm.recommended_components,  # Store recommendation
                    'weights': weights,
                    'means': means[sorted_indices],
                    'widths': widths[sorted_indices],
                    'scores': gmm.fit_result.score,
                }
                
                atom_count_estimates[element_name] = estimated_counts
                
        else:
            # Fit GMM to all cross-sections together
            gmm = GaussianMixtureModel(cross_sections)
            gmm.fit_gaussian_mixture_model(
                num_components=max_components,
                scoring_methods=[scoring_method, "nllh"],
                initialization_method=initialization_method,
                use_first_local_minimum=use_first_local_minimum,
            )
            
            # Plot results and allow component selection
            if plot_results:
                selected_components = gmm.plot_interactive_gmm_selection(
                    cross_sections, 'all_elements', 
                    save_results, interactive_selection
                )
            else:
                selected_components = gmm.get_optimal_components("recommendation")
            
            component_idx = selected_components - 1
            
            weights = gmm.fit_result.weight[component_idx]
            means = gmm.fit_result.mean[component_idx]
            widths = gmm.fit_result.width[component_idx]
            
            sorted_indices = np.argsort(means.flatten())
            atom_counts = np.arange(1, len(sorted_indices) + 1)
            
            column_assignments = gmm.fit_result.idxComponentOfScs(component_idx)
            estimated_counts = atom_counts[sorted_indices][column_assignments]
            
            gmm_results['all_elements'] = {
                'gmm_model': gmm,
                'selected_components': selected_components,  # Store user selection
                'recommended_components': gmm.recommended_components,  # Store recommendation
                'weights': weights,
                'means': means[sorted_indices],
                'widths': widths[sorted_indices],
                'scores': gmm.fit_result.score,
            }
            
            atom_count_estimates['all_elements'] = estimated_counts
        
        # Store results as instance attributes
        self.gmm_results = gmm_results
        self.atom_count_estimates = atom_count_estimates
        
        return {
            'gmm_results': gmm_results,
            'atom_count_estimates': atom_count_estimates,
        }

    def _plot_gmm_results(self, cross_sections, gmm_model, element_name, save_results=False):
        """Legacy method - redirects to GMM module plotting for compatibility."""
        return gmm_model.plot_interactive_gmm_selection(
            cross_sections, element_name, save_results, interactive_selection=False
        )

    def plot_scs_histogram(self, save=False, has_units=True):
        """Plot histogram of refined scattering cross-sections."""
        plt.figure()
        for atom_type in np.unique(self.atom_types):
            mask = self.atom_types == atom_type
            element = self.elements[atom_type]
            plt.hist(self.volume[mask], bins=100, alpha=0.5, label=element)
        plt.legend()
        if has_units:
            plt.xlabel(r"Refined SCS (A^2)")
        else:
            plt.xlabel("Integrated intensities")
        plt.ylabel("Frequency")
        plt.title("Histogram of QEM refined SCS")
        if save:
            plt.savefig("scs_histogram.svg")
            plt.savefig("scs_histogram.png", dpi=300)
    
    def plot_atom_count_map(self, element_name=None, save=False, figsize=(12, 8)):
        """Plot spatial map of estimated atom counts with proper colorbar.
        
        Args:
            element_name: Specific element to plot, or None for all elements
            save: Whether to save the plot
            figsize: Figure size tuple
        """
        if not hasattr(self, 'atom_count_estimates'):
            raise ValueError("Please run estimate_atom_counts_with_gmm first")
        
        fig, ax = plt.subplots(figsize=figsize)
        
        if element_name is None:
            # Plot all elements with different symbols/colors
            all_counts = []
            all_pos_x = []
            all_pos_y = []
            
            for atom_type in np.unique(self.atom_types):
                element = self.elements[atom_type]
                if element in self.atom_count_estimates:
                    mask = self.atom_types == atom_type
                    counts = self.atom_count_estimates[element]
                    
                    pos_x = self.params["pos_x"][mask] * self.dx
                    pos_y = self.params["pos_y"][mask] * self.dx
                    
                    pos_x_np = safe_convert_to_numpy(pos_x)
                    pos_y_np = safe_convert_to_numpy(pos_y)
                    
                    all_counts.extend(counts)
                    all_pos_x.extend(pos_x_np)
                    all_pos_y.extend(pos_y_np)
                    
                    # Plot each element with different marker
                    markers = ['o', 's', '^', 'D', 'v', '<', '>', 'p', 'h', 'H']
                    marker = markers[atom_type % len(markers)]
                    
                    scatter = ax.scatter(
                        pos_x_np, pos_y_np,
                        c=counts, s=80, alpha=0.8, 
                        marker=marker, label=f'{element}',
                        cmap='viridis', vmin=1, vmax=max(all_counts) if all_counts else 5
                    )
            
            # Create colorbar for all elements
            if all_counts:
                cbar = plt.colorbar(scatter, ax=ax, shrink=0.8)
                cbar.set_label('Number of Atoms', fontsize=14, fontweight='bold')
                # Set integer ticks on colorbar
                max_count = max(all_counts)
                cbar.set_ticks(range(1, max_count + 1))
                
            ax.legend(loc='upper right', bbox_to_anchor=(1.15, 1))
            
        else:
            # Plot specific element
            if element_name not in self.atom_count_estimates:
                raise ValueError(f"No atom count estimates found for {element_name}")
            
            atom_type = list(self.elements).index(element_name)
            mask = self.atom_types == atom_type
            counts = self.atom_count_estimates[element_name]
            
            pos_x = self.params["pos_x"][mask] * self.dx
            pos_y = self.params["pos_y"][mask] * self.dx
            
            pos_x_np = safe_convert_to_numpy(pos_x)
            pos_y_np = safe_convert_to_numpy(pos_y)
            
            scatter = ax.scatter(
                pos_x_np, pos_y_np,
                c=counts, s=100, alpha=0.8, cmap='viridis',
                edgecolors='black', linewidth=0.5
            )
            
            # Create colorbar with proper title
            cbar = plt.colorbar(scatter, ax=ax, shrink=0.8)
            cbar.set_label('Number of Atoms', fontsize=14, fontweight='bold')
            # Set integer ticks on colorbar
            unique_counts = np.unique(counts)
            cbar.set_ticks(unique_counts)
            
            ax.set_title(f'Atom Count Map - {element_name}', fontsize=16, fontweight='bold')
        
        ax.set_xlabel('X (Å)', fontsize=12)
        ax.set_ylabel('Y (Å)', fontsize=12)
        if element_name is None:
            ax.set_title('Spatial Map of Estimated Atom Counts', fontsize=16, fontweight='bold')
        
        ax.set_aspect('equal', adjustable='box')
        ax.invert_yaxis()
        ax.grid(True, alpha=0.3)
        
        # Add summary text
        if hasattr(self, 'gmm_results'):
            summary_info = []
            for elem, results in self.gmm_results.items():
                if 'selected_components' in results:
                    selected = results['selected_components']
                    recommended = results.get('recommended_components', 'N/A')
                    summary_info.append(f"{elem}: {selected} components (rec: {recommended})")
            
            if summary_info:
                summary_text = "GMM Selection: " + ", ".join(summary_info)
                ax.text(0.02, 0.02, summary_text, transform=ax.transAxes,
                       bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
                       fontsize=10, verticalalignment='bottom')
        
        plt.tight_layout()
        
        if save:
            filename = f'atom_count_map_{element_name or "all"}.png'
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            logging.info(f"Atom count map saved as {filename}")
        
        plt.show()

    def integrate_gmm_with_crystal_analyzer(self, region_index: int = 0):
        """Integrate GMM atom count estimates with crystal analyzer atomic model.
        
        This method combines the statistical atom counting from GMM with the 
        crystal structure analysis to create a 3D atomic model with realistic
        atom counts in each column. Z-spacing is automatically determined from
        the supercell structure.
        
        Args:
            region_index: Index of the region to update (default: 0)
            
        Returns:
            Updated crystal analyzer object with GMM-based atom counts
        """
        if not hasattr(self, 'atom_count_estimates'):
            raise ValueError("Please run estimate_atom_counts_with_gmm() first")
            
        if region_index not in self.regions.keys:
            raise ValueError(f"Region {region_index} not found in regions")
            
        # Get the crystal analyzer for this region
        region = self.regions[region_index]
        if not hasattr(region, 'analyzer') or region.analyzer is None:
            raise ValueError(f"No crystal analyzer found for region {region_index}. "
                           "Please run map_lattice() first.")
                           
        crystal_analyzer = region.analyzer
        
        # Filter atom count estimates for columns in this region
        column_mask = self.region_column_labels == region_index
        region_atom_counts = {}
        
        for element_name, all_counts in self.atom_count_estimates.items():
            if element_name == 'all_elements':
                # Handle case where GMM was fit to all elements together
                region_atom_counts[element_name] = all_counts[column_mask]
            else:
                # Handle per-element GMM fitting
                element_columns = column_mask & (self.atom_types == self.elements.index(element_name))
                if element_columns.any():
                    region_atom_counts[element_name] = all_counts
        
        # Update the crystal analyzer with GMM results
        updated_columns = crystal_analyzer.update_atoms_from_gmm(
            region_atom_counts
        )
        
        # Update the region's columns
        region.columns = updated_columns
        
        return crystal_analyzer
        
    def update_all_regions_with_gmm(self):
        """Update all regions with GMM atom count estimates.
        
        Z-spacing is automatically determined from the supercell structure.
            
        Returns:
            Dictionary mapping region indices to updated crystal analyzers
        """
        updated_analyzers = {}
        
        for region_index in self.regions.keys:
            try:
                analyzer = self.integrate_gmm_with_crystal_analyzer(region_index)
                updated_analyzers[region_index] = analyzer
                logging.info(f"Successfully updated region {region_index} with GMM results")
            except Exception as e:
                logging.warning(f"Could not update region {region_index}: {str(e)}")
                
        return updated_analyzers
        
    def export_gmm_updated_structure(self, region_index: int = 0, filename: str = None):
        """Export the GMM-updated atomic structure to various formats.
        
        Args:
            region_index: Index of the region to export
            filename: Output filename (without extension)
            
        Returns:
            ASE Atoms object of the updated structure
        """
        if region_index not in self.regions.keys:
            raise ValueError(f"Region {region_index} not found")
            
        region = self.regions[region_index]
        if not hasattr(region, 'columns') or region.columns is None:
            raise ValueError(f"No atomic columns found for region {region_index}. "
                           "Please run integrate_gmm_with_crystal_analyzer() first.")
        
        # Get the updated lattice
        updated_lattice = region.columns.lattice
        
        if filename:
            # Export to different formats
            from ase.io import write
            write(f"{filename}.xyz", updated_lattice)
            write(f"{filename}.cif", updated_lattice) 
            logging.info(f"Exported GMM-updated structure to {filename}.xyz and {filename}.cif")
            
        return updated_lattice

    def plot_region(self):
        plt.figure()
        plt.imshow(self.image, cmap="gray")
        plt.imshow(self.regions.region_map, alpha=0.5)
        scalebar = self.scalebar
        plt.gca().add_artist(scalebar)
        plt.axis("off")
        cbar = plt.colorbar()
        cbar.set_ticks(np.arange(self.regions.num_regions))  # type: ignore
        plt.title("Region Map")

    @property
    def atom_types(self):
        if len(self._atom_types) == 0 or self._atom_types is None:
            self._atom_types = np.zeros(self.num_coordinates, dtype=int)
        return self._atom_types

    @atom_types.setter
    def atom_types(self, atom_types: np.ndarray):
        self._atom_types = atom_types

    @property
    def coordinates(self):
        return self._coordinates

    @coordinates.setter
    def coordinates(self, coordinates: np.ndarray):
        self._coordinates = coordinates

    @property
    def num_coordinates(self):
        return len(self._coordinates) if len(self._coordinates.shape) > 0 else 0

    @property
    def num_atom_types(self):
        return len(np.unique(self._atom_types)) if len(self._atom_types) > 0 else 0

    @property
    def region_column_labels(self):
        return self.regions.region_map[
            self.coordinates[:, 1].astype(int), self.coordinates[:, 0].astype(int)
        ]

    @property
    def voronoi_volume(self):
        return self._voronoi_volume
