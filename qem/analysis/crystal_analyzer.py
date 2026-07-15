import copy

# from shapely.affinity import scale
import logging
import re

import matplotlib.pyplot as plt
import numpy as np
from ase import Atoms
from ase.io import read
from ase.neighborlist import neighbor_list
from matplotlib_scalebar.scalebar import ScaleBar
from scipy.ndimage import gaussian_filter
from scipy.spatial import ConvexHull
from shapely.geometry import Point, Polygon
from skimage.feature import peak_local_max
from skimage.transform import rescale

from qem.analysis.atomic_column import AtomicColumns
from qem.analysis.crystal_analyzer_plot import CrystalAnalyzerPlotMixin
from qem.viz.select import GetAtomSelection, InteractivePlot

logging.basicConfig(level=logging.INFO)


class CrystalAnalyzer(CrystalAnalyzerPlotMixin):
    def __init__(
        self,
        image: np.ndarray,
        dx: float,
        peak_positions: np.ndarray,
        atom_types: np.ndarray,
        elements: list[str],
        units: str = "A",
        region_mask: np.ndarray | None = None,
    ):
        self.image = image
        self.dx = dx
        self.units = units
        self.peak_positions = peak_positions
        self.atom_types = atom_types
        self.elements = elements
        self.unit_cell: Atoms = None
        self.origin = np.array([0, 0])
        self.a_vector = {"perfect": np.array([1, 0]), "affine": np.array([1, 0])}
        self.b_vector = {"perfect": np.array([1, 0]), "affine": np.array([1, 0])}
        self._min_distances = None
        self.atomic_columns = None
        if region_mask is None:
            region_mask = np.ones(image.shape, dtype=bool)
        self.region_mask = region_mask
        self.rotation_matrix = None
        self.affine_matrix = None
        self.unit_cell_transformed = {"perfect": None, "affine": None}
        self._origin_offsets = {"perfect": {}, "affine": {}, "adaptive": {}}
        self.lattice: Atoms = None
        self.lattice_ref: Atoms = None

    #region operations################


    def region_of_interest(self, sigma: float = 0.8):
        # use the current coordinates to filter the peak_positions
        # create a mask for the current coordinates with the size of input image, area within 3 sigma of the current coordinates are masked to true
        region_of_interest = np.zeros(self.image.shape, dtype=bool)
        for i in range(len(self.peak_positions)):
            x, y = self.peak_positions[i]
            region_of_interest[
                int(max(y - 3 * sigma/self.dx, 0)): int(min(y + 3 * sigma/self.dx, self.ny)),
                int(max(x - 3 * sigma/self.dx, 0)): int(min(x + 3 * sigma/self.dx, self.nx)),
            ] = True
        return region_of_interest

    def select_region(self, peak_positions: np.ndarray, atom_types: np.ndarray):
        """
        Select a region of interest from the image.

        Args:
            peak_positions (np.ndarray): Array of peak positions.
            atom_types (np.ndarray): Array of atom types.

        Returns:
            tuple: Tuple containing peak positions, atom types, and region mask.
        """
        atom_select = GetAtomSelection(
            image=self.image, atom_positions=peak_positions, invert_selection=False
        )
        # hold on until the atom_positions_selected is not empty
        while atom_select.atom_positions_selected.size == 0:
            plt.pause(0.1)
        peak_positions_selected = np.array(atom_select.atom_positions_selected)
        is_peak_selected = np.isin(peak_positions, peak_positions_selected).all(axis=1)
        atom_types_selected = atom_types[is_peak_selected]
        region_mask = atom_select.region_mask
        return peak_positions_selected, atom_types_selected, region_mask


    # I/O ################
    def read_cif(self, cif_file_path: str):
        atoms = read(cif_file_path)
        if not isinstance(atoms, Atoms):
            raise TypeError(f"CIF read did not yield an ASE Atoms object: {cif_file_path}")
        mask = [atom.symbol in self.elements for atom in atoms]  # type: ignore
        logging.info(f"Reading CIF file: {cif_file_path}. The elements selected are {self.elements} and the elements in the CIF file are {atoms.symbols}")
        self.unit_cell = atoms[mask]
        return atoms

    def get_unitcell_elements(self):
        """
        Get the elements present in the unit cell.

        Returns:
        - element_symbols: The symbols of the elements present in the unit cell.
        """
        if not isinstance(self.unit_cell, Atoms):
            raise RuntimeError(
                "unit_cell is not set; call read_cif() or map_lattice() first."
            )
        formula = str(self.unit_cell.symbols)
        # seperate the element symbols from the composition, split by numbers
        elements = re.findall(r"[A-Z][a-z]*", formula)
        return elements

    @staticmethod
    def is_element_in_unit_cell(unitcell, element_symbol: str) -> list:
        """
        Checks if the given element is present in each site of the unit cell.

        Args:
        - unitcell (Structure): The unit cell to check.
        - element_symbol (str): The symbol of the element to check for (e.g., "O" for oxygen).

        Returns:
        - A list of booleans, where each boolean indicates whether the target element
        is present in the corresponding site of the unit cell.
        """
        mask = []
        for site in unitcell:
            mask.append(site.symbol == element_symbol)
        return mask

    # lattice mapping ################
    def get_atomic_columns(
        self,
        tol: float = 0,
        reciprocal: bool = True,
        sigma: float = 0.8,
    ):
        self.select_lattice_vectors(reciprocal=reciprocal)

        # get the supercell lattice in 3d and project to 2d
        lattice_3d, lattice_3d_ref = self.get_lattice_3d(sigma)
        if not isinstance(lattice_3d, Atoms) or not isinstance(lattice_3d_ref, Atoms):
            raise TypeError("get_lattice_3d must return ASE Atoms objects")
        ref = {
            "origin": self.origin,
            "vector_a": self.a_vector["perfect"],
            "vector_b": self.b_vector["perfect"],
        }
        self.atomic_columns = AtomicColumns(
            lattice_3d, lattice_3d_ref, self.elements, tol, self.dx, ref
        )
        self.atom_types = self.atomic_columns.atom_types
        self.peak_positions = self.atomic_columns.positions_pixel
        return self.atomic_columns

    def align_unit_cell_to_image(
        self, ref: tuple = None, plot: bool = True, mode: str = "affine"
    ):
        """
        Transforms unit cell coordinates to the image coordinate system,
        aligning them with detected atomic peak positions. Optionally visualizes the
        transformation, including the origin, lattice vectors, and positions of atoms
        within the unit cell.

        Parameters:
        - plot: A boolean indicating whether to plot the mapping and unit cell visualization.
        - ref: A tuple containing the origin and lattice vectors of the reference unit cell.
        - mode: A string indicating the transformation mode. Either 'affine' or 'perfect'. Affine means affine transformation is applied to the unit cell based on the lattice vectors selected on the image. In practice, 'affine' is more robust to match with atomic column on the image, because the unit cell and image can have different angle and its a and b lattice vectors may scale differently due to scanning distortion or simply a wrong pixel size. In contrast, 'perfect' means the transformation is based on the rotation of the angle between the lattice a-vector and the x-axis on the image and the pixel size from the image. You should use 'perfect' if you want the absolute mapping of the unit cell to the image if you have an 'ideal' microscope image with no distortion and correct pixel size.

        Returns:
        - unitcell_transformed: The transformed coordinates of the unit cell.
        """
        if ref is not None:
            origin, a, b = ref
        else:
            origin = self.origin
            a = self.a_vector["affine"]
            b = self.b_vector["affine"]

        if not isinstance(self.unit_cell, Atoms):
            raise RuntimeError(
                "unit_cell is not set; call read_cif() or map_lattice() first."
            )
        if mode not in ("affine", "perfect"):
            raise ValueError(f"mode must be 'affine' or 'perfect', got {mode!r}")

        if self.unit_cell_transformed[mode] is None:
            unit_cell = copy.deepcopy(self.unit_cell)
            new_xy = np.array([a, b]).T
            old_xy = np.array([unit_cell.cell[0][:2], unit_cell.cell[1][:2]]).T
            coords_xy = unit_cell.positions[:, :2]  # type: ignore
            if mode == "perfect":
                if self.rotation_matrix is None:
                    # get the angle between a and x-axis
                    angle_a = np.arctan2(a[1], a[0]) - np.arctan2(
                        unit_cell.cell[0][1], unit_cell.cell[0][0]
                    )
                    angle_b = np.arctan2(b[1], b[0]) - np.arctan2(
                        unit_cell.cell[1][1], unit_cell.cell[1][0]
                    )

                    if (
                        abs(angle_a - angle_b) > np.pi / 2
                    ):  # consider the case when b vector is flipped
                        self.rotation_matrix = np.array(
                            [
                                [np.cos(angle_a), np.sin(angle_a)],
                                [np.sin(angle_a), -np.cos(angle_a)],
                            ]
                        )
                    else:  # normal case when a and b are in the same order as in the unit cell
                        self.rotation_matrix = np.array(
                            [
                                [np.cos(angle_a), -np.sin(angle_a)],
                                [np.sin(angle_a), np.cos(angle_a)],
                            ]
                        )
                new_coords_xy = (coords_xy @ self.rotation_matrix) / self.dx
                self.a_vector["perfect"] = (
                    unit_cell.cell[0][:2] @ self.rotation_matrix / self.dx
                )
                self.b_vector["perfect"] = (
                    unit_cell.cell[1][:2] @ self.rotation_matrix / self.dx
                )
                logging.info(
                    f"Perfect a: {self.a_vector['perfect']} pixel, Perfect b: {self.b_vector['perfect']} pixel by rotation of unit cell and scaling with pixel size."
                )
            else:  # affine transformation
                if self.affine_matrix is None:
                    self.affine_matrix = new_xy @ np.linalg.inv(old_xy)
                new_coords_xy = coords_xy @ self.affine_matrix
            positions = np.hstack(
                [new_coords_xy * self.dx, unit_cell.positions[:, 2].reshape(-1, 1)]
            )  # type: ignore
            unit_cell.set_positions(positions)
            self.unit_cell_transformed[mode] = unit_cell
        shifted_unit_cell = self.unit_cell_transformed[mode].copy()
        shifted_unit_cell.positions[:, :2] += origin * self.dx
        if plot:
            self.plot_unitcell(mode=mode)
        return shifted_unit_cell

    def get_lattice_3d(self, sigma: float = 0.8):
        """
        Generate a supercell lattice based on the given lattice vectors and limits.

        Parameters:
        - sigma: The sigma value around the intial peak positions to consider for the supercell lattice.

        Returns:
        - supercell_lattice: The supercell lattice.
        """
        supercell = Atoms()
        supercell_ref = Atoms()
        shift_origin_adaptive = self.get_origin_offset("adaptive")
        # shift_origin_affine = self.get_origin_offset('affine')
        shift_origin_perfect = self.get_origin_offset("perfect")

        for translation, new_origin in shift_origin_adaptive.items():
            unitcell = self.align_unit_cell_to_image(
                ref=(new_origin, self.a_vector["affine"], self.b_vector["affine"]), plot=False
            )
            new_origin_ref = shift_origin_perfect[translation]
            unitcell_ref = self.align_unit_cell_to_image(
                ref=(new_origin_ref, self.a_vector["perfect"], self.b_vector["perfect"]),
                plot=False,
                mode="perfect",
            )
            supercell.extend(unitcell)
            supercell_ref.extend(unitcell_ref)

        is_within_image_bounds = (supercell.positions[:, :2] / self.dx >= 0).all(
            axis=1
        ) & (supercell.positions[:, [1, 0]] / self.dx < self.image.shape).all(axis=1)
        supercell = supercell[is_within_image_bounds]
        supercell_ref = supercell_ref[is_within_image_bounds]
        supercell.set_cell(
            np.array(
                [
                    [self.nx * self.dx, 0, 0],
                    [0, self.ny * self.dx, 0],
                    self.unit_cell.cell[2],
                ]
            )
        )
        supercell_ref.set_cell(
            np.array(
                [
                    [self.nx * self.dx, 0, 0],
                    [0, self.ny * self.dx, 0],
                    self.unit_cell.cell[2],
                ]
            )
        )

        valid_region_mask = self.region_of_interest(sigma / self.dx) & self.region_mask

        peak_region_filter = np.ones(supercell.get_global_number_of_atoms(), dtype=bool)
        for i in range(supercell.get_global_number_of_atoms()):
            x, y = supercell.positions[i, :2] / self.dx
            if not valid_region_mask[int(y), int(x)]:
                peak_region_filter[i] = False

        # supercell = supercell[peak_region_filter]
        # supercell_ref = supercell_ref[peak_region_filter]
        self.lattice = supercell[peak_region_filter]
        self.lattice_ref = supercell_ref[peak_region_filter]
        return self.lattice, self.lattice_ref

    def get_closest_peak(
        self,
        candidate_peaks: np.ndarray,
        target_peak: np.ndarray,
        min_distance: float = 1.5,
    ):
        """
        Find the closest 2D peak to the given atom position.
        """
        distance = np.linalg.norm(candidate_peaks - target_peak, axis=1)
        if distance.min() < min_distance:
            closest_peak = candidate_peaks[np.argmin(distance)]
            return closest_peak
        else:
            return None

    def get_origin_offset(
        self, mode: str = "adaptive"
    ):
        if mode not in ("perfect", "affine", "adaptive"):
            raise ValueError(
                f"mode must be 'perfect', 'affine' or 'adaptive', got {mode!r}"
            )
        if not self._origin_offsets[mode]:
            self._calc_origin_offsets()
        return self._origin_offsets[mode]

    def _calc_origin_offsets(self):
        """Compute per-cell origin offsets that snap the tiled unit cell to peaks.

        Populates ``self._origin_offsets`` for the ``perfect``, ``affine`` and
        ``adaptive`` modes. The algorithm:

        1. Align the unit cell to the image in ``perfect`` mode to get a starting
           origin and lattice vectors.
        2. Bound the number of unit-cell repeats (``a_limit``, ``b_limit``) needed
           to tile the whole image from that origin.
        3. For every integer ``(i, j)`` translation of the unit cell, record the
           expected origin (``perfect``/``affine``) and, for ``adaptive``, snap it
           to the nearest detected peak within half the peak spacing — falling
           back to the expected origin when no peak is close enough.

        The offsets are cached; :meth:`get_origin_offset` triggers this lazily.
        """
        # Perfect mapping of the unit cell to the image gives the base origin.
        self.align_unit_cell_to_image(plot=False, mode="perfect")

        a_limit = 5 * np.ceil(
            max(self.nx - self.origin[0], self.origin[0])
            * self.dx
            / self.unit_cell.get_cell()[0][0]  # type: ignore
        ).astype(int)
        b_limit = 5 * np.ceil(
            max(self.ny - self.origin[1], self.origin[1])
            * self.dx
            / self.unit_cell.get_cell()[1][1]  # type: ignore
        ).astype(int)

        # generate a meshgrid
        a_axis_mesh, b_axis_mesh = np.meshgrid(
            np.arange(-a_limit, a_limit + 1), np.arange(-b_limit, b_limit + 1)
        )
        a_axis_distance_mesh = a_axis_mesh * np.linalg.norm(self.a_vector["perfect"])
        b_axis_distance_mesh = b_axis_mesh * np.linalg.norm(self.b_vector["perfect"])
        # compute the distance in such meshgrid
        distance_mesh = np.sqrt(a_axis_distance_mesh**2 + b_axis_distance_mesh**2)
        # apply the sort to the a_axis_mesh and b_axis_mesh
        a_axis_mesh_sorted = a_axis_mesh.flatten()[np.argsort(distance_mesh, axis=None)]
        b_axis_mesh_sorted = b_axis_mesh.flatten()[np.argsort(distance_mesh, axis=None)]
        order_mesh = np.array([a_axis_mesh_sorted, b_axis_mesh_sorted]).T
        neighborhood_radius = np.linalg.norm(
            self.a_vector["affine"] + self.b_vector["affine"]
        ).astype(int)

        # Calculate the convex hull of the points
        hull = ConvexHull(self.peak_positions)
        # Get the vertices of the convex hull to form the boundary polygon
        hull_points = self.peak_positions[hull.vertices]
        # Create a Shapely Polygon from the convex hull points
        polygon = Polygon(hull_points)
        # Expand the polygon outward by the threshold distance (neighborhood_radius pixels)
        expanded_polygon = polygon.buffer(neighborhood_radius)

        # Find the closest peak to the origin to correct for drift
        origin_offsets = {"perfect": {}, "affine": {}, "adaptive": {}}
        origin_offsets["adaptive"][(0, 0)] = self.origin
        origin_offsets["affine"][(0, 0)] = self.origin
        origin_offsets["perfect"][(0, 0)] = self.origin

        for a_shift, b_shift in order_mesh[1:]:
            shifted_origin_perfect = (
                self.origin
                + self.a_vector["perfect"] * a_shift
                + self.b_vector["perfect"] * b_shift
            )
            shifted_origin_affine = (
                self.origin
                + self.a_vector["affine"] * a_shift
                + self.b_vector["affine"] * b_shift
            )

            # Check if the shifted origin is within the expanded area
            is_within_expanded_area = expanded_polygon.contains(
                Point(shifted_origin_affine)
            )
            if not is_within_expanded_area:
                continue

            # if the shifted origin is within the region of interest, add it to the dictionary
            origin_offsets["perfect"][(a_shift, b_shift)] = shifted_origin_perfect
            origin_offsets["affine"][(a_shift, b_shift)] = shifted_origin_affine

            # find the closet point of the a_shift and b_shift in the current shift_orgin
            distance = np.linalg.norm(
                np.array(list(origin_offsets["adaptive"].values()))
                - shifted_origin_affine,
                axis=1,
            )
            neighbor_distance_idx = np.where(distance < 2 * neighborhood_radius)[0]
            selected_keys = [
                list(origin_offsets["adaptive"].keys())[idx]
                for idx in neighbor_distance_idx
            ]
            expect_origin_list = []
            # find the difference of a_shift and b_shift with the selected_keys
            for selected_key in selected_keys:
                a_shift_diff = a_shift - selected_key[0]
                b_shift_diff = b_shift - selected_key[1]
                expect_origin = (
                    origin_offsets["adaptive"][selected_key]
                    + self.a_vector["affine"] * a_shift_diff
                    + self.b_vector["affine"] * b_shift_diff
                )
                expect_origin_list.append(expect_origin)

            expect_origin_avg = np.array(expect_origin_list).mean(axis=0)

            # check if the expect_origin_avg is close to any of the peak_positions within a threshold
            distance_ref = (
                min(np.array([d for d in self.min_distances.values()]) / self.dx) / 2
            )

            closest_peak = self.get_closest_peak(
                self.peak_positions, expect_origin_avg, distance_ref
            )
            if closest_peak is not None:
                origin_offsets["adaptive"][(a_shift, b_shift)] = closest_peak
            else:
                origin_offsets["adaptive"][(a_shift, b_shift)] = expect_origin_avg
        self._origin_offsets = origin_offsets
        return origin_offsets

    # strain mapping #######
    def get_strain(self, cut_off: float = 5.0):
        """
        Get the strain of the atomic columns based on the given cut-off radius.

        Args:
        - cut_off (int): The cut-off radius.

        Returns:
        - The strain of the atomic columns.
        """
        return self.atomic_columns.get_strain(float(cut_off))

    def select_lattice_vectors(self, tolerance: int = 10, reciprocal: bool = False):
        """
        Choose the lattice vectors based on the given tolerance.

        Args:
        - tolerance (int): The tolerance value.

        Returns:
        - The selected origin, a, and b vectors.
        """
        real_plot = InteractivePlot(
            image=self.image,
            peaks_locations=self.peak_positions,
            atom_types=self.atom_types,
            dx=self.dx,
            units=self.units,
        )

        real_origin, real_a, real_b = real_plot.select_vectors(tolerance=tolerance)  # type: ignore

        if reciprocal:
            image_filtered = gaussian_filter(self.image, 1)
            hanning_window = np.outer(
                np.hanning(self.image.shape[0]), np.hanning(self.image.shape[1])
            )
            image_filtered *= hanning_window
            fft_image = np.abs(np.fft.fftshift(np.fft.fft2(image_filtered)))
            fft_log = np.log(fft_image)
            fft_dx = 1 / (self.dx * self.image.shape[1])
            fft_dy = 1 / (self.dx * self.image.shape[0])
            fft_tolerance_x = np.ceil(1 / np.linalg.norm(real_a * self.dx) / fft_dx / 2).astype(int)
            fft_tolerance_y = np.ceil(1 / np.linalg.norm(real_b * self.dx) / fft_dy / 2).astype(int)

            scale_y = fft_dy/fft_dx
            if scale_y < 1:
                fft_log_rescaled = rescale(
                    fft_log, (1, 1/scale_y), anti_aliasing=True
                )
                fft_peaks = peak_local_max(
                    fft_log_rescaled, min_distance=fft_tolerance_x, num_peaks=30
                )
                # fft_peaks[:, 1] = fft_peaks[:, 1] * scale_y
            else:
                fft_log_rescaled = rescale(fft_log, (scale_y, 1), anti_aliasing=True)
                fft_peaks = peak_local_max(
                    fft_log_rescaled, min_distance=fft_tolerance_y, num_peaks=30
                )
                # fft_peaks[:, 0] = fft_peaks[:, 0] / scale_y

            fft_peaks = fft_peaks[:, [1, 0]].astype(float)
            zoom = 3
            fft_plot = InteractivePlot(
                fft_log_rescaled,
                fft_peaks,
                dx=fft_dx,
                units=f"1/{self.units}",
                dimension="si-length-reciprocal",
                zoom=zoom,
            )
            _, fft_a_pixel, fft_b_pixel = fft_plot.select_vectors(tolerance=min(fft_tolerance_x, fft_tolerance_y) * zoom)  # type: ignore
            # normalize the fft vectors
            fft_pixel_size = np.min([fft_dx, fft_dy])
            fft_a = fft_a_pixel * fft_pixel_size
            fft_b = fft_b_pixel * fft_pixel_size
            # get the matrix in real space
            vec_a = fft_a / np.linalg.norm(fft_a) ** 2
            vec_b = fft_b / np.linalg.norm(fft_b) ** 2
            vec_a_pixel = vec_a / self.dx
            vec_b_pixel = vec_b / self.dx
            logging.info(
                f"FFT real a: {vec_a_pixel} pixel, Real b: {vec_b_pixel} pixel"
            )
            logging.info(
                f"FFT real a: {vec_a} {self.units}, Real b: {vec_b} {self.units}"
            )
            self.a_vector["affine"] = vec_a_pixel
            self.b_vector["affine"] = vec_b_pixel
            self.origin = real_origin
            return real_origin, vec_a_pixel, vec_b_pixel
        else:
            self.a_vector["affine"] = real_a
            self.b_vector["affine"] = real_b
            self.origin = real_origin
            return real_origin, real_a, real_b

    def measure_polarization(self, a_element: str, b_element: str, cutoff_radius: float = 5.0, num_neighbors: int = 2) -> dict:
        """Measure the polarization of B atoms relative to the center of surrounding A atoms.

        Args:
            a_element (str): Element for A atoms (e.g., 'Sr')
            b_element (str): Element for B atoms (e.g., 'Ti')
            cutoff_radius (float, optional): Radius to search for surrounding A atoms. Defaults to 5.0, unit: Å.
            num_neighbors (int, optional): Number of surrounding A atoms to consider. Defaults to 2.

        Returns:
            dict: Dictionary containing:
                - 'positions': Array of B atom positions
                - 'polarization': Array of polarization vectors for each B atom
                - 'magnitude': Array of polarization magnitudes
        """
        # Get positions of A and B atoms
        # index the atom types
        a_mask = self.atom_types == self.elements.index(a_element)
        b_mask = self.atom_types == self.elements.index(b_element)
        a_positions = self.peak_positions[a_mask]
        b_positions = self.peak_positions[b_mask]

        # Initialize arrays for results
        polarization = np.zeros((len(b_positions), 2))
        magnitude = np.zeros(len(b_positions))

        # Calculate polarization for each B atom
        for i, b_pos in enumerate(b_positions):
            # Find A atoms within cutoff radius
            distances = np.linalg.norm(a_positions - b_pos, axis=1)
            nearby_mask = distances < cutoff_radius/self.dx
            nearby_a = a_positions[nearby_mask]

            # If not enough surrounding A atoms, assign NaN
            if len(nearby_a) < num_neighbors:  # Require at least 2 surrounding A atoms
                polarization[i] = np.nan
                magnitude[i] = np.nan
                continue

            # Calculate center of surrounding A atoms
            a_center = np.mean(nearby_a, axis=0)

            # Calculate polarization vector (B position relative to A center)
            polarization[i] = b_pos - a_center
            magnitude[i] = np.linalg.norm(polarization[i])

        return {
            'positions': b_positions,
            'polarization': polarization,
            'magnitude': magnitude
        }

    def measure_oxygen_tilt(self, a_type: int, o_type: int, cutoff_radius: float = 5.0) -> dict:
        """Measure the tilt of oxygen atoms based on their coordinates and nearby A site atoms.
        
        The tilt is calculated as the angle between two nearby oxygen atoms and the line formed by two A site atoms.
        
        Args:
            a_type (int): Atom type label for A site atoms (e.g., 0)
            o_type (int): Atom type label for oxygen atoms (e.g., 2)
            cutoff_radius (float, optional): Radius to search for nearby atoms. Defaults to 5.0.
            
        Returns:
            dict: Dictionary containing:
                - 'positions': Array of oxygen atom positions
                - 'tilt_angles': Array of tilt angles in degrees for each oxygen atom
                - 'tilt_vectors': Array of unit vectors representing the tilt direction
        """
        # Get positions of A site and oxygen atoms
        a_mask = self.atom_types == a_type
        o_mask = self.atom_types == o_type
        a_positions = self.peak_positions[a_mask]
        o_positions = self.peak_positions[o_mask]

        # Initialize arrays for results
        tilt_angles = np.zeros(len(o_positions))
        tilt_vectors = np.zeros((len(o_positions), 2))
        oo_pairs = []

        # Calculate tilt for each oxygen atom
        for i, o_pos in enumerate(o_positions):
            # Find A atoms within cutoff radius
            a_distances = np.linalg.norm(a_positions - o_pos, axis=1)
            nearby_a_mask = a_distances < cutoff_radius/self.dx
            nearby_a = a_positions[nearby_a_mask]

            # Find other oxygen atoms within cutoff radius
            o_distances = np.linalg.norm(o_positions - o_pos, axis=1)
            # Exclude the current oxygen atom (which would have distance 0)
            nearby_o_mask = (o_distances < cutoff_radius/self.dx) & (o_distances > 1e-6)
            nearby_o = o_positions[nearby_o_mask]

            # If not enough nearby atoms, assign NaN
            if len(nearby_a) < 2 or len(nearby_o) < 1:
                tilt_angles[i] = np.nan
                tilt_vectors[i] = np.array([np.nan, np.nan])
                oo_pairs.append([[np.nan, np.nan], [np.nan, np.nan]])
                continue

            # Find the two closest A atoms
            sorted_indices = np.argsort(a_distances[nearby_a_mask])
            closest_a1 = nearby_a[sorted_indices[0]]
            closest_a2 = nearby_a[sorted_indices[1]]

            # Find the closest oxygen atom
            closest_o = nearby_o[np.argmin(o_distances[nearby_o_mask])]

            # Calculate the A-A line vector
            a_a_vector = closest_a2 - closest_a1
            a_a_unit = a_a_vector / np.linalg.norm(a_a_vector)

            # Calculate the O-O line vector
            o_o_vector = closest_o - o_pos
            o_o_unit = o_o_vector / np.linalg.norm(o_o_vector)

            # Calculate the angle between the two lines
            dot_product = np.clip(np.dot(a_a_unit, o_o_unit), -1.0, 1.0)
            angle_rad = np.arccos(dot_product)
            angle_deg = np.degrees(angle_rad)

            # Determine if the angle should be > 90 or < 90 degrees
            # We want the smaller angle between the lines
            if angle_deg > 90:
                angle_deg = 180 - angle_deg
                o_o_unit = -o_o_unit

            tilt_angles[i] = angle_deg
            tilt_vectors[i] = o_o_unit
            oo_pairs.append([o_pos, closest_o])

        return {
            'positions': o_positions,
            'tilt_angles': tilt_angles,
            'tilt_vectors': tilt_vectors,
            'oo_pairs': np.array(oo_pairs)
        }

    # properties #######
    @property
    def nx(self):
        return self.image.shape[1]

    @property
    def ny(self):
        return self.image.shape[0]

    @property
    def min_distances(self):
        if self._min_distances is not None:
            return self._min_distances
        else:
            min_distances = {}
            # unitcell_in_image = self.unitcell_mapping(plot=False)
            cutoff = max(self.unit_cell.cell.lengths()) * 2  # type: ignore
            i, j, d = neighbor_list("ijd", self.unit_cell, cutoff)
            for site in self.unit_cell:  # type: ignore
                neighbor_sites = j[i == site.index]
                distances = d[i == site.index]
                element = site.symbol
                neighbor_elements = [self.unit_cell[n].symbol for n in neighbor_sites]  # type: ignore
                distances_list = []
                for other_element in np.unique(neighbor_elements):
                    mask = np.array(neighbor_elements) == other_element
                    distances_element = distances[mask]
                    min_distance = distances_element[distances_element > 0].min() / 2
                    distances_list.append(min_distance)
                    # if element not in min_distances:
                    #     min_distances[element] = {}
                    # min_distances[element][other_element] = min_distance
                min_distances[element] = np.array(distances_list).min()
            self._min_distances = min_distances
            return min_distances

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

    def update_atoms_from_gmm(self, atom_count_estimates: dict):
        """Update atomic columns with atoms based on GMM estimation.
        
        The Z-spacing is automatically determined from the existing supercell 
        lattice structure created during the crystal mapping process.
        
        Args:
            atom_count_estimates: Dictionary mapping element names to atom count arrays
            
        Returns:
            Updated atomic_columns object with additional atoms
        """
        if self.atomic_columns is None:
            raise ValueError("Please run get_atomic_columns() first to create atomic columns")

        # Update the atomic columns with GMM estimates (Z-spacing auto-determined)
        updated_lattice, updated_lattice_ref = self.atomic_columns.update_atoms_from_gmm(
            atom_count_estimates
        )

        # Create new atomic columns object with updated lattices
        ref = {
            "origin": self.origin,
            "vector_a": self.a_vector["perfect"],
            "vector_b": self.b_vector["perfect"],
        }

        from qem.analysis.atomic_column import AtomicColumns
        self.atomic_columns = AtomicColumns(
            updated_lattice, updated_lattice_ref, self.elements,
            self.atomic_columns.tol, self.dx, ref
        )

        # Update peak positions and atom types based on new lattice
        self.atom_types = self.atomic_columns.atom_types
        self.peak_positions = self.atomic_columns.positions_pixel

        return self.atomic_columns

    def integrate_gmm_results(self, gmm_results: dict):
        """Integrate GMM estimation results into the crystal analysis.
        
        This is a convenience method that extracts atom count estimates from
        GMM results and updates the atomic structure accordingly.
        
        Args:
            gmm_results: Results from Fitter.estimate_atom_counts_with_gmm()
            
        Returns:
            Updated atomic_columns object
        """
        if 'atom_count_estimates' not in gmm_results:
            raise ValueError("GMM results must contain 'atom_count_estimates' key")

        atom_count_estimates = gmm_results['atom_count_estimates']

        return self.update_atoms_from_gmm(atom_count_estimates)
