import re
import matplotlib.pyplot as plt
import numpy as np
from ase import Atom, Atoms
from ase.io import read
from ase.neighborlist import neighbor_list
from skimage.feature import peak_local_max

from qem.color import get_unique_colors
from qem.gui_classes import InteractivePlot,GetAtomSelection
from qem.atomic_column import AtomicColumn, AtomicColumnList

import logging
logging.basicConfig(level=logging.INFO)

class CrystalAnalyzer:
    def __init__(
        self,
        image: np.ndarray,
        dx: float,
        peak_positions: np.ndarray,
        atom_types: np.ndarray,
        elements: list[str],
        add_missing_elements: bool = True,
        units: str = "A",
        region_mask: np.ndarray = None,
    ):
        self.image = image
        self.dx = dx
        self.units = units
        self.peak_positions = peak_positions
        self.coordinates = np.array([])
        self.atom_types = atom_types
        self.elements = elements
        self.unit_cell = Atoms
        self._scaled_positions = np.array([])
        self.origin = np.array([0, 0])
        self.a_vector= np.array([1, 0])
        self.b_vector= np.array([0, 1])
        self._min_distances = None
        self._origin_offsets = {"rigid": {}, "adaptive": {}}
        self.neighbor_site_dict = {}
        self.add_missing_elements = add_missing_elements
        self.atomic_columns = AtomicColumnList()
        if region_mask is None:
            region_mask = np.ones(image.shape, dtype=bool)
        self.region_mask = region_mask

    ######### I/O ################
    def read_cif(self, cif_file_path):
        structure = read(cif_file_path)
        assert isinstance(structure, Atoms), "structure should be a ase Atoms object"
        mask = [atom.symbol in self.elements for atom in structure]  # type: ignore
        structure = structure[mask]
        self.unit_cell = structure
        return structure

    def get_unitcell_elements(self):
        """
        Get the elements present in the unit cell.

        Returns:
        - element_symbols: The symbols of the elements present in the unit cell.
        """
        assert isinstance(self.unit_cell, Atoms), "unitcell should be a ase Atoms object"
        formula = self.unit_cell.symbols.__str__()
        assert isinstance(formula, str), "composition should be a string"
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

    def write_lammps(self, filename="ABO3.lammps"):
        coordinates = self.coordinates
        coordinates[:, :2] = coordinates[:, :2] * self.dx
        total_atoms = len(self.coordinates)
        total_types = len(np.unique(self.atom_types))
        atom_types = self.atom_types.astype(int)
        xlo = np.min(coordinates[:, 0])
        xhi = np.max(coordinates[:, 0])
        ylo = np.min(coordinates[:, 1])
        yhi = np.max(coordinates[:, 1])
        zlo = np.min(coordinates[:, 2])
        zhi = max(np.max(coordinates[:, 2]), self.unit_cell.cell[2, 2])  # type: ignore

        with open(filename, "w") as f:
            f.write("# LAMMPS data file written by QEM\n\n")
            f.write(str(total_atoms) + " atoms\n")
            f.write(str(total_types) + " atom types\n\n")

            f.write(f"{xlo} {xhi} xlo xhi\n")
            f.write(f"{ylo} {yhi} ylo yhi\n")
            f.write(f"{zlo} {zhi} zlo zhi\n\n")
            f.write("Masses\n\n")
            for atom_type in range(total_types):
                element = self.elements[atom_type]
                weight = Atom(element).mass
                f.write(f"{atom_type+1} {weight} # {element}\n")
            f.write("\n")
            f.write("Atoms  # atomic\n\n")
            idx = 0
            for atom_type in np.unique(atom_types):
                mask = atom_types == atom_type
                positions = coordinates[mask]
                for position in positions:
                    idx += 1
                    f.write(
                        str(idx)
                        + " "
                        + str(atom_type + 1)
                        + " "
                        + str(position[0])
                        + " "
                        + str(position[1])
                        + " "
                        + str(position[2])
                        + "\n"
                    )
        f.close()

    def write_xyz(self, filename="ABO3.xyz"):
        coordinates = self.coordinates
        coordinates[:, :2] = coordinates[:, :2] * self.dx
        total_atoms = len(coordinates)
        with open(filename, "w") as f:
            f.write(str(total_atoms) + "\n")
            f.write("comment line\n")
            for position in coordinates:
                atom_type = int(position[3])
                element = self.elements[atom_type]
                f.write(
                    element
                    + " "
                    + str(position[0])
                    + " "
                    + str(position[1])
                    + " "
                    + str(position[2])
                    + "\n"
                )
        f.close()

    ######### lattice mapping ################
    def map_coordinates_to_image_space(self, frac_coords, vector_a, b, origin):
        """
        Apply a transformation matrix to the lattice vectors from the crystal unit cell coordinate to the image 2D pixel space.

        Args:
        - transformation_matrix (np.ndarray): The transformation matrix to apply.
        """
        # Construct the transformation matrix with a, b, c as its columns
        # if a and b is 1D array with shape of (2,), convert to (3,) by adding 0
        if vector_a.ndim == 1 and vector_a.size == 2:
            vector_a = np.append(vector_a, 0)
        if b.ndim == 1 and b.size == 2:
            b = np.append(b, 0)
        c = self.unit_cell.cell[2]  # type: ignore
        transform_matrix = np.array([vector_a, b, c]).T
        # Convert the old coordinates to a numpy array (if not already)
        origin = np.array(origin)
        if origin.size == 2:
            origin = np.append(origin, 0)

        # Multiply the transpose of T by the old coordinates to get the new coordinates
        new_coords = np.dot(transform_matrix, frac_coords.T).T + origin
        return new_coords
    
    def get_atomic_columns(self, tol:float=0, a_limit:int=0, b_limit:int=0, reciprocal=True):
        self.select_lattice_vectors(reciprocal=reciprocal)
        # estimate the a_limit and b_limit if not provided
        if a_limit == 0:
            a_limit = np.ceil(
                max(self.nx - self.origin[0], self.origin[0])
                * self.dx
                / self.unit_cell.get_cell()[0][0]  # type: ignore
            ).astype(int)
        if b_limit == 0:
            b_limit = np.ceil(
                max(self.ny - self.origin[1], self.origin[1])
                * self.dx
                / self.unit_cell.get_cell()[1][1]  # type: ignore
            ).astype(int)

        # get the supercell lattice in 3d and project to 2d
        lattice_3d, lattice_3d_ref, atom_types_3d = self.get_lattice_3d(
                a_limit=a_limit, b_limit=b_limit, adaptive=True
            )

        coords_2d, first_indices, inverse_indices = np.unique(lattice_3d[:, [0, 1]],
            axis=0, return_index=True, return_inverse=True
        )
        coords_2d_ref = lattice_3d_ref[first_indices]
        atom_types_2d = atom_types_3d[first_indices].astype(int)
        elements = [self.elements[i] for i in atom_types_2d]

        atomic_column_list = AtomicColumnList()
        
        for i in range(len(coords_2d)):
            indices = np.where(inverse_indices == i)[0]
            z = lattice_3d[indices, 2]
            atomic_column = AtomicColumn(
                element=elements[i],
                atom_type=atom_types_2d[i],
                x=coords_2d[i, 0],
                y=coords_2d[i, 1],
                z=[],
                x_ref=coords_2d_ref[i, 0],
                y_ref=coords_2d_ref[i, 1],
                z_info={element: z[i] for i, element in enumerate([self.elements[i] for i in atom_types_2d[indices]])},
                scs=0,
                strain={}
            )
            atomic_column_list.add(atomic_column)
        self.peak_positions = coords_2d
        self.atom_types = atom_types_2d
        self.atomic_columns = atomic_column_list
        self.align_unit_cell_to_image(plot=True)
        return atomic_column_list

    def get_lattice_3d(self, a_limit:int=0, b_limit:int=0, adaptive=True):
        """
        Generate a supercell lattice based on the given lattice vectors and limits.

        Parameters:
        - a_limit: The number of times to repeat the a lattice vector.
        - b_limit: The number of times to repeat the b lattice vector.

        Returns:
        - supercell_lattice: The supercell lattice.
        """
        supercell = np.array([]).reshape(0, 3)
        supercell_ref = np.array([]).reshape(0, 3)
        supercell_atom_types = np.array([])
        shift_origin = self.get_origin_offset(a_limit, b_limit, adaptive)

        for translation, new_origin in shift_origin.items():
            unitcell, atom_types = self.align_unit_cell_to_image(
                ref=(new_origin, self.a_vector, self.b_vector), plot=False
            )
            new_origin_ref = self._origin_offsets["rigid"][translation]
            unitcell_ref, _ = self.align_unit_cell_to_image(
                ref=(new_origin_ref, self.a_vector, self.b_vector), plot=False
            )

            supercell = np.vstack([supercell, unitcell])
            supercell_ref = np.vstack([supercell_ref, unitcell_ref])
            supercell_atom_types = np.hstack([supercell_atom_types, atom_types])

        is_within_image_bounds = (supercell[:, :2] > 0).all(axis=1) & (
            supercell[:, [1, 0]] < self.image.shape
        ).all(axis=1)
        supercell = supercell[is_within_image_bounds]
        supercell_ref = supercell_ref[is_within_image_bounds]
        supercell_atom_types = supercell_atom_types[is_within_image_bounds]

        # use the current coordinates to filter the peak_positions
        # create a mask for the current coordinates with the size of input image, area within 3 sigma of the current coordinates are masked to true
        valid_region_mask = np.zeros(self.image.shape, dtype=bool)
        for i in range(len(self.peak_positions)):
            x, y = self.peak_positions[i]
            sigma = 0.8 / self.dx
            valid_region_mask[
                int(max(y - 3 * sigma, 0)) : int(min(y + 3 * sigma, self.ny)),
                int(max(x - 3 * sigma, 0)) : int(min(x + 3 * sigma, self.nx)),
            ] = True

        valid_region_mask = valid_region_mask & self.region_mask

        peak_region_filter = np.ones(supercell.shape[0], dtype=bool)
        for i in range(supercell.shape[0]):
            x, y = supercell[i,:2]
            if not valid_region_mask[int(y), int(x)]:
                peak_region_filter[i] = False

        self.coordinates = supercell[peak_region_filter]
        self.coordinates_ref = supercell_ref[peak_region_filter]
        atom_types_3d = supercell_atom_types[peak_region_filter].astype(int)
        return self.coordinates, self.coordinates_ref, atom_types_3d

    def align_unit_cell_to_image(self, ref=None, plot=True):
        """
        Transforms unit cell fractional coordinates to the image coordinate system,
        aligning them with detected atomic peak positions. Optionally visualizes the
        transformation, including the origin, lattice vectors, and positions of atoms
        within the unit cell.

        Parameters:
        - plot: A boolean indicating whether to plot the mapping and unit cell visualization.

        Returns:
        - unitcell_transformed: The transformed coordinates of the unit cell.
        """
        if ref is not None:
            origin, a, b = ref
        else:
            origin = self.origin
            a = self.a_vector
            b = self.b_vector

        unitcell_transformed = self.map_coordinates_to_image_space(self.scaled_positions, a, b, origin)
        atom_types = []
        for site in self.unit_cell:  # type: ignore
            element_symbol = site.symbol
            if element_symbol in self.elements:
                atom_type = self.elements.index(element_symbol)
                atom_types.append(atom_type)
        atom_types = np.array(atom_types).astype(int)
        if plot:
            self.plot_unitcell(unitcell_transformed)
        return unitcell_transformed, atom_types

    def align_atom_sites_to_image(self, sites, ref=None, plot=True):
        if ref is not None:
            origin, a, b = ref
        else:
            origin = self.origin
            a = self.a_vector
            b = self.b_vector

        frac_coords = np.array([site.scaled_position for site in sites])
        elements = [site.symbol for site in sites]
        atom_types = [self.elements.index(element) for element in elements]
        atom_types = np.array(atom_types)
        sites_transformed = self.map_coordinates_to_image_space(frac_coords, a, b, origin)

        if plot:
            self.plot_site(sites_transformed)

        return sites_transformed, atom_types

    def get_closest_peak(self, candidate_peaks, target_peak, min_distance=1.5):
        """
        Find the closest 2D peak to the given atom position.
        """
        distance = np.linalg.norm(candidate_peaks - target_peak, axis=1)
        if distance.min() < min_distance:
            closest_peak = candidate_peaks[np.argmin(distance)]
            return closest_peak

    def get_origin_offset(self, a_limit:int = 0, b_limit:int=0, adaptive=True):
        mode = "adaptive" if adaptive else "rigid"
        if self._origin_offsets[mode]:
            return self._origin_offsets[mode]
        else:
            # generate a meshgrid
            a_axis_mesh, b_axis_mesh = np.meshgrid(
                np.arange(-a_limit, a_limit + 1), np.arange(-b_limit, b_limit + 1)
            )
            a_axis_distance_mesh = a_axis_mesh * np.linalg.norm(self.a_vector)
            b_axis_distance_mesh = b_axis_mesh * np.linalg.norm(self.b_vector)
            # compute the distance in such meshgrid
            distance_mesh = np.sqrt(a_axis_distance_mesh**2 + b_axis_distance_mesh**2)
            # apply the sort to the a_axis_mesh and b_axis_mesh
            a_axis_mesh_sorted = a_axis_mesh.flatten()[
                np.argsort(distance_mesh, axis=None)
            ]
            b_axis_mesh_sorted = b_axis_mesh.flatten()[
                np.argsort(distance_mesh, axis=None)
            ]
            order_mesh = np.array([a_axis_mesh_sorted, b_axis_mesh_sorted]).T
            # Find the closest peak to the origin to correct for drift
            origin_offsets = {"rigid": {}, "adaptive": {}}
            for a_shift, b_shift in order_mesh[1:]:
                shifted_origin_rigid = self.origin + self.a_vector* a_shift + self.b_vector* b_shift
                # check if shifted_origin_rigid is within the image
                boudary = np.linalg.norm(self.a_vector) + np.linalg.norm(self.b_vector)
                boudaries = np.array([boudary, boudary])
                if (shifted_origin_rigid[[1, 0]] < -boudaries).any() or (
                    shifted_origin_rigid[[1, 0]] > self.image.shape + boudaries
                ).any():
                    continue
                origin_offsets["rigid"][(a_shift, b_shift)] = shifted_origin_rigid
                if not adaptive:
                    continue
                else:
                    if (0, 0) not in origin_offsets["adaptive"].keys():
                        origin_offsets["adaptive"][(0, 0)] = self.origin
                        origin_offsets['rigid'][(0, 0)] = self.origin
                    if a_shift == 0 and b_shift == 0:
                        origin_offsets["adaptive"][
                            (a_shift, b_shift)
                        ] = shifted_origin_rigid
                    else:
                        # find the closet point of the a_shift and b_shift in the current shift_orgin
                        distance = np.linalg.norm(
                            np.array(list(origin_offsets["adaptive"].values()))
                            - shifted_origin_rigid,
                            axis=1,
                        )
                        neighbor_distance_idx = np.where(distance < boudary)[0]
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
                                + self.a_vector* a_shift_diff
                                + self.b_vector* b_shift_diff
                            )
                            expect_origin_list.append(expect_origin)
                        expect_origin_avg = np.array(expect_origin_list).mean(axis=0)
                        origin_offsets["adaptive"][(a_shift, b_shift)] = expect_origin_avg

                        # check if the unitcell is close to any exisiting peak positions
                        unitcell, atom_types = self.align_unit_cell_to_image(
                            ref=(
                                origin_offsets["adaptive"][(a_shift, b_shift)],
                                self.a_vector,
                                self.b_vector,
                            ),
                            plot=False,
                        )
                        mask = (unitcell[:, :2] > 0).all(axis=1) & (
                            unitcell[:, [1, 0]] < self.image.shape
                        ).all(axis=1)
                        unitcell_in_image = unitcell[mask]
                        displacement_list = []
                        if unitcell_in_image.size > 1:
                            for site in unitcell_in_image:
                                distance = np.linalg.norm(
                                    self.peak_positions - site[:2], axis=1
                                )
                                distance_ref = (
                                    np.array([d for d in self.min_distances.values()])
                                    / self.dx
                                )
                                if (
                                    len(distance) > 0
                                    and distance.min() < distance_ref.min() / 2
                                ):
                                    peak_selected = self.peak_positions[
                                        np.argmin(distance)
                                    ]
                                    # get the displacement of the site_position
                                    displacement = peak_selected - site[:2]
                                    displacement_list.append(displacement)
                            # update the shift_orgin with the average displacement
                            if len(displacement_list) > 0:
                                origin_offsets["adaptive"][(a_shift, b_shift)][
                                    :2
                                ] += np.mean(displacement_list, axis=0)
            self._origin_offsets = origin_offsets
            return self._origin_offsets[mode]

    def get_neighbor_sites(self, site_idx, cutoff=3):
        if site_idx in self.neighbor_site_dict:
            return self.neighbor_site_dict[site_idx]
        else:
            i, j, d = neighbor_list("ijd", self.unit_cell, cutoff)
            neighbors_indices = j[i == site_idx]
            neighbors_indices = np.unique(neighbors_indices)
            neighbor_sites = [self.unit_cell[n_index] for n_index in neighbors_indices]  # type: ignore
            self.neighbor_site_dict[site_idx] = neighbor_sites
        return neighbor_sites

    ####### select region and lattice vectors #######

    def select_region(self, peak_positions, atom_types):
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

    def select_lattice_vectors(self, tolerance=10, reciprocal=False):
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
            fft_image = np.abs(np.fft.fftshift(np.fft.fft2(self.image)))
            fft_log = np.log(fft_image)
            fft_dx = 1 / (self.dx * self.image.shape[1])
            fft_dy = 1 / (self.dx * self.image.shape[0])
            fft_pixel_size = np.array([fft_dx, fft_dy])
            fft_tolerance = int(
                1
                / min(
                    np.linalg.norm(real_a * self.dx), np.linalg.norm(real_b * self.dx)
                )
                / max(fft_dx, fft_dy)
                / 2
            )
            fft_peaks = peak_local_max(fft_log, min_distance=fft_tolerance, threshold_abs = 0.6*fft_log.max())
            fft_peaks = fft_peaks[:, [1, 0]].astype(float)
            fft_plot = InteractivePlot(
                fft_log,
                fft_peaks,
                dx=fft_dx,
                units=f"1/{self.units}",
                dimension="si-length-reciprocal",
            )
            _, fft_a_pixel, fft_b_pixel = fft_plot.select_vectors(tolerance=fft_tolerance)  # type: ignore
            # normalize the fft vectors
            fft_a = fft_a_pixel * fft_pixel_size
            fft_b = fft_b_pixel * fft_pixel_size
            matrix_fft = np.vstack([fft_a, fft_b])
            matrix_real = np.linalg.inv(matrix_fft)
            # get the matrix in real space
            vec_a = matrix_real[:, 0]
            vec_b = matrix_real[:, 1]
            vec_a_pixel = vec_a / self.dx
            vec_b_pixel = vec_b / self.dx
            logging.info(f"FFT real a: {vec_a_pixel} pixel, Real b: {vec_b_pixel} pixel")
            logging.info(f"FFT real a: {vec_a} {self.units}, Real b: {vec_b} {self.units}")
            self.a_vector= vec_a_pixel
            self.b_vector= vec_b_pixel
            self.origin = real_origin
            return real_origin, vec_a_pixel, vec_b_pixel
        else:
            self.a_vector= real_a
            self.b_vector = real_b
            self.origin = real_origin
            return real_origin, real_a, real_b

    ####### plot #######
    def plot(self):
        plt.imshow(self.image, cmap="gray")
        color_iterator = get_unique_colors()
        for atom_type in np.unique(self.atom_types):
            mask = self.atom_types == atom_type
            element = self.elements[atom_type]
            plt.scatter(
                self.coordinates[mask][:, 0],
                self.coordinates[mask][:, 1],
                label=element,
                color=next(color_iterator),
            )
        # plt.gca().invert_yaxis()
        plt.legend()
        plt.show()

    def plot_unitcell(self, unitcell_transformed):
        plt.subplots()
        plt.imshow(self.image, cmap="gray")
        color_iterator = get_unique_colors()
        for atom_type in np.unique(self.atom_types):
            mask_element = self.atom_types == atom_type
            element = self.elements[atom_type]
            current_color = np.array(next(color_iterator)).reshape(1, -1)
            plt.scatter(
                self.peak_positions[:, 0][mask_element],
                self.peak_positions[:, 1][mask_element],
                label=element,
                c=current_color,
            )
        for element in self.get_unitcell_elements():
            current_color = np.array(next(color_iterator)).reshape(1, -1)
            mask_unitcell_element = self.is_element_in_unit_cell(
                self.unit_cell, element
            )
            plt.scatter(
                unitcell_transformed[:, 0][mask_unitcell_element],
                unitcell_transformed[:, 1][mask_unitcell_element],
                edgecolors="k",
                c=current_color,
                alpha=0.8,
                label=element + " unitcell",
            )
        plt.tight_layout()
        plt.legend()
        plt.setp(plt.gca(), aspect="equal", adjustable="box")
        # plt.gca().invert_yaxis()

        # plot the a and b vectors
        plt.arrow(
            self.origin[0],
            self.origin[1],
            self.a_vector[0],
            self.a_vector[1],
            color="k",
            head_width=5,
            head_length=5,
        )
        plt.arrow(
            self.origin[0],
            self.origin[1],
            self.b_vector[0],
            self.b_vector[1],
            color="k",
            head_width=5,
            head_length=5,
        )
        # label the a and b vectors
        plt.text(
            self.origin[0] + self.a_vector[0],
            self.origin[1] + self.a_vector[1],
            "a",
            fontsize=20,
        )
        plt.text(
            self.origin[0] + self.b_vector[0],
            self.origin[1] + self.b_vector[1],
            "b",
            fontsize=20,
        )

    def plot_site(self, sites_transformed):
        plt.subplots(figsize=(10, 10))
        # plot the a and b vectors
        plt.arrow(
            self.origin[0],
            self.origin[1],
            self.a_vector[0],
            self.a_vector[1],
            color="k",
            head_width=10,
            head_length=10,
        )
        plt.arrow(
            self.origin[0],
            self.origin[1],
            self.b_vector[0],
            self.b_vector[1],
            color="k",
            head_width=10,
            head_length=10,
        )
        # label the a and b vectors
        plt.text(
            self.origin[0] + self.a_vector[0],
            self.origin[1] + self.a_vector[1],
            "a",
            fontsize=20,
        )
        plt.text(
            self.origin[0] + self.b_vector[0],
            self.origin[1] + self.b_vector[1],
            "b",
            fontsize=20,
        )
        color_iterator = get_unique_colors()
        for atom_type in np.unique(self.atom_types):
            mask_element = self.atom_types == atom_type
            element = self.elements[atom_type]
            current_color = np.array(next(color_iterator)).reshape(1, -1)
            plt.scatter(
                self.peak_positions[:, 0][mask_element],
                self.peak_positions[:, 1][mask_element],
                label=element,
                c=current_color,
            )
        for element in self.get_unitcell_elements():
            current_color = np.array(next(color_iterator)).reshape(1, -1)
            mask_unitcell_element = self.is_element_in_unit_cell(
                self.unit_cell, element
            )
            plt.scatter(
                sites_transformed[:, 0][mask_unitcell_element],
                sites_transformed[:, 1][mask_unitcell_element],
                edgecolors="k",
                c=current_color,
                alpha=0.5,
                label=element + " unitcell",
            )
        plt.tight_layout()
        plt.legend()
        plt.setp(plt.gca(), aspect="equal", adjustable="box")
        plt.gca().invert_yaxis()

    ####### properties #######
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
    def scaled_positions(self):
        if self._scaled_positions.size == 0:
            self._scaled_positions = self.unit_cell.cell.scaled_positions(self.unit_cell.positions) # type: ignore
        return self._scaled_positions