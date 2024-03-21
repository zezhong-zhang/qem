import numpy as np
import matplotlib.pyplot as plt
from qem.utils import InteractivePlot
from pymatgen.core.structure import Structure
from qem.color import get_unique_colors


def transform_coordinates(frac_coords, a, b, c):
    # Construct the transformation matrix with a, b, c as its columns
    transform_matrix = np.array(
        [a, b, c]
    ).T  # Take the transpose to get the vectors as columns
    # Convert the old coordinates to a numpy array (if not already)
    frac_coords = np.array(frac_coords)
    # Multiply the transpose of T by the old coordinates to get the new coordinates
    new_coords = np.dot(transform_matrix, frac_coords.T).T
    return new_coords


def check_element_in_unitcell(unitcell: Structure, element_symbol: str) -> list:
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
        # Check if the target element is in the current site
        has_element = any(
            element.symbol == element_symbol for element in site.species.elements
        )
        mask.append(has_element)
    mask = np.array(mask)
    return mask


class CrystalAnalyzer:
    def __init__(self, image, pixel_size, peak_positions, atom_types, elements):
        self.image = image
        self.pixel_size = pixel_size
        self.peak_positions = peak_positions
        self.coordinates = np.array([])
        self.atom_types = atom_types
        self.elements = elements
        self.unitcell = Structure
        self.origin = np.array([0, 0, 0])
        self.a = np.array([1, 0, 0])
        self.b = np.array([0, 1, 0])
        self.c = np.array([0, 0, 1])
        self._min_distances = None

    def plot(self):
        plt.imshow(self.image, cmap="gray")
        color_iterator = get_unique_colors()
        for atom_type in np.unique(self.atom_types):
            mask = self.atom_types == atom_type
            element = self.elements[atom_type]
            plt.scatter(
                self.peak_positions[:, 0][mask],
                self.peak_positions[:, 1][mask],
                label=element,
                c=next(color_iterator),
            )
        # plt.gca().invert_yaxis()
        plt.legend()
        plt.show()

    def choose_lattice_vectors(self, tolerance=10):
        interactive_plot = InteractivePlot(
            peaks_locations=self.peak_positions, image=self.image, tolerance=tolerance
        )
        origin, a, b = interactive_plot.select_vectors()
        self.origin = np.append(origin, 0)
        self.a = np.append(a, 0)
        self.b = np.append(b, 0)
        return origin, a, b

    def import_crystal_structure(self, cif_file_path):
        structure = Structure.from_file(cif_file_path)
        self.unitcell = structure
        self.c = np.array([0, 0, self.unitcell.lattice.c])

    def transform(self, transformation_matrix):
        from pymatgen.transformations.advanced_transformations import (
            SupercellTransformation,
        )

        self.unitcell = SupercellTransformation(
            scaling_matrix=transformation_matrix
        ).apply_transformation(self.unitcell)
        self.c = np.array([0, 0, self.unitcell.lattice.c])

    def get_unitcell_elements(self):
        composition = self.unitcell.composition

        # Then, extract the elements as a list of unique Element objects
        elements = list(composition.element_composition.elements)

        # If you want the element symbols as strings
        element_symbols = [str(element) for element in elements]
        return element_symbols

    def unitcell_mapping(self, plot=True):
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
        unitcell_transformed = transform_coordinates(
            self.unitcell.frac_coords, self.a, self.b, self.c
        )
        atom_types = []
        for site in self.unitcell:
            species = site.species
            for specie in species:
                element = specie.element
                element_symbol = element.symbol
                if element_symbol in self.elements:
                    atom_type = self.elements.index(element_symbol)
                    atom_types.append(atom_type)
        atom_types = np.array(atom_types)
        unitcell_transformed = np.append(
            unitcell_transformed, np.array(atom_types).reshape(-1, 1), axis=1
        )
        if plot:
            plt.subplots()
            # plot the a and b vectors
            plt.arrow(
                self.origin[0],
                self.origin[1],
                self.a[0],
                self.a[1],
                color="k",
                head_width=10,
                head_length=10,
            )
            plt.arrow(
                self.origin[0],
                self.origin[1],
                self.b[0],
                self.b[1],
                color="k",
                head_width=10,
                head_length=10,
            )
            # label the a and b vectors
            plt.text(
                self.origin[0] + self.a[0],
                self.origin[1] + self.a[1],
                "a",
                fontsize=20,
            )
            plt.text(
                self.origin[0] + self.b[0],
                self.origin[1] + self.b[1],
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
                mask_unitcell_element = check_element_in_unitcell(
                    self.unitcell, element
                )
                plt.scatter(
                    unitcell_transformed[:, 0][mask_unitcell_element] + self.origin[0],
                    unitcell_transformed[:, 1][mask_unitcell_element] + self.origin[1],
                    edgecolors="k",
                    c=current_color,
                    alpha=0.5,
                    label=element + " unitcell",
                )
            plt.tight_layout()
            plt.legend()
            plt.setp(plt.gca(), aspect="equal", adjustable="box")
            plt.gca().invert_yaxis()
        return unitcell_transformed

    def sites_mapping(self, sites, plot=True):
        frac_coords = np.array([site.frac_coords for site in sites])
        atom_types = []
        for site in sites:
            species = site.species
            for specie in species:
                element = specie.element
                element_symbol = element.symbol
                if element_symbol in self.elements:
                    atom_type = self.elements.index(element_symbol)
                    atom_types.append(atom_type)
        atom_types = np.array(atom_types)
        sites_transformed = transform_coordinates(frac_coords, self.a, self.b, self.c)
        sites_transformed = np.append(
            sites_transformed, np.array(atom_types).reshape(-1, 1), axis=1
        )
        if plot:
            plt.subplots(figsize=(10, 10))
            # plot the a and b vectors
            plt.arrow(
                self.origin[0],
                self.origin[1],
                self.a[0],
                self.a[1],
                color="k",
                head_width=10,
                head_length=10,
            )
            plt.arrow(
                self.origin[0],
                self.origin[1],
                self.b[0],
                self.b[1],
                color="k",
                head_width=10,
                head_length=10,
            )
            # label the a and b vectors
            plt.text(
                self.origin[0] + self.a[0],
                self.origin[1] + self.a[1],
                "a",
                fontsize=20,
            )
            plt.text(
                self.origin[0] + self.b[0],
                self.origin[1] + self.b[1],
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
                mask_unitcell_element = check_element_in_unitcell(
                    self.unitcell, element
                )
                plt.scatter(
                    sites_transformed[:, 0][mask_unitcell_element] + self.origin[0],
                    sites_transformed[:, 1][mask_unitcell_element] + self.origin[1],
                    edgecolors="k",
                    c=current_color,
                    alpha=0.5,
                    label=element + " unitcell",
                )
            plt.tight_layout()
            plt.legend()
            plt.setp(plt.gca(), aspect="equal", adjustable="box")
            plt.gca().invert_yaxis()
        return sites_transformed

    def closest_peak(self, candidate_peaks, target_peak, min_distance=1.5):
        """
        Find the closest 2D peak to the given atom position.
        """
        distance = np.linalg.norm(candidate_peaks - target_peak, axis=1)
        if distance.min() < min_distance:
            closest_peak = candidate_peaks[np.argmin(distance)]
            return closest_peak

    def min_distances(self):
        if self._min_distances is not None:
            return self._min_distances
        else:
            # Compute the minimum 2D distance between peaks in the unit cell, output as dictionary
            min_distances = {}
            elements = self.get_unitcell_elements()
            unitcell_in_image = self.unitcell_mapping(plot=False)
            for element in elements:
                mask = check_element_in_unitcell(self.unitcell, element)
                element_positions = unitcell_in_image[mask][:, :2]
                distances = np.linalg.norm(
                    element_positions[:, None] - element_positions, axis=2
                )
                np.fill_diagonal(distances, np.inf)
                min_distances[element] = distances.min() / 2
        return min_distances

    def shift_origin_adaptive(self, a_limit, b_limit):
        # generate a meshgrid
        a_axis_mesh, b_axis_mesh = np.meshgrid(
            np.arange(-a_limit, a_limit + 1), np.arange(-b_limit, b_limit + 1)
        )
        a_axis_distance_mesh = a_axis_mesh * np.linalg.norm(self.a)
        b_axis_distance_mesh = b_axis_mesh * np.linalg.norm(self.b)
        # compute the distance in such meshgrid
        distance_mesh = np.sqrt(a_axis_distance_mesh**2 + b_axis_distance_mesh**2)
        # apply the sort to the a_axis_mesh and b_axis_mesh
        a_axis_mesh_sorted = a_axis_mesh.flatten()[np.argsort(distance_mesh, axis=None)]
        b_axis_mesh_sorted = b_axis_mesh.flatten()[np.argsort(distance_mesh, axis=None)]
        order_mesh = np.array([a_axis_mesh_sorted, b_axis_mesh_sorted]).T
        # Find the closest peak to the origin to correct for drift
        shifted_origin_adaptive = {}
        for a_shift, b_shift in order_mesh:
            shifted_origin_rigid = self.origin + self.a * a_shift + self.b * b_shift
            boundary = np.array(
                [[0, 0, 0], [self.image.shape[1], self.image.shape[0], 0]]
            )
            boundary[0, :] = boundary[0, :] - np.abs(self.a + self.b) 
            boundary[1, :] = boundary[1, :] + np.abs(self.a + self.b)
            mask = (shifted_origin_rigid[:2] > boundary[0, :2]).all() & (
                shifted_origin_rigid[:2] < boundary[1, :2]
            ).all()
            if mask is False:
                continue
            if a_shift == 0 and b_shift == 0:
                shifted_origin_adaptive[(a_shift, b_shift)] = shifted_origin_rigid
            else:
                # find the closet point of the a_shift and b_shift in the current shifted_origin_adaptive
                distance = np.linalg.norm(
                    np.array(list(shifted_origin_adaptive.values()))
                    - shifted_origin_rigid,
                    axis=1,
                )
                mask = distance < np.linalg.norm(self.a) + np.linalg.norm(self.b)
                if mask.sum() == 0:
                    shifted_origin_adaptive[(a_shift, b_shift)] = shifted_origin_rigid
                else:
                    selected_keys = list(shifted_origin_adaptive.keys())[
                        np.argmin(distance)
                    ]
                    # find the difference of a_shift and b_shift with the selected_keys
                    a_shift_diff = a_shift - selected_keys[0]
                    b_shift_diff = b_shift - selected_keys[1]
                    shifted_origin_adaptive[(a_shift, b_shift)] = (
                        shifted_origin_adaptive[selected_keys]
                        + self.a * a_shift_diff
                        + self.b * b_shift_diff
                    )
        return shifted_origin_adaptive

    def unitcell_with_refined_peaks(self, origin, search_range=3):
        coordinates = np.array([]).reshape(0, 4)
        unitcell_in_image = self.unitcell_mapping(plot=False)
        min_distances = self.min_distances()
        for idx, site in enumerate(self.unitcell):
            atom_position = unitcell_in_image[idx, :3] + origin
            elements = site.species.elements
            for element in elements:
                element_symbol = element.symbol
                if element_symbol in self.elements:
                    atom_type = self.elements.index(element_symbol)
                    mask = self.atom_types == atom_type
                    if mask.any():
                        candidate_peaks = self.peak_positions[mask]
                        closest_peak = self.closest_peak(
                            candidate_peaks,
                            atom_position[:2],
                            min_distance=min_distances[element_symbol],
                        )
                        if closest_peak is not None:
                            atom_position_3d = np.append(
                                closest_peak * self.pixel_size,
                                [atom_position[2], atom_type],
                            )
                            coordinates = (
                                np.vstack([coordinates, atom_position_3d])
                                if coordinates.size
                                else atom_position_3d
                            )
                    else:
                        if coordinates.size == 0:
                            continue
                        # check if have any close peak within the search range
                        closest_peak = self.closest_peak(
                            self.peak_positions,
                            atom_position[:2],
                            min_distance=min_distances[element_symbol] * search_range,
                        )
                        if closest_peak is None:
                            continue
                        # the element is not in the peak_positions, we will add it according to the neigboring peak positions with the symmetry preserved
                        neighbor_list = self.unitcell.get_neighbors(
                            site=site,
                            r=min_distances[element_symbol]
                            * search_range
                            * self.pixel_size,
                        )
                        # remove the same element in the neighbour_list
                        neighbor_list = [
                            neighbour
                            for neighbour in neighbor_list
                            if neighbour.species != site.species
                        ]
                        if len(neighbor_list) < 1:
                            continue
                        # find the peak positions of the neighbors in the coordinates
                        sites_transformed = self.sites_mapping(
                            sites=neighbor_list, plot=False
                        )
                        displacement_list = []
                        for site_transformed in sites_transformed:
                            atom_type_site = int(site_transformed[3])
                            element_site = self.elements[atom_type_site]
                            site_position = site_transformed[:3] + origin
                            if np.ndim(coordinates) == 1:
                                candidate_peaks = coordinates[
                                    coordinates[3] == atom_type_site
                                ]
                            else:
                                candidate_peaks = coordinates[
                                    coordinates[:, 3] == atom_type_site
                                ]
                            if candidate_peaks.size == 0:
                                continue
                            else:
                                candidate_peaks = candidate_peaks[:, :2]
                            closest_peak = self.closest_peak(
                                candidate_peaks,
                                site_position[:2],
                                min_distance=min_distances[element_site],
                            )
                            # get the displacement of the site_position
                            if closest_peak is not None:
                                displacement = closest_peak[:2] - site_position[:2]
                                displacement_list.append(displacement)
                            else:
                                continue
                        # get the average displacement of the neighbor_list
                        if len(displacement_list) > 0:
                            displacement = np.mean(displacement_list, axis=0)
                            atom_position = atom_position[:2] + displacement
                        atom_position[:2] = atom_position[:2] * self.pixel_size
                        atom_position_3d = np.append(atom_position, atom_type)
                        coordinates = (
                            np.vstack([coordinates, atom_position_3d])
                            if coordinates.size
                            else atom_position_3d
                        )
        return coordinates

    def supercell_with_refined_peaks(self, a_limit=1, b_limit=1):
        supercell_coordinates = np.array([])
        shifted_origin_adaptive = self.shift_origin_adaptive(a_limit, b_limit)
        # Determine the range for translation along a_axis and b_axis
        # This range depends on the size of the supercell you want to cover
        for a_translation, b_translation in shifted_origin_adaptive.keys():
            # Calculate new origin for the translated unit cell
            new_origin = shifted_origin_adaptive[(a_translation, b_translation)]
            # Use unitcell_with_refined_peaks for the new origin and adjusted peaks
            translated_unitcell_peaks = self.unitcell_with_refined_peaks(
                origin=new_origin
            )  # Adjust the method to accept dynamic origin and peak list
            # Combine with supercell coordinates
            if translated_unitcell_peaks.size > 0:
                supercell_coordinates = (
                    np.vstack([supercell_coordinates, translated_unitcell_peaks])
                    if supercell_coordinates.size
                    else translated_unitcell_peaks
                )
        self.coordinates = supercell_coordinates
        return supercell_coordinates

    def write_lammps(self, filename="ABO3.lammps"):
        total_atoms = len(self.coordinates)
        total_types = len(np.unique(self.coordinates[:, 3]))
        xlo = np.min(self.coordinates[:, 0])
        xhi = np.max(self.coordinates[:, 0])
        ylo = np.min(self.coordinates[:, 1])
        yhi = np.max(self.coordinates[:, 1])
        zlo = np.min(self.coordinates[:, 2])
        zhi = np.max(self.coordinates[:, 2])

        with open(filename, "w") as f:
            f.write("# LAMMPS data file written by QEM\n\n")
            f.write(str(total_atoms) + " atoms\n")
            f.write(str(total_types) + " atom types\n\n")

            f.write(f"{xlo} {xhi} xlo xhi\n")
            f.write(f"{ylo} {yhi} ylo yhi\n")
            f.write(f"{zlo} {zhi} zlo zhi\n\n")
            f.write("Masses\n\n")
            f.write("1 88.90585  # Y3+\n")
            f.write("2 26.981538  # Al3+\n")
            f.write("3 15.9994  # O2-\n\n")
            f.write("Atoms  # atomic\n\n")
            idx = 0
            for atom_type in np.unique(self.coordinates[:, 3]):
                atom_type = int(atom_type)
                mask = self.coordinates[:, 3] == atom_type
                positions = self.coordinates[mask]
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

    def write_xyz(self, filename="ABO3.xyz"):
        total_atoms = len(self.coordinates)
        with open(filename, "w") as f:
            f.write(str(total_atoms) + "\n")
            f.write("comment line\n")
            for position in self.coordinates:
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
