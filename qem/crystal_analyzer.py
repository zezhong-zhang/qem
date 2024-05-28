from distutils import dist
import numpy as np
import matplotlib.pyplot as plt
from qem.gui_classes import InteractivePlot
from pymatgen.core.structure import Structure
from qem.color import get_unique_colors
from pymatgen.transformations.advanced_transformations import SupercellTransformation
from ase.io import read
from ase import Atoms, Atom
import re
from ase.neighborlist import neighbor_list

class CrystalAnalyzer:
    def __init__(self, image:np.ndarray, pixel_size:float, peak_positions:np.ndarray, atom_types:np.ndarray, elements:list[str], add_missing_elements:bool=True, units:str="angstrom"):
        self.image = image
        self.pixel_size = pixel_size
        self.units = units
        self.peak_positions = peak_positions
        self.coordinates = np.array([])
        self.atom_types = atom_types
        self.elements = elements
        self.unitcell = Atoms
        self.origin = np.array([0, 0, 0])
        self.a = np.array([1, 0, 0])
        self.b = np.array([0, 1, 0])
        self.c = np.array([0, 0, 1])
        self._min_distances = None
        self._origin_adaptive = None
        self.neighbor_site_dict = {}
        self.add_missing_elements = add_missing_elements

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
                color=next(color_iterator),
            )
        # plt.gca().invert_yaxis()
        plt.legend()
        plt.show()

    def choose_lattice_vectors(self, tolerance=10):
        """
        Choose the lattice vectors based on the given tolerance.

        Args:
        - tolerance (int): The tolerance value.

        Returns:
        - The selected origin, a, and b vectors.
        """
        interactive_plot = InteractivePlot(
            image=self.image, 
            peaks_locations=self.peak_positions,
            atom_types=self.atom_types,
            tolerance=tolerance
        )
        origin, a, b = interactive_plot.select_vectors()
        # check origin, a, b are numpy array
        assert isinstance(origin, np.ndarray), "origin should be a numpy array"
        assert isinstance(a, np.ndarray), "a should be a numpy array"
        assert isinstance(b, np.ndarray), "b should be a numpy array"
        self.origin = np.array([origin[0], origin[1], 0])
        self.a = np.array([a[0], a[1], 0])
        self.b = np.array([b[0], b[1], 0])
        self.c = self.unitcell.cell[2]
        return origin, a, b

    def import_crystal_structure(self, cif_file_path):
        structure = read(cif_file_path)
        assert isinstance(structure, Atoms), "structure should be a ase Atoms object"
        mask = [atom.symbol in self.elements for atom in structure]
        structure = structure[mask]
        self.unitcell = structure

    # def transform(self, transformation_matrix):
    #     """
    #     Apply a transformation matrix to the lattice vectors.

    #     Args:
    #     - transformation_matrix (np.ndarray): The transformation matrix to apply.
    #     """
    #     self.unitcell.positions = np.dot(self.unitcell.positions, transformation_matrix)
    #     self.unitcell.cell = np.dot(self.unitcell.cell, transformation_matrix)

    def get_unitcell_elements(self):
        """
        Get the elements present in the unit cell.
    
        Returns:
        - element_symbols: The symbols of the elements present in the unit cell.
        """
        assert isinstance(self.unitcell, Atoms), "unitcell should be a ase Atoms object"
        formula = self.unitcell.symbols.__str__()
        assert isinstance(formula, str), "composition should be a string"
        # seperate the element symbols from the composition, split by numbers
        elements = re.findall(r'[A-Z][a-z]*', formula)
        return elements

    def generate_supercell_lattice(self, a_limit=1, b_limit=1):
        """
        Generate a supercell lattice based on the given lattice vectors and limits.

        Parameters:
        - a_limit: The number of times to repeat the a lattice vector.
        - b_limit: The number of times to repeat the b lattice vector.

        Returns:
        - supercell_lattice: The supercell lattice.
        """
        supercell = np.array([]).reshape(0, 4)
        shift_origin_adaptive = self.shift_origin_adaptive(a_limit, b_limit)

        for translation, new_origin in shift_origin_adaptive.items():
            unitcell = self.unitcell_mapping(ref=(new_origin, self.a, self.b, self.c), plot=False)
            supercell = np.vstack([supercell, unitcell])

        mask = (supercell[:, :2] > 0).all(axis=1) & (supercell[:, :2] < self.image.shape).all(axis=1)
        supercell_in_image = supercell[mask]
        mask_close = np.zeros(supercell_in_image.shape[0], dtype=bool)
        for idx, site in enumerate(supercell_in_image):
            element = self.elements[int(site[3])]
            distance = np.linalg.norm(
                self.peak_positions - site[:2], axis=1
            )
            distance_ref = np.array([d for d in self.min_distances()[element].values()])/self.pixel_size
            if distance.min() < distance_ref.max():
                mask_close[idx] = True
        self.coordinates = supercell_in_image[mask_close]
        return supercell_in_image[mask_close]
    
    def supercell_project_2d(self):
        coordinates = self.coordinates
        # get the unique 2d coordinates and atom types
        unique_coordinates = np.unique(coordinates[:, [0,1,3]], axis=0)
        peak_positions = unique_coordinates[:, :2]
        atom_types = unique_coordinates[:, 2].astype(int)
        return peak_positions, atom_types
    
    def select_region(self, peak_positions, atom_types):
        from qem.gui_classes import GetAtomSelection
        atom_select = GetAtomSelection(
            image=self.image, atom_positions=peak_positions, invert_selection=False
        )
        # hold on until the atom_positions_selected is not empty
        while atom_select.atom_positions_selected.size == 0:
            plt.pause(0.1)
        peak_positions_selected = np.array(atom_select.atom_positions_selected)
        mask = np.isin(peak_positions,peak_positions_selected).all(axis=1)
        atom_types_selected = atom_types[mask]
        return peak_positions_selected, atom_types_selected

    def unitcell_mapping(self, ref=None, plot=True):
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
            origin, a, b, c = ref
        else:   
            origin = self.origin
            a = self.a
            b = self.b
            c = self.c
        
        frac_positions = self.unitcell.cell.scaled_positions(self.unitcell.positions)
        unitcell_transformed = self.transform_coordinates(
            frac_positions, a, b, c
        ) + origin
        atom_types = []
        for site in self.unitcell:
            element_symbol = site.symbol
            if element_symbol in self.elements:
                atom_type = self.elements.index(element_symbol)
                atom_types.append(atom_type)
        atom_types = np.array(atom_types)
        unitcell_transformed = np.append(
            unitcell_transformed, np.array(atom_types).reshape(-1, 1), axis=1
        )
        if plot:
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
                mask_unitcell_element = self.check_element_in_unitcell(
                    self.unitcell, element
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
                self.a[0],
                self.a[1],
                color="k",
                head_width=5,
                head_length=5,
            )
            plt.arrow(
                self.origin[0],
                self.origin[1],
                self.b[0],
                self.b[1],
                color="k",
                head_width=5,
                head_length=5,
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
        return unitcell_transformed

    def sites_mapping(self, sites, ref=None, plot=True):
        if ref is not None:
            origin, a, b, c = ref
        else:   
            origin = self.origin
            a = self.a
            b = self.b
            c = self.c
                    
        frac_coords = np.array([site.scaled_position for site in sites])
        elements = [site.symbol for site in sites]
        atom_types = [self.elements.index(element) for element in elements]
        atom_types = np.array(atom_types)
        sites_transformed = self.transform_coordinates(frac_coords, a, b, c) + origin
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
                mask_unitcell_element = self.check_element_in_unitcell(
                    self.unitcell, element
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
            min_distances = {}
            # unitcell_in_image = self.unitcell_mapping(plot=False)
            cutoff = max(self.unitcell.cell.lengths())*2
            i, j, d = neighbor_list("ijd", self.unitcell,cutoff)
            for site in self.unitcell:
                neighbor_sites = j[i == site.index]
                distances = d[i == site.index]
                element = site.symbol
                neighbor_elements = [self.unitcell[n].symbol for n in neighbor_sites]
                distances_list = []
                for other_element in np.unique(neighbor_elements):
                    mask = np.array(neighbor_elements) == other_element
                    distances_element = distances[mask]
                    min_distance = distances_element[distances_element>0].min() / 2
                    distances_list.append(min_distance)
                    # if element not in min_distances:
                    #     min_distances[element] = {}
                    # min_distances[element][other_element] = min_distance
                min_distances[element] = np.array(distances_list).min()
            self._min_distances = min_distances
            return min_distances

    def shift_origin_adaptive(self, a_limit, b_limit):
        if self._origin_adaptive is not None:
            return self._origin_adaptive
        else:
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
                mask = (shifted_origin_rigid[:2] > boundary[0, :2]).all() and (
                    shifted_origin_rigid[:2] < boundary[1, :2]
                ).all()
                if mask == False:
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
                    # mask = distance < np.linalg.norm(self.a) + np.linalg.norm(self.b)
                    # if mask.sum() == 0:
                    #     shifted_origin_adaptive[(a_shift, b_shift)] = shifted_origin_rigid
                    # else:
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
                    # check if the unitcell is close to any exisiting peak positions
                    unitcell = self.unitcell_mapping(ref=(shifted_origin_adaptive[(a_shift, b_shift)], self.a, self.b, self.c), plot=False)
                    mask = (unitcell[:, :2] > 0).all(axis=1) & (unitcell[:, [1,0]] < self.image.shape).all(axis=1)
                    unitcell_in_image = unitcell[mask]
                    displacement_list = []
                    if unitcell_in_image.size > 1:
                        for site in unitcell_in_image:
                            distance = np.linalg.norm(
                                self.peak_positions - site[:2], axis=1
                            )
                            distance_ref = np.array([d for d in self.min_distances().values()])/self.pixel_size
                            if len(distance)>0 and distance.min() < distance_ref.min()/2:
                                peak_selected = self.peak_positions[np.argmin(distance)]
                                # get the displacement of the site_position
                                displacement = peak_selected - site[:2]
                                displacement_list.append(displacement)
                        # update the shifted_origin_adaptive with the average displacement
                        if len(displacement_list) > 0:
                            displacement= np.mean(displacement_list, axis=0)
                            displacement = np.array([displacement[0], displacement[1], 0])
                            shifted_origin_adaptive[(a_shift, b_shift)] = (
                                shifted_origin_adaptive[(a_shift, b_shift)] + displacement
                            )
            self._origin_adaptive = shifted_origin_adaptive
            return shifted_origin_adaptive

    def neighbor_site(self, site_idx, cutoff=3):
        if site_idx in self.neighbor_site_dict:
            return self.neighbor_site_dict[site_idx]
        else:
            i, j, d = neighbor_list("ijd", self.unitcell,cutoff)
            neighbors_indices = j[i == site_idx]
            neighbors_indices = np.unique(neighbors_indices)
            neighbor_sites = [self.unitcell[n_index] for n_index in neighbors_indices]
            self.neighbor_site_dict[site_idx] = neighbor_sites
        return neighbor_sites

    def unitcell_with_refined_peaks(self, origin, search_range=3):
        coordinates = np.array([]).reshape(0, 4)
        ref = (origin, self.a, self.b, self.c)
        unitcell_in_image = self.unitcell_mapping(ref=ref, plot=False)
        for idx, site in enumerate(self.unitcell):
            atom_position = unitcell_in_image[idx, :3]
            atom_type = int(unitcell_in_image[idx, 3])
            element_symbol = self.elements[int(unitcell_in_image[idx, 3])]
            if element_symbol not in self.elements and not self.add_missing_elements:
                continue
            mask = self.atom_types == atom_type
            if mask.any(): # if input peak_positions contains the element
                candidate_peaks = self.peak_positions[mask]
                distance_ref = np.array([d for d in self.min_distances()[element_symbol].values()])/self.pixel_size
                closest_peak = self.closest_peak(
                    candidate_peaks,
                    atom_position[:2],
                    min_distance=distance_ref.min(),
                )
                if closest_peak is not None:
                    atom_position_3d = np.append(
                        closest_peak,
                        [atom_position[2], atom_type],
                    )
                    coordinates = (
                        np.vstack([coordinates, atom_position_3d])
                        if coordinates.size
                        else atom_position_3d
                    )
            else: # if input peak_positions does not contain the element
                if coordinates.size == 0:
                    continue
                distance_ref = np.array([d for d in self.min_distances()[element_symbol].values()])/self.pixel_size
                # check if have any close peak within the search range
                closest_peak = self.closest_peak(
                    self.peak_positions,
                    atom_position[:2],
                    min_distance=distance_ref.min() * search_range,
                )
                if closest_peak is None:
                    continue
                # the element is not in the peak_positions, we will add it according to the neigboring peak positions with the symmetry preserved
                neighbor_sites = self.neighbor_site(site_idx=idx, cutoff=distance_ref.min() * search_range)
                # # remove the same element in the neighbour_list
                # neighbor_list = [
                #     neighbour
                #     for neighbour in neighbor_list
                #     if neighbour.species != site.species
                # ]
                if len(neighbor_sites) < 1:
                    continue
                # find the peak positions of the neighbors in the coordinates
                sites_transformed = self.sites_mapping(
                    ref=ref,
                    sites=neighbor_sites,
                    plot=False
                )
                displacement_list = []
                for site_transformed in sites_transformed:
                    atom_type_site = int(site_transformed[3])
                    element_site = self.elements[atom_type_site]
                    distance_ref = np.array([d for d in self.min_distances()[element_site].values()])/self.pixel_size
                    site_position = site_transformed[:3]
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
                        min_distance=distance_ref.min(),
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
                    atom_position[:2] = atom_position[:2] + displacement
                # atom_position[:2] = atom_position[:2] * self.pixel_size
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
        # This range depends on the size of the supercell you want to cover
        for translation, new_origin in shifted_origin_adaptive.items():
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
    
####### export atomic structure #######
    def write_lammps(self, filename="ABO3.lammps"):
        coordinates = self.coordinates
        coordinates[:, :2] = coordinates[:, :2] * self.pixel_size
        total_atoms = len(self.coordinates)
        total_types = len(np.unique(self.coordinates[:, 3]))
        atom_types = self.coordinates[:, 3].astype(int)
        xlo = np.min(coordinates[:, 0])
        xhi = np.max(coordinates[:, 0])
        ylo = np.min(coordinates[:, 1])
        yhi = np.max(coordinates[:, 1])
        zlo = np.min(coordinates[:, 2])
        zhi = max(np.max(coordinates[:, 2]), self.unitcell.cell[2, 2])

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

    def write_xyz(self, filename="ABO3.xyz"):
        coordinates = self.coordinates
        coordinates[:, :2] = coordinates[:, :2] * self.pixel_size
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

    @staticmethod   
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

    @staticmethod
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
            mask.append(site.symbol == element_symbol)
        return mask


