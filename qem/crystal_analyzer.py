import copy

# from shapely.affinity import scale
import logging
import re
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
from ase import Atoms
from ase.io import read
from ase.neighborlist import neighbor_list
from scipy.spatial import ConvexHull
from shapely.geometry import Point, Polygon
from skimage.feature import peak_local_max
from matplotlib_scalebar.scalebar import ScaleBar
from skimage.transform import rescale


from qem.atomic_column import AtomicColumns
from qem.color import get_unique_colors
from qem.gui_classes import GetAtomSelection, InteractivePlot
from qem.image_fitting import gaussian_filter

logging.basicConfig(level=logging.INFO)

class CrystalAnalyzer:
    def __init__(
        self,
        image: np.ndarray,
        dx: float,
        peak_positions: np.ndarray,
        atom_types: np.ndarray,
        elements: list[str],
        units: str = "A",
        region_mask: Optional[np.ndarray] = None,
    ):
        self.image = image
        self.dx = dx
        self.units = units
        self.peak_positions = peak_positions
        self.atom_types = atom_types
        self.elements = elements
        self.unit_cell = Atoms
        self.origin = np.array([0, 0])
        self.a_vector_affine= np.array([1, 0])
        self.b_vector_affine= np.array([0, 1])
        self.a_vector_perfect = np.array([1, 0])
        self.b_vector_perfect = np.array([0, 1])
        self._min_distances = None
        self.atomic_columns = None
        if region_mask is None:
            region_mask = np.ones(image.shape, dtype=bool)
        self.region_mask = region_mask
        self.rotation_matrix = None
        self.affine_matrix = None
        # self.add_missing_elements = add_missing_elements
        # self.neighbor_site_dict = {}
        self.unit_cell_transformed = {'perfect': None, 'affine': None}
        self._origin_offsets = {"perfect": {}, "affine":{}, "adaptive": {}}
        self.lattice = Atoms
        self.lattice_ref = Atoms



    ######### I/O ################
    def read_cif(self, cif_file_path):
        atoms = read(cif_file_path)
        assert isinstance(atoms, Atoms), "atoms should be a ase Atoms object"
        mask = [atom.symbol in self.elements for atom in atoms]  # type: ignore
        self.unit_cell = atoms[mask]
        return atoms

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

    ######### lattice mapping ################
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
        lattice_3d, lattice_3d_ref = self.get_lattice_3d(a_limit=a_limit, b_limit=b_limit)
        assert isinstance(lattice_3d, Atoms), "lattice_3d should be an Atoms object"
        assert isinstance(lattice_3d_ref, Atoms), "lattice_3d_ref should be an Atoms object"
        ref ={'origin': self.origin, 'vector_a': self.a_vector_perfect, 'vector_b': self.b_vector_perfect}
        self.atomic_columns = AtomicColumns(lattice_3d, lattice_3d_ref, self.elements, tol, self.dx, ref)
        self.atom_types = self.atomic_columns.atom_types
        self.peak_positions = self.atomic_columns.positions_pixel
        return self.atomic_columns

    def align_unit_cell_to_image(self, ref=None, plot=True, mode='affine'):
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
            a = self.a_vector_affine
            b = self.b_vector_affine

        assert isinstance(self.unit_cell, Atoms), "self.unit_cell should be a ase Atoms object"
        assert mode in ['affine', 'perfect'], "mode should be either 'affine' or 'perfect'"

        if self.unit_cell_transformed[mode] is None:
            unit_cell = copy.deepcopy(self.unit_cell)
            new_xy = np.array([a, b]).T
            old_xy = np.array([unit_cell.cell[0][:2], unit_cell.cell[1][:2]]).T
            coords_xy = unit_cell.positions[:, :2]  # type: ignore
            if mode == 'perfect':
                if self.rotation_matrix is None:
                # get the angle between a and x-axis
                    angle_a = np.arctan2(a[1], a[0]) - np.arctan2(unit_cell.cell[0][1], unit_cell.cell[0][0])
                    angle_b = np.arctan2(b[1], b[0]) - np.arctan2(unit_cell.cell[1][1], unit_cell.cell[1][0])

                    if abs(angle_a - angle_b) > np.pi / 2: # consider the case when b vector is flipped
                        self.rotation_matrix = np.array([[np.cos(angle_a), np.sin(angle_a)], [np.sin(angle_a), -np.cos(angle_a)]])
                    else: # normal case when a and b are in the same order as in the unit cell
                        self.rotation_matrix = np.array([[np.cos(angle_a), -np.sin(angle_a)], [np.sin(angle_a), np.cos(angle_a)]])
                new_coords_xy = (coords_xy @ self.rotation_matrix)/self.dx 
                self.a_vector_perfect = unit_cell.cell[0][:2] @ self.rotation_matrix / self.dx
                self.b_vector_perfect = unit_cell.cell[1][:2] @ self.rotation_matrix / self.dx
                logging.info(f"Perfect a: {self.a_vector_perfect} pixel, Perfect b: {self.b_vector_perfect} pixel by rotation of unit cell and scaling with pixel size.")
            else:  # affine transformation
                if self.affine_matrix is None:
                    self.affine_matrix = new_xy @ np.linalg.inv(old_xy)
                new_coords_xy = (coords_xy @ self.affine_matrix) 
            positions = np.hstack([new_coords_xy *self.dx, unit_cell.positions[:, 2].reshape(-1, 1)])  # type: ignore
            unit_cell.set_positions(positions)
            self.unit_cell_transformed[mode] = unit_cell
        shifted_unit_cell = self.unit_cell_transformed[mode].copy()
        shifted_unit_cell.positions[:, :2] += origin * self.dx
        if plot:
            self.plot_unitcell(mode=mode)
        return shifted_unit_cell
    
    def region_of_interest(self, sigma):
        # use the current coordinates to filter the peak_positions
        # create a mask for the current coordinates with the size of input image, area within 3 sigma of the current coordinates are masked to true
        region_of_interest = np.zeros(self.image.shape, dtype=bool)
        for i in range(len(self.peak_positions)):
            x, y = self.peak_positions[i]
            region_of_interest[
                int(max(y - 3 * sigma, 0)) : int(min(y + 3 * sigma, self.ny)),
                int(max(x - 3 * sigma, 0)) : int(min(x + 3 * sigma, self.nx)),
            ] = True
        return region_of_interest

    def get_lattice_3d(self, a_limit:int=0, b_limit:int=0):
        """
        Generate a supercell lattice based on the given lattice vectors and limits.

        Parameters:
        - a_limit: The number of times to repeat the a lattice vector.
        - b_limit: The number of times to repeat the b lattice vector.

        Returns:
        - supercell_lattice: The supercell lattice.
        """
        supercell = Atoms()
        supercell_ref = Atoms()
        shift_origin_adaptive = self.get_origin_offset(a_limit, b_limit, 'adaptive')
        # shift_origin_affine = self.get_origin_offset(a_limit, b_limit, 'affine')
        shift_origin_perfect = self.get_origin_offset(a_limit, b_limit, 'perfect')

        for translation, new_origin in shift_origin_adaptive.items():
            unitcell = self.align_unit_cell_to_image(
                ref=(new_origin, self.a_vector_affine, self.b_vector_affine), plot=False
            )
            new_origin_ref = shift_origin_perfect[translation]
            unitcell_ref = self.align_unit_cell_to_image(
                ref=(new_origin_ref, self.a_vector_perfect, self.b_vector_perfect), plot=False, mode='perfect'
            )
            supercell.extend(unitcell)
            supercell_ref.extend(unitcell_ref)

        is_within_image_bounds = (supercell.positions[:, :2] /self.dx > 0).all(axis=1) & (
            supercell.positions[:, [1, 0]] /self.dx < self.image.shape
        ).all(axis=1)
        supercell = supercell[is_within_image_bounds]
        supercell_ref = supercell_ref[is_within_image_bounds]
        supercell.set_cell(np.array([[self.nx*self.dx,0,0],[0,self.ny*self.dx,0], self.unit_cell.cell[2]]))
        supercell_ref.set_cell(np.array([[self.nx*self.dx,0,0],[0,self.ny*self.dx,0], self.unit_cell.cell[2]]))

        valid_region_mask = self.region_of_interest(0.8/self.dx) & self.region_mask

        peak_region_filter = np.ones(supercell.get_global_number_of_atoms(), dtype=bool)
        for i in range(supercell.get_global_number_of_atoms()):
            x, y = supercell.positions[i,:2] /self.dx
            if not valid_region_mask[int(y), int(x)]:
                peak_region_filter[i] = False

        # supercell = supercell[peak_region_filter]
        # supercell_ref = supercell_ref[peak_region_filter]
        self.lattice = supercell[peak_region_filter]
        self.lattice_ref = supercell_ref[peak_region_filter]
        return self.lattice, self.lattice_ref

    def get_closest_peak(self, candidate_peaks, target_peak, min_distance=1.5):
        """
        Find the closest 2D peak to the given atom position.
        """
        distance = np.linalg.norm(candidate_peaks - target_peak, axis=1)
        if distance.min() < min_distance:
            closest_peak = candidate_peaks[np.argmin(distance)]
            return closest_peak

    def get_origin_offset(self, a_limit:int = 0, b_limit:int=0, mode='adaptive'):
        assert mode in ['perfect','affine', 'adaptive'], "mode should be either 'perfect', 'affine' or 'adaptive'"
        if not self._origin_offsets[mode]:
            self._calc_origin_offsets(a_limit, b_limit)        
        return self._origin_offsets[mode]
            
        
    def _calc_origin_offsets(self, a_limit:int, b_limit:int):
        # get the perfect mapping of the unit cell to the image
        self.align_unit_cell_to_image(plot=False,mode='perfect')

        # generate a meshgrid
        a_axis_mesh, b_axis_mesh = np.meshgrid(
            np.arange(-a_limit, a_limit + 1), np.arange(-b_limit, b_limit + 1)
        )
        a_axis_distance_mesh = a_axis_mesh * np.linalg.norm(self.a_vector_perfect)
        b_axis_distance_mesh = b_axis_mesh * np.linalg.norm(self.b_vector_perfect)
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
        neighborhood_radius = np.linalg.norm(self.a_vector_affine + self.b_vector_affine).astype(int)

        # Calculate the convex hull of the points
        hull = ConvexHull(self.peak_positions)
        # Get the vertices of the convex hull to form the boundary polygon
        hull_points = self.peak_positions[hull.vertices]
        # Create a Shapely Polygon from the convex hull points
        polygon = Polygon(hull_points)
        # Expand the polygon outward by the threshold distance (neighborhood_radius pixels)
        expanded_polygon = polygon.buffer(neighborhood_radius)

        # Find the closest peak to the origin to correct for drift
        origin_offsets = {"perfect": {}, "affine":{}, "adaptive": {}}
        origin_offsets["adaptive"][(0, 0)] = self.origin
        origin_offsets['affine'][(0, 0)] = self.origin
        origin_offsets['perfect'][(0, 0)] = self.origin
        
        for a_shift, b_shift in order_mesh[1:]:
            shifted_origin_perfect = self.origin + self.a_vector_perfect* a_shift + self.b_vector_perfect* b_shift
            shifted_origin_affine = self.origin + self.a_vector_affine* a_shift + self.b_vector_affine* b_shift

            # # check if shifted_origin_rigid is within the image
            # boudaries = np.array([neighborhood_radius, neighborhood_radius])
            # if (shifted_origin_affine < -boudaries ).any() or (
            #     shifted_origin_affine > self.image.shape + boudaries
            # ).any():
            #     continue

            # check if shifted_origin_rigid is within the region of interest
            # if (shifted_origin_affine >= 0).all() and (
            #     shifted_origin_affine < self.image.shape).all():
            #     if not self.region_of_interest(sigma = 2/self.dx)[int(shifted_origin_affine[1]), int(shifted_origin_affine[0])]:
            #         continue
            # region_of_interest = self.region_of_interest(1/self.dx)
            # padded_array = np.pad(region_of_interest, neighborhood_radius, mode='constant', constant_values=False)
            # expanded_array = binary_dilation(padded_array, iterations=int(neighborhood_radius))
            # if not expanded_array[int(shifted_origin_affine[1]+neighborhood_radius), int(shifted_origin_affine[0]+neighborhood_radius)]:
            #     continue      


            # Check if the shifted origin is within the expanded area
            is_within_expanded_area = expanded_polygon.contains(Point(shifted_origin_affine))
            if not is_within_expanded_area:
                continue
            
            # if the shifted origin is within the region of interest, add it to the dictionary
            origin_offsets["perfect"][(a_shift, b_shift)] = shifted_origin_perfect
            origin_offsets["affine"][(a_shift, b_shift)] = shifted_origin_affine
            origin_offsets["adaptive"][(a_shift, b_shift)] = shifted_origin_affine
                
            # find the closet point of the a_shift and b_shift in the current shift_orgin
            distance = np.linalg.norm(
                np.array(list(origin_offsets["adaptive"].values()))
                - shifted_origin_affine,
                axis=1,
            )
            neighbor_distance_idx = np.where(distance < neighborhood_radius)[0]
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
                    + self.a_vector_affine* a_shift_diff
                    + self.b_vector_affine* b_shift_diff
                )
                expect_origin_list.append(expect_origin)

            expect_origin_avg = np.array(expect_origin_list).mean(axis=0)
            origin_offsets["adaptive"][(a_shift, b_shift)] = expect_origin_avg

            # check if the unitcell is close to any exisiting peak positions
            unitcell = self.align_unit_cell_to_image(
                ref=(
                    origin_offsets["adaptive"][(a_shift, b_shift)],
                    self.a_vector_affine,
                    self.b_vector_affine,
                ),
                plot=False,
                mode = 'affine'
            )
            mask = (unitcell.positions[:, :2] /self.dx > 0).all(axis=1) & (
                unitcell.positions[:, [1, 0]] /self.dx < self.image.shape
            ).all(axis=1)
            unitcell_in_image = unitcell[mask]
            displacement_list = []
            if len(unitcell_in_image) > 1:
                for site in unitcell_in_image:
                    distance = np.linalg.norm(
                        self.peak_positions - site.position[:2]/self.dx, axis=1
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
                        displacement = peak_selected - site.position[:2]/self.dx
                        displacement_list.append(displacement)
                # update the shift_orgin with the average displacement
                if len(displacement_list) > 0:
                    origin_offsets["adaptive"][(a_shift, b_shift)][
                        :2
                    ] += np.mean(displacement_list, axis=0)
                    if  np.linalg.norm(np.mean(displacement_list, axis=0)) > 1/self.dx:
                        logging.warning(f"Large local displacmenet detected, adaptive is not aligned properly at {origin_offsets['adaptive'][(a_shift, b_shift)]} with translation of {(a_shift, b_shift)}")
        self._origin_offsets = origin_offsets
        return origin_offsets

    # def get_neighbor_sites(self, site_idx, cutoff=5):
    #     if site_idx in self.neighbor_site_dict:
    #         return self.neighbor_site_dict[site_idx]
    #     else:
    #         i, j, d = neighbor_list("ijd", self.unit_cell, cutoff)
    #         neighbors_indices = j[i == site_idx]
    #         neighbors_indices = np.unique(neighbors_indices)
    #         neighbor_sites = [self.unit_cell[n_index] for n_index in neighbors_indices]  # type: ignore
    #         self.neighbor_site_dict[site_idx] = neighbor_sites
    #     return neighbor_sites

    ####### strain mapping #######
    def get_strain(self, cut_off:float=5.0):
        """
        Get the strain of the atomic columns based on the given cut-off radius.

        Args:
        - cut_off (int): The cut-off radius.

        Returns:
        - The strain of the atomic columns.
        """
        return self.atomic_columns.get_strain(float(cut_off))
        # return self.atomic_columns.get_strain(cut_off)

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
            image_filtered = gaussian_filter(self.image, 1)
            fft_image = np.abs(np.fft.fftshift(np.fft.fft2(image_filtered)))
            fft_log = np.log(fft_image)
            fft_dx = 1 / (self.dx * self.image.shape[1])
            fft_dy = 1 / (self.dx * self.image.shape[0])
            fft_pixel_size = np.array([fft_dx, fft_dy])
            fft_tolerance_x = int(1/ np.linalg.norm(real_a * self.dx)/ fft_dx/ 4)
            fft_tolerance_y = int(1/ np.linalg.norm(real_b * self.dx)/ fft_dy/ 4)

            scale_y = fft_tolerance_x / fft_tolerance_y
            if scale_y <1:
                fft_log_rescaled = rescale(fft_log, (1, 1/scale_y), anti_aliasing=False)
                fft_peaks = peak_local_max(fft_log_rescaled, min_distance=fft_tolerance_y, num_peaks=30)
                fft_peaks[:, 1] = fft_peaks[:, 1] * scale_y
            else:
                fft_log_rescaled = rescale(fft_log, (scale_y, 1), anti_aliasing=False)
                fft_peaks = peak_local_max(fft_log_rescaled, min_distance=fft_tolerance_x, num_peaks=30)
                fft_peaks[:, 0] = fft_peaks[:, 0] / scale_y

            fft_peaks = fft_peaks[:, [1, 0]].astype(float)
            zoom =3
            fft_plot = InteractivePlot(
                fft_log,
                fft_peaks,
                dx=fft_dx,
                units=f"1/{self.units}",
                dimension="si-length-reciprocal",
                zoom=zoom,
            )
            _, fft_a_pixel, fft_b_pixel = fft_plot.select_vectors(tolerance=min(fft_tolerance_x,fft_tolerance_y)*zoom)  # type: ignore
            # normalize the fft vectors

            fft_a = fft_a_pixel * fft_pixel_size / zoom
            fft_b = fft_b_pixel * fft_pixel_size / zoom
            # get the matrix in real space
            vec_a = fft_a / np.linalg.norm(fft_a)**2
            vec_b = fft_b / np.linalg.norm(fft_b)**2
            vec_a_pixel = vec_a / self.dx
            vec_b_pixel = vec_b / self.dx
            logging.info(f"FFT real a: {vec_a_pixel} pixel, Real b: {vec_b_pixel} pixel")
            logging.info(f"FFT real a: {vec_a} {self.units}, Real b: {vec_b} {self.units}")
            self.a_vector_affine= vec_a_pixel
            self.b_vector_affine= vec_b_pixel
            self.origin = real_origin
            return real_origin, vec_a_pixel, vec_b_pixel
        else:
            self.a_vector_affine= real_a
            self.b_vector_affine = real_b
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
                self.lattice.positions[mask, 0]/self.dx,
                self.lattice.positions[mask, 1]/self.dx,
                label=element,
                color=next(color_iterator),
            )
        # plt.gca().invert_yaxis()
        plt.legend()
        plt.show()

    def plot_unitcell(self, mode='affine'):
        if mode == 'perfect':
            unitcell_transformed = self.unit_cell_transformed['perfect'].copy()
            origin, a, b = self.origin, self.a_vector_perfect, self.b_vector_perfect
            unitcell_transformed.positions[:, :2] += origin * self.dx
        else:
            unitcell_transformed = self.unit_cell_transformed['affine'].copy()
            origin, a, b = self.origin, self.a_vector_affine, self.b_vector_affine
            unitcell_transformed.positions[:, :2] += origin * self.dx
        

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
                unitcell_transformed.positions[:, 0][mask_unitcell_element] / self.dx,
                unitcell_transformed.positions[:, 1][mask_unitcell_element] / self.dx,
                edgecolors="k",
                c=current_color,
                alpha=0.8,
                label=element + " unitcell",
            )
        plt.tight_layout()
        plt.legend()
        plt.setp(plt.gca(), aspect="equal", adjustable="box")
        plt.gca().add_artist(self.scalebar)
        # plt.gca().invert_yaxis()

        # plot the a and b vectors
        plt.arrow(
            origin[0],
            origin[1],
            a[0],
            a[1],
            color="k",
            head_width=5,
            head_length=5,
        )
        plt.arrow(
            origin[0],
            origin[1],
            b[0],
            b[1],
            color="k",
            head_width=5,
            head_length=5,
        )
        # label the a and b vectors
        plt.text(
            origin[0] + a[0],
            origin[1] + a[1],
            "a",
            fontsize=20,
        )
        plt.text(
            origin[0] + b[0],
            origin[1] + b[1],
            "b",
            fontsize=20,
        )

    def plot_displacement(self, mode='local',cut_off=5.0, units='A'):
        if mode == 'local':
            displacement = self.atomic_columns.get_local_displacement(cut_off, units)
        else:
            displacement = self.atomic_columns.get_column_displacement(units)
        plt.imshow(self.image, cmap="gray")
        plt.scatter(self.atomic_columns.x, self.atomic_columns.y, c=np.linalg.norm(displacement, axis=1), cmap='plasma')
        cbar = plt.colorbar()
        cbar.set_label(f'Displacement ({units})')
        plt.quiver(self.atomic_columns.x, self.atomic_columns.y, displacement[:, 0], displacement[:, 1], scale=1, scale_units='xy')
        plt.gca().add_artist(self.scalebar)
        plt.axis('off')

    def plot_strain(self, cut_off:float=5.0):
        epsilon_xx, epsilon_yy, epsilon_xy, omega_xy = self.get_strain(cut_off)
        plt.subplots(2,2,constrained_layout=True)
        plt.subplot(2, 2, 1)
        plt.imshow(self.image, cmap="gray")
        plt.scatter(self.atomic_columns.x, self.atomic_columns.y, c=epsilon_xx, cmap='coolwarm')
        plt.axis('off')
        plt.gca().add_artist(self.scalebar)
        plt.colorbar()
        # bounds = np.abs(epsilon_xx).max()
        # get the 95 percentile of the strain
        bounds = np.percentile(np.abs(epsilon_xx), 95)
        plt.clim(-bounds, bounds)
        plt.title(r'$\epsilon_{xx}$')
        # plt.tight_layout()
        plt.subplot(2, 2, 2)
        plt.imshow(self.image, cmap="gray")
        plt.scatter(self.atomic_columns.x, self.atomic_columns.y, c=epsilon_yy, cmap='coolwarm')
        plt.colorbar()
        # bounds = np.abs(epsilon_yy).max()
        bounds = np.percentile(np.abs(epsilon_yy), 95)
        plt.clim(-bounds, bounds)
        plt.axis('off')
        plt.title(r'$\epsilon_{yy}$')
        # plt.tight_layout()
        plt.subplot(2, 2, 3)
        plt.imshow(self.image, cmap="gray")
        plt.scatter(self.atomic_columns.x, self.atomic_columns.y, c=epsilon_xy, cmap='coolwarm')
        plt.colorbar()
        # bounds = np.abs(epsilon_xy).max()
        bounds = np.percentile(np.abs(epsilon_xy), 95)
        plt.clim(-bounds, bounds)
        plt.axis('off')
        plt.title(r'$\epsilon_{xy}$')
        # plt.tight_layout()
        plt.subplot(2, 2, 4)
        plt.imshow(self.image, cmap="gray")
        plt.scatter(self.atomic_columns.x, self.atomic_columns.y, c=omega_xy, cmap='coolwarm')
        plt.colorbar()
        # bounds = np.abs(omega_xy).max()
        bounds = np.percentile(np.abs(omega_xy), 95)
        plt.clim(-bounds, bounds)
        plt.axis('off')
        plt.title(r'$\omega_{xy}$')
        # plt.tight_layout()
        

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
    def scalebar(self):
        scalebar = ScaleBar(
            self.dx,
            units=self.units,
            location="lower right",
            length_fraction=0.2,
            font_properties={"size": 20},
        )
        return scalebar