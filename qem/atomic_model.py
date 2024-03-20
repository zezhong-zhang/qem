import numpy as np
import matplotlib.pyplot as plt
from qem.utils import InteractivePlot
from pymatgen.core.structure import Structure

def rotate_vector(vector, axis, angle):
    # Rotate a vector around a specified axis by a given angle
    axis = axis / np.linalg.norm(axis)
    rot_matrix = np.array(
        [
            [
                np.cos(angle) + axis[0] ** 2 * (1 - np.cos(angle)),
                axis[0] * axis[1] * (1 - np.cos(angle)) - axis[2] * np.sin(angle),
                axis[0] * axis[2] * (1 - np.cos(angle)) + axis[1] * np.sin(angle),
            ],
            [
                axis[1] * axis[0] * (1 - np.cos(angle)) + axis[2] * np.sin(angle),
                np.cos(angle) + axis[1] ** 2 * (1 - np.cos(angle)),
                axis[1] * axis[2] * (1 - np.cos(angle)) - axis[0] * np.sin(angle),
            ],
            [
                axis[2] * axis[0] * (1 - np.cos(angle)) - axis[1] * np.sin(angle),
                axis[2] * axis[1] * (1 - np.cos(angle)) + axis[0] * np.sin(angle),
                np.cos(angle) + axis[2] ** 2 * (1 - np.cos(angle)),
            ],
        ]
    )

    return np.dot(rot_matrix, vector.T).T


class AtomicModel:
    def __init__(self, image, pixel_size, peak_positions, atom_types, elements):
        self.image = image
        self.pixel_size = pixel_size
        self.peak_positions = peak_positions
        self.atom_types = atom_types
        self.elements = elements
        self.unitcell = None
        self.lattice_parameters = None
        self.origin = None
        self.a = None
        self.b = None

    def plot(self):
        plt.imshow(self.image, cmap="gray")
        plt.scatter(self.peak_positions[:, 1], self.peak_positions[:, 0], c="r", s=1)
        plt.show()

    def choose_lattice_vectors(self, tolerance=10):
        interactive_plot = InteractivePlot(
            peaks_locations=self.peak_positions, image=self.image, tolerance=tolerance
        )
        origin, a, b = interactive_plot.select_vectors()
        self.origin = origin
        self.a = a
        self.b = b
        return origin, a, b

    def import_crystal_structure(self, cif_file_path):
        structure = Structure.from_file(cif_file_path)
        lattice_parameters = structure.lattice.parameters
        self.unitcell = structure
        self.lattice_parameters = lattice_parameters

    def rotate_lattice(self, axis=[0, 0, 1], plot=True):
        angle = np.arctan2(self.a[1], self.a[0])
        rotated_lattice = rotate_vector(self.unitcell.cart_coords, axis, -angle)
        if plot:
            plt.subplots(figsize=(10, 10))
            # label the a and b vectors
            plt.text(
                self.origin[0] + self.a[0] / 2,
                self.origin[1] + self.a[1] / 2,
                "a",
                fontsize=20,
            )
            plt.text(
                self.origin[0] + self.b[0] / 2,
                self.origin[1] + self.b[1] / 2,
                "b",
                fontsize=20,
            )
            for atom_type in np.unique(self.atom_types):
                mask = self.atom_types == atom_type
                element = self.elements[atom_type]
                plt.scatter(
                    self.peak_positions[:, 0][mask],
                    self.peak_positions[:, 1][mask],
                    label=element,
                )
            plt.tight_layout()
            plt.setp(plt.gca(), aspect="equal", adjustable="box")

            plt.scatter(
                rotated_lattice[:, 0] + self.origin[0],
                rotated_lattice[:, 2] + self.origin[1],
                s=10,
                c="grey",
            )
            plt.scatter(self.origin[0], self.origin[1], c="r", s=100)
            # plot the a and b vectors
            plt.arrow(
                self.origin[0],
                self.origin[1],
                self.a[0],
                self.a[1],
                color="r",
                width=0.1,
            )
            plt.arrow(
                self.origin[0],
                self.origin[1],
                self.b[0],
                self.b[1],
                color="r",
                width=0.1,
            )
            plt.gca().invert_yaxis()
        return rotated_lattice
    
    def compute_3d_coordinates(self):
        element_coordinates_3d = {}
        for id, atom in enumerate(self.unitcell):
            element = atom.label[:-1]
            if element not in element_coordinates_3d.keys():
                element_coordinates_3d[element] = []
            # check if the element is in the element_coordinates_2d
            if element not in element_coordinates_2d.keys():
                coord_3d = rotated_coordinates[id, [0,2,1]] + np.append(origin,0)
                coord_2d = np.array([rotated_coordinates[id, [0,2]] + origin])
                # check if the coord_2d is away from the rest of the element_coordinates_2d by a distance within than the threshold
                if len(element_coordinates_2d) != 0:
                    distance = np.linalg.norm(coord_2d - np.vstack(list(element_coordinates_2d.values())), axis=1)
                    if distance.min() <= 2:
                        element_coordinates_3d[element].append(coord_3d)
                continue
            # compute the distance between the atom and the element_coordinates_2d 
            else:
                distance = np.linalg.norm(element_coordinates_2d[element] - origin - rotated_coordinates[id, [0,2]] , axis=1)
                mask = distance < 1.5
                if mask.sum() != 0:
                    element_coordinates_3d[element].append(rotated_coordinates[id, [0,2,1]] + np.append(origin,0))
        return element_coordinates_3d

    def map_3d(structure, element_coordinates_2d, rotated_coordinates, origin, a_axis, b_axis, a_axis_length, b_axis_length):
        # now repeat the same process for origin shifted by a_axis and b_axis
        limit_a_axis = 20
        limit_b_axis = 10
        coords_3d={}
        # generate a meshgrid 
        a_axis_mesh, b_axis_mesh = np.meshgrid(np.arange(-limit_a_axis, limit_a_axis+1), np.arange(-limit_b_axis, limit_b_axis+1))
        a_axis_distance_mesh = a_axis_mesh * a_axis_length
        b_axis_distance_mesh = b_axis_mesh * b_axis_length
        # compute the distance in such meshgrid
        distance_mesh = np.sqrt(a_axis_distance_mesh**2 + b_axis_distance_mesh**2)
        # sort the distance
        distance_mesh_sorted = np.sort(distance_mesh, axis=None)
        # apply the sort to the a_axis_mesh and b_axis_mesh
        a_axis_mesh_sorted = a_axis_mesh.flatten()[np.argsort(distance_mesh, axis=None)]
        b_axis_mesh_sorted = b_axis_mesh.flatten()[np.argsort(distance_mesh, axis=None)]
        order = np.array([a_axis_mesh_sorted, b_axis_mesh_sorted]).T
        # shifted_origin_array =np.array([origin[:,np.newaxis] + a_axis[:,np.newaxis] * a_axis_mesh_sorted.T +  b_axis[:,np.newaxis]*b_axis_mesh_sorted]).squeeze().T

        shifted_origin_adaptive = {}
        for a_shift, b_shift in order:
            shifted_origin_rigid = origin + a_axis * a_shift + b_axis * b_shift
            if a_shift == 0 and b_shift == 0:
                shifted_origin_adaptive[(a_shift, b_shift)] = shifted_origin_rigid 
            else:
                #find the closet point of the a_shift and b_shift in the current shifted_origin_adaptive
                distance = np.linalg.norm(np.array(list(shifted_origin_adaptive.values())) - shifted_origin_rigid, axis=1)
                mask = distance < 1
                if mask.sum() == 0:
                    shifted_origin_adaptive[(a_shift, b_shift)] = shifted_origin_rigid
                else:
                    selected_keys = list(shifted_origin_adaptive.keys())[np.argmin(distance)]
                    # find the difference of a_shift and b_shift with the selected_keys
                    a_shift_diff = a_shift - selected_keys[0]
                    b_shift_diff = b_shift - selected_keys[1]
                    shifted_origin_adaptive[(a_shift, b_shift)] = shifted_origin_adaptive[selected_keys] + a_axis * a_shift_diff + b_axis * b_shift_diff
            

            result = compute_3d_coordinates(structure, element_coordinates_2d, rotated_coordinates, shifted_origin_adaptive[(a_shift, b_shift)])
            # combine the result
            for l in result.keys():
                if l not in coords_3d.keys():
                    coords_3d[l] = []
                coords_3d[l].extend(result[l])
        return coords_3d
    
    def write_lammps(coords_3d, lattice_parameters, filename='ABO3.lammps'):
        with open(filename, 'w') as f:
            f.write('# LAMMPS data file written by OVITO Basic 3.8.1\n\n')
            f.write(str(len(coords_3d['Y'])+len(coords_3d['Al'])+len(coords_3d['O']))+' atoms\n')
            f.write(str(len(coords_3d.keys()))+' atom types\n\n')

            f.write(f'0 80 xlo xhi\n')
            f.write(f'0 80 ylo yhi\n')
            f.write(f'0.0 {lattice_parameters[1]} zlo zhi\n\n')
            f.write('Masses\n\n')
            f.write('1 88.90585  # Y3+\n')
            f.write('2 26.981538  # Al3+\n')
            f.write('3 15.9994  # O2-\n\n')
            f.write('Atoms  # atomic\n\n')
            id = 0
            for l in coords_3d.keys():
                for position in coords_3d[l]:
                    id += 1
                    # get the atomic number of the element
                    if l == 'Y':
                        atomic_number = 1
                    elif l == 'Al':
                        atomic_number = 2
                    elif l == 'O':
                        atomic_number = 3
                    f.write(str(id)+' '+str(atomic_number)+' '+str(position[0])+' '+str(position[1])+' '+str(position[2])+'\n')


    def write_xyz(coords_3d, filename='ABO3.xyz'):
        with open(filename, 'w') as f:
            f.write(str(len(coords_3d['Y'])+len(coords_3d['Al'])+len(coords_3d['O']))+'\n')
            f.write('comment line\n')
            for l in coords_3d.keys():
                for position in coords_3d[l]:
                    f.write(l+' '+str(position[0])+' '+str(position[1])+' '+str(position[2])+'\n')
        f.close()