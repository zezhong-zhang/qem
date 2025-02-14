from dataclasses import dataclass, field
from typing import Dict, List

import numpy as np
from ase import Atoms
from matscipy.atomic_strain import atomic_strain
from matscipy.neighbours import neighbour_list

from qem.periodic_table import chemical_symbols


@dataclass
class AtomicColumns:
    """
    A class to represent atomic columns projected from a 3D atomic lattice onto a 2D plane.

    Attributes:
        lattice (Atoms): The 3D atomic lattice.
        lattice_ref (Atoms): The reference 3D atomic lattice.
        elements (List[str]): List of element symbols.
        tol (float): Tolerance for the projection.
        pixel_size (float): Size of each pixel.
        projection_params (Dict[str, np.ndarray]): Dictionary containing origin, vector_a, and vector_b for projection.
    """
    lattice: Atoms
    lattice_ref: Atoms
    elements: List[str] = field(default_factory=list)
    tol: float = 0
    pixel_size: float = 0.1
    reference: Dict[str, np.ndarray] = field(default_factory=lambda: {
        'origin': np.array([0, 0]),
        'vector_a': np.array([1, 0]),
        'vector_b': np.array([0, 1])
    })

    def get_columns(self):
        """project the 3d atomic lattice onto the 2d plane in z direction and return the unique atomic columns

        Args:
            tol (float): tolerance for the projection, within which the atoms are considered to be in the same column

        Returns:
            coords_2d (np.ndarray): 2d coordinates of the atomic columns
            atomic_numbers (np.ndarray): atomic numbers of the atoms in the atomic columns
        """
        # project the 3d atomic lattice onto the 2d plane in z direction
        coords_2d, mask = np.unique(self.lattice.positions[:, :2], axis=0, return_index=True)
        coords_2d = coords_2d/self.pixel_size
        atomic_numbers = self.lattice.get_atomic_numbers()[mask]
        return coords_2d, atomic_numbers

    def get_columns_ref(self):
        """project the 3d atomic lattice onto the 2d plane in z direction and return the unique atomic columns

        Args:
            tol (float): tolerance for the projection, within which the atoms are considered to be in the same column

        Returns:
            coords_2d (np.ndarray): 2d coordinates of the atomic columns
            atomic_numbers (np.ndarray): atomic numbers of the atoms in the atomic columns
        """
        # project the 3d atomic lattice onto the 2d plane in z direction
        _, mask = np.unique(self.lattice.positions[:, :2], axis=0, return_index=True)
        coords_2d = self.lattice_ref.positions[:, :2][mask]/self.pixel_size
        atomic_numbers = self.lattice_ref.get_atomic_numbers()[mask]
        return coords_2d, atomic_numbers

    def get_local_displacement(self, cutoff: float, units='pixel') -> np.ndarray:
        """Return an array of local displacements."""
        # mean displacement within the cutoff radius for each column
        # distances = self.positions_pixel[:,np.newaxis] - self.positions_pixel
        # neighbour_mask = np.linalg.norm(distances, axis=-1) < cutoff/self.pixel_size
        # local_displacements = self.displacements - np.array([np.mean(self.displacements[row], axis=0) for row in neighbour_mask])
        lattice_2d = self.lattice.copy()
        _, mask = np.unique(lattice_2d.positions[:, :2], axis=0, return_index=True)
        lattice_2d = lattice_2d[mask]
        i, j = neighbour_list('ij', lattice_2d, cutoff)
        local_displacements = (
            self.get_column_displacement(units)
            -
            np.array([
                np.mean(self.get_column_displacement(units)[j[i == idx]], axis=0)
                for idx in range(len(i))
            ])
        )
        return local_displacements

    def get_column_displacement(self, units='pixel') -> np.ndarray:
        """Return the displacement of the column."""
        if units == 'pixel':
            return self.positions_pixel - self.positions_pixel_ref
        else:
            return (self.positions_pixel - self.positions_pixel_ref)*self.pixel_size

    @property
    def positions_pixel(self) -> np.ndarray:
        """Return an array of positions."""
        coords_2d, _ = self.get_columns()
        return coords_2d

    @property
    def positions_pixel_ref(self) -> np.ndarray:
        coords_2d, _ = self.get_columns_ref()
        return coords_2d

    @property
    def x(self) -> np.ndarray:
        """Return an array of x coordinates."""
        return self.positions_pixel[:, 0]

    @property
    def y(self) -> np.ndarray:
        """Return an array of y coordinates."""
        return self.positions_pixel[:, 1]

    @property
    def x_ref(self) -> np.ndarray:
        """Return an array of x_ref coordinates."""
        return self.positions_pixel_ref[:, 0]

    @property
    def y_ref(self) -> np.ndarray:
        """Return an array of y_ref coordinates."""
        return self.positions_pixel_ref[:, 1]

    @property
    def num_columns(self) -> int:
        """Return the total number of AtomicColumns."""
        return self.positions_pixel.size

    @property
    def atomic_numbers(self) -> np.ndarray:
        """Return an array of atom types."""
        _, atomic_numbers = self.get_columns()
        return atomic_numbers

    @property
    def column_elements(self) -> List[str]:
        """Return a list of column elements."""
        return [chemical_symbols[atomic_number] for atomic_number in self.atomic_numbers]

    @property
    def atom_types(self) -> np.ndarray:
        """Return a numpy array of atom types."""
        return np.array([self.elements.index(element) for element in self.column_elements]).astype(int)

    def get_strain_xy(self, cutoff: float = 0) -> np.ndarray:
        """Return the strain matrix."""
        if cutoff == 0:
            displacement = self.get_column_displacement()
        else:
            displacement = self.get_local_displacement(cutoff)

        # origin = self.reference['origin']
        vector_a = self.reference['vector_a']
        vector_b = self.reference['vector_b']

        # sign_a = np.sign((self.positions_pixel - origin) @ vector_a)
        # sign_b = np.sign((self.positions_pixel - origin) @ vector_b)
        # project th local displacements onto the lattice vectors
        deformation_gradient_tensor = np.array([
            np.dot(displacement, vector_a) / np.linalg.norm(vector_a)**2 * vector_a[:, None],
            np.dot(displacement, vector_b) / np.linalg.norm(vector_b)**2 * vector_b[:, None]
        ])
        lattice_matrix = np.array([vector_a, vector_b])
        strain_matrix = np.linalg.inv(lattice_matrix) @ deformation_gradient_tensor
        epsilon_xx = strain_matrix[0, 0]
        epsilon_yy = strain_matrix[1, 1]
        epsilon_xy = 0.5*(strain_matrix[0, 1] + strain_matrix[1, 0])
        omega_xy = 0.5*(strain_matrix[0, 1] - strain_matrix[1, 0])
        return epsilon_xx, epsilon_yy, epsilon_xy, omega_xy

    def get_strain(self, cutoff: float = 3.0) -> np.ndarray:
        """Return the strain tensor."""
        cutoff = float(cutoff)

        lattice_2d = self.lattice.copy()
        _, mask = np.unique(lattice_2d.positions[:, :2], axis=0, return_index=True)
        lattice_2d = lattice_2d[mask]
        lattice_2d_ref = self.lattice_ref[mask]
        lattice_2d.positions[:, 2] = 0
        lattice_2d_ref.positions[:, 2] = 0

        atomic_strain_3d, residual = atomic_strain(lattice_2d, lattice_2d_ref, cutoff=cutoff)
        atomic_strain_2d = atomic_strain_3d[:, :2, :2]  # only the in-plane strain

        epsilon_xx = atomic_strain_2d[:, 0, 0] - 1
        epsilon_yy = atomic_strain_2d[:, 1, 1] - 1
        epsilon_xy = 0.5*(atomic_strain_2d[:, 0, 1] + atomic_strain_2d[:, 1, 0])
        omega_xy = 0.5*(atomic_strain_2d[:, 0, 1] - atomic_strain_2d[:, 1, 0])
        return epsilon_xx, epsilon_yy, epsilon_xy, omega_xy
