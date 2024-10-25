from calendar import c
import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict
from ase import Atoms
from qem.periodic_table import chemical_symbols


@dataclass
class AtomicColumns:
    lattice: Atoms
    lattice_ref: Atoms
    elements: List[str] = field(default_factory=list)
    tol: float = 0

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
        coords_2d = self.lattice_ref.positions[:, :2][mask]
        atomic_numbers = self.lattice_ref.get_atomic_numbers()[mask]
        return coords_2d, atomic_numbers

    def get_local_displacements(self, cutoff:float) -> np.ndarray:
        """Return an array of local displacements."""
        # mean displacement within the cutoff radius for each column
        distances = self.positions[:,np.newaxis] - self.positions
        neighbour_mask = np.linalg.norm(distances, axis=-1) < cutoff
        local_displacements = self.displacements - np.array([np.mean(self.displacements[row], axis=0) for row in neighbour_mask])
        return local_displacements

    @property
    def displacements(self) -> np.ndarray:
        """Return the displacement of the column."""
        return self.positions - self.positions_ref
    
    @property
    def positions(self) -> np.ndarray:
        """Return an array of positions."""
        coords_2d, _ = self.get_columns()
        return coords_2d

    @property
    def positions_ref(self) -> np.ndarray:
        coords_2d, _ = self.get_columns_ref()
        return coords_2d

    @property
    def x(self) -> np.ndarray:
        """Return an array of x coordinates."""
        return self.positions[:, 0]
    
    @property
    def y(self) -> np.ndarray:
        """Return an array of y coordinates."""
        return self.positions[:, 1]
    
    @property
    def x_ref(self) -> np.ndarray:
        """Return an array of x_ref coordinates."""
        return self.positions_ref[:, 0]
    
    @property
    def y_ref(self) -> np.ndarray:
        """Return an array of y_ref coordinates."""
        return self.positions_ref[:, 1]
    
    @property
    def num_columns(self) -> int:
        """Return the total number of AtomicColumns."""
        return self.positions.size
    
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