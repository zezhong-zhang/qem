import numpy as np
from dataclasses import dataclass
from typing import List, Dict

@dataclass
class AtomicColumn:
    element: str
    atom_type: int
    x: float
    y: float
    z: list[float]
    x_ref: float
    y_ref: float
    z_info: Dict[str, float]
    scs: float
    strain: Dict[str, float]

@dataclass
class AtomicColumnList:
    columns: List[AtomicColumn] = []

    def add(self, column: AtomicColumn):
        """Add a new AtomicColumn to the list."""
        self.columns.append(column)

    def remove(self, index: int):
        """Remove an AtomicColumn by its index in the list."""
        if 0 <= index < len(self.columns):
            del self.columns[index]
        else:
            raise IndexError("Index out of range.")
    
    def get(self, index: int) -> AtomicColumn:
        """Get an AtomicColumn by its index."""
        if 0 <= index < len(self.columns):
            return self.columns[index]
        else:
            raise IndexError("Index out of range.")
    
    def total_columns(self) -> int:
        """Return the total number of AtomicColumns."""
        return len(self.columns)
    
    def get_x(self) -> np.ndarray:
        """Return an array of x coordinates."""
        return np.array([column.x for column in self.columns])
    
    def get_y(self) -> np.ndarray:
        """Return an array of y coordinates."""
        return np.array([column.y for column in self.columns])
    
    def get_x_ref(self) -> np.ndarray:
        """Return an array of x_ref coordinates."""
        return np.array([column.x_ref for column in self.columns])
    
    def get_y_ref(self) -> np.ndarray:
        """Return an array of y_ref coordinates."""
        return np.array([column.y_ref for column in self.columns])
    
    def get_atom_types(self) -> np.ndarray:
        """Return an array of atom types."""
        return np.array([column.atom_type for column in self.columns])