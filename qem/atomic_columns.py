import numpy as np
from ase import Atoms
from qem.periodic_table import chemical_symbols


class AtomicColumns(Atoms):
    def __init__(self, symbols, positions, scs=None, pbc=False):
        super().__init__(symbols=symbols, positions=positions, pbc=pbc)
        self.scs = scs
