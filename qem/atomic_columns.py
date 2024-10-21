import numpy as np
from ase import Atoms
from qem.periodic_table import chemical_symbols

class AtomicColumn:
    def __init__(self, symbol, x, y, z, scs=None):
        self.sym = symbol
        self.x = x
        self.y = y
        self.z = z
        self.scs = scs
