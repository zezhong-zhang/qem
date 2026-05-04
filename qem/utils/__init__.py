"""
Utility functions and helpers for QEM.
"""

from .params import *
from .log import *
from .backend import *
from .arrays import *
from .memory import *

__all__ = [
    # General utilities
    'export_params',
    
    # Backend utilities
    'safe_convert_to_numpy',
    'safe_convert_to_tensor',
    'safe_deepcopy_params',
    
    # Array utilities
    'find_duplicate_row_indices',
    'find_row_indices',
    'find_element_indices',
    'get_random_indices_in_batches',
    
    # Logging
    # Add logging exports here
]