"""
ADF-EDX-EELS joint quantification tools.

The fusion package implements Route B: projected joint least-squares
optimization over elemental concentration maps from pre-aligned ADF, EDX, and
EELS signals.
"""

from .analyzer import MultiModalAnalyzer
from .dataset import MultiModalDataset
from .io import load_fusion_result, save_fusion_result
from .route_b_joint_ls import FusionResult, JointLeastSquaresRoute

__all__ = [
    "FusionResult",
    "JointLeastSquaresRoute",
    "MultiModalAnalyzer",
    "MultiModalDataset",
    "load_fusion_result",
    "save_fusion_result",
]
