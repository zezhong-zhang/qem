"""High-level entry point for multi-modal joint quantification."""

from __future__ import annotations

from typing import Optional

import numpy as np

from .dataset import MultiModalDataset
from .route_b_joint_ls import FusionResult, JointLeastSquaresRoute


class MultiModalAnalyzer:
    """Convenience wrapper around :class:`JointLeastSquaresRoute`."""

    def __init__(self, dataset: MultiModalDataset, route: Optional[JointLeastSquaresRoute] = None):
        self.dataset = dataset
        self.route = route or JointLeastSquaresRoute(elements=dataset.elements)

    @classmethod
    def from_hspy(cls, directory: str, elements, **kwargs) -> "MultiModalAnalyzer":
        dataset = MultiModalDataset.from_hspy(directory, elements=elements, **kwargs)
        return cls(dataset)

    def fit(self, initial: Optional[np.ndarray] = None, **route_options) -> FusionResult:
        if route_options:
            self.route = JointLeastSquaresRoute(
                elements=self.dataset.elements,
                **route_options,
            )
        self.route.fit(self.dataset, initial=initial)
        return self.route.get_results()

    def get_results(self) -> FusionResult:
        return self.route.get_results()
