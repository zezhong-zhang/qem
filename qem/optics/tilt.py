"""Sample tilt transformations for STEM imaging.

This module handles the effect of sample tilt on projected atomic positions.
When a 3D sample is tilted, the projected positions of atoms shift.

For small tilt angles, the shift is approximately:
Δx ≈ z * tan(θx)
Δy ≈ z * tan(θy)

where z is the height/thickness of the atomic layer.
"""

from typing import Optional, Tuple, Union

import numpy as np


class SampleTilt:
    """Handle sample tilt effects on projected atomic positions."""

    @staticmethod
    def apply_tilt(
        positions: np.ndarray,
        tilt_x: float,
        tilt_y: float,
        thickness: float = 6.0,
        center: Optional[Tuple[float, float]] = None,
    ) -> np.ndarray:
        """
        Apply sample tilt to 2D projected positions.

        For small angles, the projection is approximately:
        x_proj ≈ x + z * tan(θx)
        y_proj ≈ y + z * tan(θy)

        Parameters
        ----------
        positions : np.ndarray (N x 2) or (2,)
            Atomic positions [x, y] in pixels or Angstroms
        tilt_x : float
            Tilt around x-axis in mrad (positive tilts +y towards -y)
        tilt_y : float
            Tilt around y-axis in mrad (positive tilts +x towards -x)
        thickness : float, optional
            Sample thickness in Angstroms. Defaults to 6.0.
        center : tuple (x_center, y_center), optional
            Center of rotation. If None, uses the center of the position array.

        Returns
        -------
        tilted_positions : np.ndarray
            Tilted positions with same shape as input
        """
        positions = np.asarray(positions)
        original_shape = positions.shape

        if positions.ndim == 1:
            positions = positions.reshape(1, -1)

        x, y = positions[:, 0], positions[:, 1]

        # Set center of rotation
        if center is None:
            x_center = np.mean(x)
            y_center = np.mean(y)
        else:
            x_center, y_center = center

        # Convert mrad to radians and calculate shifts
        # The shift depends on the z-coordinate (height)
        # For a single layer, we use the nominal thickness
        shift_x = thickness * np.tan(tilt_x * 1e-3)
        shift_y = thickness * np.tan(tilt_y * 1e-3)

        # Apply shift (rotation about the center)
        x_tilted = x - x_center + shift_x + x_center
        y_tilted = y - y_center - shift_y + y_center

        tilted = np.column_stack([x_tilted, y_tilted])
        return tilted.reshape(original_shape)

    @staticmethod
    def tilt_jacobian(
        positions: np.ndarray,
        tilt_x: float,
        tilt_y: float,
        thickness: float = 6.0,
    ) -> np.ndarray:
        """
        Calculate Jacobian matrix for tilt transformation.

        The Jacobian describes how small changes in tilt affect positions:
        J = [dx/dθx, dx/dθy]
            [dy/dθx, dy/dθy]

        Parameters
        ----------
        positions : np.ndarray (N x 2)
            Atomic positions [x, y]
        tilt_x : float
            Tilt around x-axis in mrad
        tilt_y : float
            Tilt around y-axis in mrad
        thickness : float, optional
            Sample thickness in Angstroms

        Returns
        -------
        jacobian : np.ndarray (2 x 2)
            Jacobian matrix [dx/dθ, dy/dθ]
        """
        # Derivatives of the tilt transformation
        # dx/dθx = thickness * sec²(θx) * 1e-3 ≈ thickness * 1e-3 for small angles
        # dx/dθy = 0 (x-shift doesn't depend on y-tilt)
        # dy/dθx = 0
        # dy/dθy = -thickness * sec²(θy) * 1e-3 ≈ -thickness * 1e-3

        sec_x = 1.0 / np.cos(tilt_x * 1e-3)
        sec_y = 1.0 / np.cos(tilt_y * 1e-3)

        jacobian = np.array([
            [thickness * sec_x ** 2 * 1e-3, 0],
            [0, -thickness * sec_y ** 2 * 1e-3],
        ])

        return jacobian

    @staticmethod
    def inverse_tilt(
        positions: np.ndarray,
        tilt_x: float,
        tilt_y: float,
        thickness: float = 6.0,
        center: Optional[Tuple[float, float]] = None,
    ) -> np.ndarray:
        """
        Apply inverse tilt to recover untilted positions.

        Parameters
        ----------
        positions : np.ndarray (N x 2)
            Tilted atomic positions [x, y]
        tilt_x : float
            Applied tilt around x-axis in mrad
        tilt_y : float
            Applied tilt around y-axis in mrad
        thickness : float, optional
            Sample thickness in Angstroms
        center : tuple, optional
            Center of rotation

        Returns
        -------
        untilted_positions : np.ndarray
            Positions with tilt removed
        """
        # Inverse tilt is just applying the negative tilt
        return SampleTilt.apply_tilt(
            positions, -tilt_x, -tilt_y, thickness, center
        )

    @staticmethod
    def optimize_tilt(
        positions_ref: np.ndarray,
        positions_observed: np.ndarray,
        thickness: float = 6.0,
        initial_tilt: Tuple[float, float] = (0.0, 0.0),
    ) -> Tuple[float, float, float]:
        """
        Find the tilt that best maps reference positions to observed positions.

        Uses least-squares optimization to find the tilt angles that
        minimize the difference between tilted reference positions
        and observed positions.

        Parameters
        ----------
        positions_ref : np.ndarray (N x 2)
            Reference atomic positions (untilted)
        positions_observed : np.ndarray (N x 2)
            Observed atomic positions (tilted)
        thickness : float, optional
            Sample thickness in Angstroms
        initial_tilt : tuple (tilt_x, tilt_y), optional
            Initial guess for tilt in mrad

        Returns
        -------
        tilt_x : float
            Optimized tilt around x-axis in mrad
        tilt_y : float
            Optimized tilt around y-axis in mrad
        residual : float
            Final residual error
        """
        from scipy.optimize import least_squares

        positions_ref = np.asarray(positions_ref)
        positions_observed = np.asarray(positions_observed)

        def tilt_error(tilt):
            tilted = SampleTilt.apply_tilt(
                positions_ref, tilt[0], tilt[1], thickness
            )
            return (tilted - positions_observed).ravel()

        result = least_squares(
            tilt_error,
            x0=list(initial_tilt),
            method="lm",
        )

        return result.x[0], result.x[1], float(np.sum(result.fun ** 2))

    @staticmethod
    def calculate_projected_distances(
        positions: np.ndarray,
        tilt_x: float,
        tilt_y: float,
        thickness: float = 6.0,
    ) -> np.ndarray:
        """
        Calculate projected interatomic distances under tilt.

        Parameters
        ----------
        positions : np.ndarray (N x 2)
            Atomic positions
        tilt_x : float
            Tilt around x-axis in mrad
        tilt_y : float
            Tilt around y-axis in mrad
        thickness : float, optional
            Sample thickness in Angstroms

        Returns
        -------
        distances : np.ndarray (N x N)
            Matrix of projected distances between all atom pairs
        """
        from scipy.spatial.distance import pdist, squareform

        tilted = SampleTilt.apply_tilt(positions, tilt_x, tilt_y, thickness)
        distances = squareform(pdist(tilted))
        return distances


def tilt_from_affine(
    affine_matrix: np.ndarray,
    thickness: float = 6.0,
) -> Tuple[float, float]:
    """
    Estimate tilt angles from an affine transformation matrix.

    Parameters
    ----------
    affine_matrix : np.ndarray (2 x 3) or (3 x 3)
        Affine transformation matrix
    thickness : float, optional
        Sample thickness in Angstroms

    Returns
    -------
    tilt_x : float
        Estimated tilt around x-axis in mrad
    tilt_y : float
        Estimated tilt around y-axis in mrad
    """
    # Extract translation components
    if affine_matrix.shape == (2, 3):
        tx, ty = affine_matrix[0, 2], affine_matrix[1, 2]
    elif affine_matrix.shape == (3, 3):
        tx, ty = affine_matrix[0, 2], affine_matrix[1, 2]
    else:
        raise ValueError(f"Unexpected affine matrix shape: {affine_matrix.shape}")

    # Convert translation to tilt angles
    # tx = thickness * tan(tilt_x) * 1e-3
    # tilt_x = arctan(tx / thickness / 1e-3) * 1e3
    tilt_x = np.arctan(tx / thickness) * 1e3
    tilt_y = -np.arctan(ty / thickness) * 1e3  # Negative for coordinate convention

    return tilt_x, tilt_y
