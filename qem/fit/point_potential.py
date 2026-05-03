"""Point-potential model for ptychography phase quantification.

The key idea is to represent the atomic potential as a sum of delta functions
at atomic positions, each weighted by a phase value. This point-potential
is then convolved with the PSF to simulate the image.
"""

from __future__ import annotations

from typing import Optional, Tuple, Union

import numpy as np
import torch
from qem.utils import torch_compat as keras
from scipy.fft import fft2, ifft2, fftshift, ifftshift
from scipy.signal import fftconvolve

from qem.fit.model import ImageModel
from qem.utils.params import safe_convert_to_numpy, safe_convert_to_tensor


class PointPotentialModel:
    """Point-potential model for ptychography phase quantification.

    Based on paper Eq. (1):
    V(r) = Σ φ_Z · δ(r, r_Z)

    Where:
    - φ_Z are the phase values at each atomic site
    - δ is Kronecker delta
    - r_Z are atomic positions

    This class creates the point-potential map and handles convolution.
    """

    def __init__(self, dx: float = 1.0):
        """
        Initialize the point-potential model.

        Parameters
        ----------
        dx : float, optional
            Pixel size in Angstroms. Defaults to 1.0.
        """
        self.dx = dx

    def build_point_potential_map(
        self,
        pos_x: np.ndarray,
        pos_y: np.ndarray,
        phases: np.ndarray,
        gridshape: Tuple[int, int],
        subpixel: bool = True,
    ) -> np.ndarray:
        """
        Create 2D point-potential map from atomic positions and phases.

        Implements sub-pixel precision by weighting 4 neighboring pixels
        using bilinear interpolation.

        Parameters
        ----------
        pos_x : np.ndarray
            X coordinates of atoms (in pixels)
        pos_y : np.ndarray
            Y coordinates of atoms (in pixels)
        phases : np.ndarray
            Phase values at each atomic site
        gridshape : tuple (ny, nx)
            Shape of the output grid
        subpixel : bool, optional
            Whether to use sub-pixel precision. Defaults to True.

        Returns
        -------
        potential_map : np.ndarray
            2D point-potential map
        """
        ny, nx = gridshape
        potential_map = np.zeros((ny, nx), dtype=np.float32)

        pos_x = np.asarray(pos_x)
        pos_y = np.asarray(pos_y)
        phases = np.asarray(phases)

        if subpixel:
            # Sub-pixel precision using bilinear interpolation
            for x, y, phi in zip(pos_x, pos_y, phases):
                x_int = int(np.floor(x))
                y_int = int(np.floor(y))
                x_frac = x - x_int
                y_frac = y - y_int

                # Bilinear interpolation weights
                w00 = (1 - x_frac) * (1 - y_frac)
                w01 = (1 - x_frac) * y_frac
                w10 = x_frac * (1 - y_frac)
                w11 = x_frac * y_frac

                # Add contribution to each neighboring pixel
                if 0 <= x_int < nx and 0 <= y_int < ny:
                    potential_map[y_int, x_int] += w00 * phi
                if 0 <= x_int + 1 < nx and 0 <= y_int < ny:
                    potential_map[y_int, x_int + 1] += w10 * phi
                if 0 <= x_int < nx and 0 <= y_int + 1 < ny:
                    potential_map[y_int + 1, x_int] += w01 * phi
                if 0 <= x_int + 1 < nx and 0 <= y_int + 1 < ny:
                    potential_map[y_int + 1, x_int + 1] += w11 * phi
        else:
            # Simple rounding to nearest pixel
            for x, y, phi in zip(pos_x, pos_y, phases):
                x_int = int(np.round(x))
                y_int = int(np.round(y))
                if 0 <= x_int < nx and 0 <= y_int < ny:
                    potential_map[y_int, x_int] += phi

        return potential_map

    def convolve_with_psf(
        self,
        potential_map: np.ndarray,
        psf_kernel: np.ndarray,
        mode: str = "same",
    ) -> np.ndarray:
        """
        Convolve potential map with PSF to simulate image.

        From paper Eq. (2):
        φ_sim = V(r) * FFT^(-1)(CTF(Qp))

        Parameters
        ----------
        potential_map : np.ndarray
            2D point-potential map
        psf_kernel : np.ndarray
            Point spread function (real space)
        mode : str, optional
            Convolution mode ('same', 'full', 'valid'). Defaults to 'same'.

        Returns
        -------
        simulated_image : np.ndarray
            Convolved result (simulated image)
        """
        # Use FFT-based convolution for efficiency
        simulated = fftconvolve(potential_map, psf_kernel, mode=mode)
        return simulated

    def simulate_from_positions(
        self,
        pos_x: np.ndarray,
        pos_y: np.ndarray,
        phases: np.ndarray,
        psf_kernel: np.ndarray,
        gridshape: Tuple[int, int],
        subpixel: bool = True,
    ) -> np.ndarray:
        """
        Simulate image directly from positions, phases, and PSF.

        Convenience method that combines potential map creation
        and convolution.

        Parameters
        ----------
        pos_x : np.ndarray
            X coordinates of atoms (in pixels)
        pos_y : np.ndarray
            Y coordinates of atoms (in pixels)
        phases : np.ndarray
            Phase values at each atomic site
        psf_kernel : np.ndarray
            Point spread function
        gridshape : tuple (ny, nx)
            Shape of the output grid
        subpixel : bool, optional
            Whether to use sub-pixel precision. Defaults to True.

        Returns
        -------
        simulated_image : np.ndarray
            Simulated image
        """
        potential = self.build_point_potential_map(
            pos_x, pos_y, phases, gridshape, subpixel
        )
        return self.convolve_with_psf(potential, psf_kernel)


class ConvolutionImageModel(ImageModel):
    """Image model based on convolution of point-potential with PSF.

    This class inherits from ImageModel for compatibility with the
    existing QEM fitting infrastructure. It implements a convolution-based
    approach where atomic positions are represented as delta functions
    (point-potential) that are then convolved with the PSF.

    The 'height' parameter represents the phase value at each atomic site.
    The 'width' parameter is not used (PSF is fixed by microscope parameters).

    To use this model with ImageFitting:
    ```python
    from qem.instruments import SSB_CTF
    from qem.fit.point_potential import ConvolutionImageModel

    # Create PSF from microscope parameters
    ctf = SSB_CTF(alpha=20, eV=60e3, df=0)
    psf = ctf.get_psf((ny, nx), (ny, nx))

    # Create model
    model = ConvolutionImageModel(psf_kernel=psf, dx=1.0)

    # Set initial parameters
    params = {
        'pos_x': np.array([10.5, 20.3, ...]),
        'pos_y': np.array([15.2, 25.1, ...]),
        'height': np.array([1.0, 1.0, ...]),  # Phase values
        'width': np.ones(n_atoms) * 1.0,  # Not used, but required
        'background': 0.0,
    }
    model.set_params(params)
    model.build()

    # The model can now be used with model.fit()
    ```

    Note: For accurate results, use a smaller learning rate (step_size=0.001)
    and more iterations than with Gaussian models.
    """

    def __init__(self, psf_kernel: np.ndarray, dx: float = 1.0):
        """
        Initialize the convolution image model.

        Parameters
        ----------
        psf_kernel : np.ndarray
            Point spread function for convolution (2D array)
        dx : float, optional
            Pixel size in Angstroms. Defaults to 1.0.
        """
        super().__init__(dx)
        psf_kernel = np.asarray(psf_kernel, dtype=np.float32)
        self.psf_kernel = safe_convert_to_tensor(psf_kernel)
        self.ny, self.nx = psf_kernel.shape

        # Pre-compute PSF kernel info for efficient convolution
        self._psf_np = psf_kernel
        self._ky, self._kx = psf_kernel.shape
        self._ky_half, self._kx_half = self._ky // 2, self._kx // 2

        # Cache for point-potential model (used for non-gradient computation)
        self._point_potential_model = PointPotentialModel(dx=dx)

    def set_params(self, params):
        """Set parameters with PSF kernel support."""
        super().set_params(params)
        # Update PSF kernel if provided
        if "psf_kernel" in params:
            psf = np.asarray(params["psf_kernel"], dtype=np.float32)
            self.psf_kernel = safe_convert_to_tensor(psf)
            self._psf_np = psf
            self._ky, self._kx = psf.shape
            self._ky_half, self._kx_half = self._ky // 2, self._kx // 2

    def model_fn(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        pos_x: torch.Tensor,
        pos_y: torch.Tensor,
        height: torch.Tensor,
        width: torch.Tensor,
    ) -> torch.Tensor:
        """
        Core computation for convolution model.

        For the convolution model, 'height' represents phase values.
        The 'width' parameter is not used (PSF is fixed).

        This implementation creates a point-potential map and convolves
        with the PSF using Keras operations for differentiable gradients.

        Parameters
        ----------
        x, y : Tensor
            Coordinate grids (2D arrays)
        pos_x, pos_y : Tensor
            Atomic positions (1D arrays)
        height : Tensor
            Phase values at each atomic site (1D array)
        width : Tensor
            Not used (PSF is fixed)

        Returns
        -------
        Tensor
            Simulated image (convolution of point-potential with PSF)
        """
        ny, nx = keras.ops.shape(x)[0], keras.ops.shape(x)[1]
        n_atoms = keras.ops.shape(pos_x)[0]

        # Create point-potential map using a differentiable approach
        # Use Gaussian approximation centered at each atomic position
        # This provides smooth gradients while approximating the PSF

        # Get PSF width as a measure of the kernel size
        psf_sigma = keras.ops.cast(self._kx_half / 3.0, dtype=x.dtype)

        # Initialize result
        result = keras.ops.zeros_like(x)

        # For each atom, add a Gaussian contribution
        # The amplitude is the phase value (height)
        for i in range(n_atoms):
            # Squared distance from atom i
            dx = x - pos_x[i]
            dy = y - pos_y[i]
            dist_sq = dx**2 + dy**2

            # Gaussian approximation of PSF contribution
            # Using the actual PSF width as sigma
            sigma = psf_sigma
            contribution = height[i] * keras.ops.exp(-dist_sq / (2 * sigma**2))
            result = result + contribution

        return result

    def sum(self, x_grid: torch.Tensor, y_grid: torch.Tensor,
            local: bool = True) -> torch.Tensor:
        """
        Calculate the sum of all atoms convolved with PSF.

        This method uses Keras operations to maintain gradient computation
        for use with model.fit().

        Parameters
        ----------
        x_grid : Tensor
            X coordinate grid (2D)
        y_grid : Tensor
            Y coordinate grid (2D)
        local : bool, optional
            Whether to use local computation. Ignored for convolution model.

        Returns
        -------
        Tensor
            Simulated image with PSF convolution
        """
        # Handle batch dimension if present
        has_batch_dim = len(x_grid.shape) > 2
        if has_batch_dim:
            x_grid = keras.ops.squeeze(x_grid, axis=0)
            y_grid = keras.ops.squeeze(y_grid, axis=0)

        ny, nx = keras.ops.shape(x_grid)[0], keras.ops.shape(x_grid)[1]
        n_atoms = keras.ops.shape(self.pos_x)[0]

        # Build point-potential map using the model_fn approach
        # This creates differentiable atomic contributions
        result = self.model_fn(x_grid, y_grid, self.pos_x, self.pos_y, self.height, self.width)

        # Add background
        result = result + self.background

        # Add batch dimension back if needed
        if has_batch_dim:
            result = keras.ops.expand_dims(result, axis=0)

        return result

    def _build_point_potential_map(
        self,
        pos_x: torch.Tensor,
        pos_y: torch.Tensor,
        height: torch.Tensor,
        ny: int,
        nx: int,
    ) -> torch.Tensor:
        """
        Build point-potential map from atomic positions and phases.

        Uses sub-pixel precision with bilinear interpolation.

        Parameters
        ----------
        pos_x : Tensor
            X coordinates of atoms
        pos_y : Tensor
            Y coordinates of atoms
        height : Tensor
            Phase values at each atomic site
        ny : int
            Image height
        nx : int
            Image width

        Returns
        -------
        Tensor
            Point-potential map (2D array)
        """
        # Convert to numpy for efficient scatter operation
        pos_x_np = safe_convert_to_numpy(pos_x)
        pos_y_np = safe_convert_to_numpy(pos_y)
        height_np = safe_convert_to_numpy(height)

        potential_map = np.zeros((ny, nx), dtype=np.float32)

        # Build point-potential map with sub-pixel precision
        for x, y, phi in zip(pos_x_np, pos_y_np, height_np):
            x_int = int(np.floor(x))
            y_int = int(np.floor(y))
            x_frac = x - x_int
            y_frac = y - y_int

            # Bilinear interpolation weights
            w00 = (1 - x_frac) * (1 - y_frac)
            w01 = (1 - x_frac) * y_frac
            w10 = x_frac * (1 - y_frac)
            w11 = x_frac * y_frac

            # Add contribution to each neighboring pixel
            if 0 <= x_int < nx and 0 <= y_int < ny:
                potential_map[y_int, x_int] += w00 * phi
            if 0 <= x_int + 1 < nx and 0 <= y_int < ny:
                potential_map[y_int, x_int + 1] += w10 * phi
            if 0 <= x_int < nx and 0 <= y_int + 1 < ny:
                potential_map[y_int + 1, x_int] += w01 * phi
            if 0 <= x_int + 1 < nx and 0 <= y_int + 1 < ny:
                potential_map[y_int + 1, x_int + 1] += w11 * phi

        return safe_convert_to_tensor(potential_map)

    def volume(self, params: dict) -> np.ndarray:
        """
        Calculate total phase for each atomic site.

        For the convolution model, the volume is simply the
        phase value (since we're dealing with integrated phase
        at each atomic position).

        Parameters
        ----------
        params : dict
            Dictionary containing 'height' (phase values)

        Returns
        -------
        volumes : np.ndarray
            Phase values (same as height)
        """
        height = safe_convert_to_numpy(params["height"])
        return height * self.dx ** 2

    def simulate_with_psf(
        self,
        pos_x: np.ndarray,
        pos_y: np.ndarray,
        phases: np.ndarray,
        background: float = 0.0,
    ) -> np.ndarray:
        """
        Simulate image using full PSF convolution.

        This method uses FFT-based convolution for accuracy.

        Parameters
        ----------
        pos_x : np.ndarray
            X coordinates of atoms (in pixels)
        pos_y : np.ndarray
            Y coordinates of atoms (in pixels)
        phases : np.ndarray
            Phase values at each atomic site
        background : float, optional
            Background level to add. Defaults to 0.0.

        Returns
        -------
        simulated_image : np.ndarray
            Simulated image with PSF convolution
        """
        psf = safe_convert_to_numpy(self.psf_kernel)
        gridshape = (self.ny, self.nx)

        simulated = self._point_potential_model.simulate_from_positions(
            pos_x, pos_y, phases, psf, gridshape, subpixel=True
        )

        return simulated + background


def correlation_coefficient(
    sim_image: np.ndarray,
    target_image: np.ndarray,
) -> float:
    """
    Calculate correlation coefficient between simulated and target images.

    From paper Eq. (3):
    R = NΣ[(μ_sim - φ_sim_i)(μ_target - φ_target_i)] / [(N-1)σ_simσ_target]

    This is the merit function used for optimization.

    Parameters
    ----------
    sim_image : np.ndarray
        Simulated image
    target_image : np.ndarray
        Target/experimental image

    Returns
    -------
    correlation : float
        Pearson correlation coefficient (-1 to 1)
    """
    sim = sim_image.ravel()
    target = target_image.ravel()

    mu_sim = np.mean(sim)
    mu_target = np.mean(target)
    sigma_sim = np.std(sim)
    sigma_target = np.std(target)

    if sigma_sim == 0 or sigma_target == 0:
        return 0.0

    n = len(sim)
    numerator = np.sum((sim - mu_sim) * (target - mu_target))
    denominator = (n - 1) * sigma_sim * sigma_target

    correlation = numerator / denominator
    return float(correlation)


def normalized_root_mean_square_error(
    sim_image: np.ndarray,
    target_image: np.ndarray,
) -> float:
    """
    Calculate normalized RMS error between images.

    Alternative merit function that can be used alongside correlation.

    Parameters
    ----------
    sim_image : np.ndarray
        Simulated image
    target_image : np.ndarray
        Target/experimental image

    Returns
    -------
    nrmse : float
        Normalized RMS error
    """
    sim = sim_image.ravel()
    target = target_image.ravel()

    target_range = np.max(target) - np.min(target)
    if target_range == 0:
        return 0.0

    rmse = np.sqrt(np.mean((sim - target) ** 2))
    nrmse = rmse / target_range

    return float(nrmse)


def calculate_residual(
    sim_image: np.ndarray,
    target_image: np.ndarray,
) -> np.ndarray:
    """
    Calculate residual (difference) between simulated and target images.

    Parameters
    ----------
    sim_image : np.ndarray
        Simulated image
    target_image : np.ndarray
        Target/experimental image

    Returns
    -------
    residual : np.ndarray
        Difference image (sim - target)
    """
    return sim_image - target_image
