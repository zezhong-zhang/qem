"""Optimization engine for ptychography phase quantification.

The optimization iteratively updates:
1. Atomic positions (x, y)
2. Phase values at each position
3. PSF width (convergence angle effect)
4. Sample tilt (optional)
5. Global transformations (translation, scale, rotation)

This implementation follows the PyTorch optimization pattern used in QEM
(Adam, AdamW, SGD optimizers).
"""

import copy
from dataclasses import dataclass
from typing import Optional, Tuple, Union, List, Callable

import numpy as np
import torch

from qem.optics import (
    Aberrations,
    Grid,
    Probe,
    adf_psf,
    epie_psf,
    icom_psf,
    ssb_psf,
)
from qem.processing.psf import calculate_psf_width
from qem.fit.point_potential import (
    PointPotentialModel,
    correlation_coefficient,
    normalized_root_mean_square_error,
    calculate_residual,
)
from qem.utils.params import safe_convert_to_numpy, safe_convert_to_tensor


def _bilinear_point_potential(
    pos_x: torch.Tensor,
    pos_y: torch.Tensor,
    phases: torch.Tensor,
    ny: int,
    nx: int,
) -> torch.Tensor:
    """Differentiable bilinear point-potential map.

    Each atomic site contributes its phase to the four neighbouring pixels
    weighted by the bilinear factors. The whole computation is in PyTorch
    so gradients flow back into ``pos_x``, ``pos_y`` and ``phases``.
    """
    floor_x = torch.floor(pos_x)
    floor_y = torch.floor(pos_y)
    fx = pos_x - floor_x
    fy = pos_y - floor_y

    ix0 = floor_x.long()
    iy0 = floor_y.long()
    ix1 = ix0 + 1
    iy1 = iy0 + 1

    def _scatter(yi: torch.Tensor, xi: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        in_bounds = (yi >= 0) & (yi < ny) & (xi >= 0) & (xi < nx)
        yi_safe = yi.clamp(0, ny - 1)
        xi_safe = xi.clamp(0, nx - 1)
        flat_idx = yi_safe * nx + xi_safe
        contrib = weight * phases * in_bounds.to(phases.dtype)
        flat = torch.zeros(ny * nx, dtype=phases.dtype, device=phases.device)
        flat = flat.scatter_add(0, flat_idx, contrib)
        return flat.reshape(ny, nx)

    return (
        _scatter(iy0, ix0, (1 - fx) * (1 - fy))
        + _scatter(iy0, ix1, fx * (1 - fy))
        + _scatter(iy1, ix0, (1 - fx) * fy)
        + _scatter(iy1, ix1, fx * fy)
    )


def _fft_convolve_same(image: torch.Tensor, kernel: torch.Tensor) -> torch.Tensor:
    """FFT-based 'same'-mode 2-D convolution that preserves autograd."""
    image = image.to(dtype=torch.float32)
    kernel = kernel.to(dtype=torch.float32, device=image.device)

    H, W = image.shape
    kh, kw = kernel.shape
    full_h = H + kh - 1
    full_w = W + kw - 1

    image_f = torch.fft.rfft2(image, s=(full_h, full_w))
    kernel_f = torch.fft.rfft2(kernel, s=(full_h, full_w))
    full = torch.fft.irfft2(image_f * kernel_f, s=(full_h, full_w))

    start_y = (kh - 1) // 2
    start_x = (kw - 1) // 2
    return full[start_y : start_y + H, start_x : start_x + W]


@dataclass
class OptimizationResult:
    """Result of ptychography optimization."""

    positions: np.ndarray  # Optimized atomic positions (N x 2)
    phases: np.ndarray  # Optimized phase values (N,)
    correlation: float  # Final correlation coefficient
    nrmse: float  # Final normalized RMS error
    tilt_x: float  # Optimized x-tilt in mrad
    tilt_y: float  # Optimized y-tilt in mrad
    psf_width: float  # Optimized PSF width
    n_iterations: int  # Number of iterations performed
    converged: bool  # Whether optimization converged
    history: Optional[dict]  # Optimization history


class ConvolutionModel(torch.nn.Module):
    """PyTorch model for convolution-based ptychography optimization."""

    def __init__(self, psf_kernel, potential_model):
        super().__init__()
        self.psf_kernel = torch.as_tensor(psf_kernel.astype(np.float32))
        self.potential_model = potential_model
        self.input_params: dict | None = None
        self.built = False

    def set_params(self, params):
        """Set model parameters; build lazily on first use."""
        self.input_params = {k: torch.as_tensor(v) for k, v in params.items()}
        if self.built:
            self.update_params(self.input_params)

    def update_params(self, params):
        """Copy new values into the existing nn.Parameter slots."""
        with torch.no_grad():
            for key, value in params.items():
                if hasattr(self, key):
                    target = getattr(self, key)
                    target.copy_(torch.as_tensor(value, dtype=target.dtype).to(target.device))

    def build(self, input_shape=None):
        if self.input_params is None:
            raise ValueError("set_params() must run before build().")
        if self.built:
            return
        ip = self.input_params
        # Always-present parameters
        self.pos_x = torch.nn.Parameter(torch.as_tensor(ip['pos_x'], dtype=torch.float32).clone())
        self.pos_y = torch.nn.Parameter(torch.as_tensor(ip['pos_y'], dtype=torch.float32).clone())
        self.phases = torch.nn.Parameter(torch.as_tensor(ip['phases'], dtype=torch.float32).clone())
        # Optional parameters
        for opt_key in ("tilt_x", "tilt_y", "psf_scale"):
            if opt_key in ip:
                value = torch.as_tensor(ip[opt_key], dtype=torch.float32).clone()
                if value.dim() == 0:
                    pass  # scalar
                self.register_parameter(opt_key, torch.nn.Parameter(value))
        self.built = True

    def get_params(self):
        out = {
            'pos_x': self.pos_x.detach(),
            'pos_y': self.pos_y.detach(),
            'phases': self.phases.detach(),
        }
        for opt_key in ("tilt_x", "tilt_y", "psf_scale"):
            if hasattr(self, opt_key):
                out[opt_key] = getattr(self, opt_key).detach()
        return out

    def forward(self, inputs):
        return self.call(inputs)

    def call(self, inputs):
        """Forward pass — fully differentiable PyTorch simulation.

        Builds the point-potential map with bilinear scatter (so gradients
        flow through both ``pos_x``/``pos_y`` and ``phases``) and convolves
        it with the PSF via FFT. The legacy NumPy/SciPy path in
        :class:`PointPotentialModel` is retained for non-differentiable
        callers but is not used inside the optimisation loop.
        """
        gridshape = inputs[0] if isinstance(inputs, (list, tuple)) else inputs
        if isinstance(gridshape, torch.Tensor):
            ny, nx = int(gridshape[0]), int(gridshape[1])
        else:
            ny, nx = int(gridshape[0]), int(gridshape[1])

        pos_x = self.pos_x
        pos_y = self.pos_y
        phases = self.phases

        if hasattr(self, "tilt_x") and hasattr(self, "tilt_y"):
            # Tilt is applied as a constant shift; preserves autograd through phases/positions.
            thickness = 6.0
            shift_x = thickness * torch.tan(self.tilt_x * 1e-3)
            shift_y = thickness * torch.tan(self.tilt_y * 1e-3)
            pos_x = pos_x + shift_x
            pos_y = pos_y + shift_y

        potential = _bilinear_point_potential(pos_x, pos_y, phases, ny, nx)

        psf = self.psf_kernel
        if hasattr(self, "psf_scale"):
            scale = self.psf_scale
            if not torch.equal(scale, torch.ones_like(scale)):
                # Scaling the PSF is a non-differentiable resampling step in
                # the legacy NumPy implementation; keep gradients off this
                # branch by detaching, optimise it via finite differences if
                # ever needed.
                psf_np = self._scale_psf(safe_convert_to_numpy(psf), float(scale.detach()))
                psf = safe_convert_to_tensor(psf_np.astype(np.float32))

        return _fft_convolve_same(potential, psf)

    def _apply_tilt(self, pos_x, pos_y, tilt_x, tilt_y, thickness=6.0):
        """Apply sample tilt to positions."""
        shift_x = thickness * np.tan(tilt_x * 1e-3)
        shift_y = thickness * np.tan(tilt_y * 1e-3)
        return pos_x + shift_x, pos_y + shift_y

    def _scale_psf(self, psf, scale):
        """Scale PSF by a factor."""
        if scale == 1.0:
            return psf

        from scipy.ndimage import zoom

        ny, nx = psf.shape
        new_ny = int(round(ny * scale))
        new_nx = int(round(nx * scale))

        if new_ny < 3 or new_nx < 3:
            return psf

        psf_zoomed = zoom(psf, scale, order=1)
        psf_scaled = np.zeros_like(psf)

        ny_zoom, nx_zoom = psf_zoomed.shape
        y_start = max(0, (ny_zoom - ny) // 2)
        x_start = max(0, (nx_zoom - nx) // 2)

        y_end = min(y_start + ny, ny_zoom)
        x_end = min(x_start + nx, nx_zoom)

        out_y_start = max(0, (ny - ny_zoom) // 2)
        out_x_start = max(0, (nx - nx_zoom) // 2)

        out_y_end = min(out_y_start + (y_end - y_start), ny)
        out_x_end = min(out_x_start + (x_end - x_start), nx)

        if (y_end > y_start) and (x_end > x_start):
            psf_scaled[out_y_start:out_y_end, out_x_start:out_x_end] = \
                psf_zoomed[y_start:y_end, x_start:x_end]

        return psf_scaled


class PtychographyOptimizer:
    """Optimizer for ptychographic phase quantification using Keras.

    Implements the method from Figure 3 in the paper:
    1. Create point-potential from atomic model
    2. Convolve with PSF to simulate image
    3. Calculate correlation with target
    4. Update positions, phases, PSF width
    5. Repeat until convergence

    This implementation uses Keras optimizers (Adam, AdamW, SGD) for
    gradient-based optimization, following the existing QEM codebase pattern.
    """

    def __init__(
        self,
        target_image: np.ndarray,
        ctf_type: str = "SSB",
        alpha: float = 20.0,
        eV: float = 60e3,
        df: float = 0.0,
        aberrations: Optional[list] = None,
        detector_inner: Optional[float] = None,
        detector_outer: Optional[float] = None,
        high_pass_cutoff: Optional[float] = None,
        psf_kernel: Optional[np.ndarray] = None,
    ):
        """
        Initialize the optimizer.

        Parameters
        ----------
        target_image : np.ndarray
            Target ptychographic phase image to fit
        ctf_type : str, optional
            Type of CTF ('SSB', 'ADF', 'ePIE', 'iCoM'). Defaults to 'SSB'.
        alpha : float, optional
            Convergence angle in mrad. Defaults to 20.0.
        eV : float, optional
            Acceleration voltage in eV. Defaults to 60e3 (60 kV).
        df : float, optional
            Defocus in Angstroms. Defaults to 0.0.
        aberrations : list of aberration objects, optional
            List of aberrations to include
        detector_inner : float, optional
            Inner detector angle for ADF (mrad)
        detector_outer : float, optional
            Outer detector angle for ADF (mrad)
        high_pass_cutoff : float, optional
            High-pass filter cutoff for iCoM (mrad)
        psf_kernel : np.ndarray, optional
            Pre-computed PSF kernel (overrides CTF calculation)
        """
        self.target_image = target_image.astype(np.float32)
        self.ny, self.nx = target_image.shape
        self.ctf_type = ctf_type
        self.real_dim = (self.ny, self.nx)  # Assuming pixel size = 1 Angstrom

        # Build PSF directly via the new optics API
        self.ctf = None  # legacy attr; PSF is what downstream code consumes
        if psf_kernel is not None:
            self.psf = psf_kernel
        else:
            self.psf = self._compute_psf(
                ctf_type, alpha, eV, df, aberrations,
                detector_inner, detector_outer, high_pass_cutoff,
            )

        self.psf_width = calculate_psf_width(self.psf)

        # Initialize point-potential model
        self.potential_model = PointPotentialModel()

        # Create Keras model
        self.model = ConvolutionModel(self.psf, self.potential_model)

    def _compute_psf(
        self,
        ctf_type: str,
        alpha: float,
        eV: float,
        df: float,
        aberrations: Optional[list],
        detector_inner: Optional[float],
        detector_outer: Optional[float],
        high_pass_cutoff: Optional[float],
    ) -> np.ndarray:
        """Build a real-space PSF for the requested STEM imaging mode."""
        if aberrations is None or len(aberrations) == 0:
            ab = Aberrations(defocus=df) if df else Aberrations()
        elif isinstance(aberrations, Aberrations):
            ab = aberrations  # already the new dataclass
        else:
            raise TypeError(
                f"aberrations must be an Aberrations dataclass or empty, "
                f"got {type(aberrations).__name__}"
            )
        probe = Probe(energy=eV, aperture=alpha, aberrations=ab)
        grid = Grid(pixels=(self.ny, self.nx), extent=tuple(self.real_dim))
        if ctf_type == "SSB":
            psf = ssb_psf(grid, probe)
        elif ctf_type == "ADF":
            if detector_inner is None or detector_outer is None:
                raise ValueError(
                    "ADF requires detector_inner and detector_outer angles"
                )
            psf = adf_psf(grid, probe)
        elif ctf_type == "ePIE":
            psf = epie_psf(grid, probe)
        elif ctf_type == "iCoM":
            psf = icom_psf(grid, probe, high_pass_mrad=high_pass_cutoff)
        else:
            raise ValueError(f"Unknown CTF type: {ctf_type}")
        return psf.detach().cpu().numpy()

    def _loss_function(self, y_true, y_pred):
        """
        Loss function for optimization.

        Uses negative correlation as the loss (to maximize correlation).

        Parameters
        ----------
        y_true : tensor
            Target image
        y_pred : tensor
            Simulated image

        Returns
        -------
        loss : tensor
            Negative correlation coefficient
        """
        # Flatten images
        y_true_flat = torch.reshape(y_true, (-1,))
        y_pred_flat = torch.reshape(y_pred, (-1,))

        # Calculate means
        mu_true = torch.mean(y_true_flat)
        mu_pred = torch.mean(y_pred_flat)

        # Calculate standard deviations
        sigma_true = torch.std(y_true_flat)
        sigma_pred = torch.std(y_pred_flat)

        # Avoid division by zero
        epsilon = 1e-10
        sigma_true = torch.maximum(sigma_true, epsilon)
        sigma_pred = torch.maximum(sigma_pred, epsilon)

        # Calculate correlation
        n = tuple(y_true_flat.shape)[0]
        centered_true = y_true_flat - mu_true
        centered_pred = y_pred_flat - mu_pred
        numerator = torch.sum(centered_true * centered_pred)
        denominator = (n - 1) * sigma_true * sigma_pred

        correlation = numerator / denominator

        # Return negative correlation (to minimize)
        return -correlation

    def optimize(
        self,
        initial_positions: np.ndarray,
        initial_phases: np.ndarray,
        optimize_tilt: bool = False,
        optimize_psf_width: bool = False,
        max_iterations: int = 100,
        tolerance: float = 1e-6,
        step_size: float = 0.01,
        optimizer: str = "adam",
        verbose: bool = True,
        **optimizer_kwargs
    ) -> OptimizationResult:
        """
        Run optimization to fit the target image.

        Parameters
        ----------
        initial_positions : np.ndarray (N x 2)
            Initial atomic positions [x, y] in pixels
        initial_phases : np.ndarray (N,)
            Initial phase values
        optimize_tilt : bool, optional
            Whether to optimize sample tilt. Defaults to False.
        optimize_psf_width : bool, optional
            Whether to optimize PSF width. Defaults to False.
        max_iterations : int, optional
            Maximum number of iterations. Defaults to 100.
        tolerance : float, optional
            Convergence tolerance. Defaults to 1e-6.
        step_size : float, optional
            Learning rate. Defaults to 0.01.
        optimizer : str, optional
            Optimizer type ('adam', 'adamw', 'sgd'). Defaults to 'adam'.
        verbose : bool, optional
            Whether to print progress. Defaults to True.
        **optimizer_kwargs : additional arguments passed to optimizer

        Returns
        -------
        result : OptimizationResult
            Optimization results
        """
        n_atoms = len(initial_phases)
        initial_positions = np.asarray(initial_positions)
        initial_phases = np.asarray(initial_phases)

        # Prepare initial parameters
        params = {
            'pos_x': initial_positions[:, 0].astype(np.float32),
            'pos_y': initial_positions[:, 1].astype(np.float32),
            'phases': initial_phases.astype(np.float32),
        }

        if optimize_tilt:
            params['tilt_x'] = np.array(0.0, dtype=np.float32)
            params['tilt_y'] = np.array(0.0, dtype=np.float32)

        if optimize_psf_width:
            params['psf_scale'] = np.array(1.0, dtype=np.float32)

        # Set parameters and build model
        self.model.set_params(params)
        self.model.build()

        # Prepare target image tensor
        target_tensor = safe_convert_to_tensor(self.target_image)
        gridshape = np.array([self.ny, self.nx], dtype=np.int32)

        # Pure-PyTorch training loop with explicit progress tracking and
        # early stopping (replaces the legacy keras.callbacks plumbing).
        from qem.fit._loop import make_optimizer

        opt = make_optimizer(optimizer, self.model.parameters(), step_size)
        history: dict[str, list] = {'loss': [], 'correlation': [], 'nrmse': []}

        best_loss = float('inf')
        best_state: dict | None = None
        epochs_no_improve = 0
        patience = 20

        for epoch in range(max_iterations):
            opt.zero_grad(set_to_none=True)
            simulated = self.model(gridshape)
            loss = self._loss_function(target_tensor, simulated)
            loss.backward()
            opt.step()

            loss_val = float(loss.detach())
            correlation = -loss_val
            simulated_np = safe_convert_to_numpy(simulated)
            nrmse = normalized_root_mean_square_error(simulated_np, self.target_image)
            history['loss'].append(loss_val)
            history['correlation'].append(correlation)
            history['nrmse'].append(nrmse)

            if verbose and (epoch % 10 == 0 or epoch == max_iterations - 1):
                print(f"Epoch {epoch}: correlation={correlation:.6f}, nrmse={nrmse:.6f}")

            if loss_val < best_loss - tolerance:
                best_loss = loss_val
                best_state = {k: v.detach().clone() for k, v in self.model.state_dict().items()}
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= patience:
                    break

        if best_state is not None:
            self.model.load_state_dict(best_state)

        # Extract optimized parameters
        opt_params = self.model.get_params()
        opt_positions = np.column_stack([
            safe_convert_to_numpy(opt_params['pos_x']),
            safe_convert_to_numpy(opt_params['pos_y']),
        ])
        opt_phases = safe_convert_to_numpy(opt_params['phases'])

        # Extract optional parameters
        opt_tilt_x = 0.0
        opt_tilt_y = 0.0
        if 'tilt_x' in opt_params:
            opt_tilt_x = float(safe_convert_to_numpy(opt_params['tilt_x']))
        if 'tilt_y' in opt_params:
            opt_tilt_y = float(safe_convert_to_numpy(opt_params['tilt_y']))

        opt_psf_scale = 1.0
        if 'psf_scale' in opt_params:
            opt_psf_scale = float(safe_convert_to_numpy(opt_params['psf_scale']))

        # Calculate final metrics
        simulated = self.model(gridshape)
        final_sim = safe_convert_to_numpy(simulated)
        final_corr = correlation_coefficient(final_sim, self.target_image)
        final_nrmse = normalized_root_mean_square_error(final_sim, self.target_image)

        # Check convergence
        converged = len(history['loss']) > 0 and history['loss'][-1] < tolerance

        return OptimizationResult(
            positions=opt_positions,
            phases=opt_phases,
            correlation=final_corr,
            nrmse=final_nrmse,
            tilt_x=opt_tilt_x,
            tilt_y=opt_tilt_y,
            psf_width=self.psf_width * opt_psf_scale,
            n_iterations=len(history['loss']),
            converged=converged,
            history=history,
        )


class ADFConvolutionFitting:
    """ADF fitting treating image as probe ⊗ potential convolution."""

    def __init__(
        self,
        image: np.ndarray,
        eV: float,
        alpha: float,
        detector_inner: float,
        detector_outer: float,
        df: float = 0.0,
        aberrations: Optional[list] = None,
    ):
        """
        Initialize ADF convolution fitting.

        Parameters
        ----------
        image : np.ndarray
            ADF image to fit
        eV : float
            Acceleration voltage (eV)
        alpha : float
            Probe convergence angle (mrad)
        detector_inner : float
            Inner detector angle (mrad)
        detector_outer : float
            Outer detector angle (mrad)
        df : float, optional
            Defocus (Å)
        aberrations : list, optional
            List of aberration objects
        """
        self.image = image.astype(np.float32)
        self.ny, self.nx = image.shape

        # Remember construction params so :meth:`fit` can pass them through
        self.alpha = alpha
        self.eV = eV
        self.df = df
        self.aberrations = aberrations
        self.detector_inner = detector_inner
        self.detector_outer = detector_outer

        # Build ADF PSF directly via the new optics API
        ab = aberrations if isinstance(aberrations, Aberrations) else (
            Aberrations(defocus=df) if df else Aberrations()
        )
        probe = Probe(energy=eV, aperture=alpha, aberrations=ab)
        grid = Grid(pixels=(self.ny, self.nx), extent=(self.ny, self.nx))
        self.psf = adf_psf(grid, probe).detach().cpu().numpy()
        self.potential_model = PointPotentialModel()

    def fit(
        self,
        initial_positions: np.ndarray,
        initial_intensities: np.ndarray,
        optimize_tilt: bool = False,
        optimize_psf_width: bool = False,
        **kwargs,
    ) -> OptimizationResult:
        """
        Fit ADF image using convolution model.

        Parameters
        ----------
        initial_positions : np.ndarray (N x 2)
            Initial atomic positions [x, y] in pixels
        initial_intensities : np.ndarray (N,)
            Initial intensity values
        optimize_tilt : bool, optional
            Whether to optimize sample tilt. Defaults to False.
        optimize_psf_width : bool, optional
            Whether to optimize PSF width. Defaults to False.
        **kwargs : additional arguments passed to optimizer

        Returns
        -------
        result : OptimizationResult
            Fitting results
        """
        # Create a specialized optimizer for ADF
        optimizer = PtychographyOptimizer(
            target_image=self.image,
            ctf_type="ADF",
            alpha=self.alpha,
            eV=self.eV,
            df=self.df,
            aberrations=self.aberrations,
            detector_inner=self.detector_inner,
            detector_outer=self.detector_outer,
            psf_kernel=self.psf,
        )

        # Run optimization (phases = intensities for ADF)
        result = optimizer.optimize(
            initial_positions=initial_positions,
            initial_phases=initial_intensities,
            optimize_tilt=optimize_tilt,
            optimize_psf_width=optimize_psf_width,
            **kwargs,
        )

        return result
