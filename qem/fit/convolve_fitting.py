"""Convolution-based fitting for STEM images.

This module implements correlation-based optimization for different STEM imaging modes:
- SSB Ptychography
- ePIE (extended Ptychographic Iterative Engine)
- iCoM (integrated Center of Mass)
- ADF (Annular Dark Field)

Based on:
Hofer & Pennycook, "Reliable phase quantification in focused probe
electron ptychography of thin materials", Ultramicroscopy 254 (2023) 113829.

The key idea is that all these imaging modes can be treated as a convolution
of a point-potential (atomic positions with values) with a PSF derived from
the microscope's CTF.

This implementation uses the standard QEM ImageFitting infrastructure with
native PyTorch optimization.
"""

from dataclasses import dataclass
from typing import Optional, Tuple, Union, List

import numpy as np
from qem.utils import torch_compat as keras

from qem.instruments.ctf import (
    SSB_CTF,
    ADF_CTF,
    ePIE_CTF,
    iCoM_CTF,
    calculate_psf_width,
    ProbeParameters,
    create_probe_parameters,
)
from qem.fit.point_potential import (
    PointPotentialModel,
    ConvolutionImageModel,
    correlation_coefficient,
    normalized_root_mean_square_error,
    calculate_residual,
)
from qem.fit.image_fitting import ImageFitting
from qem.utils.params import safe_convert_to_numpy, safe_convert_to_tensor


@dataclass
class OptimizationResult:
    """Result of convolution-based optimization."""

    positions: np.ndarray  # Optimized atomic positions (N x 2)
    values: np.ndarray  # Optimized values (phase/intensity) (N,)
    correlation: float  # Final correlation coefficient
    nrmse: float  # Final normalized RMS error
    tilt_x: float  # Optimized x-tilt in mrad (if enabled)
    tilt_y: float  # Optimized y-tilt in mrad (if enabled)
    psf_width: float  # PSF width used
    n_iterations: int  # Number of iterations performed
    converged: bool  # Whether optimization converged
    history: Optional[dict]  # Optimization history

    @property
    def phases(self) -> np.ndarray:
        """Alias for values (for backward compatibility)."""
        return self.values


class ConvolutionFitting(ImageFitting):
    """General convolution-based fitting for STEM images.

    This class extends ImageFitting to support convolution-based fitting
    where the image is modeled as a convolution of a point-potential
    with a PSF derived from the microscope's CTF.

    The fitting works for:
    - SSB ptychography: phases are the optimized values
    - ADF imaging: intensities are the optimized values
    - ePIE: phases
    - iCoM: phases

    Parameters
    ----------
    image : np.ndarray
        Target image to fit
    dx : float, optional
        Pixel size in Angstroms. Defaults to 1.0.
    units : str, optional
        Units of image. Defaults to "A".
    elements : list of str, optional
        Element types in the image.
    ctf_type : str, optional
        Type of CTF ('SSB', 'ADF', 'ePIE', 'iCoM'). Defaults to 'SSB'.
    probe_params : ProbeParameters or dict, optional
        Microscope probe parameters
    psf_kernel : np.ndarray, optional
        Pre-computed PSF kernel (overrides CTF calculation)
    **kwargs : additional arguments passed to ImageFitting

    Examples
    --------
    >>> from qem.fit import ConvolutionFitting
    >>> from qem.instruments import create_probe_parameters
    >>>
    >>> # Create probe parameters
    >>> probe = create_probe_parameters(alpha=20, eV=60e3, defocus=30)
    >>>
    >>> # Initialize fitter
    >>> fitter = ConvolutionFitting(
    ...     image=ssb_image,
    ...     ctf_type='SSB',
    ...     probe_params=probe,
    ... )
    >>>
    >>> # Set initial atomic positions
    >>> positions = np.array([[10.5, 20.3], [15.2, 25.1], ...])
    >>>
    >>> # Run fitting
    >>> result = fitter.fit_positions(
    ...     positions=positions,
    ...     initial_values=np.ones(len(positions)),
    ...     maxiter=100,
    ...     step_size=0.001,
    ... )
    """

    def __init__(
        self,
        image: np.ndarray,
        dx: float = 1.0,
        units: str = "A",
        elements: List[str] = None,
        ctf_type: str = "SSB",
        probe_params: Union[ProbeParameters, dict] = None,
        psf_kernel: np.ndarray = None,
        **kwargs
    ):
        """
        Initialize convolution fitter.

        Parameters
        ----------
        image : np.ndarray
            Target image to fit
        dx : float, optional
            Pixel size in Angstroms. Defaults to 1.0.
        units : str, optional
            Units of image. Defaults to "A".
        elements : list of str, optional
            Element types in the image.
        ctf_type : str, optional
            Type of CTF ('SSB', 'ADF', 'ePIE', 'iCoM'). Defaults to 'SSB'.
        probe_params : ProbeParameters or dict, optional
            Microscope probe parameters
        psf_kernel : np.ndarray, optional
            Pre-computed PSF kernel
        **kwargs : additional arguments passed to ImageFitting
        """
        # Store CTF and probe parameters
        self.ctf_type = ctf_type
        self.ny, self.nx = image.shape
        self.real_dim = (self.ny * dx, self.nx * dx)

        # Convert probe_params to ProbeParameters if needed
        if probe_params is None:
            self.probe_params = create_probe_parameters(alpha=20, eV=60e3)
        elif isinstance(probe_params, dict):
            self.probe_params = ProbeParameters.from_dict(probe_params)
        else:
            self.probe_params = probe_params

        # Get or create PSF
        if psf_kernel is not None:
            self.psf = psf_kernel
            self.ctf = None
        else:
            self.ctf = self._create_ctf()
            self.psf = self.ctf.get_psf((self.ny, self.nx), self.real_dim)

        self.psf_width = calculate_psf_width(self.psf)

        # Store the PSF for model creation
        self._psf_kernel = self.psf

        # Initialize parent ImageFitting with model_type='convolution'
        super().__init__(
            image=image,
            dx=dx,
            units=units,
            elements=elements,
            model_type='convolution',
            **kwargs
        )

    def _create_ctf(self):
        """Create CTF calculator based on type and probe parameters."""
        p = self.probe_params

        if self.ctf_type == "SSB":
            return SSB_CTF(
                alpha=p.alpha,
                eV=p.eV,
                df=p.df,
                aberrations=p.aberrations,
            )
        elif self.ctf_type == "ADF":
            if p.detector_inner is None or p.detector_outer is None:
                raise ValueError("ADF requires detector_inner and detector_outer angles")
            return ADF_CTF(
                alpha=p.alpha,
                eV=p.eV,
                detector_inner=p.detector_inner,
                detector_outer=p.detector_outer,
                df=p.df,
                aberrations=p.aberrations,
            )
        elif self.ctf_type == "ePIE":
            return ePIE_CTF(
                alpha=p.alpha,
                eV=p.eV,
                df=p.df,
            )
        elif self.ctf_type == "iCoM":
            return iCoM_CTF(
                alpha=p.alpha,
                eV=p.eV,
                high_pass_cutoff=p.high_pass_cutoff,
                df=p.df,
                aberrations=p.aberrations,
            )
        else:
            raise ValueError(f"Unknown CTF type: {self.ctf_type}")

    def _select_model(self):
        """Create convolution model."""
        return ConvolutionImageModel(
            psf_kernel=self._psf_kernel,
            dx=float(self.dx),
        )

    def init_params_from_positions(
        self,
        positions: np.ndarray,
        initial_values: Optional[np.ndarray] = None,
        background: float = 0.0,
    ) -> dict:
        """
        Initialize parameters dictionary from atomic positions.

        Parameters
        ----------
        positions : np.ndarray (N x 2)
            Atomic positions [x, y] in pixels
        initial_values : np.ndarray, optional
            Initial values (phases or intensities). Defaults to ones.
        background : float, optional
            Background level. Defaults to 0.0.

        Returns
        -------
        params : dict
            Parameters dictionary for ImageFitting
        """
        positions = np.asarray(positions)
        if positions.ndim == 1:
            positions = positions.reshape(-1, 1)
        if positions.shape[1] != 2:
            raise ValueError("Positions must be N x 2 array")

        n_atoms = len(positions)
        pos_x = positions[:, 0]
        pos_y = positions[:, 1]

        if initial_values is None:
            values = np.ones(n_atoms, dtype=np.float32)
        else:
            values = np.asarray(initial_values, dtype=np.float32)

        # Width is not used for convolution model but required by interface
        width = np.ones(n_atoms, dtype=np.float32)

        return {
            'pos_x': pos_x.astype(np.float32),
            'pos_y': pos_y.astype(np.float32),
            'height': values,  # 'height' represents phase/intensity for convolution model
            'width': width,
            'background': np.float32(background),
        }

    def fit_positions(
        self,
        positions: np.ndarray,
        initial_values: Optional[np.ndarray] = None,
        maxiter: int = 100,
        step_size: float = 0.001,
        optimizer: str = "adam",
        tol: float = 1e-6,
        verbose: bool = True,
        **optimizer_kwargs
    ) -> OptimizationResult:
        """
        Fit atomic positions and values to the target image.

        This method uses Keras model.fit() for optimization, following
        the standard QEM ImageFitting pattern.

        Parameters
        ----------
        positions : np.ndarray (N x 2)
            Initial atomic positions [x, y] in pixels
        initial_values : np.ndarray, optional
            Initial values (phases or intensities). Defaults to ones.
        maxiter : int, optional
            Maximum number of iterations. Defaults to 100.
        step_size : float, optional
            Learning rate. Default is 0.001 (smaller than Gaussian fitting).
        optimizer : str, optional
            Optimizer type ('adam', 'adamw', 'sgd'). Defaults to 'adam'.
        tol : float, optional
            Convergence tolerance. Defaults to 1e-6.
        verbose : bool, optional
            Whether to print progress. Defaults to True.
        **optimizer_kwargs : additional optimizer arguments

        Returns
        -------
        result : OptimizationResult
            Fitting results
        """
        # Initialize parameters
        params = self.init_params_from_positions(positions, initial_values)

        # Run optimization using ImageFitting.fit_global
        optimized_params = self.fit_global(
            params=params,
            maxiter=maxiter,
            step_size=step_size,
            optimizer=optimizer,
            tol=tol,
            verbose=verbose,
            **optimizer_kwargs
        )

        # Extract results
        opt_positions = np.column_stack([
            safe_convert_to_numpy(optimized_params['pos_x']),
            safe_convert_to_numpy(optimized_params['pos_y']),
        ])
        opt_values = safe_convert_to_numpy(optimized_params['height'])

        # Calculate final metrics
        sim_image = safe_convert_to_numpy(self.predict(optimized_params))
        corr = correlation_coefficient(sim_image, self.image)
        nrmse_val = normalized_root_mean_square_error(sim_image, self.image)

        return OptimizationResult(
            positions=opt_positions,
            values=opt_values,
            correlation=corr,
            nrmse=nrmse_val,
            tilt_x=0.0,  # Tilt optimization not yet implemented
            tilt_y=0.0,
            psf_width=self.psf_width,
            n_iterations=maxiter,
            converged=self.converged,
            history=None,
        )

    def get_values(self) -> np.ndarray:
        """Get optimized values (phases/intensities) from current parameters."""
        if self.params is None:
            raise ValueError("No parameters available. Run fit_positions() first.")
        return safe_convert_to_numpy(self.params['height'])

    def get_positions(self) -> np.ndarray:
        """Get optimized atomic positions from current parameters."""
        if self.params is None:
            raise ValueError("No parameters available. Run fit_positions() first.")
        return np.column_stack([
            safe_convert_to_numpy(self.params['pos_x']),
            safe_convert_to_numpy(self.params['pos_y']),
        ])

    def get_correlation(self) -> float:
        """Get current correlation coefficient."""
        if self.params is None:
            raise ValueError("No parameters available. Run fit_positions() first.")
        sim_image = safe_convert_to_numpy(self.predict(self.params))
        return correlation_coefficient(sim_image, self.image)

    def get_nrmse(self) -> float:
        """Get current normalized RMS error."""
        if self.params is None:
            raise ValueError("No parameters available. Run fit_positions() first.")
        sim_image = safe_convert_to_numpy(self.predict(self.params))
        return normalized_root_mean_square_error(sim_image, self.image)


class PtychographyFitting(ConvolutionFitting):
    """Ptychography phase quantification using convolution-based fitting.

    This is a specialized version of ConvolutionFitting for ptychography
    phase images (SSB, ePIE, iCoM). The optimized values represent phase
    values at each atomic site.

    Examples
    --------
    >>> from qem.fit import PtychographyFitting
    >>>
    >>> fitter = PtychographyFitting(
    ...     image=ssb_image,
    ...     alpha=20,           # mrad convergence angle
    ...     eV=60e3,            # 60 kV
    ...     defocus=30,         # Angstroms
    ... )
    >>>
    >>> result = fitter.fit_positions(
    ...     positions=initial_positions,
    ...     initial_phases=np.ones(n_atoms),
    ... )
    >>>
    >>> # Access optimized phases
    >>> phases = result.values
    """

    def __init__(
        self,
        image: np.ndarray,
        dx: float = 1.0,
        units: str = "A",
        elements: List[str] = None,
        ctf_type: str = "SSB",
        probe_params: Union[ProbeParameters, dict] = None,
        psf_kernel: np.ndarray = None,
        **kwargs
    ):
        """
        Initialize ptychography fitter.

        Parameters
        ----------
        image : np.ndarray
            Ptychography phase image (SSB, ePIE, or iCoM)
        dx : float, optional
            Pixel size in Angstroms. Defaults to 1.0.
        units : str, optional
            Units of image. Defaults to "A".
        elements : list of str, optional
            Element types in the image.
        ctf_type : str, optional
            Type of CTF ('SSB', 'ePIE', 'iCoM'). Defaults to 'SSB'.
        probe_params : ProbeParameters or dict, optional
            Microscope probe parameters
        psf_kernel : np.ndarray, optional
            Pre-computed PSF kernel
        **kwargs : additional arguments passed to ConvolutionFitting

        Quick Parameters
        ---------------
        For convenience, you can pass probe parameters directly as kwargs:
        alpha : float
            Convergence angle (mrad)
        eV : float
            Acceleration voltage (eV)
        defocus : float
            Defocus (Angstroms)
        aberrations : list
            List of aberration objects
        """
        # Extract probe parameters from kwargs if provided
        if probe_params is None:
            # Check if common probe parameters are in kwargs
            probe_kwargs = {}
            for key in ['alpha', 'eV', 'df', 'defocus', 'aberrations']:
                if key in kwargs:
                    probe_kwargs[key] = kwargs.pop(key)
            if probe_kwargs:
                probe_params = ProbeParameters.from_dict(probe_kwargs)
            else:
                # Default parameters
                probe_params = create_probe_parameters(alpha=20, eV=60e3)

        super().__init__(
            image=image,
            dx=dx,
            units=units,
            elements=elements,
            ctf_type=ctf_type,
            probe_params=probe_params,
            psf_kernel=psf_kernel,
            **kwargs
        )

    def fit_positions(
        self,
        positions: np.ndarray,
        initial_phases: Optional[np.ndarray] = None,
        maxiter: int = 100,
        step_size: float = 0.001,
        optimizer: str = "adam",
        tol: float = 1e-6,
        verbose: bool = True,
        **optimizer_kwargs
    ) -> OptimizationResult:
        """
        Fit atomic positions and phases to the ptychography image.

        Parameters
        ----------
        positions : np.ndarray (N x 2)
            Initial atomic positions [x, y] in pixels
        initial_phases : np.ndarray, optional
            Initial phase values. Defaults to ones.
        maxiter : int, optional
            Maximum number of iterations. Defaults to 100.
        step_size : float, optional
            Learning rate. Default is 0.001.
        optimizer : str, optional
            Optimizer type ('adam', 'adamw', 'sgd'). Defaults to 'adam'.
        tol : float, optional
            Convergence tolerance. Defaults to 1e-6.
        verbose : bool, optional
            Whether to print progress. Defaults to True.
        **optimizer_kwargs : additional optimizer arguments

        Returns
        -------
        result : OptimizationResult
            Fitting results with result.values containing phases
        """
        return super().fit_positions(
            positions=positions,
            initial_values=initial_phases,
            maxiter=maxiter,
            step_size=step_size,
            optimizer=optimizer,
            tol=tol,
            verbose=verbose,
            **optimizer_kwargs
        )

    def get_phases(self) -> np.ndarray:
        """Get optimized phase values from current parameters."""
        return self.get_values()


class ADFConvolutionFitting(ConvolutionFitting):
    """ADF image fitting using convolution model.

    This treats ADF images as a convolution of the probe with the
    sample potential. The optimized values represent intensity values
    at each atomic site.

    Examples
    --------
    >>> from qem.fit import ADFConvolutionFitting
    >>>
    >>> fitter = ADFConvolutionFitting(
    ...     image=adf_image,
    ...     alpha=20,           # mrad convergence angle
    ...     eV=60e3,            # 60 kV
    ...     detector_inner=50,  # mrad
    ...     detector_outer=200, # mrad
    ... )
    >>>
    >>> result = fitter.fit_positions(
    ...     positions=initial_positions,
    ...     initial_intensities=np.ones(n_atoms),
    ... )
    """

    def __init__(
        self,
        image: np.ndarray,
        alpha: float,
        eV: float,
        detector_inner: float,
        detector_outer: float,
        df: float = 0.0,
        aberrations: Optional[list] = None,
        dx: float = 1.0,
        units: str = "A",
        elements: List[str] = None,
        probe_params: Union[ProbeParameters, dict] = None,
        **kwargs
    ):
        """
        Initialize ADF convolution fitter.

        Parameters
        ----------
        image : np.ndarray
            ADF image to fit
        alpha : float
            Probe convergence angle (mrad)
        eV : float
            Acceleration voltage (eV)
        detector_inner : float
            Inner detector angle (mrad)
        detector_outer : float
            Outer detector angle (mrad)
        df : float, optional
            Defocus (Å). Defaults to 0.0.
        aberrations : list, optional
            List of aberration objects
        dx : float, optional
            Pixel size (Å). Defaults to 1.0.
        units : str, optional
            Units. Defaults to "A".
        elements : list of str, optional
            Element types
        probe_params : ProbeParameters or dict, optional
            Override probe parameters (if provided, other parameters ignored)
        **kwargs : additional arguments passed to ConvolutionFitting
        """
        # Create probe parameters for ADF if not provided
        if probe_params is None:
            probe_params = ProbeParameters(
                alpha=alpha,
                eV=eV,
                df=df,
                aberrations=aberrations,
                detector_inner=detector_inner,
                detector_outer=detector_outer,
            )

        super().__init__(
            image=image,
            dx=dx,
            units=units,
            elements=elements,
            ctf_type='ADF',
            probe_params=probe_params,
            **kwargs
        )

    def fit_positions(
        self,
        positions: np.ndarray,
        initial_intensities: Optional[np.ndarray] = None,
        maxiter: int = 100,
        step_size: float = 0.001,
        optimizer: str = "adam",
        tol: float = 1e-6,
        verbose: bool = True,
        **optimizer_kwargs
    ) -> OptimizationResult:
        """
        Fit atomic positions and intensities to the ADF image.

        Parameters
        ----------
        positions : np.ndarray (N x 2)
            Initial atomic positions [x, y] in pixels
        initial_intensities : np.ndarray, optional
            Initial intensity values. Defaults to ones.
        maxiter : int, optional
            Maximum number of iterations. Defaults to 100.
        step_size : float, optional
            Learning rate. Default is 0.001.
        optimizer : str, optional
            Optimizer type ('adam', 'adamw', 'sgd'). Defaults to 'adam'.
        tol : float, optional
            Convergence tolerance. Defaults to 1e-6.
        verbose : bool, optional
            Whether to print progress. Defaults to True.
        **optimizer_kwargs : additional optimizer arguments

        Returns
        -------
        result : OptimizationResult
            Fitting results with result.values containing intensities
        """
        return super().fit_positions(
            positions=positions,
            initial_values=initial_intensities,
            maxiter=maxiter,
            step_size=step_size,
            optimizer=optimizer,
            tol=tol,
            verbose=verbose,
            **optimizer_kwargs
        )

    def get_intensities(self) -> np.ndarray:
        """Get optimized intensity values from current parameters."""
        return self.get_values()


# Convenience functions for quick fitting

def fit_ssb_ptychography(
    image: np.ndarray,
    positions: np.ndarray,
    initial_phases: Optional[np.ndarray] = None,
    alpha: float = 20.0,
    eV: float = 60e3,
    df: float = 0.0,
    aberrations: Optional[list] = None,
    maxiter: int = 100,
    step_size: float = 0.001,
    verbose: bool = True,
) -> OptimizationResult:
    """
    Quick fit for SSB ptychography images.

    Parameters
    ----------
    image : np.ndarray
        SSB ptychography phase image
    positions : np.ndarray (N x 2)
        Initial atomic positions [x, y] in pixels
    initial_phases : np.ndarray, optional
        Initial phase values
    alpha : float, optional
        Convergence angle (mrad). Default is 20.0.
    eV : float, optional
        Acceleration voltage (eV). Default is 60e3.
    df : float, optional
        Defocus (Å). Default is 0.0.
    aberrations : list, optional
        List of aberration objects
    maxiter : int, optional
        Maximum iterations. Default is 100.
    step_size : float, optional
        Learning rate. Default is 0.001.
    verbose : bool, optional
        Print progress. Default is True.

    Returns
    -------
    result : OptimizationResult
        Fitting results

    Examples
    --------
    >>> from qem.fit.convolve_fitting import fit_ssb_ptychography
    >>>
    >>> result = fit_ssb_ptychography(
    ...     image=ssb_image,
    ...     positions=initial_positions,
    ...     alpha=20,
    ...     eV=60e3,
    ... )
    >>>
    >>> print(f"Correlation: {result.correlation:.4f}")
    >>> print(f"Phases: {result.values}")
    """
    probe_params = create_probe_parameters(
        alpha=alpha, eV=eV, df=df, aberrations=aberrations
    )

    fitter = ConvolutionFitting(
        image=image,
        ctf_type='SSB',
        probe_params=probe_params,
    )

    return fitter.fit_positions(
        positions=positions,
        initial_values=initial_phases,
        maxiter=maxiter,
        step_size=step_size,
        verbose=verbose,
    )


def fit_adf_image(
    image: np.ndarray,
    positions: np.ndarray,
    initial_intensities: Optional[np.ndarray] = None,
    alpha: float = 20.0,
    eV: float = 60e3,
    detector_inner: float = 50,
    detector_outer: float = 200,
    df: float = 0.0,
    maxiter: int = 100,
    step_size: float = 0.001,
    verbose: bool = True,
) -> OptimizationResult:
    """
    Quick fit for ADF images using convolution model.

    Parameters
    ----------
    image : np.ndarray
        ADF image
    positions : np.ndarray (N x 2)
        Initial atomic positions [x, y] in pixels
    initial_intensities : np.ndarray, optional
        Initial intensity values
    alpha : float, optional
        Convergence angle (mrad). Default is 20.0.
    eV : float, optional
        Acceleration voltage (eV). Default is 60e3.
    detector_inner : float, optional
        Inner detector angle (mrad). Default is 50.
    detector_outer : float, optional
        Outer detector angle (mrad). Default is 200.
    df : float, optional
        Defocus (Å). Default is 0.0.
    maxiter : int, optional
        Maximum iterations. Default is 100.
    step_size : float, optional
        Learning rate. Default is 0.001.
    verbose : bool, optional
        Print progress. Default is True.

    Returns
    -------
    result : OptimizationResult
        Fitting results
    """
    fitter = ADFConvolutionFitting(
        image=image,
        alpha=alpha,
        eV=eV,
        detector_inner=detector_inner,
        detector_outer=detector_outer,
        df=df,
    )

    return fitter.fit_positions(
        positions=positions,
        initial_values=initial_intensities,
        maxiter=maxiter,
        step_size=step_size,
        verbose=verbose,
    )


# Backward compatibility aliases
PtychographyOptimizer = PtychographyFitting
