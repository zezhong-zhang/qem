"""Contrast Transfer Functions for different STEM imaging modes.

This module implements CTF calculations for:
- SSB (Single Side Band) ptychography
- ADF (Annular Dark Field) imaging
- ePIE (extended Ptychographic Iterative Engine)
- iCoM (integrated Center of Mass) imaging

Based on:
Hofer & Pennycook, "Reliable phase quantification in focused probe
electron ptychography of thin materials", Ultramicroscopy 254 (2023) 113829.

This module integrates tightly with probe.py for:
- Aberration calculations via the Probe class
- Wavenumber calculations via wavev()
- Aberration objects from the Aberration class
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional, Tuple, Union, List

import numpy as np
from scipy.fft import fft2, ifft2, fftshift, ifftshift

from qem.processing import q_space_array
from qem.instruments.probe import (
    wavev,
    Probe,
    Aberration,
    aberration_starter_pack,
)


@dataclass
class ProbeParameters:
    """Microscope probe parameters for CTF/PSF calculation.

    This dataclass provides a clean way to pass probe parameters to
    PtychographyFitting and other fitting classes.

    Attributes
    ----------
    alpha : float
        Probe convergence semi-angle in mrad
    eV : float
        Acceleration voltage in eV
    df : float, optional
        Defocus in Angstroms (negative for overfocus). Default is 0.0.
    aberrations : list of aberration objects, optional
        List of aberrations to include. Default is None (no aberrations).
    detector_inner : float, optional
        Inner detector angle for ADF (mrad). Required for ADF.
    detector_outer : float, optional
        Outer detector angle for ADF (mrad). Required for ADF.
    high_pass_cutoff : float, optional
        High-pass filter cutoff for iCoM (mrad). Default is None.

    Partial coherence parameters:
    Cc : float, optional
        Chromatic aberration coefficient in Angstroms. Default is None.
    deltaE : float, optional
        Energy spread in eV (1/e convention). Default is None.
    df_spread : float, optional
        Direct defocus spread in Angstroms (alternative to Cc + deltaE).
    source_size : float, optional
        Effective source size in mrad for spatial coherence. Default is None.
    """

    alpha: float
    eV: float
    df: float = 0.0
    aberrations: Optional[List[Aberration]] = None
    detector_inner: Optional[float] = None
    detector_outer: Optional[float] = None
    high_pass_cutoff: Optional[float] = None
    # Partial coherence parameters
    Cc: Optional[float] = None
    deltaE: Optional[float] = None
    df_spread: Optional[float] = None
    source_size: Optional[float] = None

    def to_dict(self) -> dict:
        """Convert parameters to dictionary."""
        return {
            'alpha': self.alpha,
            'eV': self.eV,
            'df': self.df,
            'aberrations': self.aberrations,
            'detector_inner': self.detector_inner,
            'detector_outer': self.detector_outer,
            'high_pass_cutoff': self.high_pass_cutoff,
            'Cc': self.Cc,
            'deltaE': self.deltaE,
            'df_spread': self.df_spread,
            'source_size': self.source_size,
        }

    @classmethod
    def from_dict(cls, params: dict) -> 'ProbeParameters':
        """Create ProbeParameters from dictionary."""
        return cls(
            alpha=params.get('alpha', 20.0),
            eV=params.get('eV', 60e3),
            df=params.get('df', 0.0),
            aberrations=params.get('aberrations'),
            detector_inner=params.get('detector_inner'),
            detector_outer=params.get('detector_outer'),
            high_pass_cutoff=params.get('high_pass_cutoff'),
            Cc=params.get('Cc'),
            deltaE=params.get('deltaE'),
            df_spread=params.get('df_spread'),
            source_size=params.get('source_size'),
        )


def create_probe_parameters(
    alpha: float = 20.0,
    eV: float = 60e3,
    df: float = 0.0,
    defocus: float = None,
    spherical_aberration: float = None,
    two_fold_astigmatism: float = None,
    two_fold_angle: float = None,
    three_fold_astigmatism: float = None,
    three_fold_angle: float = None,
    coma: float = None,
    coma_angle: float = None,
    detector_inner: float = None,
    detector_outer: float = None,
    high_pass_cutoff: float = None,
    aberrations: Optional[List[Aberration]] = None,
    Cc: float = None,
    deltaE: float = None,
    df_spread: float = None,
    source_size: float = None,
) -> ProbeParameters:
    """
    Convenience function to create ProbeParameters.

    Parameters
    ----------
    alpha : float, optional
        Probe convergence semi-angle in mrad. Default is 20.0.
    eV : float, optional
        Acceleration voltage in eV. Default is 60e3 (60 kV).
    df : float, optional
        Defocus in Angstroms. Default is 0.0.
    defocus : float, optional
        Alias for df (defocus in Angstroms).
    spherical_aberration : float, optional
        Third-order spherical aberration Cs in Angstroms.
    two_fold_astigmatism : float, optional
        Two-fold astigmatism amplitude in Angstroms.
    two_fold_angle : float, optional
        Two-fold astigmatism angle in radians.
    three_fold_astigmatism : float, optional
        Three-fold astigmatism amplitude in Angstroms.
    three_fold_angle : float, optional
        Three-fold astigmatism angle in radians.
    coma : float, optional
        Coma aberration amplitude in Angstroms.
    coma_angle : float, optional
        Coma aberration angle in radians.
    detector_inner : float, optional
        Inner detector angle for ADF (mrad).
    detector_outer : float, optional
        Outer detector angle for ADF (mrad).
    high_pass_cutoff : float, optional
        High-pass filter cutoff for iCoM (mrad).
    aberrations : list of aberration objects, optional
        Pre-defined list of aberrations (takes precedence).
    Cc : float, optional
        Chromatic aberration coefficient in Angstroms.
    deltaE : float, optional
        Energy spread in eV (1/e convention).
    df_spread : float, optional
        Direct defocus spread in Angstroms (alternative to Cc + deltaE).
    source_size : float, optional
        Effective source size in mrad for spatial coherence.

    Returns
    -------
    ProbeParameters
        Probe parameters dataclass

    Examples
    --------
    >>> # Simple SSB with defocus
    >>> params = create_probe_parameters(alpha=20, eV=60e3, defocus=50)

    >>> # With partial coherence
    >>> params = create_probe_parameters(
    ...     alpha=20, eV=60e3, defocus=50,
    ...     Cc=2e7, deltaE=0.5,  # Chromatic aberration and energy spread
    ...     source_size=0.1,      # Spatial coherence
    ... )

    >>> # ADF with detector angles
    >>> params = create_probe_parameters(
    ...     alpha=20, eV=60e3,
    ...     detector_inner=50, detector_outer=200
    ... )

    >>> # With aberrations
    >>> params = create_probe_parameters(
    ...     alpha=20, eV=60e3,
    ...     defocus=30,
    ...     spherical_aberration=1e7,  # 1 mm Cs
    ...     two_fold_astigmatism=10,
    ...     two_fold_angle=np.pi/4,
    ... )
    """
    # Handle defocus alias
    if defocus is not None:
        df = defocus

    # Create aberrations list if not provided
    if aberrations is None:
        aberrations = []
        # Add aberrations based on provided parameters
        if df != 0:
            # Defocus is handled separately (C10), but we can add explicit C10
            pass  # Defocus is passed directly to CTF
        if spherical_aberration is not None:
            # C30 - third order spherical
            aberrations.append(Aberration(spherical_aberration, 0, 3, 0))
        if two_fold_astigmatism is not None:
            angle = two_fold_angle if two_fold_angle is not None else 0
            aberrations.append(Aberration(two_fold_astigmatism, angle, 2, 2))
        if three_fold_astigmatism is not None:
            angle = three_fold_angle if three_fold_angle is not None else 0
            aberrations.append(Aberration(three_fold_astigmatism, angle, 3, 3))
        if coma is not None:
            angle = coma_angle if coma_angle is not None else 0
            aberrations.append(Aberration(coma, angle, 3, 1))

    # If aberrations list is still empty after processing, set to None
    if len(aberrations) == 0:
        aberrations = None

    return ProbeParameters(
        alpha=alpha,
        eV=eV,
        df=df,
        aberrations=aberrations,
        detector_inner=detector_inner,
        detector_outer=detector_outer,
        high_pass_cutoff=high_pass_cutoff,
        Cc=Cc,
        deltaE=deltaE,
        df_spread=df_spread,
        source_size=source_size,
    )


class ContrastTransferFunction(ABC):
    """Base class for CTF calculations."""

    def __init__(
        self,
        alpha: float,
        eV: float,
        df: float = 0.0,
        aberrations: Optional[list] = None,
        Cc: Optional[float] = None,
        deltaE: Optional[float] = None,
        df_spread: Optional[float] = None,
        source_size: Optional[float] = None,
    ):
        """
        Initialize CTF calculator.

        Parameters
        ----------
        alpha : float
            Probe convergence semi-angle in mrad
        eV : float
            Acceleration voltage in eV
        df : float, optional
            Defocus in Angstroms (negative for overfocus)
        aberrations : list of aberration objects, optional
            List of aberrations to include
        Cc : float, optional
            Chromatic aberration coefficient in Angstroms
        deltaE : float, optional
            Energy spread in eV (1/e convention)
        df_spread : float, optional
            Direct defocus spread in Angstroms (alternative to Cc + deltaE)
        source_size : float, optional
            Effective source size in mrad for spatial coherence
        """
        self.alpha = alpha  # mrad
        self.eV = eV
        self.df = df
        # Use None as default to be consistent with probe.py conventions
        # Empty list means no aberrations
        self.aberrations = aberrations if aberrations is not None else []

        # Partial coherence parameters
        self.Cc = Cc
        self.deltaE = deltaE
        self.df_spread = df_spread
        self.source_size = source_size

        # Calculate wavenumber (inverse wavelength)
        self.k = wavev(eV)

        # Internal Probe instance for aberration / envelope calculations
        self._probe = Probe(
            eV=eV,
            aperture=alpha,
            df=df,
            aberrations=self.aberrations,
            aperture_units="mrad",
            Cc=Cc,
            deltaE=deltaE,
            df_spread=df_spread,
            source_size=source_size,
        )

    @abstractmethod
    def calculate_ctf(
        self,
        pix_dim: Tuple[int, int],
        real_dim: Tuple[float, float],
    ) -> np.ndarray:
        """
        Calculate CTF in reciprocal space.

        Parameters
        ----------
        pix_dim : tuple (ny, nx)
            Pixel dimensions of the grid
        real_dim : tuple (dy, dx)
            Real space dimensions in Angstroms

        Returns
        -------
        ctf : np.ndarray (complex)
            Complex CTF in reciprocal space
        """
        pass

    def get_psf(
        self,
        pix_dim: Tuple[int, int],
        real_dim: Tuple[float, float],
    ) -> np.ndarray:
        """
        Get Point Spread Function via inverse FFT of CTF.

        The PSF is the real-space representation of the CTF and
        represents the shape of a single atom in the reconstructed image.

        Partial coherence envelope is applied before the inverse FFT.

        Parameters
        ----------
        pix_dim : tuple (ny, nx)
            Pixel dimensions of the grid
        real_dim : tuple (dy, dx)
            Real space dimensions in Angstroms

        Returns
        -------
        psf : np.ndarray (float)
            Real-space PSF (can have negative values for coherent imaging)
        """
        ctf = self.calculate_ctf(pix_dim, real_dim)

        # Apply partial coherence envelope if parameters are provided
        if self.Cc is not None or self.deltaE is not None or self.df_spread is not None or self.source_size is not None:
            q = q_space_array(pix_dim, real_dim)
            q_mag = np.sqrt(q[0] ** 2 + q[1] ** 2)
            envelope = self._probe.partial_coherence_envelope(q_mag)
            ctf = ctf * envelope

        psf = np.real(ifft2(ifftshift(ctf)))
        # Shift so PSF is centered
        psf = fftshift(psf)
        return psf


class SSB_CTF(ContrastTransferFunction):
    """Single Side Band ptychography CTF.

    Implements the analytical SSB CTF formula from the paper.
    The SSB CTF has a characteristic shape that produces a negative
    halo around atomic sites in the reconstructed phase image.

    From paper Eq. for SSB CTF:
    CTF(Qp) = (4/π) * [acos(Qp/2) - (Qp/2)√(1-(Qp/2)²)
                         - acos(Qp) + Qp√(1-Qp²)]     for 0 ≤ Qp ≤ α
    CTF(Qp) = (4/π) * [acos(Qp/2) - (Qp/2)√(1-(Qp/2)²)] for α < Qp ≤ 2α
    """

    def calculate_ctf(
        self,
        pix_dim: Tuple[int, int],
        real_dim: Tuple[float, float],
    ) -> np.ndarray:
        """
        Calculate SSB CTF using analytical formula.

        Parameters
        ----------
        pix_dim : tuple (ny, nx)
            Pixel dimensions of the grid
        real_dim : tuple (dy, dx)
            Real space dimensions in Angstroms

        Returns
        -------
        ctf : np.ndarray (complex)
            Complex CTF in reciprocal space
        """
        # Get reciprocal space array
        q = q_space_array(pix_dim, real_dim)

        # Calculate spatial frequency magnitude
        q_mag = np.sqrt(q[0] ** 2 + q[1] ** 2)

        # Convert convergence angle from mrad to inverse Angstroms
        # alpha_invA = alpha * 1e-3 * k
        alpha_invA = self.alpha * 1e-3 * self.k

        # Normalize by convergence angle
        q_norm = q_mag / alpha_invA

        # Initialize CTF array
        ctf = np.zeros(pix_dim, dtype=complex)

        # Region 1: 0 ≤ Qp ≤ α (0 ≤ q_norm ≤ 1)
        mask1 = (q_norm >= 0) & (q_norm <= 1)
        if np.any(mask1):
            qn = q_norm[mask1]
            # Avoid numerical issues near boundaries
            qn = np.clip(qn, 0, 0.9999)
            term1 = np.arccos(qn / 2)
            term2 = (qn / 2) * np.sqrt(1 - (qn / 2) ** 2)
            term3 = np.arccos(qn)
            term4 = qn * np.sqrt(1 - qn ** 2)
            ctf[mask1] = (4 / np.pi) * (term1 - term2 - term3 + term4)

        # Region 2: α < Qp ≤ 2α (1 < q_norm ≤ 2)
        mask2 = (q_norm > 1) & (q_norm <= 2)
        if np.any(mask2):
            qn = q_norm[mask2]
            qn = np.clip(qn, 0, 0.9999)
            term1 = np.arccos(qn / 2)
            term2 = (qn / 2) * np.sqrt(1 - (qn / 2) ** 2)
            ctf[mask2] = (4 / np.pi) * (term1 - term2)

        # Apply aberrations via the internal Probe instance
        if self.df != 0 or self.aberrations:
            qphi = np.arctan2(q[0], q[1])
            phase_factor = np.exp(-1j * self._probe.chi(q_mag, qphi))
            ctf *= phase_factor

        return ctf


class ADF_CTF(ContrastTransferFunction):
    """ADF CTF using the existing probe CTF infrastructure.

    ADF imaging is incoherent, so the CTF is squared to produce
    the final PSF. The ADF PSF is always positive (no negative halo).

    This implementation uses make_contrast_transfer_function() from probe.py
    to ensure consistency with existing QEM infrastructure and proper
    integration with aberrations.

    The detector geometry is specified by inner and outer collection angles.
    """

    def __init__(
        self,
        alpha: float,
        eV: float,
        detector_inner: float,
        detector_outer: float,
        df: float = 0.0,
        aberrations: Optional[list] = None,
        Cc: Optional[float] = None,
        deltaE: Optional[float] = None,
        df_spread: Optional[float] = None,
        source_size: Optional[float] = None,
    ):
        """
        Initialize ADF CTF calculator.

        Parameters
        ----------
        alpha : float
            Probe convergence semi-angle in mrad
        eV : float
            Acceleration voltage in eV
        detector_inner : float
            Inner detector collection angle in mrad
        detector_outer : float
            Outer detector collection angle in mrad
        df : float, optional
            Defocus in Angstroms
        aberrations : list of aberration objects, optional
            List of aberrations to include
        Cc : float, optional
            Chromatic aberration coefficient in Angstroms
        deltaE : float, optional
            Energy spread in eV (1/e convention)
        df_spread : float, optional
            Direct defocus spread in Angstroms
        source_size : float, optional
            Effective source size in mrad
        """
        super().__init__(alpha, eV, df, aberrations, Cc, deltaE, df_spread, source_size)
        self.detector_inner = detector_inner  # mrad
        self.detector_outer = detector_outer  # mrad

    def calculate_ctf(
        self,
        pix_dim: Tuple[int, int],
        real_dim: Tuple[float, float],
    ) -> np.ndarray:
        """
        Calculate ADF CTF (probe intensity).

        For ADF, the CTF is essentially the probe wavefunction squared,
        integrated over the detector angles.

        Parameters
        ----------
        pix_dim : tuple (ny, nx)
            Pixel dimensions of the grid
        real_dim : tuple (dy, dx)
            Real space dimensions in Angstroms

        Returns
        -------
        ctf : np.ndarray (float)
            Real, non-negative CTF for incoherent ADF imaging
        """
        # Create a clean probe (no aberrations) for basic ADF PSF
        adf_probe = Probe(
            eV=self.eV,
            aperture=self.alpha,
            df=self.df,
            aberrations=[],
        )
        ctf = adf_probe.make_ctf(pix_dim=pix_dim, real_dim=real_dim)

        # For incoherent imaging, we square the amplitude
        # The detector integration is handled by the angular range
        ctf_adf = np.abs(ctf) ** 2

        # Apply detector geometry (simple binary mask for now)
        # A more sophisticated implementation would integrate over
        # the detector angular range
        q = q_space_array(pix_dim, real_dim)
        q_mag = np.sqrt(q[0] ** 2 + q[1] ** 2)

        # Convert detector angles to inverse Angstroms
        inner_invA = self.detector_inner * 1e-3 * self.k
        outer_invA = self.detector_outer * 1e-3 * self.k

        # Detector mask (1 within detector range, 0 outside)
        detector_mask = (q_mag >= inner_invA) & (q_mag <= outer_invA)

        # Normalize by detector area
        if np.any(detector_mask):
            ctf_adf *= detector_mask
            # Normalize to unit sum
            total = np.sum(ctf_adf)
            if total > 0:
                ctf_adf = ctf_adf / total * np.prod(pix_dim)

        return ctf_adf.astype(np.float64)

    def get_psf(
        self,
        pix_dim: Tuple[int, int],
        real_dim: Tuple[float, float],
    ) -> np.ndarray:
        """
        Get ADF PSF.

        The ADF PSF is the squared magnitude of the probe wavefunction,
        which is always positive (Gaussian-like shape).

        Parameters
        ----------
        pix_dim : tuple (ny, nx)
            Pixel dimensions of the grid
        real_dim : tuple (dy, dx)
            Real space dimensions in Angstroms

        Returns
        -------
        psf : np.ndarray (float)
            Real, positive PSF for ADF imaging
        """
        ctf = self.calculate_ctf(pix_dim, real_dim)
        psf = np.real(ifft2(ifftshift(ctf)))
        psf = fftshift(psf)

        # Ensure PSF is non-negative (clip small FFT artifacts)
        psf = np.maximum(psf, 0)

        return psf


class ePIE_CTF(ContrastTransferFunction):
    """ePIE CTF - extracted from simulation or analytical approximation.

    ePIE (extended Ptychographic Iterative Engine) produces a different
    CTF than SSB. It can be obtained by:
    1. Simulating a single atom with ePIE and extracting the profile
    2. Using an analytical approximation

    Like SSB, ePIE produces a PSF with a negative halo.
    """

    def __init__(
        self,
        alpha: float,
        eV: float,
        df: float = 0.0,
        defocus: float = 1.0,  # ePIE typically uses defocus
        step_size: float = 0.5,  # Probe position step in Angstroms
        aberrations: Optional[list] = None,
        Cc: Optional[float] = None,
        deltaE: Optional[float] = None,
        df_spread: Optional[float] = None,
        source_size: Optional[float] = None,
    ):
        """
        Initialize ePIE CTF calculator.

        Parameters
        ----------
        alpha : float
            Probe convergence semi-angle in mrad
        eV : float
            Acceleration voltage in eV
        df : float, optional
            Nominal defocus in Angstroms
        defocus : float, optional
            ePIE defocus value in nm (typically 1 nm)
        step_size : float, optional
            Probe position step size in Angstroms
        aberrations : list of aberration objects, optional
            List of aberrations to include
        Cc : float, optional
            Chromatic aberration coefficient in Angstroms
        deltaE : float, optional
            Energy spread in eV (1/e convention)
        df_spread : float, optional
            Direct defocus spread in Angstroms
        source_size : float, optional
            Effective source size in mrad
        """
        super().__init__(alpha, eV, df, aberrations, Cc, deltaE, df_spread, source_size)
        self.defocus = defocus  # nm
        self.step_size = step_size  # Angstroms

    def calculate_ctf(
        self,
        pix_dim: Tuple[int, int],
        real_dim: Tuple[float, float],
        psf_from_simulation: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Calculate ePIE CTF.

        If psf_from_simulation is provided, it will be used directly.
        Otherwise, an analytical approximation is used.

        Parameters
        ----------
        pix_dim : tuple (ny, nx)
            Pixel dimensions of the grid
        real_dim : tuple (dy, dx)
            Real space dimensions in Angstroms
        psf_from_simulation : np.ndarray, optional
            Pre-computed PSF from single atom ePIE simulation

        Returns
        -------
        ctf : np.ndarray (complex)
            Complex CTF in reciprocal space
        """
        if psf_from_simulation is not None:
            # Use PSF from simulation (forward FFT to get CTF)
            psf_shifted = ifftshift(psf_from_simulation)
            ctf = fft2(psf_shifted)
            return ctf

        # Analytical approximation: similar to SSB but modified
        # based on defocus and step size
        q = q_space_array(pix_dim, real_dim)
        q_mag = np.sqrt(q[0] ** 2 + q[1] ** 2)

        alpha_invA = self.alpha * 1e-3 * self.k
        q_norm = q_mag / alpha_invA

        # ePIE CTF approximation (similar form to SSB but different coefficients)
        # The exact shape depends on defocus and step size
        ctf = np.zeros(pix_dim, dtype=complex)

        mask = (q_norm >= 0) & (q_norm <= 2)
        if np.any(mask):
            # Create full-size arrays for broadcasting
            qn_clipped = np.clip(q_norm, 0, 1.9999)

            # ePIE produces a somewhat different CTF than SSB
            # The negative halo is typically less pronounced
            term1 = np.arccos(np.clip(qn_clipped / 2, 0, 0.9999))
            term2 = (qn_clipped / 2) * np.sqrt(1 - np.clip((qn_clipped / 2) ** 2, 0, 1))

            # Additional factor for defocus dependence
            defocus_factor = 1.0 / (1.0 + (self.defocus * q_mag / 10) ** 2)

            ctf = np.where(mask, (4 / np.pi) * (term1 - term2) * defocus_factor, ctf)

        # Apply aberrations via the internal Probe instance
        if self.df != 0 or self.aberrations:
            qphi = np.arctan2(q[0], q[1])
            phase_factor = np.exp(-1j * self._probe.chi(q_mag, qphi))
            ctf *= phase_factor

        return ctf


class iCoM_CTF(ContrastTransferFunction):
    """iCoM (integrated Center of Mass) CTF.

    iCoM imaging starts with CTF = 1 at zero frequency.
    High-pass filters are often applied, which introduce negative
    components to the PSF.

    The CTF can be modified by:
    - High-pass filter cutoff
    - riCoM kernel parameters
    """

    def __init__(
        self,
        alpha: float,
        eV: float,
        high_pass_cutoff: Optional[float] = None,
        filter_type: str = "none",
        df: float = 0.0,
        aberrations: Optional[list] = None,
        Cc: Optional[float] = None,
        deltaE: Optional[float] = None,
        df_spread: Optional[float] = None,
        source_size: Optional[float] = None,
    ):
        """
        Initialize iCoM CTF calculator.

        Parameters
        ----------
        alpha : float
            Probe convergence semi-angle in mrad
        eV : float
            Acceleration voltage in eV
        high_pass_cutoff : float, optional
            High-pass filter cutoff in mrad
        filter_type : str, optional
            Type of filter ('none', 'highpass', 'ricom')
        df : float, optional
            Defocus in Angstroms
        aberrations : list of aberration objects, optional
            List of aberrations to include
        Cc : float, optional
            Chromatic aberration coefficient in Angstroms
        deltaE : float, optional
            Energy spread in eV (1/e convention)
        df_spread : float, optional
            Direct defocus spread in Angstroms
        source_size : float, optional
            Effective source size in mrad
        """
        super().__init__(alpha, eV, df, aberrations, Cc, deltaE, df_spread, source_size)
        self.high_pass_cutoff = high_pass_cutoff
        self.filter_type = filter_type

    def calculate_ctf(
        self,
        pix_dim: Tuple[int, int],
        real_dim: Tuple[float, float],
    ) -> np.ndarray:
        """
        Calculate iCoM CTF.

        Parameters
        ----------
        pix_dim : tuple (ny, nx)
            Pixel dimensions of the grid
        real_dim : tuple (dy, dx)
            Real space dimensions in Angstroms

        Returns
        -------
        ctf : np.ndarray (float)
            Real CTF for iCoM imaging
        """
        q = q_space_array(pix_dim, real_dim)
        q_mag = np.sqrt(q[0] ** 2 + q[1] ** 2)

        # Raw iCoM CTF starts at 1 for zero frequency
        ctf = np.ones(pix_dim, dtype=complex)

        # Apply filter if specified
        if self.filter_type == "highpass" and self.high_pass_cutoff is not None:
            cutoff_invA = self.high_pass_cutoff * 1e-3 * self.k
            # High-pass filter: 0 below cutoff, 1 above
            # Use smooth transition (Butterworth-like)
            filter_order = 2
            ctf_real = 1.0 - 1.0 / (1.0 + (q_mag / cutoff_invA) ** (2 * filter_order))
            ctf = ctf_real.astype(complex)

        elif self.filter_type == "ricom":
            # riCoM has its own intrinsic filter based on kernel
            # This is a simplified model
            if self.high_pass_cutoff is not None:
                cutoff_invA = self.high_pass_cutoff * 1e-3 * self.k
                ctf_real = q_mag / (q_mag + cutoff_invA)
                ctf = ctf_real.astype(complex)

        # Apply aberrations via the internal Probe instance
        if self.df != 0 or self.aberrations:
            qphi = np.arctan2(q[0], q[1])
            phase_factor = np.exp(-1j * self._probe.chi(q_mag, qphi))
            ctf *= phase_factor

        return ctf


def calculate_psf_width(psf: np.ndarray, dx: float = 1.0) -> float:
    """
    Calculate the effective width of a PSF.

    Uses the second moment of the PSF distribution.
    For PSFs with negative values (like SSB), uses absolute values.

    Parameters
    ----------
    psf : np.ndarray
        Point spread function
    dx : float, optional
        Pixel size in Angstroms

    Returns
    -------
    width : float
        PSF width (standard deviation) in pixels
    """
    # Calculate center of mass using absolute values
    ny, nx = psf.shape
    y, x = np.indices(psf.shape)

    # Use absolute values for PSFs with negative halos
    psf_abs = np.abs(psf)
    total = np.sum(psf_abs)

    if total == 0:
        return 1.0

    x_c = np.sum(x * psf_abs) / total
    y_c = np.sum(y * psf_abs) / total

    # Calculate second moment using absolute values
    var_x = np.sum((x - x_c) ** 2 * psf_abs) / total
    var_y = np.sum((y - y_c) ** 2 * psf_abs) / total

    width = np.sqrt(var_x + var_y)
    return width


def extract_psf_from_atom_image(
    atom_image: np.ndarray,
    background: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Extract PSF from an image of a single isolated atom.

    This is useful when the analytical CTF is not available or
    when you want to use an experimentally measured PSF.

    Parameters
    ----------
    atom_image : np.ndarray
        Image containing a single isolated atom
    background : np.ndarray, optional
        Background image to subtract

    Returns
    -------
    psf : np.ndarray
        Normalized PSF extracted from the atom image
    """
    psf = atom_image.copy()

    # Subtract background if provided
    if background is not None:
        psf = psf - background

    # Remove negative values
    psf = np.maximum(psf, 0)

    # Center the PSF
    ny, nx = psf.shape
    cy, cx = ny // 2, nx // 2

    # Find center of mass
    y, x = np.indices(psf.shape)
    total = np.sum(psf)
    if total > 0:
        x_c = np.sum(x * psf) / total
        y_c = np.sum(y * psf) / total

        # Shift to center
        shift_y = int(round(cy - y_c))
        shift_x = int(round(cx - x_c))
        psf = np.roll(psf, shift_y, axis=0)
        psf = np.roll(psf, shift_x, axis=1)

    # Normalize to unit sum
    psf = psf / np.sum(psf)

    return psf


def create_aberration_list(
    defocus: float = 0.0,
    two_fold_astigmatism: float = 0.0,
    two_fold_angle: float = 0.0,
    three_fold_astigmatism: float = 0.0,
    three_fold_angle: float = 0.0,
    coma: float = 0.0,
    coma_angle: float = 0.0,
    spherical_aberration: float = 0.0,
) -> list:
    """
    Create a list of aberrations for CTF calculation.

    This is a convenience function that creates aberration objects
    compatible with the probe.py infrastructure, making it easy to
    specify common aberrations without directly constructing aberration objects.

    Parameters
    ----------
    defocus : float, optional
        Defocus (C10) in Angstroms (negative for overfocus)
    two_fold_astigmatism : float, optional
        2-fold astigmatism (C12) amplitude in Angstroms
    two_fold_angle : float, optional
        2-fold astigmatism angle in radians
    three_fold_astigmatism : float, optional
        3-fold astigmatism (C23) amplitude in Angstroms
    three_fold_angle : float, optional
        3-fold astigmatism angle in radians
    coma : float, optional
        Axial coma (C21) amplitude in Angstroms
    coma_angle : float, optional
        Coma angle in radians
    spherical_aberration : float, optional
        3rd order spherical aberration (C30) in Angstroms (typically ~1 mm = 1e7 A)

    Returns
    -------
    aberrations : list
        List of aberration objects from probe.py

    Examples
    --------
    >>> # Create aberrations with 50 Angstrom defocus
    >>> ab_list = create_aberration_list(defocus=50)
    >>> # Use in CTF calculation
    >>> from qem.instruments.ctf import SSB_CTF
    >>> ctf = SSB_CTF(alpha=20, eV=60e3, aberrations=ab_list)

    >>> # Create with multiple aberrations
    >>> ab_list = create_aberration_list(
    ...     defocus=50,              # 50 A defocus
    ...     spherical_aberration=1e7,  # 1 mm C30 (typical for uncorrected STEM)
    ...     two_fold_astigmatism=10,   # 10 A 2-fold astigmatism
    ...     two_fold_angle=np.pi/4,     # at 45 degrees
    ... )

    Notes
    -----
    This function uses the aberration class from probe.py, which follows
    the Krivanek notation system:
    - C10: Defocus (n=1, m=0)
    - C12: 2-fold astigmatism (n=1, m=2)
    - C23: 3-fold astigmatism (n=2, m=3)
    - C21: Axial coma (n=2, m=1)
    - C30: 3rd order spherical aberration (n=3, m=0)

    For a complete set of aberrations up to 5th order, use:
    >>> from qem.instruments.probe import aberration_starter_pack
    >>> aberrations = aberration_starter_pack()
    """
    aberrations = []

    if defocus != 0:
        aberrations.append(Aberration("C10", "C1", "Defocus", defocus, 0.0, 1, 0))

    if two_fold_astigmatism != 0:
        aberrations.append(Aberration("C12", "A1", "2-Fold astig.", two_fold_astigmatism, two_fold_angle, 1, 2))

    if three_fold_astigmatism != 0:
        aberrations.append(Aberration("C23", "A2", "3-Fold astig.", three_fold_astigmatism, three_fold_angle, 2, 3))

    if coma != 0:
        aberrations.append(Aberration("C21", "B2", "Axial coma", coma, coma_angle, 2, 1))

    if spherical_aberration != 0:
        aberrations.append(Aberration("C30", "C3", "3rd order spher.", spherical_aberration, 0.0, 3, 0))

    return aberrations


def demonstrate_aberration_effects():
    """
    Demonstrate the effects of different aberrations on the SSB PSF.

    This function creates a visual comparison showing how different
    aberrations affect the PSF shape, which is useful for understanding
    the impact of microscope alignment on ptychography reconstructions.

    Returns
    -------
    fig : matplotlib.figure.Figure
        Figure showing PSF comparison

    Examples
    --------
    >>> fig = demonstrate_aberration_effects()
    >>> fig.savefig('aberration_effects.png', dpi=150)
    """
    import matplotlib.pyplot as plt

    # Parameters
    alpha = 20  # mrad
    eV = 60e3  # 60 kV

    # Create figure
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    # 1. No aberrations
    ax = axes[0, 0]
    ctf = SSB_CTF(alpha, eV)
    psf = ctf.get_psf((64, 64), (64, 64))
    im = ax.imshow(psf, cmap='RdBu', vmin=-np.max(np.abs(psf)), vmax=np.max(np.abs(psf)))
    ax.set_title('No Aberrations')
    ax.axis('off')
    plt.colorbar(im, ax=ax)

    # 2. Defocus only
    ax = axes[0, 1]
    aberrations = create_aberration_list(defocus=50)
    ctf = SSB_CTF(alpha, eV, aberrations=aberrations)
    psf = ctf.get_psf((64, 64), (64, 64))
    im = ax.imshow(psf, cmap='RdBu', vmin=-np.max(np.abs(psf)), vmax=np.max(np.abs(psf)))
    ax.set_title('Defocus: 50 Å')
    ax.axis('off')
    plt.colorbar(im, ax=ax)

    # 3. 2-fold astigmatism
    ax = axes[0, 2]
    aberrations = create_aberration_list(
        defocus=50,
        two_fold_astigmatism=20,
        two_fold_angle=np.pi/4
    )
    ctf = SSB_CTF(alpha, eV, aberrations=aberrations)
    psf = ctf.get_psf((64, 64), (64, 64))
    im = ax.imshow(psf, cmap='RdBu', vmin=-np.max(np.abs(psf)), vmax=np.max(np.abs(psf)))
    ax.set_title('Defocus + 2-fold Astig.')
    ax.axis('off')
    plt.colorbar(im, ax=ax)

    # 4. Spherical aberration
    ax = axes[1, 0]
    aberrations = create_aberration_list(
        defocus=50,
        spherical_aberration=0.5e7  # 0.5 mm
    )
    ctf = SSB_CTF(alpha, eV, aberrations=aberrations)
    psf = ctf.get_psf((64, 64), (64, 64))
    im = ax.imshow(psf, cmap='RdBu', vmin=-np.max(np.abs(psf)), vmax=np.max(np.abs(psf)))
    ax.set_title('Defocus + Cs (0.5 mm)')
    ax.axis('off')
    plt.colorbar(im, ax=ax)

    # 5. Coma
    ax = axes[1, 1]
    aberrations = create_aberration_list(
        defocus=50,
        coma=1000,
        coma_angle=np.pi/3
    )
    ctf = SSB_CTF(alpha, eV, aberrations=aberrations)
    psf = ctf.get_psf((64, 64), (64, 64))
    im = ax.imshow(psf, cmap='RdBu', vmin=-np.max(np.abs(psf)), vmax=np.max(np.abs(psf)))
    ax.set_title('Defocus + Coma')
    ax.axis('off')
    plt.colorbar(im, ax=ax)

    # 6. Full aberration set
    ax = axes[1, 2]
    aberrations = aberration_starter_pack()
    # Set some non-zero values for demonstration
    for ab in aberrations:
        if ab.Krivanek == "C10":
            ab.amplitude = 50  # Defocus
        elif ab.Krivanek == "C30":
            ab.amplitude = 0.5e7  # 0.5 mm Cs
        elif ab.Krivanek == "C12":
            ab.amplitude = 20  # 2-fold astig
            ab.angle = np.pi/4
    ctf = SSB_CTF(alpha, eV, aberrations=aberrations)
    psf = ctf.get_psf((64, 64), (64, 64))
    im = ax.imshow(psf, cmap='RdBu', vmin=-np.max(np.abs(psf)), vmax=np.max(np.abs(psf)))
    ax.set_title('Full Aberration Set')
    ax.axis('off')
    plt.colorbar(im, ax=ax)

    plt.suptitle('Effect of Aberrations on SSB PSF\n(60 kV, 20 mrad)', fontsize=14)
    plt.tight_layout()

    return fig

