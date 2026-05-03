"""Numpy wave-function helpers for multislice / coherent-imaging code.

Multislice-adjacent helpers (focused-probe constructors, plane-wave
illumination, chromatic-aberration averaging).  Kept on numpy because
they're not on the optimisation hot path; for PSFs and fitting use
:mod:`qem.optics` instead.
"""

from __future__ import annotations

import copy
import math

import numpy as np
import torch

from qem.optics import Aberrations, chi as _chi, wavelength, wavev
from qem.processing import q_space_array


def _chi_numpy(q, qphi, lam: float, df: float, aberrations: Aberrations | None) -> np.ndarray:
    """Adapter: evaluate :func:`qem.optics.chi` for numpy q-grids."""
    if aberrations is None:
        ab = Aberrations(defocus=df) if df else Aberrations()
    else:
        # Add user-facing defocus on top of any C10 already in the dataclass.
        if df:
            ab = Aberrations(C10=aberrations.C10 - float(df), **{
                k: getattr(aberrations, k)
                for k in ("C12","phi12","C21","phi21","C23","phi23","C30",
                          "C32","phi32","C34","phi34","C41","phi41","C43",
                          "phi43","C45","phi45","C50","C52","phi52","C54",
                          "phi54","C56","phi56")
            })
        else:
            ab = aberrations
    alpha_t = torch.as_tensor(np.asarray(q, dtype=float) * lam)
    phi_t = torch.as_tensor(np.asarray(qphi, dtype=float))
    return _chi(alpha_t, phi_t, wavelength=lam, aberrations=ab).cpu().numpy()


def depth_of_field(eV: float, alpha: float) -> float:
    """Probe depth-of-field FWHM in Å.

    ``alpha`` is the convergence semi-angle in mrad.  See Kirkland.
    """
    return 1.77 / wavev(eV) / (alpha ** 2) * 1e6


def convert_tilt_angles(tilt, tilt_units, rsize, eV, invA_out=False):
    """Convert tilt to pixel or inverse-Å units regardless of input units."""
    if tilt_units == "mrad":
        tilt_ = np.asarray(tilt) * 1e-3 * wavev(eV)
    else:
        tilt_ = np.asarray(tilt)
    if invA_out:
        return tilt_
    if tilt_units != "pixels":
        tilt_ = np.round(tilt_ * np.asarray(rsize[:2])).astype(int)
    return tilt_


def make_contrast_transfer_function(
    pix_dim,
    real_dim,
    eV,
    app,
    optic_axis=None,
    aperture_shift=None,
    tilt_units="mrad",
    df=0.0,
    aberrations=None,
    q=None,
    app_units="mrad",
):
    """Build an electron-lens CTF on a numpy grid.

    Returns the q-space wave-function ``ψ(q) = aperture(q)·exp(-iχ(q))``
    in fft-natural ordering.

    Accepts the historical ``aberrations`` argument as either a list of
    legacy :class:`qem.instruments.Aberration` objects or any iterable
    that the legacy ``chi`` understands; ``df`` is the user-facing
    defocus (positive ⇒ under-focus, abtem convention).
    """
    if aberrations is None:
        aberrations = []
    if aperture_shift is None:
        aperture_shift = [0, 0]
    if optic_axis is None:
        optic_axis = [0, 0]
    if q is None:
        q = q_space_array(pix_dim, real_dim[:2])

    k = wavev(eV)
    optic_axis_ = convert_tilt_angles(optic_axis, tilt_units, real_dim, eV, invA_out=True)
    aperture_shift_ = convert_tilt_angles(aperture_shift, tilt_units, real_dim, eV, invA_out=True)

    if app is None:
        app_ = np.amax(np.abs(q))
    else:
        app_ = convert_tilt_angles(app, app_units, real_dim, eV, invA_out=True)

    qarray1 = np.sqrt((q[0] - optic_axis_[0]) ** 2 + (q[1] - optic_axis_[1]) ** 2)
    qarray2 = (q[0] - optic_axis_[0] - aperture_shift_[0]) ** 2 \
              + (q[1] - optic_axis_[1] - aperture_shift_[1]) ** 2
    qphi = np.arctan2(q[0] - optic_axis_[0], q[1] - optic_axis_[1])

    mask = qarray2 <= app_ ** 2
    ctf = np.zeros(pix_dim, dtype=complex)
    ctf[mask] = np.exp(
        -1j * _chi_numpy(qarray1[mask], qphi[mask], 1.0 / k, df, aberrations)
    )
    return ctf


def focused_probe(
    gridshape,
    rsize,
    eV,
    app,
    beam_tilt=None,
    aperture_shift=None,
    tilt_units="mrad",
    df=0.0,
    aberrations=None,
    q=None,
    app_units="mrad",
    qspace=False,
):
    """Focused electron-probe wave function on a numpy grid.

    Normalised so that ``Σ |ψ(r)|² = 1``.  Returns real-space probe by
    default; pass ``qspace=True`` for the q-space wave function.
    """
    if aberrations is None:
        aberrations = []
    if aperture_shift is None:
        aperture_shift = [0, 0]
    if beam_tilt is None:
        beam_tilt = [0, 0]
    probe = make_contrast_transfer_function(
        gridshape, rsize, eV, app,
        beam_tilt, aperture_shift, tilt_units,
        df, aberrations, q, app_units,
    )
    probe *= np.sqrt(np.prod(gridshape)) / np.sqrt(np.sum(np.abs(probe) ** 2))
    return probe if qspace else np.fft.ifft2(probe)


def plane_wave_illumination(
    gridshape, gridsize, eV, tilt=None, tilt_units="mrad", qspace=False,
):
    """Plane wave illumination, normalised so ``Σ |ψ(r)|² = 1``."""
    if tilt is None:
        tilt = [0, 0]
    illum = np.zeros(gridshape, dtype=complex)
    tilt_ = convert_tilt_angles(tilt, tilt_units, gridsize, eV)
    if tilt[0] == 0 and tilt[1] == 0:
        illum[:, :] = 1.0 / np.sqrt(np.prod(gridshape))
        return np.fft.fft2(illum) if qspace else illum
    illum[tilt_[0], tilt_[1]] = np.sqrt(np.prod(gridshape))
    return illum if qspace else np.fft.ifft2(illum)


def convert_deltaE(deltaE: float, deltaEconv: str) -> float:
    """Convert an energy-spread input to the 1/e half-width convention.

    Recognises ``'1/e'`` (passthrough), ``'FWHM'`` (divide by 2√ln2),
    and ``'std'`` (multiply by √2).
    """
    if deltaEconv == "1/e":
        return deltaE
    if deltaEconv == "FWHM":
        return deltaE / (2 * np.sqrt(np.log(2)))
    if deltaEconv == "std":
        return deltaE * np.sqrt(2)
    raise ValueError(
        f"deltaEconv {deltaEconv!r} not recognised; expected '1/e', 'FWHM', or 'std'."
    )


def Cc_integration_points(
    Cc: float, deltaE: float, eV: float, npoints: int = 7, deltaEconv: str = "1/e",
):
    """Defocus integration points for chromatic-aberration averaging.

    Splits the Gaussian focal-spread distribution into ``npoints`` regions
    of equal probability and returns the mean defocus in each region.
    """
    from scipy.special import erf, erfinv

    partitions = erfinv(2 * (np.arange(npoints - 1) + 1) / npoints - 1)
    x = np.zeros(npoints)
    pre = 1 / (2 * np.sqrt(np.pi))
    x[0] = -pre * np.exp(-partitions[0] ** 2) / (1 + erf(partitions[0])) * 2
    x[1:-1] = (
        pre
        * (np.exp(-partitions[:-1] ** 2) - np.exp(-partitions[1:] ** 2))
        / (erf(partitions[1:]) - erf(partitions[:-1]))
        * 2
    )
    x[-1] = pre * np.exp(-partitions[-1] ** 2) / (1 - erf(partitions[-1])) * 2
    return x * Cc * convert_deltaE(deltaE, deltaEconv) / eV


def Cc_defocus_spread(
    df, Cc: float, deltaE: float, eV: float, deltaEconv: str = "1/e",
):
    """Gaussian focal-spread PDF evaluated at ``df``."""
    df_spread = Cc * convert_deltaE(deltaE, deltaEconv) / eV
    return np.exp(-(df / df_spread) ** 2) / (np.sqrt(np.pi) * df_spread)


def simulation_result_with_Cc(
    func, Cc, deltaE, eV, args=None, kwargs=None,
    npoints: int = 7, deltaEconv: str = "1/e",
):
    """Average a simulation function over a Gaussian focal-spread distribution.

    ``func`` must accept ``df`` somewhere in its keyword arguments.
    Results are averaged with equal weights over ``npoints`` defocus
    samples.  Numpy arrays, dicts of arrays, and lists of arrays are all
    supported as ``func`` outputs.
    """
    if kwargs is None:
        kwargs = {}
    if args is None:
        args = []
    nominal_df = kwargs.get("df", 0.0)
    defocii = Cc_integration_points(Cc, deltaE, eV, npoints, deltaEconv) + nominal_df
    ndf = len(defocii)
    average = None
    for df in defocii:
        kwargs["df"] = df
        result = func(*args, **kwargs)
        if isinstance(result, np.ndarray):
            average = result / ndf if average is None else average + result / ndf
        elif isinstance(result, dict):
            if average is None:
                average = {k: (v / ndf if v is not None else None)
                           for k, v in copy.deepcopy(result).items()}
            else:
                for k in average:
                    if average[k] is not None:
                        average[k] = average[k] + result[k] / ndf
        else:
            average = ([x / ndf for x in result] if average is None
                       else [a + x / ndf for a, x in zip(average, result)])
    return average


__all__ = [
    "depth_of_field",
    "convert_tilt_angles",
    "make_contrast_transfer_function",
    "focused_probe",
    "plane_wave_illumination",
    "convert_deltaE",
    "Cc_integration_points",
    "Cc_defocus_spread",
    "simulation_result_with_Cc",
]
