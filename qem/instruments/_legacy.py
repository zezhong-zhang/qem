"""Compatibility shim for the pre-2026-05-03 optics public API.

Every name re-exported here delegates to the new functional core under
:mod:`qem.instruments.optics`.  Calling any wrapper emits a
``DeprecationWarning`` so users can migrate at their own pace.

The shim covers what was previously accessible from
``qem.instruments.{probe,ctf}`` and ``qem.instruments``.  For the
internal qem.fit / tests / examples migration we use the new API
directly — this file is for external users (notebooks, downstream
scripts).
"""

from __future__ import annotations

import warnings
from typing import Optional

import numpy as np

from ..processing.psf import (
    calculate_psf_width,
    extract_psf_from_atom_image,
)
from .optics import (
    Aberrations,
    Grid,
    Probe,
    adf_psf,
    chi as _new_chi,
    epie_psf,
    focal_spread_from_chromatic,
    icom_psf,
    invA_to_mrad,
    mrad_to_invA,
    relativistic_mass_correction,
    ssb_psf,
    wavelength,
    wavev,
)


def _warn(old: str, new: str) -> None:
    warnings.warn(
        f"{old} is deprecated; use {new} instead.",
        DeprecationWarning,
        stacklevel=3,
    )


# ---------------------------------------------------------------------------
# Old Aberration / aberration_starter_pack
# ---------------------------------------------------------------------------


class Aberration:
    """Legacy single-aberration record.

    .. deprecated::
        Use :class:`qem.instruments.optics.Aberrations`, which carries
        every coefficient up to fifth order in one immutable dataclass.
    """

    def __init__(self, *args):
        # Accept the two historical argument shapes:
        #   Aberration(amplitude, angle, n, m)
        #   Aberration(Krivanek, Haider, Description, amplitude, angle, n, m)
        if len(args) == 4:
            self.Krivanek = ""
            self.Haider = ""
            self.Description = ""
            self.amplitude, self.angle, self.n, self.m = args
        elif len(args) == 7:
            (self.Krivanek, self.Haider, self.Description,
             self.amplitude, self.angle, self.n, self.m) = args
        else:
            raise TypeError(
                "Aberration() takes 4 args (amplitude, angle, n, m) or "
                f"7 args (Krivanek, Haider, Description, ...); got {len(args)}."
            )
        if self.m == 0:
            self.angle = 0.0
        # No deprecation warning here — too noisy when constructing
        # starter packs with all zeros.

    def __repr__(self) -> str:  # pragma: no cover
        return (f"Aberration(Krivanek={self.Krivanek!r}, amplitude={self.amplitude}, "
                f"angle={self.angle}, n={self.n}, m={self.m})")


_STARTER_SYMBOLS = (
    ("C10", "C1", "Defocus          ", 1, 0),
    ("C12", "A1", "2-Fold astig.    ", 1, 2),
    ("C23", "A2", "3-Fold astig.    ", 2, 3),
    ("C21", "B2", "Axial coma       ", 2, 1),
    ("C30", "C3", "3rd order spher. ", 3, 0),
    ("C34", "A3", "4-Fold astig.    ", 3, 4),
    ("C32", "S3", "Axial star aber. ", 3, 2),
    ("C45", "A4", "5-Fold astig.    ", 4, 5),
    ("C43", "D4", "3-Lobe aberr.    ", 4, 3),
    ("C41", "B4", "4th order coma   ", 4, 1),
    ("C50", "C5", "5th order spher. ", 5, 0),
    ("C56", "A5", "6-Fold astig.    ", 5, 6),
    ("C52", "S5", "5th order star   ", 5, 2),
    ("C54", "R5", "5th order rosette", 5, 4),
)


def aberration_starter_pack() -> list[Aberration]:
    """Legacy list of all-zero-amplitude aberrations up to fifth order.

    .. deprecated::
        Use ``Aberrations()`` (all coefficients default to 0); set the
        ones you want with kwargs.
    """
    return [Aberration(K, H, D, 0.0, 0.0, n, m)
            for (K, H, D, n, m) in _STARTER_SYMBOLS]


def create_aberration_list(
    defocus: float = 0.0,
    two_fold_astigmatism: float = 0.0,
    two_fold_angle: float = 0.0,
    three_fold_astigmatism: float = 0.0,
    three_fold_angle: float = 0.0,
    coma: float = 0.0,
    coma_angle: float = 0.0,
    spherical_aberration: float = 0.0,
) -> list[Aberration]:
    """Legacy aberration-list factory.

    .. deprecated::
        Use ``Aberrations(defocus=…, Cs=…, astigmatism=…, …)`` —
        a single immutable dataclass replaces the list.

    Maintains the post-fix sign convention: ``defocus > 0`` ⇒
    under-focus, internally stored as ``C10 = -defocus``.
    """
    abs_: list[Aberration] = []
    if defocus != 0:
        abs_.append(Aberration("C10", "C1", "Defocus", -defocus, 0.0, 1, 0))
    if two_fold_astigmatism != 0:
        abs_.append(Aberration(
            "C12", "A1", "2-Fold astig.",
            two_fold_astigmatism, two_fold_angle, 1, 2,
        ))
    if three_fold_astigmatism != 0:
        abs_.append(Aberration(
            "C23", "A2", "3-Fold astig.",
            three_fold_astigmatism, three_fold_angle, 2, 3,
        ))
    if coma != 0:
        abs_.append(Aberration("C21", "B2", "Axial coma", coma, coma_angle, 2, 1))
    if spherical_aberration != 0:
        abs_.append(Aberration(
            "C30", "C3", "3rd order spher.",
            spherical_aberration, 0.0, 3, 0,
        ))
    return abs_


# ---------------------------------------------------------------------------
# Old `Probe` class — used by tests / examples to query envelopes only
# ---------------------------------------------------------------------------


class LegacyProbe:
    """Old qem.instruments.probe.Probe wrapper around the new API.

    .. deprecated::
        Use :class:`qem.instruments.optics.Probe` directly.
    """

    def __init__(
        self,
        eV: float,
        aperture: float = 20.0,
        df: float = 0.0,
        aberrations: Optional[list] = None,
        aperture_units: str = "mrad",
        Cc: Optional[float] = None,
        deltaE: Optional[float] = None,
        df_spread: Optional[float] = None,
        source_size: Optional[float] = None,
    ):
        self.eV = eV
        self.aperture = aperture
        self.df = df
        self.aberrations = aberrations or []
        self.aperture_units = aperture_units
        self.Cc = Cc
        self.deltaE = deltaE
        self.df_spread = df_spread
        self.source_size = source_size
        self.lam = wavelength(eV)
        self._abs = Aberrations.from_legacy_list(self.aberrations, df=df)
        if df_spread is not None:
            self._fs = float(df_spread)
        elif Cc is not None and deltaE is not None:
            self._fs = focal_spread_from_chromatic(Cc, deltaE, eV)
        else:
            self._fs = 0.0
        self._sigma = float(source_size) if source_size is not None else 0.0

    def chi(self, q, qphi):
        """Legacy chi(q, qphi) → np.ndarray."""
        import torch
        q_t = torch.as_tensor(np.asarray(q, dtype=float))
        p_t = torch.as_tensor(np.asarray(qphi, dtype=float))
        alpha = q_t * self.lam
        return _new_chi(
            alpha, p_t, wavelength=self.lam, aberrations=self._abs
        ).cpu().numpy()

    def temporal_coherence_envelope(self, q_mag):
        from .optics import temporal_envelope as _te
        import torch
        q = np.asarray(q_mag, dtype=float)
        a = torch.as_tensor(q * self.lam)
        return _te(a, wavelength=self.lam, focal_spread=self._fs).cpu().numpy()

    def spatial_coherence_envelope(self, q_mag, qphi=None):
        from .optics import spatial_envelope as _se
        import torch
        q = np.asarray(q_mag, dtype=float)
        a = torch.as_tensor(q * self.lam)
        if qphi is None:
            p = torch.zeros_like(a)
        else:
            p = torch.as_tensor(np.asarray(qphi, dtype=float))
        return _se(
            a, p, wavelength=self.lam, aberrations=self._abs,
            angular_spread_mrad=self._sigma,
        ).cpu().numpy()

    def partial_coherence_envelope(self, q_mag, qphi=None):
        return (self.temporal_coherence_envelope(q_mag)
                * self.spatial_coherence_envelope(q_mag, qphi=qphi))


# Re-export with the historical name.
LegacyProbe.__name__ = "Probe"
LegacyProbe.__qualname__ = "Probe"


# ---------------------------------------------------------------------------
# ProbeParameters dataclass + factory
# ---------------------------------------------------------------------------


class ProbeParameters:
    """Legacy probe-parameter holder.

    .. deprecated::
        Use :class:`qem.instruments.optics.Probe` directly — it's a
        frozen dataclass and serves the same purpose without a
        ``to_dict`` / ``from_dict`` round-trip.
    """

    def __init__(
        self,
        alpha: float,
        eV: float,
        df: float = 0.0,
        aberrations: Optional[list] = None,
        detector_inner: Optional[float] = None,
        detector_outer: Optional[float] = None,
        high_pass_cutoff: Optional[float] = None,
        Cc: Optional[float] = None,
        deltaE: Optional[float] = None,
        df_spread: Optional[float] = None,
        source_size: Optional[float] = None,
    ):
        self.alpha = alpha
        self.eV = eV
        self.df = df
        self.aberrations = aberrations
        self.detector_inner = detector_inner
        self.detector_outer = detector_outer
        self.high_pass_cutoff = high_pass_cutoff
        self.Cc = Cc
        self.deltaE = deltaE
        self.df_spread = df_spread
        self.source_size = source_size

    def to_dict(self) -> dict:
        return self.__dict__.copy()

    @classmethod
    def from_dict(cls, params: dict) -> "ProbeParameters":
        return cls(**{k: v for k, v in params.items()
                      if k in cls.__init__.__code__.co_varnames})

    def to_probe(self) -> Probe:
        ab_obj = Aberrations.from_legacy_list(self.aberrations or [], df=self.df)
        if self.df_spread is not None:
            fs = float(self.df_spread)
        elif self.Cc is not None and self.deltaE is not None:
            fs = focal_spread_from_chromatic(self.Cc, self.deltaE, self.eV)
        else:
            fs = 0.0
        return Probe(
            energy=self.eV,
            aperture=self.alpha,
            aberrations=ab_obj,
            focal_spread=fs,
            angular_spread=float(self.source_size or 0.0),
        )


def create_probe_parameters(
    alpha: float = 20.0,
    eV: float = 60e3,
    df: float = 0.0,
    defocus: Optional[float] = None,
    spherical_aberration: Optional[float] = None,
    two_fold_astigmatism: Optional[float] = None,
    two_fold_angle: Optional[float] = None,
    three_fold_astigmatism: Optional[float] = None,
    three_fold_angle: Optional[float] = None,
    coma: Optional[float] = None,
    coma_angle: Optional[float] = None,
    detector_inner: Optional[float] = None,
    detector_outer: Optional[float] = None,
    high_pass_cutoff: Optional[float] = None,
    aberrations: Optional[list] = None,
    Cc: Optional[float] = None,
    deltaE: Optional[float] = None,
    df_spread: Optional[float] = None,
    source_size: Optional[float] = None,
) -> ProbeParameters:
    """Legacy factory.

    .. deprecated::
        Build :class:`Probe` and :class:`Aberrations` directly.
    """
    if defocus is not None:
        df = defocus
    if aberrations is None:
        ab_list: list[Aberration] = []
        if spherical_aberration is not None:
            ab_list.append(Aberration(
                "C30", "C3", "3rd order spher.",
                spherical_aberration, 0.0, 3, 0,
            ))
        if two_fold_astigmatism is not None:
            ab_list.append(Aberration(
                "C12", "A1", "2-Fold astig.",
                two_fold_astigmatism,
                two_fold_angle if two_fold_angle is not None else 0.0,
                1, 2,
            ))
        if three_fold_astigmatism is not None:
            ab_list.append(Aberration(
                "C23", "A2", "3-Fold astig.",
                three_fold_astigmatism,
                three_fold_angle if three_fold_angle is not None else 0.0,
                2, 3,
            ))
        if coma is not None:
            ab_list.append(Aberration(
                "C21", "B2", "Axial coma",
                coma, coma_angle if coma_angle is not None else 0.0, 2, 1,
            ))
        aberrations = ab_list or None
    return ProbeParameters(
        alpha=alpha, eV=eV, df=df, aberrations=aberrations,
        detector_inner=detector_inner, detector_outer=detector_outer,
        high_pass_cutoff=high_pass_cutoff,
        Cc=Cc, deltaE=deltaE, df_spread=df_spread, source_size=source_size,
    )


# ---------------------------------------------------------------------------
# Legacy CTF classes — wrap the new psf functions
# ---------------------------------------------------------------------------


class _LegacyCTFBase:
    """Common scaffolding for old SSB/ADF/ePIE/iCoM_CTF classes."""

    _psf_fn = None  # set by subclass

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
        self.alpha = alpha
        self.eV = eV
        self.df = df
        self.aberrations = aberrations or []
        self.Cc = Cc
        self.deltaE = deltaE
        self.df_spread = df_spread
        self.source_size = source_size
        self.k = wavev(eV)
        self._probe = self._build_probe()

    def _build_probe(self) -> Probe:
        ab_obj = Aberrations.from_legacy_list(self.aberrations, df=self.df)
        if self.df_spread is not None:
            fs = float(self.df_spread)
        elif self.Cc is not None and self.deltaE is not None:
            fs = focal_spread_from_chromatic(self.Cc, self.deltaE, self.eV)
        else:
            fs = 0.0
        return Probe(
            energy=self.eV,
            aperture=self.alpha,
            aberrations=ab_obj,
            focal_spread=fs,
            angular_spread=float(self.source_size or 0.0),
        )

    def calculate_ctf(self, pix_dim, real_dim):
        from .optics import adf_ctf, epie_ctf, icom_ctf, ssb_ctf
        ctf_fn = {
            "SSB_CTF": ssb_ctf,
            "ADF_CTF": adf_ctf,
            "iCoM_CTF": icom_ctf,
            "ePIE_CTF": epie_ctf,
        }[type(self).__name__]
        grid = Grid(pixels=tuple(pix_dim), extent=tuple(real_dim))
        if type(self).__name__ == "iCoM_CTF":
            ctf = ctf_fn(grid, self._probe,
                         high_pass_mrad=getattr(self, "high_pass_cutoff", None))
        else:
            ctf = ctf_fn(grid, self._probe)
        return ctf.cpu().numpy().astype(np.complex128)

    def get_psf(self, pix_dim, real_dim):
        grid = Grid(pixels=tuple(pix_dim), extent=tuple(real_dim))
        if type(self).__name__ == "iCoM_CTF":
            psf = self._psf_fn(
                grid, self._probe,
                high_pass_mrad=getattr(self, "high_pass_cutoff", None),
            )
        else:
            psf = self._psf_fn(grid, self._probe)
        return psf.cpu().numpy().astype(np.float64)


class SSB_CTF(_LegacyCTFBase):
    """.. deprecated:: Use ``ssb_psf`` / ``ssb_ctf`` from qem.instruments.optics."""
    _psf_fn = staticmethod(ssb_psf)


class ADF_CTF(_LegacyCTFBase):
    """.. deprecated:: Use ``adf_psf`` / ``adf_ctf`` from qem.instruments.optics."""
    _psf_fn = staticmethod(adf_psf)

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
        super().__init__(alpha, eV, df, aberrations, Cc, deltaE, df_spread, source_size)
        self.detector_inner = detector_inner
        self.detector_outer = detector_outer


class ePIE_CTF(_LegacyCTFBase):
    """.. deprecated:: Use ``epie_psf`` / ``epie_ctf`` from qem.instruments.optics."""
    _psf_fn = staticmethod(epie_psf)

    def __init__(
        self,
        alpha: float,
        eV: float,
        df: float = 0.0,
        defocus: float = 1.0,
        step_size: float = 0.5,
        aberrations: Optional[list] = None,
        Cc: Optional[float] = None,
        deltaE: Optional[float] = None,
        df_spread: Optional[float] = None,
        source_size: Optional[float] = None,
    ):
        super().__init__(alpha, eV, df, aberrations, Cc, deltaE, df_spread, source_size)
        self.defocus = defocus
        self.step_size = step_size


class iCoM_CTF(_LegacyCTFBase):
    """.. deprecated:: Use ``icom_psf`` / ``icom_ctf`` from qem.instruments.optics."""
    _psf_fn = staticmethod(icom_psf)

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
        super().__init__(alpha, eV, df, aberrations, Cc, deltaE, df_spread, source_size)
        self.filter_type = filter_type
        # filter_type='none' or 'highpass' both currently produce a high-pass
        # CTF in the legacy code when a cutoff is set; mirror that here.
        self.high_pass_cutoff = high_pass_cutoff if filter_type != "none" else high_pass_cutoff


# ---------------------------------------------------------------------------
# Free-function legacy chi(q, qphi, lam, df, aberrations) wrapper
# ---------------------------------------------------------------------------


def chi(q, qphi, lam, df=0.0, aberrations=None):
    """Legacy chi(q, qphi, lam, df, aberrations) → np.ndarray.

    .. deprecated::
        Use :func:`qem.instruments.optics.chi` directly.  Note that the
        new function takes ``alpha = q · lam`` (in radians) instead of
        ``q`` (in 1/Å).
    """
    import torch
    q_arr = np.asarray(q, dtype=float)
    p_arr = np.asarray(qphi, dtype=float)
    a = torch.as_tensor(q_arr * lam)
    p = torch.as_tensor(p_arr)
    ab = Aberrations.from_legacy_list(aberrations or [], df=df)
    return _new_chi(a, p, wavelength=lam, aberrations=ab).cpu().numpy()


__all__ = [
    "Aberration",
    "aberration_starter_pack",
    "create_aberration_list",
    "LegacyProbe",
    "ProbeParameters",
    "create_probe_parameters",
    "SSB_CTF",
    "ADF_CTF",
    "ePIE_CTF",
    "iCoM_CTF",
    "chi",
    "wavev",
    "wavelength",
    "relativistic_mass_correction",
    "mrad_to_invA",
    "invA_to_mrad",
    "calculate_psf_width",
    "extract_psf_from_atom_image",
]
