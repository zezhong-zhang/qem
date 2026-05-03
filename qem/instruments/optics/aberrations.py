"""Polar aberration coefficients (abtem-compatible).

A single :class:`Aberrations` dataclass holds every named coefficient up
to fifth order.  Coefficients use Krivanek polar notation
(``Cnm`` magnitudes in Å, ``phinm`` angles in radians).

Sign convention
---------------
Aligned with abtem: ``defocus = -C10`` (positive defocus ⇒ under-focus).
Helper aliases on the dataclass let callers say ``Aberrations(defocus=50)``
or ``Aberrations(C10=-50)`` interchangeably; both store the same C10.
"""

from __future__ import annotations

from dataclasses import dataclass, field, fields
from typing import Iterable, Mapping


# Symbol → (n, m) order pairs.  Magnitudes (`Cnm`) carry units of Å,
# angles (`phinm`) of radians.  Order-1 ⇒ α² term, order-2 ⇒ α³, …
_POLAR_SYMBOLS: tuple[tuple[str, int, int], ...] = (
    ("C10", 1, 0),
    ("C12", 1, 2),
    ("C21", 2, 1),
    ("C23", 2, 3),
    ("C30", 3, 0),
    ("C32", 3, 2),
    ("C34", 3, 4),
    ("C41", 4, 1),
    ("C43", 4, 3),
    ("C45", 4, 5),
    ("C50", 5, 0),
    ("C52", 5, 2),
    ("C54", 5, 4),
    ("C56", 5, 6),
)

# All the named magnitude / phase fields that the dataclass exposes.
MAGNITUDE_SYMBOLS: tuple[str, ...] = tuple(sym for sym, _, _ in _POLAR_SYMBOLS)
PHASE_SYMBOLS: tuple[str, ...] = tuple(
    "phi" + sym[1:] for sym, _, m in _POLAR_SYMBOLS if m > 0
)
ALL_SYMBOLS: tuple[str, ...] = MAGNITUDE_SYMBOLS + PHASE_SYMBOLS

# Friendly aliases that callers can pass to ``Aberrations(...)`` or
# look up through __getattr__.  ``defocus`` carries an explicit sign flip
# (``defocus = -C10``); the rest are pure renames.
_ALIASES: Mapping[str, str] = {
    "defocus": "C10",          # negated
    "Cs": "C30",
    "C5": "C50",
    "astigmatism": "C12",
    "astigmatism_angle": "phi12",
    "astigmatism3": "C32",
    "astigmatism3_angle": "phi32",
    "astigmatism5": "C52",
    "astigmatism5_angle": "phi52",
    "coma": "C21",
    "coma_angle": "phi21",
    "coma4": "C41",
    "coma4_angle": "phi41",
    "trefoil": "C23",
    "trefoil_angle": "phi23",
    "trefoil4": "C43",
    "trefoil4_angle": "phi43",
    "quadrafoil": "C34",
    "quadrafoil_angle": "phi34",
    "quadrafoil5": "C54",
    "quadrafoil5_angle": "phi54",
    "pentafoil": "C45",
    "pentafoil_angle": "phi45",
    "hexafoil": "C56",
    "hexafoil_angle": "phi56",
}

_NEGATED_ALIASES = {"defocus"}


def _symbol_metadata() -> dict[str, tuple[int, int]]:
    return {sym: (n, m) for sym, n, m in _POLAR_SYMBOLS}


_SYM_NM = _symbol_metadata()


@dataclass(frozen=True)
class Aberrations:
    """Polar aberration coefficients.  Magnitudes in Å, angles in radians.

    Construct with named coefficients (Krivanek symbols) and/or aliases:

        >>> ab = Aberrations(defocus=50, Cs=1e7, astigmatism=10, astigmatism_angle=0.5)
        >>> ab.C10                  # internally stored as -defocus
        -50.0
        >>> ab.defocus              # alias property
        50.0
    """

    C10: float = 0.0
    C12: float = 0.0
    phi12: float = 0.0
    C21: float = 0.0
    phi21: float = 0.0
    C23: float = 0.0
    phi23: float = 0.0
    C30: float = 0.0
    C32: float = 0.0
    phi32: float = 0.0
    C34: float = 0.0
    phi34: float = 0.0
    C41: float = 0.0
    phi41: float = 0.0
    C43: float = 0.0
    phi43: float = 0.0
    C45: float = 0.0
    phi45: float = 0.0
    C50: float = 0.0
    C52: float = 0.0
    phi52: float = 0.0
    C54: float = 0.0
    phi54: float = 0.0
    C56: float = 0.0
    phi56: float = 0.0

    # ``frozen=True`` blocks normal assignment; we patch __init__ to
    # accept aliases and rewrite to the canonical field names.
    def __init__(self, **kwargs: float) -> None:
        canonical: dict[str, float] = {}
        for key, value in kwargs.items():
            if key in _ALIASES:
                target = _ALIASES[key]
                stored = -float(value) if key in _NEGATED_ALIASES else float(value)
                canonical[target] = canonical.get(target, 0.0) + stored
            elif key in ALL_SYMBOLS:
                canonical[key] = canonical.get(key, 0.0) + float(value)
            else:
                raise TypeError(
                    f"Aberrations() got an unexpected keyword argument {key!r}. "
                    f"Expected one of {ALL_SYMBOLS} or aliases {tuple(_ALIASES)}."
                )
        for f in fields(self):
            object.__setattr__(self, f.name, float(canonical.get(f.name, 0.0)))

    # Aliases as read-only properties that respect the same sign rules.
    @property
    def defocus(self) -> float:
        return -self.C10

    @property
    def Cs(self) -> float:
        return self.C30

    @property
    def C5(self) -> float:
        return self.C50

    def coefficients(self) -> dict[str, float]:
        """Return non-zero polar coefficients as a plain dict."""
        return {
            f.name: getattr(self, f.name)
            for f in fields(self)
            if getattr(self, f.name) != 0.0
        }

    def is_zero(self) -> bool:
        return all(getattr(self, f.name) == 0.0 for f in fields(self))

    @classmethod
    def from_mapping(cls, params: Mapping[str, float]) -> "Aberrations":
        return cls(**dict(params))

    @classmethod
    def from_legacy_list(
        cls, ablist: Iterable, df: float = 0.0
    ) -> "Aberrations":
        """Convert the old-style ``[Aberration(...), ...]`` list + ``df``.

        ``df`` follows the post-fix convention (``df = -C10``).  Each
        legacy aberration object is identified by its ``(n, m)`` tuple
        or, if available, its ``Krivanek`` tag.
        """
        kwargs: dict[str, float] = {}
        if df:
            kwargs["defocus"] = float(df)
        for ab in ablist or ():
            sym = getattr(ab, "Krivanek", "") or f"C{int(ab.n)}{int(ab.m)}"
            sym = sym.strip()
            if sym not in MAGNITUDE_SYMBOLS:
                raise ValueError(
                    f"Unknown aberration symbol {sym!r} in legacy list "
                    f"(n={ab.n}, m={ab.m})."
                )
            kwargs[sym] = kwargs.get(sym, 0.0) + float(ab.amplitude)
            if int(ab.m) > 0:
                phi_sym = "phi" + sym[1:]
                kwargs[phi_sym] = float(getattr(ab, "angle", 0.0))
        return cls(**kwargs)

    def __repr__(self) -> str:
        nz = self.coefficients()
        if not nz:
            return "Aberrations()"
        body = ", ".join(f"{k}={v:g}" for k, v in nz.items())
        return f"Aberrations({body})"
