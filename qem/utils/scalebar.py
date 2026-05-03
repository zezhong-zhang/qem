"""Helpers for matplotlib-scalebar integration.

``matplotlib_scalebar.ScaleBar`` only accepts SI-derived length units, but the
QEM codebase uses ``"A"`` (Ångström) as the canonical unit. This module
converts an ``(scale, units)`` pair stored in QEM convention into one that
``ScaleBar`` will accept.
"""

from __future__ import annotations

from typing import Tuple


_ANGSTROM_ALIASES = ("A", "Å")
_ANGSTROM_TO_METRES = 1e-10


def to_scalebar_units(scale: float, units: str) -> Tuple[float, str]:
    """Translate a ``(scale, units)`` pair into one ``ScaleBar`` accepts.

    Ångström values are converted to metres so ``matplotlib_scalebar`` can
    pick a sensible SI prefix (nm, pm, …) on its own. All other units pass
    through unchanged so existing callers that already use SI units (``m``,
    ``nm``, …) keep their behaviour.
    """
    if units in _ANGSTROM_ALIASES:
        return scale * _ANGSTROM_TO_METRES, "m"
    return scale, units
