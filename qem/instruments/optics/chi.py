"""Aberration phase χ(α, φ) and its gradient (PyTorch primitives).

Both functions are pure: in-tensors → out-tensors, no hidden state.
``alpha`` is the polar semi-angle in radians (= q · λ); ``phi`` is the
azimuthal angle in radians.  Coefficients come from
:class:`qem.instruments.optics.aberrations.Aberrations`.

Sign convention matches abtem:
``ψ(k) = aperture(k) · exp(-i · χ(k))`` and ``defocus = -C10``.
The polynomial expansion is the abtem one (Aberrations._evaluate_…).
"""

from __future__ import annotations

import math

import torch

from .aberrations import Aberrations


def _coeffs(ab: Aberrations) -> dict[str, float]:
    """Mapping every polar symbol → its float value (defaults to 0.0)."""
    return {
        "C10": ab.C10, "C12": ab.C12, "phi12": ab.phi12,
        "C21": ab.C21, "phi21": ab.phi21,
        "C23": ab.C23, "phi23": ab.phi23,
        "C30": ab.C30,
        "C32": ab.C32, "phi32": ab.phi32,
        "C34": ab.C34, "phi34": ab.phi34,
        "C41": ab.C41, "phi41": ab.phi41,
        "C43": ab.C43, "phi43": ab.phi43,
        "C45": ab.C45, "phi45": ab.phi45,
        "C50": ab.C50,
        "C52": ab.C52, "phi52": ab.phi52,
        "C54": ab.C54, "phi54": ab.phi54,
        "C56": ab.C56, "phi56": ab.phi56,
    }


def chi(
    alpha: torch.Tensor,
    phi: torch.Tensor,
    *,
    wavelength: float,
    aberrations: Aberrations,
) -> torch.Tensor:
    """Aberration phase χ in radians.

    Parameters
    ----------
    alpha
        Semi-angle |q| · λ in radians.  Any tensor shape.
    phi
        Azimuth in radians.  Broadcastable with ``alpha``.
    wavelength
        Electron wavelength λ in Å.
    aberrations
        Polar coefficients (Krivanek convention).

    Returns
    -------
    Tensor with the same broadcast shape as ``alpha`` and ``phi``.

    Notes
    -----
    Each order-n term contributes ``α^(n+1)/(n+1) · Σ_m Cnm cos(m(φ − φnm))``;
    the whole polynomial is multiplied by ``2π/λ``.  See abtem
    ``Aberrations._evaluate_from_angular_grid`` (transfer.py:1366).
    """
    if aberrations.is_zero():
        return torch.zeros_like(alpha)
    p = _coeffs(aberrations)
    a2 = alpha * alpha
    a3 = a2 * alpha
    a4 = a2 * a2
    a5 = a4 * alpha
    a6 = a4 * a2
    expansion = torch.zeros_like(alpha)
    if p["C10"] or p["C12"]:
        expansion = expansion + 0.5 * a2 * (
            p["C10"] + p["C12"] * torch.cos(2 * (phi - p["phi12"]))
        )
    if p["C21"] or p["C23"]:
        expansion = expansion + (1.0 / 3.0) * a3 * (
            p["C21"] * torch.cos(phi - p["phi21"])
            + p["C23"] * torch.cos(3 * (phi - p["phi23"]))
        )
    if p["C30"] or p["C32"] or p["C34"]:
        expansion = expansion + 0.25 * a4 * (
            p["C30"]
            + p["C32"] * torch.cos(2 * (phi - p["phi32"]))
            + p["C34"] * torch.cos(4 * (phi - p["phi34"]))
        )
    if p["C41"] or p["C43"] or p["C45"]:
        expansion = expansion + 0.2 * a5 * (
            p["C41"] * torch.cos(phi - p["phi41"])
            + p["C43"] * torch.cos(3 * (phi - p["phi43"]))
            + p["C45"] * torch.cos(5 * (phi - p["phi45"]))
        )
    if p["C50"] or p["C52"] or p["C54"] or p["C56"]:
        expansion = expansion + (1.0 / 6.0) * a6 * (
            p["C50"]
            + p["C52"] * torch.cos(2 * (phi - p["phi52"]))
            + p["C54"] * torch.cos(4 * (phi - p["phi54"]))
            + p["C56"] * torch.cos(6 * (phi - p["phi56"]))
        )
    return expansion * (2.0 * math.pi / wavelength)


def grad_chi(
    alpha: torch.Tensor,
    phi: torch.Tensor,
    *,
    wavelength: float,
    aberrations: Aberrations,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Polar components of ``∇_α χ`` used by the spatial-coherence envelope.

    Returns ``(g_α, g_φ)`` where

        g_α = ∂χ/∂α        (radial part, rad / rad)
        g_φ = (1/α) ∂χ/∂φ  (metric-corrected azimuthal part)

    so that ``|∇_α χ|² = g_α² + g_φ²``.  This matches the convention
    abtem uses in ``SpatialEnvelope._evaluate_from_angular_grid`` (where
    the components are confusingly named ``dchi_dk`` and ``dchi_dphi``
    but are in fact the polar gradient pieces above).
    """
    if aberrations.is_zero():
        zeros = torch.zeros_like(alpha)
        return zeros, zeros.clone()
    p = _coeffs(aberrations)
    a2 = alpha * alpha
    a3 = a2 * alpha
    a4 = a2 * a2
    a5 = a4 * alpha
    pre = 2.0 * math.pi / wavelength
    dk = pre * (
        ((p["C10"] + p["C12"] * torch.cos(2 * (phi - p["phi12"]))) * alpha)
        + ((p["C21"] * torch.cos(phi - p["phi21"])
            + p["C23"] * torch.cos(3 * (phi - p["phi23"]))) * a2)
        + ((p["C30"]
            + p["C32"] * torch.cos(2 * (phi - p["phi32"]))
            + p["C34"] * torch.cos(4 * (phi - p["phi34"]))) * a3)
        + ((p["C41"] * torch.cos(phi - p["phi41"])
            + p["C43"] * torch.cos(3 * (phi - p["phi43"]))
            + p["C45"] * torch.cos(5 * (phi - p["phi45"]))) * a4)
        + ((p["C50"]
            + p["C52"] * torch.cos(2 * (phi - p["phi52"]))
            + p["C54"] * torch.cos(4 * (phi - p["phi54"]))
            + p["C56"] * torch.cos(6 * (phi - p["phi56"]))) * a5)
    )
    dphi = -pre * (
        0.5 * (2.0 * p["C12"] * torch.sin(2 * (phi - p["phi12"]))) * alpha
        + (1.0 / 3.0) * (
            3.0 * p["C23"] * torch.sin(3 * (phi - p["phi23"]))
            + p["C21"] * torch.sin(phi - p["phi21"])
        ) * a2
        + (1.0 / 4.0) * (
            4.0 * p["C34"] * torch.sin(4 * (phi - p["phi34"]))
            + 2.0 * p["C32"] * torch.sin(2 * (phi - p["phi32"]))
        ) * a3
        + (1.0 / 5.0) * (
            5.0 * p["C45"] * torch.sin(5 * (phi - p["phi45"]))
            + 3.0 * p["C43"] * torch.sin(3 * (phi - p["phi43"]))
            + p["C41"] * torch.sin(phi - p["phi41"])
        ) * a4
        + (1.0 / 6.0) * (
            6.0 * p["C56"] * torch.sin(6 * (phi - p["phi56"]))
            + 4.0 * p["C54"] * torch.sin(4 * (phi - p["phi54"]))
            + 2.0 * p["C52"] * torch.sin(2 * (phi - p["phi52"]))
        ) * a5
    )
    return dk, dphi
