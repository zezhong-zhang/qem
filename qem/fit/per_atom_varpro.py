"""StatSTEM-style alternating fit, vectorised per-atom on GPU.

The reference implementation in ``StatSTEM/functions/@inputStatSTEM/
fitGauss_samerho.m`` does, per outer iteration:

  1. For each atom i (in a parfor loop):
     a. Cut a local box around the atom of radius ``2.5·dist``.
     b. Subtract neighbour-Gaussian contributions from the local image.
     c. ``lsqnonlin`` (trust-region-reflective) on (BetaX_i, BetaY_i)
        with the height ``eta_i`` profiled out via closed-form linear LS
        (variable projection, *Golub-Pereyra*).
  2. Global linear LS refit of all heights and the scalar background
     (``getLinFitParam``).

This module implements the same algorithm but vectorised across all
atoms at once on the active torch device, so the whole sweep is a
handful of fused tensor ops instead of N MATLAB function calls. The
key trick is that each per-atom box is the same shape, so the boxes
stack into an ``(N, 2W+1, 2W+1)`` tensor and the per-atom 2×2
Gauss-Newton system is a batched linear solve.

Why this is robust: the per-atom local objective is quasi-convex
(one Gaussian against an essentially-isolated patch of the image
once neighbours are subtracted), so position updates can't drift
into a neighbour's basin during a step. The global linear refit
re-anchors heights and background between sweeps.

Usage::

    from qem.fit.per_atom_varpro import fit_per_atom_varpro
    fit_per_atom_varpro(fitter, max_iter=30)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from qem.utils.tensors import to_numpy, to_tensor

log = logging.getLogger(__name__)


@dataclass
class VarProResult:
    iters_run: int
    converged: bool
    final_max_dpos_px: float


def _build_local_boxes(
    image: torch.Tensor,
    pos_x: torch.Tensor,
    pos_y: torch.Tensor,
    half: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Stack per-atom (2W+1, 2W+1) windows into ``(N, 2W+1, 2W+1)`` tensors.

    Out-of-bounds pixels get masked to zero. Returns (obs_local, mask,
    int-corner) — int-corner is the integer floor of (pos_x, pos_y),
    needed to recover sub-pixel offsets for the Gaussian.
    """
    H, W = image.shape
    N = pos_x.shape[0]
    K = 2 * half + 1

    pos_x_int = torch.floor(pos_x).to(torch.int64)
    pos_y_int = torch.floor(pos_y).to(torch.int64)
    offsets = torch.arange(-half, half + 1, device=pos_x.device, dtype=torch.int64)
    dxg, dyg = torch.meshgrid(offsets, offsets, indexing="xy")  # (K, K)

    gx = pos_x_int[:, None, None] + dxg[None]  # (N, K, K)
    gy = pos_y_int[:, None, None] + dyg[None]

    in_bounds = (gx >= 0) & (gx < W) & (gy >= 0) & (gy < H)
    gx_clamped = gx.clamp(0, W - 1)
    gy_clamped = gy.clamp(0, H - 1)

    flat_idx = gy_clamped * W + gx_clamped
    obs = image.reshape(-1)[flat_idx.reshape(-1)].reshape(N, K, K)
    obs = obs * in_bounds.to(obs.dtype)
    return obs, in_bounds.to(obs.dtype), torch.stack([pos_x_int, pos_y_int], dim=-1)


def _per_atom_gauss_step(
    obs_local: torch.Tensor,         # (N, K, K) — neighbours-subtracted local target
    mask: torch.Tensor,              # (N, K, K)
    frac_x: torch.Tensor,            # (N,)
    frac_y: torch.Tensor,            # (N,)
    sigma: torch.Tensor,             # (N,) or scalar
    *,
    lam: float = 1e-6,
    delta_clip_px: float = 0.5,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """One Gauss-Newton step on per-atom (frac_x, frac_y) with η profiled out.

    Returns (delta_x, delta_y, eta_per_atom).
    """
    N, K, _ = obs_local.shape
    half = (K - 1) // 2
    offsets = torch.arange(-half, half + 1, device=obs_local.device, dtype=obs_local.dtype)
    dxg, dyg = torch.meshgrid(offsets, offsets, indexing="xy")  # (K, K)

    rx = dxg[None] - frac_x[:, None, None]  # (N, K, K)
    ry = dyg[None] - frac_y[:, None, None]
    sigma2 = (sigma if sigma.dim() else sigma)
    sigma2 = (sigma2 ** 2).reshape(-1, 1, 1) if sigma.dim() else sigma2 ** 2

    G = torch.exp(-(rx * rx + ry * ry) / (2 * sigma2)) * mask

    # VarPro: η = (G·obs) / (G·G)
    sum_Gt = (G * obs_local).sum(dim=(1, 2))
    sum_G2 = (G * G).sum(dim=(1, 2)).clamp_min(1e-12)
    eta = sum_Gt / sum_G2

    # Residual (Kaufman approximation: drop the dη/dβ chain term —
    # cheap and well-behaved on this kind of data; full Golub-Pereyra
    # adds a small correction but ~2× the per-step cost).
    model = eta[:, None, None] * G
    res = model - obs_local

    # Jacobian columns for (frac_x, frac_y).
    dGdx = G * rx / sigma2
    dGdy = G * ry / sigma2
    Jx = eta[:, None, None] * dGdx
    Jy = eta[:, None, None] * dGdy

    H_xx = (Jx * Jx).sum(dim=(1, 2))
    H_yy = (Jy * Jy).sum(dim=(1, 2))
    H_xy = (Jx * Jy).sum(dim=(1, 2))
    g_x = (Jx * res).sum(dim=(1, 2))
    g_y = (Jy * res).sum(dim=(1, 2))

    # 2×2 closed-form Levenberg–Marquardt step.
    a = H_xx + lam
    d = H_yy + lam
    b = H_xy
    det = (a * d - b * b).clamp_min(1e-30)
    delta_x = -(d * g_x - b * g_y) / det
    delta_y = -(-b * g_x + a * g_y) / det

    delta_x = delta_x.clamp(-delta_clip_px, delta_clip_px)
    delta_y = delta_y.clamp(-delta_clip_px, delta_clip_px)
    return delta_x, delta_y, eta


def fit_per_atom_varpro(
    fitter,
    *,
    max_iter: int = 30,
    pos_tol_px: float = 5e-3,
    alpha: float = 0.5,
    box_radius_factor: float = 3.0,
    inner_iter: int = 1,
    refit_eta_each_outer: bool = True,
    verbose: bool = False,
) -> VarProResult:
    """StatSTEM-style alternating fit, vectorised per-atom.

    Args:
        fitter: a :class:`Fitter` whose params are already initialised.
        max_iter: maximum number of outer iterations.
        pos_tol_px: stop when the max position move falls below this
            (in pixels).
        alpha: damping on the position update (0.5 by default — same
            as StatSTEM's ``alfa``).
        box_radius_factor: half-window in pixels = ``factor·width_px``.
            ``3.0`` covers 99.7% of an isotropic Gaussian, plenty for
            the per-atom fit.
        inner_iter: number of GN steps on positions per outer
            iteration before re-rendering the neighbour model and
            globally refitting heights.
        refit_eta_each_outer: call :meth:`Fitter.linear_estimator`
            after each outer iteration to refit heights + background
            globally given the new positions. Highly recommended
            (this is the StatSTEM ``getLinFitParam`` step).
        verbose: log per-iteration progress.
    """
    device = fitter.device
    image_t = fitter.image_tensor.to(device)
    H, W = int(fitter.ny), int(fitter.nx)

    pos_x = fitter.params["pos_x"].detach().clone().to(device)
    pos_y = fitter.params["pos_y"].detach().clone().to(device)
    height = fitter.params["height"].detach().clone().to(device)

    width = fitter.params["width"].detach().clone().to(device)
    same_width = bool(fitter.params.get("same_width", True))
    atom_types = fitter.params["atom_types"].detach().clone().to(device, dtype=torch.int64)
    if same_width:
        sigma_per_atom = width[atom_types]
    else:
        sigma_per_atom = width

    bg_key = "background" if "background" in fitter.params else "background_scale"
    bg_val = float(to_numpy(fitter.params[bg_key]).reshape(-1)[0])

    box_half = max(int(float(sigma_per_atom.max().item()) * box_radius_factor), 4)
    converged = False
    iters_run = 0
    last_max_dpos = float("inf")

    for outer in range(max_iter):
        # Render the FULL current model (using current η, σ, bg) once.
        # We use the existing local-window renderer — same prediction as the loss.
        with torch.inference_mode():
            full_pred = fitter.predict(
                {**fitter.params, "pos_x": pos_x, "pos_y": pos_y,
                 "height": height, "width": width, "background": to_tensor(bg_val)},
                local=True,
            )
            full_pred = full_pred.to(device)

        # Per-atom local target = obs - (full_pred - own_contribution).
        obs_box, mask, corners = _build_local_boxes(image_t, pos_x, pos_y, box_half)
        pred_box, _, _ = _build_local_boxes(full_pred, pos_x, pos_y, box_half)

        # Own contribution at each atom in its own window.
        frac_x = pos_x - corners[..., 0].to(pos_x.dtype)
        frac_y = pos_y - corners[..., 1].to(pos_y.dtype)
        offsets = torch.arange(
            -box_half, box_half + 1, device=device, dtype=obs_box.dtype,
        )
        dxg, dyg = torch.meshgrid(offsets, offsets, indexing="xy")
        rx = dxg[None] - frac_x[:, None, None]
        ry = dyg[None] - frac_y[:, None, None]
        s2 = (sigma_per_atom ** 2).reshape(-1, 1, 1)
        Gself = torch.exp(-(rx * rx + ry * ry) / (2 * s2)) * mask
        own_contrib = height[:, None, None] * Gself
        # Local target: obs - (others + bg) where (others + bg) = pred_box - own_contrib.
        target = obs_box - (pred_box - own_contrib)
        target = target * mask

        # `inner_iter` per-atom Gauss-Newton steps on positions.
        for _ in range(inner_iter):
            dx, dy, eta_new = _per_atom_gauss_step(
                target, mask, frac_x, frac_y, sigma_per_atom,
            )
            # Update fractional positions; let the integer corner re-snap
            # next outer iteration when we re-extract the box.
            frac_x = (frac_x + alpha * dx).clamp(-box_half + 1, box_half - 1)
            frac_y = (frac_y + alpha * dy).clamp(-box_half + 1, box_half - 1)

        new_pos_x = corners[..., 0].to(pos_x.dtype) + frac_x
        new_pos_y = corners[..., 1].to(pos_y.dtype) + frac_y

        max_dpos = float((torch.maximum((new_pos_x - pos_x).abs().max(),
                                        (new_pos_y - pos_y).abs().max())).item())
        last_max_dpos = max_dpos

        pos_x = new_pos_x
        pos_y = new_pos_y

        # Optional: re-anchor heights / bg with a global linear LS.
        if refit_eta_each_outer:
            params = {
                "pos_x": pos_x.detach(),
                "pos_y": pos_y.detach(),
                "height": height.detach(),
                "width": width.detach(),
                "background": to_tensor(bg_val),
                "same_width": fitter.params["same_width"],
                "atom_types": fitter.params["atom_types"],
            }
            try:
                fitter.params = params
                params = fitter.linear_estimator(params, best_effort=True)
                height = params["height"].detach().to(device)
                if "background" in params:
                    bg_val = float(to_numpy(params["background"]).reshape(-1)[0])
            except Exception as exc:  # pragma: no cover
                log.warning("VarPro inner LE failed: %s", exc)
        else:
            # At least update η using the per-atom VarPro estimate.
            height = eta_new.detach().to(device)

        if verbose:
            log.info("VarPro outer %3d: max|Δpos|=%.4f px, η median=%.1f, bg=%.1f",
                     outer + 1, max_dpos, float(height.median().item()), bg_val)

        iters_run = outer + 1
        if max_dpos < pos_tol_px and outer >= 1:
            converged = True
            break

    # Push final state back into the Fitter.
    final_params = {
        "pos_x": pos_x.detach(),
        "pos_y": pos_y.detach(),
        "height": height.detach(),
        "width": width.detach(),
        "background": to_tensor(bg_val),
        "same_width": fitter.params["same_width"],
        "atom_types": fitter.params["atom_types"],
    }
    fitter.params = final_params
    fitter.prediction = to_numpy(fitter.predict(final_params, local=True))
    return VarProResult(
        iters_run=iters_run,
        converged=converged,
        final_max_dpos_px=last_max_dpos,
    )


__all__ = ["fit_per_atom_varpro", "VarProResult"]
