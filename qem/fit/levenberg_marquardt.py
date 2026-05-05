"""Levenberg-Marquardt (Gauss-Newton with damping) for nonlinear least squares.

Matrix-free: uses ``torch.func.jvp`` + ``torch.func.vjp`` to evaluate
``J·v`` and ``J^T·v`` on demand, and solves the trust-region subproblem
``(J^T J + λ I) δ = -J^T r`` with conjugate gradient.

This is the right tool for STEM atom-fitting — smooth Gaussian
forward model, 10²-10⁴ parameters, 10⁶ residuals — and it
consistently beats Adam/AdamW and PyTorch's stock LBFGS in both
final loss and convergence rate. Same algorithm family that scipy's
``least_squares(method='trf')`` and StatSTEM's MATLAB fitter use.

Robust loss support (``loss="huber"|"soft_l1"|"cauchy"``) follows the
IRLS reformulation: each LM iteration freezes the per-pixel weight
``w_i = ρ'(r_i)/r_i`` from the current residual, then takes a
Gauss-Newton step on the weighted least-squares problem
``min ½ ||√w · r||²``. Equivalent in matrix-free form to multiplying
cotangents by ``w`` before the VJP — almost free.

Usage::

    from qem.fit.levenberg_marquardt import fit_lm
    fit_lm(model, inputs, target, max_iter=30)               # plain LS
    fit_lm(model, inputs, target, max_iter=30, loss="huber") # robust

Or via :class:`Fitter` by passing ``optimizer="lm"`` (and
``loss="huber"`` etc. via ``**optimizer_kwargs``).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Callable

import torch
from torch import nn
from torch.func import functional_call, jvp, vjp

log = logging.getLogger(__name__)


@dataclass
class LMResult:
    final_loss: float
    iters_run: int
    converged: bool


def _trainable_named_params(model: nn.Module):
    return [(n, p) for n, p in model.named_parameters() if p.requires_grad]


def _flat_trainable(model: nn.Module) -> torch.Tensor:
    parts = [p.detach().reshape(-1) for _, p in _trainable_named_params(model)]
    if not parts:
        raise RuntimeError("Model has no trainable parameters.")
    return torch.cat(parts)


def _write_back(model: nn.Module, theta: torch.Tensor) -> None:
    offset = 0
    with torch.inference_mode():
        for _, p in _trainable_named_params(model):
            n = p.numel()
            p.data.copy_(theta[offset : offset + n].reshape(p.shape))
            offset += n


def _make_residual_fn(
    model: nn.Module,
    inputs: Any,
    target: torch.Tensor,
) -> tuple[Callable[[torch.Tensor], torch.Tensor], list[str], list[torch.Size]]:
    """Build ``f(theta_flat) -> residual_vector`` using ``functional_call``.

    Returns the function plus the names/shapes used to flatten/unflatten.
    Frozen parameters are bound at capture time and not part of theta.
    """
    trainable = _trainable_named_params(model)
    names = [n for n, _ in trainable]
    shapes = [p.shape for _, p in trainable]
    sizes = [p.numel() for _, p in trainable]

    frozen = {n: p.detach() for n, p in model.named_parameters() if not p.requires_grad}
    buffers = {n: b.detach() for n, b in model.named_buffers()}
    target_flat = target.detach().reshape(-1)

    def unflatten(theta: torch.Tensor) -> dict[str, torch.Tensor]:
        out: dict[str, torch.Tensor] = {**frozen, **buffers}
        offset = 0
        for n, s, k in zip(names, shapes, sizes):
            out[n] = theta[offset : offset + k].reshape(s)
            offset += k
        return out

    def f(theta: torch.Tensor) -> torch.Tensor:
        params = unflatten(theta)
        pred = functional_call(model, params, inputs)
        return pred.reshape(-1) - target_flat

    return f, names, shapes


def _robust_weight(
    r: torch.Tensor,
    loss: str,
    scale: float | None,
) -> tuple[torch.Tensor, float, torch.Tensor]:
    """IRLS weight ``w_i = ρ'(r_i) / r_i`` and per-pixel cost ``ρ(r_i)``.

    Auto-estimates the scale from a robust MAD when not provided
    (``s = 1.4826 · median(|r|)``). Returns ``(w, scale_used, cost_per_pixel)``.
    """
    loss = loss.lower()
    if loss in ("l2", "ls", "lsq", "none"):
        ones = torch.ones_like(r)
        return ones, 0.0, 0.5 * r * r

    if scale is None or scale <= 0:
        med = float(r.detach().abs().median().item())
        s = max(1.4826 * med, 1e-6)
    else:
        s = float(scale)

    if loss == "huber":
        c = 1.345 * s
        absr = r.abs()
        inside = absr <= c
        w = torch.where(inside, torch.ones_like(r),
                        c / absr.clamp(min=1e-12))
        cost = torch.where(inside, 0.5 * r * r, c * absr - 0.5 * c * c)
        return w, s, cost
    if loss in ("soft_l1", "softl1"):
        # ρ(r) = c²·(√(1+(r/c)²) − 1),  ρ'(r)/r = 1/√(1+(r/c)²)
        z2 = (r / s) ** 2
        denom = torch.sqrt(1.0 + z2)
        w = 1.0 / denom
        cost = (s * s) * (denom - 1.0)
        return w, s, cost
    if loss == "cauchy":
        # ρ(r) = (c²/2)·log(1+(r/c)²),  ρ'(r)/r = 1/(1+(r/c)²)
        z2 = (r / s) ** 2
        denom = 1.0 + z2
        w = 1.0 / denom
        cost = 0.5 * (s * s) * torch.log(denom)
        return w, s, cost
    raise ValueError(
        f"Unknown loss {loss!r}; expected 'l2', 'huber', 'soft_l1', 'cauchy'."
    )


def _estimate_diag_jtj(
    f: Callable[[torch.Tensor], torch.Tensor],
    theta: torch.Tensor,
    *,
    n_probes: int = 4,
) -> torch.Tensor:
    """Hutchinson estimate of ``diag(J^T J)`` (column norms² of the Jacobian).

    Uses Rademacher probes ``v ∈ {±1}^p``: ``E[v ⊙ J^T J v] = diag(J^T J)``.
    Each probe needs one JVP+VJP. With ``n_probes=4`` the estimate is
    typically within ~30% relative error per component — plenty for
    Marquardt-style per-parameter damping (which only needs the
    *relative* scale across parameters).
    """
    diag = torch.zeros_like(theta)
    for _ in range(n_probes):
        v = torch.empty_like(theta).bernoulli_(0.5).mul_(2).sub_(1)
        _out, jv = jvp(f, (theta,), (v,))
        _, vjp_fn = vjp(f, theta)
        (jtjv,) = vjp_fn(jv)
        diag = diag + v * jtjv
    return (diag / n_probes).clamp_min(0.0)


def cg_solve(
    matvec: Callable[[torch.Tensor], torch.Tensor],
    b: torch.Tensor,
    *,
    max_iter: int = 30,
    tol: float = 1e-3,
) -> tuple[torch.Tensor, int]:
    """Conjugate gradient for symmetric positive-definite ``A x = b``.

    Returns ``(x, iters_used)``. Stops on relative residual ``< tol``.
    """
    x = torch.zeros_like(b)
    r = b.clone()
    p = r.clone()
    rs_old = r.dot(r)
    b_norm = b.norm()
    if b_norm.item() == 0.0:
        return x, 0
    target_sq = (tol * b_norm) ** 2
    iters = 0
    for k in range(max_iter):
        iters = k + 1
        Ap = matvec(p)
        denom = p.dot(Ap)
        if not torch.isfinite(denom) or denom.item() <= 0.0:
            break
        alpha = rs_old / denom
        x = x + alpha * p
        r = r - alpha * Ap
        rs_new = r.dot(r)
        if rs_new <= target_sq:
            break
        p = r + (rs_new / rs_old) * p
        rs_old = rs_new
    return x, iters


def fit_lm(
    model: nn.Module,
    inputs: Any,
    target: torch.Tensor,
    *,
    max_iter: int = 30,
    cg_max_iter: int = 30,
    cg_tol: float = 1e-3,
    ftol: float = 1e-9,
    gtol: float = 1e-9,
    lam_init: float = 1e-3,
    lam_min: float = 1e-12,
    lam_max: float = 1e12,
    loss: str = "l2",
    loss_scale: float | None = None,
    scale: bool = True,
    scale_n_probes: int = 4,
    verbose: bool = False,
    progress: bool = True,
    use_amp: bool = False,
) -> LMResult:
    """Levenberg-Marquardt with matrix-free CG. Updates ``model`` in place.

    Args:
        model: ``nn.Module`` whose ``forward(inputs)`` returns the
            prediction tensor.
        inputs: arguments forwarded to ``model``.
        target: target tensor (any shape; flattened internally).
        max_iter: maximum number of LM outer iterations.
        cg_max_iter: maximum CG iterations per LM step.
        cg_tol: relative residual tolerance for the CG inner solve.
        ftol: stop if ``|cost reduction| < ftol * cost``.
        gtol: stop if ``||J^T r||_∞ < gtol``.
        lam_init: initial Marquardt damping.
        lam_min, lam_max: clamps for adaptive damping.
        loss: residual loss. ``"l2"`` is plain Gauss-Newton.
            ``"huber"``, ``"soft_l1"`` and ``"cauchy"`` are robust
            variants that down-weight outlier pixels via IRLS.
        loss_scale: scale ``c`` in the robust loss. ``None`` ⇒
            estimated each LM iteration as ``1.4826·median(|r|)``
            (MAD-based, scale-equivariant).
        scale: use Marquardt's per-parameter diagonal damping
            ``(J^T J + λ·diag(J^T J))δ = -J^T r`` instead of the plain
            isotropic form ``(J^T J + λI)δ = -J^T r``. The diagonal
            entries are estimated once at start via Hutchinson probes.
            This is what scipy ``least_squares`` and StatSTEM's
            lsqnonlin (TRF) do internally; it auto-balances damping
            across parameters with very different physical scales
            (positions in pixels vs heights in image units vs
            background near zero). Default True.
        scale_n_probes: number of Hutchinson probes for the
            ``diag(J^T J)`` estimate (default 4). Each probe costs one
            JVP+VJP, so this is amortised over the whole fit.
        verbose: log per-iteration progress.
    """
    f, _names, _shapes = _make_residual_fn(model, inputs, target)
    theta = _flat_trainable(model).clone()

    r = f(theta).detach()
    w, s, cost_pp = _robust_weight(r, loss, loss_scale)
    cost = float(cost_pp.sum().item())
    lam = float(lam_init)
    iters_run = 0
    converged = False

    # Marquardt's per-parameter scaling: diagonal of J^T J. Estimated
    # once at the well-initialised starting point — by the time LM is
    # invoked the parameters are close enough to the optimum that
    # column norms don't shift much, and re-estimating each iteration
    # would double the LM cost. Falls back to the isotropic form when
    # ``scale=False`` is requested.
    if scale:
        with torch.inference_mode():
            damp_diag = _estimate_diag_jtj(f, theta, n_probes=scale_n_probes)
        # Floor by mean(diag) * 1e-6 so unobservable parameters (which
        # would otherwise get zero damping and ill-conditioned CG) still
        # get *some* regularisation.
        diag_mean = float(damp_diag.mean().item()) if damp_diag.numel() else 0.0
        floor = max(diag_mean * 1e-6, 1e-12)
        damp_diag = damp_diag.clamp_min(floor)
    else:
        damp_diag = torch.ones_like(theta)

    # AMP gating: only meaningful on CUDA; no-op on CPU/MPS.
    _amp_enabled = use_amp and torch.cuda.is_available()
    scaler = torch.cuda.amp.GradScaler() if _amp_enabled else None

    if verbose:
        log.info("LM start: cost=%.6e, n_params=%d, n_resid=%d, loss=%s, amp=%s",
                 cost, theta.numel(), r.numel(), loss, _amp_enabled)

    # Local import to avoid pulling tqdm into module-load if unused.
    from tqdm.auto import tqdm
    bar = tqdm(total=max_iter, desc=f"LM ({loss})", leave=False, disable=not progress)

    # Early-stop if too many consecutive rejected steps — means we've
    # ramped λ into the noise floor and further iterations won't help.
    consecutive_rejects = 0
    max_consecutive_rejects = 5

    for it in range(max_iter):
        # vjp_fn closes over theta's forward graph, so subsequent
        # vjp_fn(c) calls only re-run the backward — no new forward.
        with torch.amp.autocast("cuda", enabled=_amp_enabled):
            out, vjp_fn = vjp(f, theta)
        # Gradient g = J^T (w*r). For L2, w = 1 and out = r.
        (g,) = vjp_fn(w * out)

        g_inf = g.abs().max().item()
        if g_inf < gtol:
            converged = True
            iters_run = it
            if verbose:
                log.info("LM stop: |g|_inf=%.3e < gtol", g_inf)
            break

        # Weighted Marquardt normal equations:
        #   (J^T diag(w) J + λ·D) δ = -J^T (w r)
        # where D = diag(J^T J) gives per-parameter damping (auto-
        # balances positions in pixels vs heights in image units vs
        # background near zero — the classical Marquardt 1963 form).
        # Matrix-free: jv = J·v then vjp(w·jv) = J^T diag(w) J · v.
        def matvec(v: torch.Tensor) -> torch.Tensor:
            with torch.amp.autocast("cuda", enabled=_amp_enabled):
                _out2, jv = jvp(f, (theta,), (v,))
                (jtjv,) = vjp_fn(w * jv)
            return jtjv + lam * damp_diag * v

        delta, cg_iters = cg_solve(matvec, -g, max_iter=cg_max_iter, tol=cg_tol)

        # Predicted reduction under the weighted GN model. Using the
        # quadratic on the L2 surrogate Σ w r²/2 — sufficient for the
        # ρ accept/reject step.
        with torch.inference_mode():
            with torch.amp.autocast("cuda", enabled=_amp_enabled):
                _, jdelta = jvp(f, (theta,), (delta,))
            wjd = w * jdelta
            pred_red = (-r.dot(wjd) - 0.5 * jdelta.dot(wjd)).item()

        theta_new = theta + delta
        with torch.amp.autocast("cuda", enabled=_amp_enabled):
            r_new = f(theta_new).detach()
        # Re-weight against the trial residual so that cost/cost_new are
        # measured under the SAME loss function (otherwise we'd compare
        # apples to oranges across robust losses).
        w_new, _, cost_pp_new = _robust_weight(r_new, loss, loss_scale)
        cost_new = float(cost_pp_new.sum().item())
        actual_red = cost - cost_new
        rho = actual_red / pred_red if abs(pred_red) > 0 else 0.0

        if rho > 0.75:
            lam = max(lam * 0.33, lam_min)
        elif rho < 0.25:
            lam = min(lam * 3.0, lam_max)

        accepted = rho > 0
        if accepted:
            theta = theta_new
            r = r_new
            cost = cost_new
            w = w_new
            consecutive_rejects = 0
        else:
            consecutive_rejects += 1

        bar.set_postfix(
            cost=f"{cost:.3e}", lam=f"{lam:.1e}",
            rho=f"{rho:+.2f}", step="✓" if accepted else "✗",
        )
        bar.update(1)
        if verbose:
            log.info(
                "LM iter %3d: cost=%.6e |δ|=%.3e |g|_∞=%.3e λ=%.2e "
                "ρ=%+.2f cg=%d %s",
                it + 1, cost, delta.norm().item(), g_inf, lam, rho, cg_iters,
                "accept" if accepted else "reject",
            )

        iters_run = it + 1

        if accepted and abs(actual_red) < ftol * max(cost, 1.0):
            converged = True
            if verbose:
                log.info("LM stop: relative cost change < ftol")
            break

        if lam >= lam_max and not accepted:
            if verbose:
                log.info("LM stop: damping saturated, no progress.")
            break

        if consecutive_rejects >= max_consecutive_rejects:
            if verbose:
                log.info(
                    "LM stop: %d consecutive rejected steps — "
                    "in a flat / saddle region, won't improve.",
                    consecutive_rejects,
                )
            break

    bar.close()
    _write_back(model, theta)
    return LMResult(final_loss=cost, iters_run=iters_run, converged=converged)


__all__ = ["fit_lm", "cg_solve", "LMResult"]
