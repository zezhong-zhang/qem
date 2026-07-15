"""Optimisation mixin for :class:`qem.fit.fitter.Fitter`.

This is a self-contained slice of Fitter — the high-level optimisation
API (``fit_stochastic`` / ``fit_global`` / ``optimize``) and its helpers
(``convergence``, ``select_params``, ``update_from_local_params``).

Why a mixin and not the existing ``_bind`` monkey-patch
(``cls.fit_stochastic = fit_stochastic``)?

  * **Visible to type-checkers**. Pyright/mypy walk the MRO and find
    these methods; the "Cannot access attribute fit_stochastic" noise
    we kept seeing on every test script is gone.
  * **Subclassing works**. ``class MyFitter(Fitter): def fit_global(...)``
    via ``super()`` does the right thing — with module-level monkey-patch
    you can't override cleanly because the rebound function is on the
    base class, not in the MRO.
  * **Jump-to-definition works** in the IDE — the method lives in a
    class, not as a lambda assignment.
  * **Same call-site API** — ``fitter.fit_stochastic(...)`` is unchanged.
    Composition (``fitter.opt.fit_stochastic(...)``) would break every
    caller in the notebooks, examples, and benchmark module.

The mixin documents which host attributes it reads and writes. The
attributes themselves are populated by the Fitter base ``__init__``;
the type annotations below are *declarations* (no default), so Python
doesn't try to assign them at mixin-import time.
"""

from __future__ import annotations

import logging
import math
from collections.abc import Callable
from contextlib import nullcontext
from typing import TYPE_CHECKING, Any

import numpy as np
import torch
from tqdm.auto import tqdm  # auto picks the JS bar in Jupyter, ASCII in TTY

from qem.utils.arrays import get_random_indices_in_batches
from qem.utils.tensors import clone_params, release_memory, stop_grad, to_numpy

# Inverse golden-ratio squared, used as Brent's golden-section fallback step.
_INV_PHI2: float = 0.5 * (3.0 - math.sqrt(5.0))   # ≈ 0.381966


def _brent_minimize(
    f: Callable[[float], float],
    lo: float,
    hi: float,
    *,
    xtol: float = 5e-3,
    ftol: float = 1e-4,
    max_evals: int = 12,
) -> tuple[float, float, int]:
    """Bounded 1-D minimiser using Brent's method (parabolic interp + GS fallback).

    Pure-Python (no scipy). Adapted from Numerical Recipes §10.2: at each
    step try inverse parabolic interpolation through the three best
    points; accept if the step is bounded and at least halves the
    previous step, otherwise fall back to a golden-section bisection.

    On a smooth, near-quadratic residual surface this converges in
    5-7 evals for ``xtol=5e-3`` — vs ~15 evals for plain golden section.

    Returns ``(x*, f(x*), n_evals)``. Stops early when (a) the bracket is
    smaller than ``xtol``, or (b) the last two parabolic steps each
    improved the loss by less than ``ftol`` relative to the best.
    """
    a, b = float(lo), float(hi)
    # Initial best-guess probe at the golden-ratio interior point.
    x = a + _INV_PHI2 * (b - a)
    w = v = x
    fx = f(x)
    fw = fv = fx
    n_evals = 1
    e = 0.0   # step from the previous-but-one iteration
    d = 0.0   # step from the previous iteration
    last_improve = float("inf")

    while n_evals < max_evals:
        m = 0.5 * (a + b)
        tol1 = xtol * abs(x) + 1e-10
        tol2 = 2.0 * tol1
        if abs(x - m) <= tol2 - 0.5 * (b - a):
            break

        use_parabolic = False
        if abs(e) > tol1:
            # Try inverse parabolic interpolation through (v, fv), (w, fw), (x, fx).
            r = (x - w) * (fx - fv)
            q = (x - v) * (fx - fw)
            p = (x - v) * q - (x - w) * r
            q2 = 2.0 * (q - r)
            if q2 > 0.0:
                p = -p
            q2 = abs(q2)
            e_prev = e
            e = d
            # Accept if step is bounded and < half the previous-but-one step.
            if (
                abs(p) < abs(0.5 * q2 * e_prev)
                and p > q2 * (a - x)
                and p < q2 * (b - x)
            ):
                d = p / q2
                u = x + d
                if (u - a) < tol2 or (b - u) < tol2:
                    d = tol1 if (m - x) >= 0 else -tol1
                use_parabolic = True

        if not use_parabolic:
            # Golden-section step into the larger sub-interval.
            e = (b - x) if x < m else (a - x)
            d = _INV_PHI2 * e

        u = x + (d if abs(d) >= tol1 else (tol1 if d >= 0 else -tol1))
        fu = f(u)
        n_evals += 1

        if fu <= fx:
            improve = fx - fu
            if u >= x:
                a = x
            else:
                b = x
            v, fv = w, fw
            w, fw = x, fx
            x, fx = u, fu
            # Early exit on stalled improvement: two consecutive steps
            # each contributing < ftol·|fx| improvement.
            rel = improve / (abs(fx) + 1e-30)
            if rel < ftol and last_improve < ftol:
                break
            last_improve = rel
        else:
            if u < x:
                a = u
            else:
                b = u
            if fu <= fw or w == x:
                v, fv = w, fw
                w, fw = u, fu
            elif fu <= fv or v == x or v == w:
                v, fv = u, fu
            last_improve = 0.0

    return x, fx, n_evals

if TYPE_CHECKING:
    from qem.fit.model import ImageModel
    from qem.utils.memory import MemoryMonitor

log = logging.getLogger(__name__)


class FitterOptimizationMixin:
    """High-level optimisation API mixed into :class:`Fitter`.

    Reads (provided by the host class):
        ``image_tensor``, ``device``, ``x_grid_batched``,
        ``y_grid_batched``, ``memory_monitor``, ``params``,
        ``same_width``, ``num_coordinates``.

        Methods: ``loss``, ``predict``, ``linear_estimator``,
        ``init_params``, ``_create_fitting_model``,
        ``_params_to_device``, ``_plot_progress``.

    Writes:
        ``params``, ``prediction``, ``converged``,
        ``_optimization_model``.
    """

    # ------------------------------------------------------------------
    # Attribute declarations (host class supplies these — no defaults).
    # Listing them gives Pyright/mypy enough type information to check
    # the method bodies without runtime overhead.
    # ------------------------------------------------------------------
    image_tensor: torch.Tensor
    device: torch.device
    x_grid_batched: torch.Tensor
    y_grid_batched: torch.Tensor
    memory_monitor: MemoryMonitor | None
    params: dict[str, Any]
    same_width: bool
    num_coordinates: int
    converged: bool
    prediction: Any
    _optimization_model: ImageModel | None

    # ------------------------------------------------------------------
    # Core dispatch
    # ------------------------------------------------------------------

    def optimize(
        self,
        model: ImageModel,
        image_tensor: torch.Tensor | None = None,
        params: dict | None = None,
        maxiter: int = 1000,
        tol: float = 1e-4,
        step_size: float = 0.01,
        optimizer: str = "adam",
        **optimizer_kwargs: Any,
    ) -> dict[str, Any]:
        """Optimise ``model`` against ``image_tensor`` using ``optimizer``.

        Dispatches to:
        - ``"lm"`` / ``"gn"``  → matrix-free Levenberg–Marquardt
          (:func:`qem.fit.levenberg_marquardt.fit_lm`).
        - anything else        → :func:`qem.fit.loop.fit_loop` with the
          named first-order optimiser (built-ins, ``pytorch_optimizer``
          or ``torch_optimizer`` — see :func:`qem.fit.loop.make_optimizer`).
        """
        if image_tensor is None:
            image_tensor = self.image_tensor
        if params is not None:
            model.set_params(params)
        if not model.built:
            model.build()
        # Move to active accelerator before the inner loop.
        model.to(self.device)
        # Loss closure reads the model from this attribute.
        self._optimization_model = model

        image_tensor = image_tensor.to(self.device).unsqueeze(0)
        model_inputs = [self.x_grid_batched, self.y_grid_batched]
        op_ctx = (
            self.memory_monitor.monitor_operation("optimize")
            if self.memory_monitor else nullcontext()
        )

        with op_ctx:
            if optimizer.lower() in {"lm", "gn"}:
                # Levenberg–Marquardt — bypass first-order fit_loop.
                from qem.fit.levenberg_marquardt import fit_lm

                fit_lm(
                    model=model,
                    inputs=model_inputs,
                    target=image_tensor,
                    max_iter=maxiter,
                    cg_max_iter=optimizer_kwargs.get("cg_max_iter", 30),
                    cg_tol=optimizer_kwargs.get("cg_tol", 1e-3),
                    ftol=tol,
                    gtol=optimizer_kwargs.get("gtol", 1e-9),
                    lam_init=optimizer_kwargs.get("lam_init", 1e-3),
                    loss=optimizer_kwargs.get("loss", "l2"),
                    loss_scale=optimizer_kwargs.get("loss_scale"),
                    scale=optimizer_kwargs.get("scale", True),
                    verbose=optimizer_kwargs.get("verbose", False),
                    progress=optimizer_kwargs.get("progress", True),
                )
            else:
                from qem.fit.loop import fit_loop, make_optimizer

                opt = make_optimizer(optimizer, model.parameters(), step_size,
                                     **{k: v for k, v in optimizer_kwargs.items()
                                        if k not in {"cg_max_iter", "cg_tol", "gtol",
                                                     "lam_init", "loss", "loss_scale"}})
                fit_loop(
                    model=model,
                    inputs=model_inputs,
                    target=image_tensor,
                    loss_fn=self.loss,  # type: ignore[attr-defined]
                    optimizer=opt,
                    epochs=maxiter,
                    tol=tol,
                    patience=100,
                    lr_patience=10,
                    lr_factor=0.1,
                    min_lr=1e-6,
                    verbose=optimizer_kwargs.get("verbose", False),
                )

        self._optimization_model = None
        return model.get_params()

    # ------------------------------------------------------------------
    # High-level fit entry points
    # ------------------------------------------------------------------

    def fit_global(
        self,
        params: dict | None = None,
        maxiter: int = 1000,
        tol: float = 1e-3,
        step_size: float = 0.01,
        optimizer: str = "adam",
        local: bool = True,
        verbose: bool = True,
        **optimizer_kwargs: Any,
    ) -> dict[str, Any]:
        """Joint fit of all parameters against the full image."""
        if params is None:
            params = self.params if self.params is not None else self.init_params()  # type: ignore[attr-defined]

        fitting_model = self._create_fitting_model(params)  # type: ignore[attr-defined]
        params = self.optimize(
            model=fitting_model,
            image_tensor=self.image_tensor,
            params=params,
            maxiter=maxiter,
            tol=tol,
            step_size=step_size,
            optimizer=optimizer,
            verbose=verbose,
            **optimizer_kwargs,
        )

        self.params = params
        self.prediction = to_numpy(self.predict(params, local=local))  # type: ignore[attr-defined]
        return params

    def fit_stochastic(
        self,
        params: dict | None = None,
        num_epoch: int = 5,
        batch_size: int = 500,
        maxiter: int = 50,
        tol: float = 1e-3,
        step_size: float = 1e-2,
        optimizer: str = "adam",
        verbose: bool = True,
        local: bool = True,
        plot: bool = False,
        progress: bool = True,
        **optimizer_kwargs: Any,
    ) -> dict[str, Any]:
        """Mini-batch fit: random batches of atoms, neighbour subtraction."""
        if params is None:
            params = self.params if self.params is not None else self.init_params()  # type: ignore[attr-defined]
        params = {k: stop_grad(v) for k, v in params.items()}

        self.converged = False
        op_ctx = (
            self.memory_monitor.monitor_operation("fit_stochastic")
            if self.memory_monitor else nullcontext()
        )

        # Pre-condition heights via NNLS — best-effort to keep the
        # warmup robust to occasional solver hiccups.
        params = self.linear_estimator(params, best_effort=True)  # type: ignore[attr-defined]
        params = self._params_to_device(params)  # type: ignore[attr-defined]

        # Estimate total batches up front so a single flat bar shows
        # both epoch and batch progress without nested-bar flicker.
        # ``get_random_indices_in_batches`` ceiling-divides; mirror it.
        n_batches = (self.num_coordinates + batch_size - 1) // batch_size
        total_steps = num_epoch * n_batches

        with op_ctx:
            bar = tqdm(
                total=total_steps, desc=f"stochastic {optimizer}",
                leave=False, disable=not progress,
            )
            for epoch in range(num_epoch):
                pre_params = clone_params(params)
                random_batches = get_random_indices_in_batches(self.num_coordinates, batch_size)

                for b_idx, batch_indices in enumerate(random_batches):
                    if batch_size < self.num_coordinates:
                        # Subtract other atoms' contributions to get this
                        # batch's local target.
                        params_without_batch = clone_params(params)
                        height_tensor = params_without_batch["height"]
                        batch_idx_t = torch.as_tensor(
                            batch_indices, dtype=torch.int64, device=height_tensor.device,
                        )
                        new_height = height_tensor.clone()
                        new_height.view(-1)[batch_idx_t] = 0.0
                        params_without_batch["height"] = new_height
                        params_without_batch["background"] = torch.zeros_like(
                            params_without_batch["background"],
                        )
                        model_others = self._create_fitting_model(params_without_batch)  # type: ignore[attr-defined]
                        prediction_from_others = self.predict(  # type: ignore[attr-defined]
                            params_without_batch, model=model_others, local=local,
                        )
                        local_target = (self.image_tensor - prediction_from_others).detach()
                        del params_without_batch, prediction_from_others, height_tensor
                        release_memory()
                    else:
                        local_target = self.image_tensor

                    atoms_selected_mask = np.zeros(self.num_coordinates, dtype=bool)
                    atoms_selected_mask[batch_indices] = True
                    sel_params = self.select_params(params, atoms_selected_mask)
                    local_model = self._create_fitting_model(sel_params)  # type: ignore[attr-defined]

                    optimized_params = self.optimize(
                        model=local_model,
                        image_tensor=local_target,
                        params=sel_params,
                        maxiter=maxiter,
                        tol=tol,
                        step_size=step_size,
                        optimizer=optimizer,
                        verbose=verbose,
                        **optimizer_kwargs,
                    )
                    del local_target
                    release_memory()
                    params = self.update_from_local_params(params, optimized_params, atoms_selected_mask)
                    if plot:
                        self._plot_progress(params, batch_indices, sel_params)  # type: ignore[attr-defined]
                    bar.set_postfix_str(
                        f"epoch {epoch + 1}/{num_epoch}  batch {b_idx + 1}/{n_batches}"
                    )
                    bar.update(1)

                if self.convergence(params, pre_params, tol):
                    log.info("Convergence criteria met.")
                    self.converged = True
                    # Skip the bar to completion so the user sees an
                    # explicit early-exit rather than a half-filled bar.
                    bar.update(total_steps - bar.n)
                    break
            bar.close()

        self.params = params
        self.prediction = to_numpy(self.predict(params, local=local))  # type: ignore[attr-defined]
        log.info("Stochastic fitting complete.")
        return self.params

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def convergence(self, params: dict, pre_params: dict, tol: float = 1e-2) -> bool:
        """Convergence test on a parameter snapshot.

        Position params (``pos_x``, ``pos_y``): max absolute update < 1 px.
        Other tensor params: mean relative update < ``tol``.
        Non-tensor entries are skipped.
        """
        for key, value in params.items():
            if key not in pre_params or not torch.is_tensor(value):
                continue
            other = pre_params[key]
            if torch.is_tensor(other) and other.device != value.device:
                other = other.to(value.device)
            update = torch.abs(value - other)
            if key in ("pos_x", "pos_y"):
                if update.max() > 1:
                    log.info("Convergence not reached on %s (max=%.4f)", key, float(update.max()))
                    return False
            else:
                rate = torch.abs(update / (value + 1e-10)).mean()
                if rate > tol:
                    log.info("Convergence not reached on %s (rate=%.4e)", key, float(rate))
                    return False
        log.info("Convergence reached")
        return True

    def select_params(self, params: dict, mask: np.ndarray) -> dict:
        """Slice per-atom params by ``mask``; copy shared params whole.

        Shared params (``background``, ``same_width``, ``atom_types``) are
        never indexed. Width/ratio are shared when ``self.same_width`` is
        True (one entry per atom-type) and per-atom otherwise.
        """
        out: dict[str, Any] = {"background": params["background"]}
        non_atom = {"background", "same_width", "atom_types"}
        if self.same_width:
            if "width" in params:
                out["width"] = params["width"]
            if "ratio" in params:
                out["ratio"] = params["ratio"]
            for key in ("pos_x", "pos_y", "height"):
                out[key] = params[key][mask]
        else:
            for key, value in params.items():
                if key in non_atom:
                    continue
                out[key] = value[mask]
        out["same_width"] = params["same_width"]
        out["atom_types"] = params["atom_types"][mask]
        return out

    # ------------------------------------------------------------------
    # Width-first warmup (StatSTEM's fitWidth.m equivalent)
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # Loss closure — used by Fitter.optimize via fit_loop's loss_fn arg.
    # Lives here (not in a separate "loss.py") because it's tightly
    # coupled to the optimisation step. Edge / boundary helpers used by
    # this loss live in qem.fit.edge.
    # ------------------------------------------------------------------

    def loss(self, y_true, y_pred, *, adaptive_edge_loss: bool | None = None) -> torch.Tensor:
        """RMSE between target and prediction, optionally edge-aware.

        * Multiplies residuals by ``self.window`` (typically a Butterworth
          window — set to ``np.ones_like(image)`` to disable dampening).
        * If ``self.adaptive_edge_loss`` (or the kwarg) is True, scales
          the loss by the inverse-sqrt of mean peak visibility so the
          optimiser still moves partially-visible peaks.
        * If ``self.boundary_strength > 0``, adds a smooth quadratic
          penalty for atoms drifting past the image edges.
        """
        if adaptive_edge_loss is None:
            adaptive_edge_loss = bool(getattr(self, "adaptive_edge_loss", False))

        diff = y_true - y_pred
        # Cache the window as a torch tensor on the active device.
        if self._window_t is None or self._window_t.device != diff.device:
            self._window_t = torch.as_tensor(
                self.window, dtype=torch.float32, device=diff.device,
            )
        diff = diff * self._window_t
        mse = torch.sqrt(torch.mean(torch.square(diff)))

        # Adaptive edge boost: amplify the gradient on partially-visible peaks.
        if adaptive_edge_loss:
            model = getattr(self, "_optimization_model", None) or self.model  # type: ignore[attr-defined]
            if model is not None:
                params = model.get_params()
                visibility = self.calculate_peak_visibility(  # type: ignore[attr-defined]
                    params["pos_x"], params["pos_y"], params["width"],
                )
                mse = mse * torch.mean(1.0 / torch.sqrt(visibility))

        # Soft boundary penalty: keep atoms from drifting outside the image.
        boundary_strength = float(getattr(self, "boundary_strength", 0.0))
        if boundary_strength > 0:
            model = getattr(self, "_optimization_model", None) or self.model  # type: ignore[attr-defined]
            if model is not None:
                params = model.get_params()
                penalty = self.calculate_boundary_penalty(  # type: ignore[attr-defined]
                    params["pos_x"], params["pos_y"], params["width"],
                    max_distance=3.0,
                )
                mse = mse + boundary_strength * penalty

        return mse

    def fit_width_first(
        self,
        *,
        sigma_lo: float | None = None,
        sigma_hi: float | None = None,
        xtol: float = 5e-3,
        ftol: float = 1e-4,
        max_evals: int = 10,
        verbose: bool = False,
        progress: bool = True,
    ) -> float:
        """Optimise shared width σ before any position fit.

        Mirrors StatSTEM's ``fitWidth.m``: positions held fixed, σ
        optimised by Brent's method on the residual sum-of-squares with
        (η, ζ) profiled out via :meth:`linear_estimator` at every
        evaluation. Updates ``params['width']`` and ``params['height']``
        in place. Returns the converged σ in pixel units.

        Args:
            sigma_lo, sigma_hi: search bracket in pixels. ``None`` ⇒
                ``[0.5, 2.0]·current_width`` (atom_size init usually
                lands within ±50% of optimum, so a tight bracket cuts
                evals without sacrificing robustness).
            xtol: tolerance on σ (relative).
            ftol: relative-improvement floor for early exit.
            max_evals: maximum residual evaluations (each calls
                ``linear_estimator``).
            verbose: log convergence trace.
        """
        width_param = self.params["width"]
        sigma0 = float(to_numpy(width_param).reshape(-1)[0])
        if sigma_lo is None:
            sigma_lo = max(0.5, sigma0 * 0.5)
        if sigma_hi is None:
            sigma_hi = max(sigma_lo + 0.5, sigma0 * 2.0)

        history: list[tuple[float, float]] = []
        bar = tqdm(
            total=max_evals, desc="width-first σ search", leave=False,
            disable=not progress,
        )

        def loss_at_sigma(sigma_val: float) -> float:
            if sigma_val <= 0.0:
                bar.update(1)
                return 1e30
            try:
                r = self._residual_sum_for_width(float(sigma_val))
            # Numerical/shape failures in the estimator + predict path;
            # torch surfaces these as RuntimeError, numpy as ValueError.
            except (RuntimeError, ValueError, IndexError, np.linalg.LinAlgError) as exc:
                log.warning(
                    "fit_width_first: failed at σ=%.3f (%s): %s",
                    sigma_val, type(exc).__name__, exc,
                )
                bar.update(1)
                return 1e30
            history.append((float(sigma_val), r))
            bar.set_postfix(σ=f"{sigma_val:.3f}", lsq=f"{r:.2e}")
            bar.update(1)
            return r

        # Brent's parabolic interpolation — pure Python, no scipy.
        # On a smooth near-quadratic residual the parabolic step snaps
        # to the optimum in 2-3 iterations; total ~5-7 evals vs ~15 for
        # plain golden section. Each eval is ~1 s (linear_estimator +
        # predict), so this is the practical speedup lever.
        sigma_opt, fun_opt, _ = _brent_minimize(
            loss_at_sigma, sigma_lo, sigma_hi,
            xtol=xtol, ftol=ftol, max_evals=max_evals,
        )
        bar.close()
        # Final commit: ensure params reflect the optimal σ + corresponding η, bg.
        self._residual_sum_for_width(sigma_opt)
        if verbose:
            log.info(
                "fit_width_first: σ %.3f → %.3f px (Δlsq=%.3e, %d evals)",
                sigma0, sigma_opt,
                (history[0][1] - fun_opt) if history else 0.0,
                len(history),
            )
        return sigma_opt

    def _residual_sum_for_width(self, sigma_px: float) -> float:
        """Set width = sigma_px (shared), refit (η, bg) by NNLS, return ||r||²."""
        width_param = self.params["width"]
        with torch.inference_mode():
            width_param.fill_(float(sigma_px))
        params = self.linear_estimator(self.params, best_effort=True)  # type: ignore[attr-defined]
        pred = self.predict(params, local=True)  # type: ignore[attr-defined]
        res = np.asarray(self.image, dtype=np.float64) - to_numpy(pred)  # type: ignore[attr-defined]
        return float(np.sum(res * res))

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def update_from_local_params(
        self, params: dict, local_params: dict, mask: np.ndarray,
    ) -> dict:
        """Merge per-batch optimised params back into the global dict.

        Per-atom params are scattered into ``mask`` rows. Shared params
        (background, optionally width/ratio) are blended toward the new
        value with weight = batch fraction.
        """
        shared = {"background"}
        if getattr(self, "same_width", True):
            shared.update({"width", "ratio"})
        const = {"same_width", "atom_types"}
        for key, value in local_params.items():
            if key in const:
                continue
            if key in shared:
                weight = mask.sum() / self.num_coordinates
                params[key] = params[key] * (1 - weight) + value * weight
            else:
                new_param = params[key].clone()
                update_indices = torch.as_tensor(
                    np.where(mask)[0], dtype=torch.int64, device=new_param.device,
                )
                value_tensor = torch.as_tensor(
                    value, dtype=new_param.dtype, device=new_param.device,
                )
                new_param.view(-1)[update_indices] = value_tensor
                params[key] = new_param
        return params


__all__ = ["FitterOptimizationMixin"]
