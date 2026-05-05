"""High-level fit pipeline — orchestrates the full workflow.

This module owns the *recipe*: peak refinement → init → width-first
warmup → stochastic warm fit → LM polish. The individual stages live
in their proper homes (peak detection in :mod:`qem.fit.peaks`,
σ-warmup and LM polish in :mod:`qem.fit.optimization`); this module
just composes them.

Two entry points:

* Free function :func:`fit_pipeline` — operates on an existing
  :class:`Fitter`. Useful when you've already built one and want to
  run the recommended workflow.

* Class method :meth:`Fitter.fit` — convenience constructor that
  builds a Fitter and runs the pipeline in one line.

Both accept the same kwargs. Each stage is gated by a flag so the
legacy stochastic-only path is still reachable
(``width_first=False, subpixel=False, lm_polish=False``).
"""

from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from qem.fit.fitter import Fitter

log = logging.getLogger(__name__)


def _residual_std(fitter: "Fitter") -> float:
    """L2 std of (image − prediction); ``nan`` if no prediction yet."""
    pred = getattr(fitter, "prediction", None)
    if pred is None or np.asarray(pred).size == 0:
        return float("nan")
    return float(np.std(np.asarray(fitter.image) - np.asarray(pred)))


class _Stage:
    """Context manager that times a pipeline stage and prints a one-line
    summary on exit.

    Format::

        ▸ width-first        σ:5.78→6.42 px   res 473.14 → 437.21   12.4 s
    """

    def __init__(self, fitter: "Fitter", name: str, *, verbose: bool):
        self.fitter = fitter
        self.name = name
        self.verbose = verbose
        self.t0 = 0.0
        self.res0 = float("nan")

    def __enter__(self) -> "_Stage":
        self.t0 = time.perf_counter()
        self.res0 = _residual_std(self.fitter)
        return self

    def __exit__(self, exc_type, exc_value, tb) -> None:
        if not self.verbose:
            return
        dt = time.perf_counter() - self.t0
        res1 = _residual_std(self.fitter)
        if np.isnan(self.res0) and np.isnan(res1):
            res_str = ""
        elif np.isnan(self.res0):
            res_str = f"  res {res1:.2f}"
        else:
            res_str = f"  res {self.res0:.2f} → {res1:.2f}"
        log.info("▸ %-20s%s   %5.1f s", self.name, res_str, dt)
        # Also print to stdout when no logging handler is attached so
        # users running in a notebook see the stage updates without
        # configuring logging.
        if not log.handlers and not logging.getLogger().handlers:
            print(f"▸ {self.name:<20s}{res_str}   {dt:5.1f} s")


def fit_pipeline(
    fitter: "Fitter",
    *,
    atom_size: float = 0.7,
    subpixel: bool = True,
    subpixel_window: int = 0,
    width_first: bool = True,
    num_epoch: int = 10,
    batch_size: int = 2000,
    stochastic_maxiter: int = 50,
    stochastic_step_size: float = 1e-2,
    stochastic_tol: float = 1e-3,
    stochastic_optimizer: str = "adam",
    stochastic_optimizer_kwargs: dict | None = None,
    lm_polish: bool = True,
    lm_maxiter: int = 20,
    lm_tol: float = 1e-8,
    lm_loss: str = "l2",
    verbose: bool = False,
) -> "Fitter":
    """End-to-end recommended fit recipe.

    Mirrors StatSTEM's ``fitGauss.m`` flow plus our LM polish. The
    default settings empirically match or beat StatSTEM across the
    benchmark suite.

    Pipeline stages (each gated by a flag):

    1. ``subpixel``  — :meth:`Fitter.refine_peaks_subpixel` (±0.05 px)
    2. ``init_params`` — always runs
    3. ``width_first`` — :meth:`Fitter.fit_width_first` (Brent on σ)
    4. stochastic warmup — :meth:`Fitter.fit_stochastic`
    5. ``lm_polish`` — :meth:`Fitter.fit_global` with optimizer="lm"

    Examples::

        fit_pipeline(f)                                       # defaults
        fit_pipeline(f, width_first=False)                    # legacy
        fit_pipeline(f, stochastic_optimizer="Ranger",
                    stochastic_optimizer_kwargs=dict(
                        betas=(0.95, 0.999), weight_decay=1e-4))
        fit_pipeline(f, lm_loss="huber")                      # robust polish

    Returns the same ``fitter`` for chaining.
    """
    show = bool(verbose) or True   # always print the per-stage banner —
    # it's two lines max per stage and gives users immediate feedback
    # in notebooks. Set verbose=False to also silence sub-stage logs.

    if subpixel:
        with _Stage(fitter, "subpixel refine", verbose=show):
            fitter.refine_peaks_subpixel(search_window=subpixel_window)

    with _Stage(fitter, "init params", verbose=show):
        fitter.init_params(atom_size=atom_size)

    if width_first:
        with _Stage(fitter, "width-first σ", verbose=show):
            fitter.fit_width_first(verbose=verbose)

    with _Stage(fitter, "stochastic Adam", verbose=show):
        fitter.fit_stochastic(
            num_epoch=num_epoch,
            batch_size=batch_size,
            maxiter=stochastic_maxiter,
            tol=stochastic_tol,
            step_size=stochastic_step_size,
            optimizer=stochastic_optimizer,
            plot=False,
            verbose=verbose,
            **(stochastic_optimizer_kwargs or {}),
        )

    if lm_polish:
        with _Stage(fitter, f"LM polish ({lm_loss})", verbose=show):
            fitter.fit_global(
                maxiter=lm_maxiter,
                tol=lm_tol,
                optimizer="lm",
                loss=lm_loss,
                verbose=verbose,
            )

    return fitter


class FitterPipelineMixin:
    """Adds :meth:`fit_pipeline` and the :meth:`fit` classmethod to Fitter.

    Both delegate to the free :func:`fit_pipeline` function. The mixin
    pattern (rather than the legacy ``_bind`` monkey-patch) makes the
    methods visible to type-checkers and to ``super()`` in subclasses.
    """

    def fit_pipeline(self, **kwargs: Any) -> "Fitter":
        """Run the recommended end-to-end fit recipe in place.

        See :func:`qem.fit.pipeline.fit_pipeline` for the full kwarg
        list. Returns ``self`` for chaining.
        """
        return fit_pipeline(self, **kwargs)  # type: ignore[arg-type]

    @classmethod
    def fit(
        cls,
        image,
        dx: float = 1.0,
        *,
        units: str = "A",
        elements: list[str] | None = None,
        model_type: str = "gaussian",
        same_width: bool = True,
        fit_background: bool = True,
        # Pipeline kwargs forwarded to fit_pipeline:
        **pipeline_kwargs: Any,
    ) -> "Fitter":
        """One-liner: build a Fitter and run the recommended fit pipeline.

        Equivalent to::

            f = Fitter(image, dx=dx, units=units, elements=elements,
                       model_type=model_type, same_width=same_width,
                       fit_background=fit_background)
            f.find_peaks()                # default peak detection
            fit_pipeline(f, **pipeline_kwargs)

        ``pipeline_kwargs`` are forwarded to
        :func:`qem.fit.pipeline.fit_pipeline` — see its docstring for
        the full list (``atom_size``, ``subpixel``, ``width_first``,
        ``stochastic_optimizer``, ``lm_loss``, etc.).

        Returns the fitted :class:`Fitter`.
        """
        fitter = cls(  # type: ignore[call-arg]
            image,
            dx=dx,
            units=units,
            elements=elements,
            model_type=model_type,
            same_width=same_width,
            fit_background=fit_background,
        )
        if getattr(fitter, "coordinates", None) is None or len(fitter.coordinates) == 0:
            # Run default peak detection if the caller didn't pre-seed
            # ``fitter.coordinates`` themselves.
            fitter.find_peaks()
        return fit_pipeline(fitter, **pipeline_kwargs)


__all__ = ["fit_pipeline", "FitterPipelineMixin"]
