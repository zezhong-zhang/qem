"""Explicit PyTorch training loops used by image-fitting.

The legacy code piggybacked on Keras's ``model.compile()`` /
``model.fit()`` plumbing.  We replace that with two small, explicit
loops that do exactly what we need: minimise an objective with a
torch.optim optimiser, optional early-stopping on the loss, and an
optional ReduceLROnPlateau-style schedule.  No callbacks framework, no
hidden state.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Callable

import torch
from torch import nn

LossFn = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]
log = logging.getLogger(__name__)


@dataclass
class FitResult:
    final_loss: float
    epochs_run: int
    converged: bool


def make_optimizer(
    name: str,
    parameters: Any,
    learning_rate: float,
    **kwargs: Any,
) -> torch.optim.Optimizer:
    """Build a torch optimiser by short name (``'adam' / 'adamw' / 'sgd' / 'lbfgs'``)."""
    cls = {
        "adam": torch.optim.Adam,
        "adamw": torch.optim.AdamW,
        "sgd": torch.optim.SGD,
        "lbfgs": torch.optim.LBFGS,
    }.get(name.lower())
    if cls is None:
        raise ValueError(
            f"Unknown optimizer {name!r}; expected one of "
            "'adam', 'adamw', 'sgd', 'lbfgs'."
        )
    return cls(parameters, lr=learning_rate, **kwargs)


def fit_loop(
    model: nn.Module,
    inputs: Any,
    target: torch.Tensor,
    loss_fn: LossFn,
    optimizer: torch.optim.Optimizer,
    epochs: int,
    *,
    tol: float = 1e-4,
    patience: int = 100,
    lr_patience: int = 10,
    lr_factor: float = 0.1,
    min_lr: float = 1e-6,
    snapshot_every: int = 50,
    post_step: Callable[[nn.Module], None] | None = None,
    verbose: bool = False,
) -> FitResult:
    """Adam-style training loop with early stopping and LR reduction.

    Mirrors the behaviour the old keras callbacks gave us (EarlyStopping
    with restore-best-weights + ReduceLROnPlateau), without their
    overhead.
    """
    is_lbfgs = isinstance(optimizer, torch.optim.LBFGS)
    if is_lbfgs:
        # L-BFGS reevaluates the loss multiple times per .step(); the closure
        # pattern is the only supported API. ReduceLROnPlateau is meaningful
        # for first-order optimisers but not for L-BFGS — it manages its own
        # line search.
        scheduler = None
    else:
        # Use relative-threshold mode so the scheduler doesn't crash the LR
        # when the loss already starts near the optimum.
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=lr_factor,
            patience=lr_patience,
            threshold=1e-2,
            threshold_mode="rel",
            min_lr=min_lr,
        )

    best_loss = float("inf")
    best_state: dict[str, torch.Tensor] | None = None
    epochs_no_improve = 0
    last_loss = float("inf")
    epochs_run = 0

    for epoch in range(epochs):
        if is_lbfgs:
            def closure():
                optimizer.zero_grad(set_to_none=True)
                pred = model(inputs)
                _loss = loss_fn(target, pred)
                _loss.backward()
                return _loss
            loss = optimizer.step(closure)
        else:
            optimizer.zero_grad(set_to_none=True)
            prediction = model(inputs)
            loss = loss_fn(target, prediction)
            loss.backward()
            optimizer.step()

        if post_step is not None:
            with torch.no_grad():
                post_step(model)

        loss_val = float(loss.detach())
        if scheduler is not None:
            scheduler.step(loss_val)
        last_loss = loss_val
        epochs_run = epoch + 1

        # Track best (relative-improvement criterion to mirror the LR scheduler).
        if loss_val < best_loss * (1.0 - 1e-3):
            best_loss = loss_val
            should_snapshot = (
                best_state is None
                or (epoch + 1) % snapshot_every == 0
                or (epoch + 1) == epochs
            )
            if should_snapshot:
                best_state = {
                    k: v.detach().clone()
                    for k, v in model.state_dict().items()
                }
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        if verbose and (epoch == 0 or (epoch + 1) % 100 == 0):
            log.info("Epoch %d/%d — loss %.6f", epoch + 1, epochs, loss_val)

        if epochs_no_improve >= patience:
            if verbose:
                log.info("Early stopping at epoch %d (best loss %.6f).",
                         epoch + 1, best_loss)
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    return FitResult(
        final_loss=best_loss if best_loss < float("inf") else last_loss,
        epochs_run=epochs_run,
        converged=epochs_no_improve >= patience,
    )


__all__ = ["fit_loop", "make_optimizer", "FitResult"]
