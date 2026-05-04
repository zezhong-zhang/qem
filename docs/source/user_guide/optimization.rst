Optimization strategy
=====================

QEM uses three layers of optimization. This page describes what each
layer does, when each is used, and what to swap in if the defaults
don't match your problem.

The three layers
----------------

1. **Background estimation** — robust statistical (median / Sextractor),
   not gradient-based. See :class:`qem.fit.background.Background`.

2. **Linear height estimation** — a sparse least-squares pass that
   refines per-atom heights given current positions/widths. Stable
   convex sub-problem, runs once per ``fit_global`` and once per
   ``fit_stochastic`` (as a pre-conditioner).
   See :func:`qem.fit.fitter.Fitter.linear_estimator`.

3. **Joint Adam refinement** — stochastic-gradient descent on positions,
   widths, heights, background. Non-convex, this is where the time
   goes. See :func:`qem.fit.fitter.Fitter.fit_global` and
   :func:`qem.fit.fitter.Fitter.fit_stochastic`.

Linear estimator: stability knobs
---------------------------------

The linear estimator (``Fitter.linear_estimator``) solves a sparse
ridge-regularized least-squares problem:

.. math::

   \\min_x \\|A x - b\\|^2 + \\lambda \\|x\\|^2 \\quad \\text{s.t.} \\quad x \\geq 0

Two knobs control stability:

``non_negative=True`` (default)
    Enforce per-atom non-negativity via real bounded LS
    (``scipy.optimize.lsq_linear`` with bounds, TRF method).
    The historic ``lsqr → max(0, x)`` post-hoc projection caused
    fit_stochastic preconditioning to bounce; the bounded solver
    converges cleanly.

``ridge=1e-4`` (default)
    Tikhonov regularization. ``0`` disables. Set higher (``1e-3``,
    ``1e-2``) if the design matrix is ill-conditioned (atoms close
    enough to overlap windows). Higher ridge biases scales toward 1.

Adam refinement: choosing an optimizer
--------------------------------------

``Fitter.optimize`` (called by ``fit_global`` / ``fit_stochastic``)
takes an ``optimizer`` argument that maps to one of:

- ``"adam"`` (default) — Adam with ReduceLROnPlateau.
  Good for non-convex landscapes; tolerant of bad initialization.
- ``"adamw"`` — Adam with decoupled weight decay.
  Same convergence; mostly useful when you've added explicit L2.
- ``"sgd"`` — plain SGD. Use for projected-gradient patterns where
  you care about exact step direction (see :mod:`qem.fusion`).
- ``"lbfgs"`` — quasi-Newton with line search. Converges in fewer
  steps on smooth, well-initialized problems. Good after a coarse
  Adam run; use ``maxiter=20-50`` instead of hundreds.

Beyond ``torch.optim``: pytorch-minimize
----------------------------------------

For sub-problems that benefit from richer optimizers (BFGS variants
with bounds, Newton-CG, trust-region), install
`pytorch-minimize <https://github.com/rfeinman/pytorch-minimize>`_:

.. code-block:: bash

   pip install pytorch-minimize

It exposes a ``scipy.optimize``-compatible API on torch tensors with
autograd:

.. code-block:: python

   from torchmin import minimize

   def closure(x):
       return loss_fn(model_with_params(x))

   result = minimize(closure, x0, method="l-bfgs", max_iter=200)

When to reach for it:

- Heights-only refinement after positions converge — convex,
  bounded, low-dim → L-BFGS-B in seconds.
- Voronoi cell sub-fits — replace the per-cell ``scipy.optimize.curve_fit``
  with batched torch + L-BFGS for big speedups on GPU/MPS.

For comparison:

================================ ========== ================== ===========================
Library                          Backend    Methods            Best for
================================ ========== ================== ===========================
``torch.optim``                  torch      Adam, AdamW, SGD,  Non-convex / streaming /
                                            LBFGS              huge parameter counts
``pytorch-minimize`` (torchmin)  torch      BFGS, L-BFGS,      Smooth + small/medium dim
                                            L-BFGS-B,
                                            Newton-CG,
                                            trust-region
``scipy.optimize.lsq_linear``    numpy      Bounded LS         Used by ``linear_estimator``
``scipy.optimize.nnls``          numpy      Active-set NNLS    Small dense NNLS
``cvxpylayers``                  torch      Convex (full)      Differentiable convex layers
================================ ========== ================== ===========================

Device selection
----------------

QEM auto-picks the best torch device via
:func:`qem.utils.tensors.best_device`:

1. CUDA (NVIDIA + AMD via ROCm) — full speed and accuracy.
2. MPS (Apple Silicon) — ~3× faster than CPU on the StatSTEM Au
   benchmark, but ``scatter_add`` reduction has float32 precision
   issues on MPS that can cost 5–10% on residual fits. If you need
   maximum accuracy and are on Apple Silicon, set
   ``QEM_DEVICE=cpu``.
3. CPU — slowest but always correct.

Override:

.. code-block:: bash

   QEM_DEVICE=cpu python my_script.py
   QEM_DEVICE=mps python my_script.py
   QEM_DEVICE=cuda python my_script.py
