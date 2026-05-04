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

Batched optimizers for many independent sub-problems
----------------------------------------------------

When you have many small, *independent* nonlinear least-squares
sub-problems (one per atom, per cell, per region), the right
algorithm is **batched Levenberg-Marquardt** with an analytic
block-diagonal Jacobian. Generic batched torch optimizers
(``torch.optim.LBFGS``, ``Adam``) fail here because they treat all
parameters as coupled — and library wrappers like
``pytorch-minimize`` are even worse because the bounded LBFGS-B
active-set machinery is O(n²) in the parameter count.

QEM ships its own batched LM in :func:`qem.fit.voronoi._batched_gaussian_lm`
for the per-atom 2-D Gaussian fit. Pattern:

.. code-block:: python

   # crops, masks: (N, k, k) — one cell per atom
   # px, py, h, w, bg: (N,) initial parameters
   px, py = _batched_gaussian_lm(
       crops, masks, x_grid, y_grid,
       px, py, h, w, bg,
       max_iter=15,
   )

Internally:

- Forward + analytic Jacobian per pixel: ``(N, k, k, P)``
- Normal equations per atom: ``(JᵀJ + λI) δ = Jᵀ r`` —
  ``torch.linalg.solve`` on ``(N, P, P)`` systems
- Adaptive per-atom LM damping: accept step → halve λ; reject → ×4 λ

Pure torch, no scipy/numpy in the inner loop. Runs on CUDA / MPS / CPU.

For comparison:

================================ ========== ================== ===========================
Library                          Backend    Methods            Best for
================================ ========== ================== ===========================
``torch.optim``                  torch      Adam, AdamW, SGD,  Non-convex / streaming /
                                            LBFGS              huge parameter counts
``qem.fit.voronoi``              torch      batched LM         Many independent NLLS
                                                               (block-diagonal Hessian)
``scipy.optimize.lsq_linear``    numpy      Bounded LS         Used by ``linear_estimator``
``scipy.optimize.nnls``          numpy      Active-set NNLS    Small dense NNLS
``scipy.optimize.curve_fit``     numpy      LM (per-call)      Reference per-atom path in
                                                               ``fit_voronoi(batched=False)``
================================ ========== ================== ===========================

Why we don't use generic batched optimizers for per-atom problems:

- ``torch.optim.LBFGS`` / ``Adam`` track a single quasi-Hessian
  history across all parameters. For block-diagonal problems this
  contaminates each atom's search direction with information from
  unrelated atoms.
- ``pytorch-minimize`` (BFGS, L-BFGS-B) wraps scipy's solvers; on
  a 14k-parameter bounded problem we measured a 140× slowdown vs
  the legacy per-cell scipy path.
- A handwritten batched LM with analytic Jacobian sidesteps both
  problems — and gives a clean per-atom acceptance criterion.

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
