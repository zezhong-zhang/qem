Optimisation strategy
=====================

The recommended entry point — :meth:`Fitter.fit_pipeline`
---------------------------------------------------------

For typical usage, run the bundled pipeline. It mirrors StatSTEM's
``fitGauss.m`` flow plus a Levenberg–Marquardt polish, and matches
or beats StatSTEM on every benchmark sample we tested:

.. code-block:: python

   from qem.fit import Fitter

   fitter = Fitter(image, dx=dx)
   fitter.find_peaks()         # or set fitter.coordinates yourself
   fitter.fit_pipeline()        # subpixel + width-first + Adam + LM polish

Or in one line, with default peak detection included:

.. code-block:: python

   fitter = Fitter.fit(image, dx=dx)

Pipeline stages
^^^^^^^^^^^^^^^

The pipeline runs five stages, each gated by a flag:

1. **Sub-pixel peak refinement** — :meth:`Fitter.refine_peaks_subpixel`
   replaces the integer-pixel peak list with sub-pixel positions
   via 2-D parabolic fit on the 3×3 around each local maximum
   (±0.05 px accuracy).

2. **Parameter init** — :meth:`Fitter.init_params` builds the
   parameter dict (``pos_x``, ``pos_y``, ``height``, ``width``,
   ``background``).

3. **Width-first warmup** — :meth:`Fitter.fit_width_first`
   optimises σ alone (positions held fixed) by Brent's method on
   the residual sum-of-squares with η and ζ profiled out via
   :meth:`Fitter.linear_estimator`. This is the StatSTEM-equivalent
   ``fitWidth`` step. Empirically the most important stage on
   nanoparticle-on-substrate samples — without it, the joint fit
   converges to a local minimum 50 % worse on substrate-edge atoms.

4. **Stochastic Adam warmup** — :meth:`Fitter.fit_stochastic`
   refines all parameters jointly on random batches of atoms with
   neighbour-subtraction. Default optimiser is Adam; any
   ``pytorch_optimizer`` (kozistr) or ``torch_optimizer`` (jettify)
   class is accepted by name.

5. **LM polish** — :meth:`Fitter.fit_global` with
   ``optimizer="lm"``. Matrix-free Gauss–Newton with conjugate
   gradient as the inner linear solver. Same algorithm family as
   ``scipy.least_squares(method='trf')``; 5–10× faster than PyTorch
   LBFGS at strictly better residual.

Tuning
^^^^^^

Common knobs (see :func:`qem.fit.pipeline.fit_pipeline` for the full
list):

.. code-block:: python

   fitter.fit_pipeline(
       atom_size=0.7,                           # initial σ in Å
       width_first=True,                        # disable for legacy
       stochastic_optimizer="adam",             # or "Lion", "MADGRAD", …
       stochastic_optimizer_kwargs={"betas": (0.9, 0.999)},
       lm_loss="l2",                            # "huber" / "soft_l1" / "cauchy"
       lm_maxiter=30,
   )

Optimiser dispatch
------------------

:func:`qem.fit.loop.make_optimizer` resolves the ``optimizer`` argument
in this order:

1. **Built-ins**: ``"adam"``, ``"adamw"``, ``"sgd"``, ``"lbfgs"``.
2. **Levenberg–Marquardt**: ``"lm"`` or ``"gn"`` (matrix-free
   Gauss–Newton, see :mod:`qem.fit.levenberg_marquardt`).
3. **pytorch_optimizer** (kozistr): AdaBelief, Lion, MADGRAD, Adan,
   AdamP, DAdaptAdam, Yogi, Ranger, SophiaH, …
4. **torch_optimizer** (jettify): NovoGrad, PID, Apollo, AccSGD,
   QHAdam, SWATS, Lamb, …

Examples:

.. code-block:: python

   fitter.fit_global(optimizer="lm", loss="huber")
   fitter.fit_stochastic(optimizer="Lion", betas=(0.9, 0.99))
   fitter.fit_global(optimizer="MADGRAD", maxiter=200)

The matrix-free LM
------------------

For the polish step, :func:`qem.fit.levenberg_marquardt.fit_lm` solves

.. math::

   \min_\theta \tfrac{1}{2} \| r(\theta) \|^2

via Gauss–Newton with adaptive Marquardt damping. Each iteration
solves the trust-region subproblem

.. math::

   (J^\top J + \lambda I) \delta = - J^\top r

with conjugate gradient. Matrix-free: ``J·v`` and ``J^T·v`` are
computed on the fly via ``torch.func.jvp`` / ``vjp``, so the
Jacobian is never materialised (it would be 10⁶×10⁴ for typical
STEM data — infeasible).

Robust losses (``loss="huber" / "soft_l1" / "cauchy"``) follow the
IRLS reformulation: each LM iteration freezes the per-pixel weight
``w_i = ρ'(r_i) / r_i`` from the current residual, then takes a
Gauss–Newton step on the weighted least-squares problem.

The linear estimator
--------------------

:meth:`Fitter.linear_estimator` is the inner workhorse — solves
the sparse non-negative ridge problem

.. math::

   \min_x \| A x - b \|^2 + \lambda \| x \|^2 \quad
   \text{s.t.} \quad x \ge 0

where ``A`` has one column per atom (unit-amplitude Gaussian
window) plus an optional background column, and ``b`` is the
flattened image. Used in three places:

* As a pre-conditioner in :meth:`fit_stochastic` (best-effort, runs
  once at the start).
* Inside :meth:`fit_width_first` at every Brent evaluation.
* On its own — call it directly to refine heights without touching
  positions.

Two knobs:

``non_negative=True`` (default)
    Solved via :func:`qem.fit.sparse_torch.pg_nnls` — torch sparse
    CSR + projected gradient with Barzilai-Borwein step. ~5× faster
    than ``scipy.optimize.lsq_linear`` with bounds on the design
    matrices QEM builds.

``ridge=1e-4`` (default)
    Tikhonov regulariser. ``0`` disables. Set higher (``1e-3``,
    ``1e-2``) if the design matrix is ill-conditioned (atoms close
    enough to overlap windows).

Per-atom batched Levenberg–Marquardt
------------------------------------

For independent NLLS sub-problems (e.g. per-atom Voronoi-cell fits),
:func:`qem.fit.voronoi._batched_gaussian_lm` does an analytic
block-diagonal Jacobian batched LM:

.. code-block:: python

   px, py = _batched_gaussian_lm(
       crops, masks, x_grid, y_grid,
       px, py, h, w, bg,
       max_iter=15,
   )

Pure torch, no scipy/numpy in the inner loop. Runs on CUDA / MPS / CPU.

Library comparison:

================================ ========== ================== ===========================
Library                          Backend    Methods            Best for
================================ ========== ================== ===========================
``torch.optim``                  torch      Adam, AdamW, SGD,  Non-convex / streaming /
                                            LBFGS              huge parameter counts
``qem.fit.levenberg_marquardt``  torch      Matrix-free LM-CG  Joint NLLS polish
``qem.fit.voronoi``              torch      Batched LM         Many independent NLLS
                                                               (block-diagonal Hessian)
``qem.fit.per_atom_varpro``      torch      VarPro per-atom    StatSTEM-style
``scipy.optimize.lsq_linear``    numpy      Bounded LS         Used in legacy code paths
``pytorch_optimizer`` /          torch      Lion, MADGRAD,     Fast warmup; experimental
``torch_optimizer``                         AdaBelief, Adan,   step rules
                                            Ranger, NovoGrad …
================================ ========== ================== ===========================

Why we don't use generic batched optimisers for per-atom problems:

* ``torch.optim.LBFGS`` / ``Adam`` track a single quasi-Hessian
  history across all parameters. For block-diagonal problems this
  contaminates each atom's search direction with information from
  unrelated atoms.
* A handwritten batched LM with analytic Jacobian sidesteps both
  problems and gives a clean per-atom acceptance criterion.

Edge handling (boundary penalty, adaptive edge loss)
----------------------------------------------------

Set the attribute on the fitter to enable; defaults are off:

.. code-block:: python

   fitter.boundary_strength = 0.05    # add soft boundary penalty
   fitter.adaptive_edge_loss = True    # gradient boost on partial peaks
   fitter.fit_global()                 # picks them up automatically

Disable the Butterworth dampening window (sometimes helps edge atoms):

.. code-block:: python

   fitter.window = np.ones_like(fitter.image)

Device selection
----------------

QEM auto-picks the best torch device via
:func:`qem.utils.tensors.best_device`:

1. **CUDA** (NVIDIA + AMD via ROCm) — full speed and accuracy.
2. **MPS** (Apple Silicon) — ~3× faster than CPU on the StatSTEM Au
   benchmark, but ``scatter_add`` reduction has float32 precision
   issues on MPS that can cost 5–10 % on residual fits. If you need
   maximum accuracy on Apple Silicon, set ``QEM_DEVICE=cpu``.
3. **CPU** — slowest but always correct.

Override via environment variable:

.. code-block:: bash

   QEM_DEVICE=cpu python my_script.py
   QEM_DEVICE=mps python my_script.py
   QEM_DEVICE=cuda python my_script.py
