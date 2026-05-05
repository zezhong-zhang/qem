Basic analysis tutorial
=======================

Walks through the recommended QEM workflow on a STEM image:
detect peaks → fit → analyse → visualise. Uses
:meth:`qem.fit.Fitter.fit_pipeline` (or its one-line variant
:meth:`Fitter.fit`), which mirrors StatSTEM's ``fitGauss.m`` flow plus
a Levenberg–Marquardt polish.

Learning objectives
-------------------

By the end of this tutorial you will:

* Build a :class:`qem.fit.Fitter` from raw STEM data.
* Detect atomic columns and refine to sub-pixel accuracy.
* Run the bundled fit pipeline (or each stage individually).
* Extract scattering cross-sections and atom counts.
* Visualise model, residual, and SCS distributions.

Core components
---------------

* :class:`qem.fit.Fitter` — the analysis engine. Composed of capability
  mixins (see :doc:`/user_guide/optimization` for the optimisation
  layer in particular).
* Peak shape models in :mod:`qem.fit.model`: ``GaussianModel``,
  ``LorentzianModel``, ``VoigtModel``. Selected via the
  ``model_type=`` constructor argument; you rarely need to touch the
  classes directly.

Prerequisites
-------------

* QEM installed (``pip install -e .``).
* A STEM image (numpy array). The example below uses the bundled
  ``Example_Au.mat`` from a StatSTEM dataset.

Step 1 — load data
------------------

.. code-block:: python

   import numpy as np
   import matplotlib.pyplot as plt
   from qem.fit import Fitter
   from qem.io import read_statstem

   legacy = read_statstem("data/Au/Example_Au.mat")
   image = legacy["input"]["obs"]      # (H, W) float
   dx = legacy["input"]["dx"]          # Å per pixel

   plt.imshow(image, cmap="gray")
   plt.title("Raw STEM image"); plt.colorbar(); plt.show()

Step 2 — fit in one line
------------------------

.. code-block:: python

   fitter = Fitter.fit(image, dx=dx, atom_size=0.7, elements=["Au"])

That's the whole pipeline: default peak detection, sub-pixel
refinement, σ warmup, stochastic Adam, then a Gauss–Newton polish.
``fitter`` now holds:

* ``fitter.coordinates`` — refined atomic positions (px).
* ``fitter.params`` — fitted parameter dict (``pos_x``, ``pos_y``,
  ``height``, ``width``, ``background``).
* ``fitter.prediction`` — model rendering of the image.
* ``fitter.volume`` — analytical scattering cross-sections (Å²).

Step 3 (alternative) — explicit stages, when you want control
-------------------------------------------------------------

The one-liner is identical to running each stage by hand:

.. code-block:: python

   fitter = Fitter(image, dx=dx, model_type="gaussian", elements=["Au"])

   # 3a. Peak detection
   coords = fitter.find_peaks(
       min_distance=8,        # min separation in pixels
       threshold_abs=0.3,     # absolute intensity threshold
       threshold_rel=0.1,     # fraction of max intensity
   )
   print(f"Detected {len(coords)} atomic columns")

   plt.imshow(image, cmap="gray")
   plt.scatter(coords[:, 0], coords[:, 1], c="r", s=20, marker="+")
   plt.title("Detected peaks"); plt.show()

   # 3b. Sub-pixel refinement (parabolic fit on the 3×3 around each max)
   fitter.refine_peaks_subpixel(search_window=2)

   # 3c. Initialise parameters from current coordinates
   fitter.init_params(atom_size=0.7)   # σ in Å

   # 3d. Width-first warmup — Brent on σ with η, ζ profiled out
   sigma = fitter.fit_width_first()
   print(f"σ converged to {sigma:.3f} px")

   # 3e. Stochastic Adam warmup on random batches of atoms
   fitter.fit_stochastic(num_epoch=10, batch_size=2000, step_size=1e-2)

   # 3f. Levenberg–Marquardt polish
   fitter.fit_global(maxiter=30, optimizer="lm")

Either route ends up with the same ``fitter`` state.

Step 4 — analyse the fit
------------------------

.. code-block:: python

   # Fitted parameters
   pos_x = fitter.params["pos_x"].cpu().numpy()
   pos_y = fitter.params["pos_y"].cpu().numpy()
   heights = fitter.params["height"].cpu().numpy()
   width = float(fitter.params["width"].cpu().numpy()[0])

   print(f"N atoms: {len(pos_x)}")
   print(f"shared σ:  {width:.3f} px ({width * dx:.3f} Å)")
   print(f"Height range: {heights.min():.0f} – {heights.max():.0f}")

   # Per-atom scattering cross-section (Å²) — analytic Gaussian volume
   scs = fitter.volume          # 2π · h · σ² · dx²
   print(f"SCS range: {scs.min():.2f} – {scs.max():.2f} Å²")

Step 5 — goodness of fit
------------------------

.. code-block:: python

   from qem.benchmarks.benchmark import goodness_of_fit

   gof = goodness_of_fit(image, fitter.prediction)
   print(f"L2 std        = {gof['L2_std']:.2f}")
   print(f"L1 mean       = {gof['L1_mean']:.2f}")
   print(f"χ²_red        = {gof['chi2_red']:.2f}")
   print(f"PSD whiteness = {gof['psd_white_ratio']:.3f}")

For per-atom diagnostics:

.. code-block:: python

   from qem.benchmarks.benchmark import residual_per_atom, crlb_per_atom

   rpa = residual_per_atom(fitter)        # local χ² inside each atom's window
   crlb = crlb_per_atom(fitter)            # closed-form Cramér-Rao lower bound

Step 6 — visualise
------------------

.. code-block:: python

   fig, axes = plt.subplots(1, 3, figsize=(15, 5))
   axes[0].imshow(image, cmap="gray"); axes[0].set_title("Image")
   axes[1].imshow(fitter.prediction, cmap="gray"); axes[1].set_title("Fit")
   axes[2].imshow(image - fitter.prediction, cmap="RdBu_r")
   axes[2].set_title("Residual")
   plt.tight_layout(); plt.show()

Built-in plotters:

.. code-block:: python

   fitter.plot_fitting()           # 6-panel image / model / residual
   fitter.plot_scs()               # SCS scatter map
   fitter.plot_scs_histogram()     # SCS distribution
   fitter.plot_coordinates()       # peak positions overlaid

Atom counting (GMM)
-------------------

.. code-block:: python

   fitter.estimate_atom_counts_with_gmm(max_components=20)
   fitter.plot_atom_count_map()

Tweaking the pipeline
---------------------

The pipeline accepts kwargs to swap stages or change optimisers:

.. code-block:: python

   # Robust polish (Huber loss inside LM)
   fitter.fit_pipeline(lm_loss="huber")

   # Try a different first-order optimiser for the warmup
   fitter.fit_pipeline(stochastic_optimizer="Lion",
                       stochastic_optimizer_kwargs={"betas": (0.9, 0.99)})

   # Disable any stage:
   fitter.fit_pipeline(width_first=False)   # legacy / debugging
   fitter.fit_pipeline(subpixel=False)
   fitter.fit_pipeline(lm_polish=False)

See :func:`qem.fit.pipeline.fit_pipeline` for the full kwarg list.

Different peak shapes
---------------------

.. code-block:: python

   for model_type in ("gaussian", "lorentzian", "voigt"):
       f = Fitter.fit(image, dx=dx, model_type=model_type)
       res_std = (image - f.prediction).std()
       print(f"{model_type:10s}  residual std = {res_std:.2f}")

Common issues
-------------

**Peak detection misses atoms or finds spurious ones**
    Adjust ``min_distance``, ``threshold_abs``, ``threshold_rel`` on
    :meth:`Fitter.find_peaks`. For very noisy images, smooth first
    with :func:`qem.processing.signal.butterworth_window`.

**Fit converges to a wrong local minimum (high residual at edges)**
    Make sure ``width_first=True`` (default). Without it, σ is wrong
    during the joint fit and edge atoms commit to wrong basins.

**``fit_pipeline`` raises a Brent-bracket error**
    Older versions used ``scipy.optimize.minimize_scalar`` whose
    Brent method needed a strict ``f(mid) < f(both endpoints)``
    bracket condition; current code uses a torch-pipeline-native
    golden-section search that has no such requirement. Update QEM.

**Memory issues on large images**
    Reduce ``batch_size`` in ``fit_stochastic``, or run on CPU
    (``QEM_DEVICE=cpu python script.py``) — MPS in particular has
    larger transient allocations than CUDA / CPU.

Next steps
----------

* :doc:`/user_guide/optimization` — pick the right optimiser /
  loss for your data.
* :doc:`/user_guide/Analysis` — Voronoi integration, GMM atom
  counting, strain mapping.
* :doc:`/api/index` — full API reference.
