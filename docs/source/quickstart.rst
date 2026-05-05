Quick Start Guide
=================

Get a STEM image fitted in five lines.

Core idea
---------

QEM's analysis revolves around one class:

* :class:`qem.fit.Fitter` — peak detection, optimisation, analysis,
  visualisation. All capabilities are mixed in via composition; the
  call site stays a simple ``fitter.method(...)``.

The recommended end-to-end recipe is :meth:`Fitter.fit_pipeline`
(StatSTEM-equivalent + LM polish), or the one-liner
:meth:`Fitter.fit` classmethod that builds a Fitter and runs the
pipeline in one shot.

The five-line recipe
--------------------

.. code-block:: python

   import numpy as np
   from qem.fit import Fitter

   image = np.load("hadf.npy")            # 2-D STEM image, float
   fitter = Fitter.fit(image, dx=0.121)   # ← detection + fit
   prediction = fitter.prediction          # fitted image
   residual = image - prediction           # what's left
   scs = fitter.volume                     # scattering cross-sections (Å²)

That's it. ``Fitter.fit`` runs:

1. Default peak detection (:meth:`Fitter.find_peaks`)
2. Sub-pixel parabolic refinement (:meth:`Fitter.refine_peaks_subpixel`)
3. Parameter init (:meth:`Fitter.init_params`)
4. Width-first warmup (:meth:`Fitter.fit_width_first`) — Brent's
   method on σ with η/bg profiled out via the linear estimator
5. Stochastic Adam warmup (:meth:`Fitter.fit_stochastic`)
6. Levenberg–Marquardt polish (:meth:`Fitter.fit_global`
   ``optimizer="lm"``)

Each stage is gated by a flag — see :doc:`user_guide/optimization` for
the full kwarg list.

Step-by-step (when you need control)
------------------------------------

If you want to inspect the state between stages, build the
:class:`Fitter` yourself and call the steps individually:

.. code-block:: python

   import numpy as np
   from qem.fit import Fitter

   image = np.load("hadf.npy")
   fitter = Fitter(image, dx=0.121, model_type="gaussian")

   # 1. Peak detection
   coords = fitter.find_peaks(min_distance=10, threshold_abs=0.3)
   print(f"Found {len(coords)} atomic columns")

   # 2. Sub-pixel refinement
   fitter.refine_peaks_subpixel(search_window=2)

   # 3. Initialise parameters from current coordinates
   fitter.init_params(atom_size=0.7)   # σ in Å

   # 4. Width-first warmup (StatSTEM's fitWidth)
   sigma = fitter.fit_width_first()
   print(f"σ converged to {sigma:.3f} px")

   # 5. Stochastic Adam warmup
   fitter.fit_stochastic(num_epoch=10, batch_size=2000, step_size=1e-2)

   # 6. LM polish
   fitter.fit_global(maxiter=30, optimizer="lm")

Either route ends with the same ``fitter`` state (parameters in
``fitter.params``, prediction in ``fitter.prediction``).

Choosing an optimiser
---------------------

The polish step accepts any name :func:`qem.fit.loop.make_optimizer`
resolves — built-ins, ``pytorch_optimizer`` (kozistr), or
``torch_optimizer`` (jettify):

.. code-block:: python

   # Robust loss (Huber) inside our matrix-free LM:
   fitter.fit_pipeline(lm_loss="huber")

   # Swap Adam for Lion in the warmup phase:
   fitter.fit_pipeline(stochastic_optimizer="Lion",
                       stochastic_optimizer_kwargs={"betas": (0.9, 0.99)})

   # Skip width-first (legacy behaviour):
   fitter.fit_pipeline(width_first=False)

Visualise
---------

.. code-block:: python

   import matplotlib.pyplot as plt

   fig, axes = plt.subplots(1, 3, figsize=(15, 5))
   axes[0].imshow(image, cmap="gray"); axes[0].set_title("Image")
   axes[1].imshow(fitter.prediction, cmap="gray"); axes[1].set_title("Fit")
   axes[2].imshow(image - fitter.prediction, cmap="RdBu_r")
   axes[2].set_title("Residual")
   plt.show()

   # Or use the bundled scatter / histogram plotters:
   fitter.plot_scs()              # scattering cross-section map
   fitter.plot_scs_histogram()    # SCS distribution

Atom counting
-------------

.. code-block:: python

   fitter.estimate_atom_counts_with_gmm(max_components=20)
   fitter.plot_atom_count_map()

Loading real data
-----------------

.. code-block:: python

   import hyperspy.api as hs

   signal = hs.load("your_stem_data.dm3")
   image = signal.data
   dx = signal.axes_manager[0].scale  # Å / px

   fitter = Fitter.fit(image, dx=dx)

Common parameters
-----------------

**Constructor** (:class:`Fitter` ``__init__``):

* ``dx``: pixel size in ``units`` (default ``"A"``).
* ``model_type``: ``"gaussian"`` / ``"lorentzian"`` / ``"voigt"``.
* ``same_width``: share σ across atoms of the same type (default
  ``True``).
* ``fit_background``: include a background term in the fit
  (default ``True``).

**Pipeline** (:meth:`Fitter.fit_pipeline`):

* ``atom_size``: initial σ in Å (default ``0.7``).
* ``subpixel`` / ``width_first`` / ``lm_polish``: enable each stage
  (all default ``True``).
* ``stochastic_optimizer``: any name accepted by
  :func:`qem.fit.loop.make_optimizer`. Default ``"adam"``.
* ``lm_loss``: ``"l2"`` (default) / ``"huber"`` / ``"soft_l1"`` /
  ``"cauchy"`` for robust polish.

Next steps
----------

* :doc:`user_guide/optimization` — fitting recipes in depth
* :doc:`user_guide/Analysis` — voronoi, GMM, strain
* :doc:`tutorials/index` — full worked examples
* :doc:`api/index` — complete API reference
