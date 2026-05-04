Visualization stack
===================

Two surfaces, two stacks:

* **Jupyter notebook / scripts → plotly** — non-blocking HTML output.
  Use for exploratory plots, embedded figures in reports, and any
  pipeline where ``matplotlib.pyplot.show()`` would freeze on a Tk/Qt
  event loop. See :mod:`qem.viz.interactive`.

* **Desktop GUI → napari** — full microscopy-style viewer with image
  layers, point overlays per element, voronoi cell labels, and
  workflow-aligned dock widgets that drive the Fitter directly. See
  :mod:`qem.viz.napari_app`.

The matplotlib helpers in :mod:`qem.fit.plot` remain for static
publication figures (PDF/SVG vector output, mature LaTeX support).

Notebook: ``qem.viz.interactive``
---------------------------------

Three plotly drop-ins for the most-used static figures, all bound onto
``Fitter`` so the OO style works:

.. code-block:: python

    fitter.plot_fitting_interactive()           # image / model / residual
    fitter.plot_coordinates_interactive()       # atoms on image
    fitter.plot_scs_histogram_interactive()     # SCS per element

Each returns a Plotly ``Figure``. In Jupyter it auto-displays; from a
script call ``fig.write_html("out.html")`` for a self-contained file.

Desktop GUI: ``qem-app``
------------------------

The desktop app is a real napari viewer with a workflow-aligned right
sidebar:

::

    +----------------------------------+----------------------+
    | napari image canvas              |  📂 Data             |
    |   - Image (STEM scan)            |  🎯 Peaks            |
    |   - Model           (toggle)     |  ⚙️  Fit              |
    |   - Residual        (toggle)     |  📊 Voronoi          |
    |   - Atoms (Points, per element)  |  🔬 Analysis (GMM)   |
    |   - Voronoi cells (Labels, opt)  |                      |
    +----------------------------------+----------------------+

Each dock corresponds to one Fitter step:

* **📂 Data** — set ``dx`` / units, save fit (HDF5), export params (NPZ).
* **🎯 Peaks** — :meth:`Fitter.find_peaks` with min-distance, threshold,
  smoothing; ``Refine via CoM`` button calls
  :meth:`Fitter.refine_center_of_mass`.
* **⚙️ Fit** — pick ``fit_global`` / ``fit_stochastic`` and an optimizer
  (``adam`` / ``adamw`` / ``sgd`` / ``lbfgs``). Long ops run in
  ``napari.qt.thread_worker`` so the UI stays responsive. The model
  and residual layers auto-toggle visible after the fit completes.
* **📊 Voronoi** — :meth:`Fitter.fit_voronoi` (with optional
  Levenberg-Marquardt refine), then :meth:`Fitter.voronoi_integration`.
  ``Colour atoms by`` lets you switch the Points layer's face_color
  between element / SCS / height in one click.
* **🔬 Analysis** — :meth:`Fitter.estimate_atom_counts_with_gmm` and
  the matplotlib atom-count map.

Run from the CLI:

.. code-block:: bash

    pip install qem[gui]              # napari[pyqt5] + magicgui + plotly
    qem-app                           # empty viewer (drag-drop image)
    qem-app path/to/image.tif         # opens .tif / .tiff
    qem-app path/to/image.npy         # opens raw NumPy
    qem-app path/to/example.mat       # legacy StatSTEM .mat

From Python:

.. code-block:: python

    import qem
    from qem.fit.fitter import Fitter

    image = qem.io.read_statstem("data/Au/Example_Au.mat")["input"]["obs"]
    fitter = Fitter(image, dx=0.1, units="A", elements=["Au"])
    fitter.show_in_napari()           # returns the napari.Viewer

From a Jupyter notebook (Qt event loop runs alongside the kernel):

.. code-block:: python

    %gui qt
    fitter.show_in_napari()

Closing the viewer does NOT close the underlying Fitter — it remains in
memory with its updated state. Re-open with ``fitter.show_in_napari()``.

When to use what
----------------

================================  ============================  =========================
You want…                         Library                       Why
================================  ============================  =========================
Publication-quality PDF/PNG       matplotlib (qem.fit.plot)     Vector output, LaTeX
Notebook plot, embed in report    **plotly** (qem.viz.interactive)
                                                                HTML, hover, non-blocking
Full desktop app                  **napari** (qem.viz.napari_app)
                                                                Image+layer model fits
                                                                STEM workflows
>50k-atom scatter                 HoloViews + Datashader        Rasterises huge clouds
3-D crystal lattice               PyVista                       VTK, million-atom interactive
================================  ============================  =========================

Optional dependencies
---------------------

The interactive layer ships in the ``gui`` extra:

.. code-block:: bash

    pip install qem[gui]

This pulls ``napari[pyqt5]`` (desktop), ``magicgui`` (dock widgets),
``plotly`` (notebook), and ``imageio`` (image loaders).

For the larger-scale or 3-D options:

.. code-block:: bash

    pip install holoviews datashader bokeh   # >50k atom point clouds
    pip install pyvista                       # 3-D crystal viewer
