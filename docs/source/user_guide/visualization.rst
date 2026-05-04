Visualization stack
===================

QEM ships matplotlib helpers in :mod:`qem.fit.plot` (publication-quality
static figures, the default). For interactive or large-scale work the
modern Python viz ecosystem has better-suited tools — this page
compares them and maps QEM's existing plots to the right one.

Pain point: matplotlib's ``plt.show()`` and ``plt.show(block=False) +
plt.pause(...)`` patterns block on Tk/Qt event loops. That's fine for
end-user inspection but kills automated testing, headless scripts, and
notebook flow. Switching to HTML-based plots fixes this.

When to use what
----------------

==============================  ============================  ============================
You want…                       Library                       Why
==============================  ============================  ============================
Publication-quality PDF/PNG     matplotlib                    Mature, vector output, LaTeX
Interactive notebook plots      **plotly**                    HTML output, hover, zoom,
                                                              non-blocking, ships with QEM
                                                              GUI extra
3-D image + label overlays      **napari**                    Built for microscopy:
                                                              image stacks + point/shape
                                                              layers + Qt UI for free
>50k-atom scatter / heatmaps    **HoloViews + Datashader**    Rasterises huge point clouds
                                                              to images in milliseconds
Real-time (live fitting)        **PyQtGraph** or **Vispy**    OpenGL, designed for
                                                              streaming updates
Web dashboard                   **Plotly Dash** or            Plotly already in stack;
                                **Streamlit** (current)       Streamlit is simpler
==============================  ============================  ============================

QEM-specific recommendations
----------------------------

**Image + model + residual triptych** (``plot_fitting``)
  Today: matplotlib ``imshow`` × 3.
  Better: :func:`qem.viz.interactive.plot_fitting_interactive` —
  plotly heatmaps with linked pan/zoom across the three panels and
  intensity hover. Returns a Figure; calls ``fig.show()`` in Jupyter,
  writes ``fig.write_html("…")`` from scripts.

**Atom positions on the image** (``plot_coordinates``)
  Today: matplotlib ``imshow`` + per-element ``scatter``.
  Better: :func:`qem.viz.interactive.plot_coordinates_interactive` —
  plotly ``Scattergl`` (WebGL-backed) handles ~50k points smoothly.
  For >50k atoms, switch to HoloViews+Datashader (raster pipeline,
  ms-scale render time):

  .. code-block:: python

      import holoviews as hv
      from holoviews.operation.datashader import datashade
      hv.extension("bokeh")
      points = hv.Points((coords[:, 0], coords[:, 1]))
      datashade(points, cmap="viridis", width=800, height=800)

**SCS histogram** (``plot_scs_histogram``)
  Today: matplotlib ``hist`` per element.
  Better: :func:`qem.viz.interactive.plot_scs_histogram_interactive` —
  plotly with hover bin-edges and click-to-toggle elements. Same
  shape as the matplotlib version.

**Atom-count map** (``plot_atom_count_map``)
  Today: matplotlib ``scatter`` over ``imshow``.
  Better: napari for exploration. The image goes in as an
  ``Image`` layer; the per-atom counts go in as a ``Points`` layer
  with the count as ``properties``. napari's per-point coloring
  handles 50k+ atoms and gives box-select / threshold-by-count for
  free.

  .. code-block:: python

      import napari
      v = napari.Viewer()
      v.add_image(fitter.image)
      v.add_points(
          fitter.coordinates,
          properties={"count": atom_counts},
          face_color="count", face_colormap="viridis",
          size=3,
      )

**3-D crystal lattice** (``view_3d``)
  Today: ASE viewer (matplotlib + Tk).
  Better: PyVista ``Plotter`` or ``ipyvtklink`` for in-notebook 3-D.
  PyVista handles million-atom structures interactively via VTK.

**Region selection** (``select_atoms``, ``select_region``)
  Today: matplotlib widgets via ``InteractivePlot``.
  Better: napari's Polygon / Lasso layer tools — built for exactly
  this and integrate with the image stack viewer.

**GUI dashboard** (``qem-app``, currently Streamlit)
  Streamlit + plotly is the right stack today. Plotly is non-blocking
  by construction and Streamlit re-runs the script on widget change.
  Migrate any matplotlib figures inside the app to the
  ``qem.viz.interactive`` plotly versions to drop the Streamlit
  ``st.pyplot`` overhead and get hover/zoom for free.

Migration approach
------------------

The matplotlib helpers (``qem.fit.plot``) are not deprecated. They
remain the right tool for static publication figures. The
``qem.viz.interactive`` module is purely additive:

.. code-block:: python

    fitter.plot_fitting()                    # matplotlib (PDF target)
    fig = fitter.plot_fitting_interactive()  # plotly (notebook target)

Optional dependencies
---------------------

The interactive layer ships in the ``gui`` extra:

.. code-block:: bash

    pip install qem[gui]   # plotly + Streamlit

For the big-data and 3-D options you'd add yourself:

.. code-block:: bash

    pip install holoviews datashader bokeh   # very large point clouds
    pip install napari                       # image + layer exploration
    pip install pyvista                      # 3-D crystal viewer
