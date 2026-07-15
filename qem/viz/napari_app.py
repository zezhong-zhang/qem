"""napari-based desktop GUI for QEM.

Built on `napari <https://napari.org>`_ — a multi-dimensional image
viewer designed for microscopy. Same layer model QEM needs:
``Image`` for the STEM scan, ``Image`` overlay for the model and
residual, ``Points`` per element for atomic columns, ``Labels`` for
voronoi / region maps. All non-blocking — napari's event loop runs
in the background and re-renders layers as Fitter state changes.

Design (workflow-aligned dock order)
------------------------------------

::

    +----------------------------------+----------------------+
    | napari image canvas (4 layers)   |  📂 Data             |
    |   - Image (STEM scan)            |  🎯 Peaks            |
    |   - Model         (toggle)       |  ⚙️  Fit              |
    |   - Residual      (toggle)       |  📊 Voronoi          |
    |   - Atoms (Points, per element)  |  🔬 Analysis (GMM)   |
    |   - Voronoi cells (Labels, opt)  |                      |
    |                                  +----------------------+
    |                                  |  Status / loss curve |
    +----------------------------------+----------------------+

* The right docks step the user through the workflow top-to-bottom.
* The bottom dock shows the active Fitter status: device,
  num_coordinates, last residual, and the loss curve from the most
  recent ``optimize()`` call (rendered with plotly into a QWebView).
* Long ops run in ``napari.qt.thread_worker`` so the UI stays
  responsive during multi-second fits.

Entry points
------------

From a script::

    >>> import qem
    >>> from qem.fit.fitter import Fitter
    >>> fitter = Fitter(image, dx=0.1, units="A")
    >>> fitter.show_in_napari()    # opens viewer, returns Viewer instance

From a Jupyter notebook (Qt event loop runs alongside the kernel)::

    >>> %gui qt
    >>> fitter.show_in_napari()

From the CLI::

    $ qem-app                          # empty viewer
    $ qem-app path/to/image.tif        # opens with image loaded
    $ qem-app path/to/fit.h5           # opens saved fit

Closing the viewer does NOT close the underlying Fitter — it remains
in memory with its updated state. Re-open with ``fitter.show_in_napari()``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import napari  # noqa: F401

    from qem.fit.fitter import Fitter  # noqa: F401


# Layer names used across napari_app + widgets (single source of truth).
LAYER_IMAGE = "Image"
LAYER_MODEL = "Model"
LAYER_RESIDUAL = "Residual"
LAYER_ATOMS = "Atoms"
LAYER_VORONOI = "Voronoi cells"


# --------------------------------------------------------------------------
# Layer construction helpers
# --------------------------------------------------------------------------

def _as_np(x):
    if hasattr(x, "detach"):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def _color_palette(n: int) -> np.ndarray:
    """``n`` distinct RGBA colours from matplotlib's qualitative palette."""
    import matplotlib

    cmap = matplotlib.colormaps["tab10"]
    return np.array([cmap(i % 10) for i in range(n)])


def _atoms_to_points(fitter: Fitter) -> tuple[np.ndarray, dict, np.ndarray]:
    """Build ``(coords_yx, properties, face_color)`` for the Points layer.

    napari Points uses (row, col) = (y, x). ``properties`` carries
    per-atom values for hover and colour-by-property; ``face_color``
    is set per-atom by element type (default).
    """
    coords = fitter.coordinates
    if coords is None or len(coords) == 0:
        return np.zeros((0, 2)), {}, np.zeros((0, 4))
    yx = np.column_stack([coords[:, 1], coords[:, 0]])

    elements = fitter.elements or []
    atom_types = fitter.atom_types
    if atom_types is None or len(atom_types) == 0:
        atom_types = np.zeros(len(coords), dtype=int)

    element_names = np.array([
        elements[int(t)] if int(t) < len(elements) else f"type{int(t)}"
        for t in atom_types
    ])
    props: dict = {
        "element": element_names,
        "atom_type": atom_types.astype(int),
    }
    if fitter.params is not None and "height" in fitter.params:
        props["height"] = _as_np(fitter.params["height"])
    voronoi_volume = getattr(fitter, "_voronoi_volume", None)
    if voronoi_volume is not None:
        props["scs"] = _as_np(voronoi_volume)

    palette = _color_palette(int(np.max(atom_types)) + 1)
    face_color = palette[atom_types.astype(int) % len(palette)]
    return yx, props, face_color


def _model_image(fitter: Fitter) -> np.ndarray:
    """Render the current model on the full grid."""
    pred = fitter.prediction
    if pred is None or (hasattr(pred, "size") and pred.size == 0):
        return np.zeros_like(fitter.image)
    return _as_np(pred)


# --------------------------------------------------------------------------
# Viewer construction
# --------------------------------------------------------------------------

def open_in_napari(
    fitter: Fitter,
    *,
    show_model: bool = True,
    show_atoms: bool = True,
    title: str = "QEM",
):
    """Open ``fitter`` in a napari Viewer with QEM-aware widgets.

    Returns the napari ``Viewer`` instance. Caller is responsible for
    keeping the reference alive (assign to a variable in Jupyter; the
    CLI :func:`run_app` blocks on ``napari.run()``).
    """
    try:
        import napari
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "napari not installed — `pip install qem[gui]` (or "
            "`pip install napari[pyqt5] magicgui`)."
        ) from exc

    viewer = napari.Viewer(title=f"{title} — {fitter.image.shape}")
    _add_image_layers(viewer, fitter, show_model=show_model)
    if show_atoms:
        _add_atoms_layer(viewer, fitter)

    # Workflow-aligned dock widgets.
    from qem.viz._napari_widgets import build_widgets

    docks = build_widgets(viewer, fitter)
    for name, widget in docks.items():
        viewer.window.add_dock_widget(widget, name=name, area="right")

    # Status bar.
    _set_status_summary(viewer, fitter)
    return viewer


def _add_image_layers(viewer, fitter: Fitter, *, show_model: bool) -> None:
    image = np.asarray(fitter.image)
    viewer.add_image(image, name=LAYER_IMAGE, colormap="gray")
    if show_model:
        # Hidden by default — toggled on once the user runs a fit.
        model = _model_image(fitter)
        viewer.add_image(
            model, name=LAYER_MODEL, colormap="gray",
            visible=False, opacity=0.7,
        )
        residual = image - model
        viewer.add_image(
            residual, name=LAYER_RESIDUAL, colormap="magma",
            visible=False, opacity=0.7,
        )


def _add_atoms_layer(viewer, fitter: Fitter):
    yx, props, face_color = _atoms_to_points(fitter)
    if len(yx) == 0:
        return None
    return viewer.add_points(
        yx,
        name=LAYER_ATOMS,
        properties=props,
        face_color=face_color,
        size=4,
        edge_width=0,
    )


# --------------------------------------------------------------------------
# Layer / status refresh — called by widgets after each fit / voronoi step
# --------------------------------------------------------------------------

def refresh_layers(viewer, fitter: Fitter) -> None:
    """Re-pull state from ``fitter`` into the viewer's layers."""
    image = np.asarray(fitter.image)
    if LAYER_MODEL in viewer.layers:
        model = _model_image(fitter)
        viewer.layers[LAYER_MODEL].data = model
        if LAYER_RESIDUAL in viewer.layers:
            viewer.layers[LAYER_RESIDUAL].data = image - model
    if LAYER_ATOMS in viewer.layers:
        yx, props, face_color = _atoms_to_points(fitter)
        atoms = viewer.layers[LAYER_ATOMS]
        atoms.data = yx
        atoms.properties = props
        atoms.face_color = face_color
    elif fitter.coordinates is not None and len(fitter.coordinates):
        _add_atoms_layer(viewer, fitter)
    _set_status_summary(viewer, fitter)


def _set_status_summary(viewer, fitter: Fitter) -> None:
    """One-line status string in napari's status bar."""
    n = len(fitter.coordinates) if fitter.coordinates is not None else 0
    device = getattr(fitter, "device", "?")
    parts = [f"device={device}", f"atoms={n}"]
    pred = fitter.prediction
    if pred is not None and (not hasattr(pred, "size") or pred.size > 0):
        residual = float(np.std(np.asarray(fitter.image) - _model_image(fitter)))
        parts.append(f"residual_std={residual:.2f}")
    try:
        viewer.status = " | ".join(parts)
    except Exception:
        pass


# --------------------------------------------------------------------------
# CLI entry: `qem-app [path]`
# --------------------------------------------------------------------------

def run_app(image_path: str | None = None) -> None:
    """CLI entry — opens a Fitter on the given image and runs napari.

    Wired in ``pyproject.toml`` as the ``qem-app`` console script.
    """
    import napari

    if image_path is None:
        # No image given — open an empty viewer; the user can drag-drop
        # an image, then construct a Fitter via Python or wait for a
        # forthcoming "Open image" widget.
        viewer = napari.Viewer(title="QEM — drag an image here")  # noqa: F841
        napari.run()
        return

    from qem.fit.fitter import Fitter

    image = _load_image(image_path)
    fitter = Fitter(image, dx=1.0, units="A")
    open_in_napari(fitter)  # noqa: F841
    napari.run()


def _load_image(path: str) -> np.ndarray:
    """Load an image from a path that QEM understands."""
    p = str(path).lower()
    if p.endswith((".tif", ".tiff")):
        from skimage.io import imread

        return np.asarray(imread(path))
    if p.endswith(".npy"):
        return np.load(path)
    if p.endswith(".mat"):
        import qem

        legacy = qem.io.read_statstem(path)
        return np.asarray(legacy["input"]["obs"])
    raise ValueError(
        f"Unsupported image format: {path!r}. Use .tif/.tiff/.npy/.mat."
    )


__all__ = [
    "open_in_napari",
    "refresh_layers",
    "run_app",
    "LAYER_IMAGE",
    "LAYER_MODEL",
    "LAYER_RESIDUAL",
    "LAYER_ATOMS",
    "LAYER_VORONOI",
]
