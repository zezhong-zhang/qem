"""magicgui dock widgets for the napari QEM viewer.

Workflow-aligned: Data → Peaks → Fit → Voronoi → Analysis. Each dock
calls the underlying ``Fitter`` method directly and refreshes the
napari layers via :func:`qem.viz.napari_app.refresh_layers`.

Long-running operations (fit / voronoi) run in
``napari.qt.thread_worker`` so the Qt event loop stays responsive.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from napari import Viewer  # noqa: F401

    from qem.fit.fitter import Fitter  # noqa: F401


def build_widgets(viewer: Viewer, fitter: Fitter) -> dict:
    """Build the QEM workflow dock widgets, return ``{name: widget}``.

    Order matters — napari stacks docks top-to-bottom in the order
    ``add_dock_widget`` is called. The order here is the workflow:
    Data → Peaks → Fit → Voronoi → Analysis.
    """
    return {
        "📂 Data": _data_widget(viewer, fitter),
        "🎯 Peaks": _peaks_widget(viewer, fitter),
        "⚙️  Fit": _fit_widget(viewer, fitter),
        "📊 Voronoi": _voronoi_widget(viewer, fitter),
        "🔬 Analysis": _analysis_widget(viewer, fitter),
    }


# --------------------------------------------------------------------------
# Data — open / save / export
# --------------------------------------------------------------------------

def _data_widget(viewer, fitter):
    from magicgui.widgets import (
        Container,
        FileEdit,
        FloatSpinBox,
        LineEdit,
        PushButton,
    )

    dx = FloatSpinBox(label="dx (Å/px)", value=float(fitter.dx), min=1e-6, max=100.0, step=0.001)
    units = LineEdit(label="Units", value=str(fitter.units))
    save_path = FileEdit(label="Save .h5", mode="w", filter="*.h5")
    save = PushButton(text="Save fit")
    export_path = FileEdit(label="Export NumPy", mode="w", filter="*.npz")
    export = PushButton(text="Export params")

    @dx.changed.connect
    def _on_dx_change() -> None:
        fitter.dx = float(dx.value)
        _status(viewer, f"dx = {fitter.dx:.4f} {fitter.units}")

    @units.changed.connect
    def _on_units_change() -> None:
        fitter.units = str(units.value)

    @save.changed.connect
    def _on_save() -> None:
        path = str(save_path.value)
        if not path:
            _status(viewer, "No save path set.")
            return
        fitter.save(path)
        _status(viewer, f"Saved fit → {path}")

    @export.changed.connect
    def _on_export() -> None:
        path = str(export_path.value)
        if not path:
            _status(viewer, "No export path set.")
            return
        snapshot = fitter.detach()
        np.savez(path, **{k: v for k, v in snapshot.items() if isinstance(v, np.ndarray)})
        _status(viewer, f"Exported params → {path}")

    return Container(widgets=[dx, units, save_path, save, export_path, export])


# --------------------------------------------------------------------------
# Peaks — find atomic columns from the current image
# --------------------------------------------------------------------------

def _peaks_widget(viewer, fitter):
    from magicgui.widgets import Container, FloatSpinBox, PushButton, SpinBox

    min_distance = SpinBox(label="Min distance (px)", value=10, min=1, max=200)
    threshold = FloatSpinBox(
        label="Threshold (× max)", value=0.1, min=0.0, max=1.0, step=0.05,
    )
    smoothing = FloatSpinBox(
        label="Smoothing σ (px)", value=1.0, min=0.0, max=10.0, step=0.5,
    )
    refine_com = PushButton(text="Refine via CoM")
    run = PushButton(text="Find peaks")

    @run.changed.connect
    def _on_find() -> None:
        fitter.find_peaks(
            min_distance=int(min_distance.value),
            threshold_rel=float(threshold.value),
            smoothing=float(smoothing.value),
        )
        _refresh(viewer, fitter)
        _status(viewer, f"Found {len(fitter.coordinates)} peaks.")

    @refine_com.changed.connect
    def _on_refine() -> None:
        fitter.refine_center_of_mass()
        _refresh(viewer, fitter)
        _status(viewer, "Refined positions via CoM.")

    return Container(widgets=[min_distance, threshold, smoothing, run, refine_com])


# --------------------------------------------------------------------------
# Fit — linear estimator + Adam loop, threaded
# --------------------------------------------------------------------------

def _fit_widget(viewer, fitter):
    from magicgui.widgets import (
        CheckBox,
        ComboBox,
        Container,
        FloatSpinBox,
        ProgressBar,
        PushButton,
        SpinBox,
    )

    method = ComboBox(label="Method", choices=["fit_global", "fit_stochastic"])
    optimizer = ComboBox(
        label="Optimizer", choices=["adam", "adamw", "sgd", "lbfgs"],
    )
    maxiter = SpinBox(label="Max iter", value=500, min=1, max=20_000)
    step_size = FloatSpinBox(
        label="Step size (LR)", value=0.01, min=1e-5, max=1.0, step=0.01,
    )
    same_width = CheckBox(label="Same width per element", value=fitter.same_width)
    fit_bg = CheckBox(label="Fit background", value=fitter.fit_background)
    show_overlay = CheckBox(label="Show model overlay after fit", value=True)
    progress = ProgressBar(label="Progress", min=0, max=1, value=0)
    run = PushButton(text="Run fit")

    @run.changed.connect
    def _on_run() -> None:
        fitter.same_width = bool(same_width.value)
        fitter.fit_background = bool(fit_bg.value)
        params = fitter.params if fitter.params is not None else fitter.init_params()
        progress.max = int(maxiter.value)
        progress.value = 0

        from napari.qt import thread_worker

        @thread_worker
        def _do_fit():
            if method.value == "fit_global":
                fitter.fit_global(
                    params=params, maxiter=int(maxiter.value),
                    step_size=float(step_size.value),
                    optimizer=str(optimizer.value), verbose=False,
                )
            else:
                fitter.fit_stochastic(
                    params=params, maxiter=int(maxiter.value),
                    step_size=float(step_size.value),
                    optimizer=str(optimizer.value), verbose=False,
                    batch_size=fitter.num_coordinates, num_epoch=10,
                )

        worker = _do_fit()
        worker.finished.connect(
            lambda: _on_fit_done(viewer, fitter, show_overlay.value, progress)
        )
        worker.start()
        _status(viewer, f"Running {method.value}…")

    return Container(widgets=[
        method, optimizer, maxiter, step_size,
        same_width, fit_bg, show_overlay, progress, run,
    ])


def _on_fit_done(viewer, fitter, show_overlay: bool, progress) -> None:
    from qem.viz.napari_app import LAYER_MODEL, LAYER_RESIDUAL

    progress.value = progress.max
    _refresh(viewer, fitter)
    if show_overlay:
        if LAYER_MODEL in viewer.layers:
            viewer.layers[LAYER_MODEL].visible = True
        if LAYER_RESIDUAL in viewer.layers:
            viewer.layers[LAYER_RESIDUAL].visible = True


# --------------------------------------------------------------------------
# Voronoi — integration + cell map overlay + colour-by-property
# --------------------------------------------------------------------------

def _voronoi_widget(viewer, fitter):
    from magicgui.widgets import (
        CheckBox,
        ComboBox,
        Container,
        FloatSpinBox,
        PushButton,
    )

    max_radius = FloatSpinBox(
        label="Cell max radius (px)", value=10.0, min=1.0, max=200.0, step=1.0,
    )
    refine = CheckBox(label="Levenberg-Marquardt refine", value=False)
    color_by = ComboBox(label="Colour atoms by", choices=["element", "scs", "height"])
    show_cells = CheckBox(label="Overlay voronoi cell map", value=False)
    run = PushButton(text="Compute Voronoi")

    @run.changed.connect
    def _on_run() -> None:
        fitter.fit_voronoi(max_radius=int(max_radius.value), refine=bool(refine.value))
        fitter.voronoi_integration(max_radius=float(max_radius.value))

        from qem.viz.napari_app import LAYER_ATOMS, LAYER_VORONOI

        _refresh(viewer, fitter)

        if LAYER_ATOMS in viewer.layers:
            atoms = viewer.layers[LAYER_ATOMS]
            mode = str(color_by.value)
            if mode != "element" and mode in atoms.properties:
                atoms.face_color = mode
                atoms.face_colormap = "viridis"
                atoms.refresh_colors()

        if show_cells.value:
            from qem.fit.voronoi import voronoi_point_record

            coords = np.column_stack([
                fitter.coordinates[:, 1], fitter.coordinates[:, 0],
            ]).T
            point_record = voronoi_point_record(
                fitter.image, coords, float(max_radius.value),
            )
            if LAYER_VORONOI in viewer.layers:
                viewer.layers[LAYER_VORONOI].data = point_record
            else:
                viewer.add_labels(point_record, name=LAYER_VORONOI, opacity=0.4)

        _status(viewer, "Voronoi integration complete.")

    return Container(widgets=[max_radius, refine, color_by, show_cells, run])


# --------------------------------------------------------------------------
# Analysis — GMM atom counting
# --------------------------------------------------------------------------

def _analysis_widget(viewer, fitter):
    from magicgui.widgets import ComboBox, Container, PushButton, SpinBox

    max_components = SpinBox(label="GMM max components", value=10, min=1, max=50)
    scoring = ComboBox(label="Scoring", choices=["icl", "bic", "aic"])
    run_gmm = PushButton(text="Run GMM atom counting")
    show_count_map = PushButton(text="Show atom-count map (matplotlib)")

    @run_gmm.changed.connect
    def _on_gmm() -> None:
        fitter.estimate_atom_counts_with_gmm(
            max_components=int(max_components.value),
            scoring_method=str(scoring.value),
            interactive_selection=False,
            plot_results=False,
        )
        _refresh(viewer, fitter)
        _status(viewer, "GMM atom counting complete.")

    @show_count_map.changed.connect
    def _on_show_count_map() -> None:
        # External matplotlib window — non-blocking thanks to napari's
        # Qt loop running independently.
        fitter.plot_atom_count_map()

    return Container(widgets=[max_components, scoring, run_gmm, show_count_map])


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------

def _refresh(viewer, fitter) -> None:
    from qem.viz.napari_app import refresh_layers

    refresh_layers(viewer, fitter)


def _status(viewer, msg: str) -> None:
    """Push a message to the napari status bar (and stdout for logging)."""
    try:
        viewer.status = msg
    except Exception:
        pass
    print(f"[QEM] {msg}")


__all__ = ["build_widgets"]
