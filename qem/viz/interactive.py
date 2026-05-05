"""Interactive (non-blocking) visualisations using plotly.

Drop-in alternatives to the matplotlib functions in :mod:`qem.fit.plot`
that:

* render to HTML — no Tk / Qt / macOS event loop, never blocks
* show on first display in Jupyter, save to ``.html`` from scripts
* support hover (per-atom info, residual values), pan, zoom, box-select

These are *additions*, not replacements. The matplotlib helpers
remain for static figures (publication PDFs etc) — call whichever
fits the surface you're rendering to.

API mirror::

    fitter.plot_fitting()           # matplotlib (publication-quality)
    fitter_fitting_interactive(fitter)   # plotly (notebook / Streamlit)

To bind onto Fitter for the OO style ``fitter.plot_fitting_interactive()``,
import this module — the bottom-of-file ``_bind`` is wired in
``qem.fit.fitter`` next to the other ``_bind_plot`` calls.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

import numpy as np

if TYPE_CHECKING:
    from qem.fit.fitter import Fitter  # noqa: F401


def _require_plotly():
    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
        return go, make_subplots
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "plotly not installed — `pip install plotly` (or "
            "`pip install qem[gui]` once the extra is updated)."
        ) from exc


def _atoms_per_type(self):
    """Yield ``(element_name, mask)`` per atom type."""
    elements = self.elements or []
    for atom_type in np.unique(self.atom_types):
        atom_type = int(atom_type)
        name = elements[atom_type] if atom_type < len(elements) else f"type {atom_type}"
        yield name, self.atom_types == atom_type


# --------------------------------------------------------------------------
# image / model / residual triptych
# --------------------------------------------------------------------------

def plot_fitting_interactive(self, *, save: Optional[str] = None):
    """Interactive image / model / residual triptych (plotly).

    Linked pan + zoom across the three panels; hover shows pixel
    coordinates and intensity. Replaces ``plot_fitting`` for screen
    viewing. Pass ``save="figure.html"`` to write a self-contained
    HTML file.
    """
    go, make_subplots = _require_plotly()

    image = self.image
    pred = (
        self.prediction.detach().cpu().numpy()
        if hasattr(self.prediction, "cpu")
        else np.asarray(self.prediction)
    )
    residual = image - pred

    fig = make_subplots(
        rows=1, cols=3, shared_xaxes=True, shared_yaxes=True,
        subplot_titles=("Image", "Model", "Residual"),
        horizontal_spacing=0.04,
    )
    common = dict(colorscale="Greys", showscale=True)
    vmin, vmax = float(image.min()), float(image.max())
    fig.add_trace(go.Heatmap(z=image, zmin=vmin, zmax=vmax, **common), row=1, col=1)
    fig.add_trace(go.Heatmap(z=pred, zmin=vmin, zmax=vmax, **common), row=1, col=2)
    fig.add_trace(go.Heatmap(z=residual, **common), row=1, col=3)
    for col in (1, 2, 3):
        fig.update_yaxes(autorange="reversed", scaleanchor="x", row=1, col=col)
    fig.update_layout(height=500, width=1500)

    if save is not None:
        fig.write_html(save, include_plotlyjs="cdn")
    return fig


# --------------------------------------------------------------------------
# atom-position scatter on the image
# --------------------------------------------------------------------------

def plot_coordinates_interactive(self, *, save: Optional[str] = None):
    """Image with atomic columns overlaid, coloured by element (plotly).

    Hover shows ``(x, y, element)`` per atom. Box-select / lasso-select
    available out of the box. For very large atom counts (>50k) prefer
    ``plot_coordinates_datashader`` (TODO).
    """
    go, _ = _require_plotly()

    fig = go.Figure()
    fig.add_trace(go.Heatmap(z=self.image, colorscale="Greys", showscale=False))
    for name, mask in _atoms_per_type(self):
        coords = self.coordinates[mask]
        fig.add_trace(
            go.Scattergl(
                x=coords[:, 0], y=coords[:, 1], mode="markers",
                marker=dict(size=4),
                name=name,
                hovertemplate="x=%{x:.1f}<br>y=%{y:.1f}<br>" + name,
            )
        )
    fig.update_yaxes(autorange="reversed", scaleanchor="x")
    fig.update_layout(width=700, height=700, legend=dict(orientation="h"))

    if save is not None:
        fig.write_html(save, include_plotlyjs="cdn")
    return fig


# --------------------------------------------------------------------------
# scattering cross-section histogram
# --------------------------------------------------------------------------

def plot_scs_histogram_interactive(self, *, save: Optional[str] = None):
    """Per-element scattering cross-section histogram (plotly).

    Hover shows the bin edges and counts; legend toggles element
    overlays. Replaces ``plot_scs_histogram`` for screen viewing.
    """
    go, _ = _require_plotly()

    if not hasattr(self, "scs_voronoi") or self.scs_voronoi is None:
        raise ValueError("Run fitter.voronoi_integration() before plotting SCS.")
    scs = (
        self.scs_voronoi.detach().cpu().numpy()
        if hasattr(self.scs_voronoi, "cpu")
        else np.asarray(self.scs_voronoi)
    )

    fig = go.Figure()
    for name, mask in _atoms_per_type(self):
        fig.add_trace(
            go.Histogram(
                x=scs[mask], name=name, opacity=0.7,
                nbinsx=80,
            )
        )
    fig.update_layout(
        barmode="overlay",
        xaxis_title=f"SCS [{self.units}²]",
        yaxis_title="Count",
        legend=dict(orientation="h"),
        width=900, height=500,
    )

    if save is not None:
        fig.write_html(save, include_plotlyjs="cdn")
    return fig


class FitterInteractiveMixin:
    """Plotly-based interactive plot helpers for :class:`Fitter`."""

    plot_fitting_interactive = plot_fitting_interactive
    plot_coordinates_interactive = plot_coordinates_interactive
    plot_scs_histogram_interactive = plot_scs_histogram_interactive


__all__ = [
    "FitterInteractiveMixin",
    "plot_fitting_interactive",
    "plot_coordinates_interactive",
    "plot_scs_histogram_interactive",
]
