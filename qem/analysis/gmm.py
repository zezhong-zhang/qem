"""GMM-based atom counting — extracted from qem.fit.fitter (Linus #9).

Methods bind back onto Fitter via `_bind(Fitter)` from qem.fit.fitter,
preserving `fitter.estimate_atom_counts_with_gmm(...)` etc.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from qem.fit.fitter import Fitter  # noqa: F401


def estimate_atom_counts_with_gmm(
    self,
    max_components: int = 5,
    scoring_method: str = "icl",
    initialization_method: str = "middle",
    plot_results: bool = True,
    per_element: bool = True,
    save_results: bool = False,
    interactive_selection: bool = True,
    use_first_local_minimum: bool = True,
):
    """Estimate atom counts using Gaussian Mixture Model on cross-section histograms.
    
    This method applies GMM to the refined cross-section histogram to statistically
    determine the number of atoms in each atomic column based on scattering cross-sections.
    
    Args:
        max_components: Maximum number of Gaussian components to test
        scoring_method: Information criterion for model selection ('icl', 'aic', 'bic')
        initialization_method: Method for initializing GMM means
        plot_results: Whether to plot the GMM fitting results
        per_element: Whether to fit GMM separately for each element type
        save_results: Whether to save plots and results
        interactive_selection: Whether to allow interactive component selection
        use_first_local_minimum: Whether to use first local minimum instead of global
        
    Returns:
        dict: Dictionary containing GMM results and atom count estimates
    """
    if not hasattr(self, 'params') or self.params is None:
        raise ValueError("Please run fitting first to obtain refined cross-sections")

    from qem.analysis.gaussian_mixture_model import GaussianMixtureModel

    # Get refined cross-sections (volumes)
    cross_sections = self.volume.reshape(-1, 1)  # Reshape for GMM input

    gmm_results = {}
    atom_count_estimates = {}

    if per_element:
        # Fit GMM separately for each element type
        for atom_type in np.unique(self.atom_types):
            element_name = self.elements[atom_type]
            mask = self.atom_types == atom_type
            element_cross_sections = cross_sections[mask]

            if len(element_cross_sections) < 10:  # Skip if too few data points
                logging.warning(f"Skipping GMM for {element_name}: insufficient data points")
                continue

            # Initialize and fit GMM
            gmm = GaussianMixtureModel(element_cross_sections)
            gmm.fit_gaussian_mixture_model(
                num_components=max_components,
                scoring_methods=[scoring_method, "nllh"],
                initialization_method=initialization_method,
                use_first_local_minimum=use_first_local_minimum,
            )

            # Plot results and allow component selection
            if plot_results:
                selected_components = gmm.plot_interactive_gmm_selection(
                    element_cross_sections, element_name,
                    save_results, interactive_selection
                )
            else:
                # Use recommendation if no plotting
                selected_components = gmm.get_optimal_components("recommendation")

            # Get component parameters using user-selected components
            component_idx = selected_components - 1
            weights = gmm.fit_result.weight[component_idx]
            means = gmm.fit_result.mean[component_idx]
            widths = gmm.fit_result.width[component_idx]

            # Estimate atom counts based on component means
            # Assume components correspond to different atom counts (1, 2, 3, etc.)
            sorted_indices = np.argsort(means.flatten())
            atom_counts = np.arange(1, len(sorted_indices) + 1)

            # Assign atom counts to each atomic column
            column_assignments = gmm.fit_result.idxComponentOfScs(component_idx)
            estimated_counts = atom_counts[sorted_indices][column_assignments]

            gmm_results[element_name] = {
                'gmm_model': gmm,
                'selected_components': selected_components,  # Store user selection
                'recommended_components': gmm.recommended_components,  # Store recommendation
                'weights': weights,
                'means': means[sorted_indices],
                'widths': widths[sorted_indices],
                'scores': gmm.fit_result.score,
            }

            atom_count_estimates[element_name] = estimated_counts

    else:
        # Fit GMM to all cross-sections together
        gmm = GaussianMixtureModel(cross_sections)
        gmm.fit_gaussian_mixture_model(
            num_components=max_components,
            scoring_methods=[scoring_method, "nllh"],
            initialization_method=initialization_method,
            use_first_local_minimum=use_first_local_minimum,
        )

        # Plot results and allow component selection
        if plot_results:
            selected_components = gmm.plot_interactive_gmm_selection(
                cross_sections, 'all_elements',
                save_results, interactive_selection
            )
        else:
            selected_components = gmm.get_optimal_components("recommendation")

        component_idx = selected_components - 1

        weights = gmm.fit_result.weight[component_idx]
        means = gmm.fit_result.mean[component_idx]
        widths = gmm.fit_result.width[component_idx]

        sorted_indices = np.argsort(means.flatten())
        atom_counts = np.arange(1, len(sorted_indices) + 1)

        column_assignments = gmm.fit_result.idxComponentOfScs(component_idx)
        estimated_counts = atom_counts[sorted_indices][column_assignments]

        gmm_results['all_elements'] = {
            'gmm_model': gmm,
            'selected_components': selected_components,  # Store user selection
            'recommended_components': gmm.recommended_components,  # Store recommendation
            'weights': weights,
            'means': means[sorted_indices],
            'widths': widths[sorted_indices],
            'scores': gmm.fit_result.score,
        }

        atom_count_estimates['all_elements'] = estimated_counts

    # Store results as instance attributes
    self.gmm_results = gmm_results
    self.atom_count_estimates = atom_count_estimates

    return {
        'gmm_results': gmm_results,
        'atom_count_estimates': atom_count_estimates,
    }

def integrate_gmm_with_crystal_analyzer(self, region_index: int = 0):
    """Integrate GMM atom counts by stacking atoms in-place on the mapped lattice.

    This is the *legacy* path: it mutates the region's ``AtomicColumns.lattice``,
    placing extra atoms around each column's existing z with a heuristic spacing.
    For a crystal-consistent model whose atoms follow the CIF spacing and stacking
    order and are centred symmetrically about ``z = 0``, prefer
    :meth:`build_symmetric_3d_model`, which returns a fresh ASE ``Atoms`` object
    without mutating the mapped lattice.

    Args:
        region_index: Index of the region to update (default: 0)

    Returns:
        Updated crystal analyzer object with GMM-based atom counts
    """
    if not hasattr(self, 'atom_count_estimates'):
        raise ValueError("Please run estimate_atom_counts_with_gmm() first")

    if region_index not in self.regions.keys:
        raise ValueError(f"Region {region_index} not found in regions")

    # Get the crystal analyzer for this region
    region = self.regions[region_index]
    if not hasattr(region, 'analyzer') or region.analyzer is None:
        raise ValueError(f"No crystal analyzer found for region {region_index}. "
                       "Please run map_lattice() first.")

    crystal_analyzer = region.analyzer

    # Filter atom count estimates for columns in this region
    column_mask = self.region_column_labels == region_index
    region_atom_counts = {}

    for element_name, all_counts in self.atom_count_estimates.items():
        if element_name == 'all_elements':
            # Handle case where GMM was fit to all elements together
            region_atom_counts[element_name] = all_counts[column_mask]
        else:
            # Handle per-element GMM fitting
            element_columns = column_mask & (self.atom_types == self.elements.index(element_name))
            if element_columns.any():
                region_atom_counts[element_name] = all_counts

    # Update the crystal analyzer with GMM results
    updated_columns = crystal_analyzer.update_atoms_from_gmm(
        region_atom_counts
    )

    # Update the region's columns
    region.columns = updated_columns

    return crystal_analyzer


def update_all_regions_with_gmm(self):
    """Update all regions with GMM atom count estimates.
    
    Z-spacing is automatically determined from the supercell structure.
        
    Returns:
        Dictionary mapping region indices to updated crystal analyzers
    """
    updated_analyzers = {}

    for region_index in self.regions.keys:
        try:
            analyzer = self.integrate_gmm_with_crystal_analyzer(region_index)
            updated_analyzers[region_index] = analyzer
            logging.info(f"Successfully updated region {region_index} with GMM results")
        except (ValueError, RuntimeError, KeyError, AttributeError) as e:
            logging.warning(
                f"Could not update region {region_index} ({type(e).__name__}): {e}"
            )

    return updated_analyzers


def export_gmm_updated_structure(self, region_index: int = 0, filename: str = None):
    """Export the GMM-updated atomic structure to various formats.
    
    Args:
        region_index: Index of the region to export
        filename: Output filename (without extension)
        
    Returns:
        ASE Atoms object of the updated structure
    """
    if region_index not in self.regions.keys:
        raise ValueError(f"Region {region_index} not found")

    region = self.regions[region_index]
    if not hasattr(region, 'columns') or region.columns is None:
        raise ValueError(f"No atomic columns found for region {region_index}. "
                       "Please run integrate_gmm_with_crystal_analyzer() first.")

    # Get the updated lattice
    updated_lattice = region.columns.lattice

    if filename:
        # Export to different formats
        from ase.io import write
        write(f"{filename}.xyz", updated_lattice)
        write(f"{filename}.cif", updated_lattice)
        logging.info(f"Exported GMM-updated structure to {filename}.xyz and {filename}.cif")

    return updated_lattice



def _unit_cell_for_element(unit_cell, element):
    """Return an ASE unit cell containing ``element`` — the mapped cell if it has
    that element, otherwise the element's bulk crystal (ASE reference state)."""
    from ase.build import bulk
    from ase.data import atomic_numbers, reference_states

    if unit_cell is not None and getattr(unit_cell, "cell", None) is not None:
        if element in set(unit_cell.get_chemical_symbols()):
            return unit_cell

    try:
        ref = reference_states[atomic_numbers[element]]
    except (KeyError, IndexError):
        return None
    if not ref or "a" not in ref:
        return None
    symmetry, a = ref.get("symmetry"), ref["a"]
    try:
        return bulk(element, symmetry, a=a, cubic=True)
    except (ValueError, RuntimeError, KeyError, NotImplementedError):
        # Some structures (e.g. hcp) can't be built as a cubic cell; retry plain.
        try:
            return bulk(element, symmetry, a=a)
        except (ValueError, RuntimeError, KeyError, NotImplementedError):
            return None


def _element_column_line(unit_cell, element, zone_axis=(0, 0, 1), half_span=1):
    """Crystal z-positions of one element within a column, from the unit cell.

    Tiles the unit cell, views it along ``zone_axis``, groups atoms of ``element``
    that share an in-plane ``(x, y)`` position, and returns the sorted list of their
    z-coordinates for the densest (most complete) such column. This captures both
    the **spacing** and the **stacking order** of that element along the beam
    exactly as defined by the crystal (CIF) — not a hardcoded value. The line is
    shifted so a reference atom sits at ``z = 0``.

    Args:
        unit_cell: ASE ``Atoms`` unit cell (mapped from a CIF, or bulk).
        element: element symbol to extract the column line for.
        zone_axis: beam direction in the unit-cell frame (e.g. the CIF c-axis is
            ``(0, 0, 1)``).
        half_span: half the largest atom count expected — sets how many crystal
            repeats are tiled so the line is long enough to centre any column.

    Returns:
        np.ndarray | None: sorted z-positions (Angstrom), centred on a reference
        atom at 0, or None if the element/structure is unavailable.
    """
    from collections import defaultdict

    cell = _unit_cell_for_element(unit_cell, element)
    if cell is None:
        return None

    reps = int(2 * max(int(half_span), 1) + 5)
    reps = max(3, min(reps, 60))
    super_cell = cell * (reps, reps, reps)

    symbols = np.asarray(super_cell.get_chemical_symbols())
    keep = symbols == element
    if not keep.any():
        return None
    coords = super_cell.positions[keep]

    # Rotate the zone axis onto z with an orthonormal frame.
    z_dir = np.asarray(zone_axis, dtype=float)
    z_dir /= np.linalg.norm(z_dir)
    x_dir = np.array([1.0, 0.0, 0.0]) if abs(z_dir[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    x_dir -= z_dir * (x_dir @ z_dir)
    x_dir /= np.linalg.norm(x_dir)
    y_dir = np.cross(z_dir, x_dir)
    projected = coords @ np.vstack([x_dir, y_dir, z_dir]).T

    columns = defaultdict(list)
    for (px, py), pz in zip(np.round(projected[:, :2], 2), projected[:, 2], strict=False):
        columns[(px, py)].append(pz)

    # Densest column = most complete crystal line for this element.
    line = np.sort(np.asarray(max(columns.values(), key=len), dtype=float))
    if len(line) == 0:
        return None

    # Shift so the central site (reference atom) sits at z = 0.
    centre = 0.5 * (line[0] + line[-1])
    line -= line[int(np.argmin(np.abs(line - centre)))]
    return line


def _infer_z_spacing(self, analyzer=None, zone_axis=(0, 0, 1)) -> float:
    """Median inter-atom spacing along the beam for the first element, in Angstrom.

    Derived from the crystal structure via :func:`_element_column_line` (the mapped
    unit cell when available, otherwise the element's bulk crystal). Used for the
    plot annotation and as a uniform fallback; the model itself is built directly
    from the per-element crystal lines. Falls back to 2.0 A if unavailable.
    """
    unit_cell = getattr(analyzer, "unit_cell", None) if analyzer is not None else None
    for element in getattr(self, "elements", []):
        line = _element_column_line(unit_cell, element, zone_axis, half_span=4)
        if line is not None and len(line) > 1:
            return float(np.median(np.diff(line)))
    return 2.0


def build_symmetric_3d_model(
    self,
    region_index: int = 0,
    z_spacing: float | None = None,
    zone_axis: tuple = (0, 0, 1),
    plot: bool = True,
    ase_view: bool = False,
    elev: float = 12.0,
    azim: float = -72.0,
):
    """Combine GMM atom counts with the crystal structure into a Z-symmetric 3D model.

    Each atomic column keeps its refined ``(x, y)`` position, and the ``N`` atoms
    estimated by the GMM (:meth:`estimate_atom_counts_with_gmm`) are placed on the
    real crystal z-positions of that column, as symmetrically as possible about the
    ``z = 0`` plane.

    Both the **layer spacing and the stacking order along z come from the crystal
    structure** — the mapped unit cell (the CIF passed to :meth:`map_lattice`), or
    the element's bulk crystal when no crystal was mapped. For each element the
    sequence of z-positions within a column is read from the unit cell
    (:func:`_element_column_line`); the ``N`` atoms are the ``N`` crystal sites
    centred on a reference atom at ``z = 0``. Because atoms must land on crystal
    planes, a column with an even count cannot split evenly, so it is centred with a
    one-atom imbalance — e.g. an 11-atom column becomes 5 below / reference / 5
    above, and a 12-atom column becomes 5 below / reference / 6 above (6 about
    ``z``, 5 under). This reflects that ADF-STEM is insensitive to the ordering of
    atoms along the beam, so the central-symmetric lattice configuration is the
    natural unbiased guess.

    Args:
        region_index: Region whose columns and GMM counts to use.
        z_spacing: Optional override for the layer spacing (Angstrom). If None, the
            spacing and order are read from the crystal structure. Passing a value
            forces a uniform grid at that spacing instead.
        zone_axis: Beam direction in the unit-cell frame used to read the crystal
            column (the CIF c-axis is ``(0, 0, 1)``).
        plot: Draw a matplotlib 3D scatter of the model, coloured per element.
        ase_view: Also open the interactive ASE viewer.
        elev, azim: Elevation / azimuth of the matplotlib 3D view.

    Returns:
        ase.Atoms: the reconstructed 3D model. It is also stored on the region as
        ``region.columns_symmetric``.
    """
    from ase import Atoms

    if not getattr(self, "atom_count_estimates", None):
        raise ValueError(
            "No atom counts found. Run estimate_atom_counts_with_gmm() first."
        )
    if region_index not in self.regions.keys:
        raise ValueError(f"Region {region_index} not found in regions.")

    region = self.regions[region_index]
    # The mapped crystal (from map_lattice/CIF) supplies the z spacing and order.
    analyzer = getattr(region, "analyzer", None)
    unit_cell = getattr(analyzer, "unit_cell", None) if analyzer is not None else None

    # Gather, per element symbol, the columns (x, y) and their GMM atom counts.
    columns_by_symbol: dict[str, list] = {}
    for element, counts in self.atom_count_estimates.items():
        counts = np.asarray(counts).ravel().astype(int)

        if element == "all_elements":
            column_mask = np.ones(len(self.atom_types), dtype=bool)
            column_symbols = [self.elements[t] for t in self.atom_types]
        else:
            if element not in self.elements:
                logging.warning(f"Element '{element}' not in {self.elements}; skipping.")
                continue
            column_mask = self.atom_types == self.elements.index(element)
            column_symbols = [element] * int(column_mask.sum())

        xy_angstrom = self.coordinates[column_mask] * self.dx  # pixels -> Angstrom
        if len(counts) != len(xy_angstrom):
            raise ValueError(
                f"Atom-count array for '{element}' has length {len(counts)} but "
                f"{len(xy_angstrom)} columns were found. Re-run "
                "estimate_atom_counts_with_gmm() after map_lattice()."
            )

        for (x, y), n_atoms, symbol in zip(
            xy_angstrom, counts, column_symbols, strict=False
        ):
            columns_by_symbol.setdefault(symbol, []).append(
                (float(x), float(y), max(int(n_atoms), 1))
            )

    if not columns_by_symbol:
        raise ValueError(
            "No atoms were placed — the GMM produced no counts for the mapped "
            "elements. Check estimate_atom_counts_with_gmm() results."
        )

    symbols: list[str] = []
    positions: list[tuple[float, float, float]] = []
    per_element_atoms: dict[str, int] = {}
    per_element_columns: dict[str, int] = {}
    spacings: list[float] = []

    for symbol, cols in columns_by_symbol.items():
        max_count = max(n for _, _, n in cols)

        # Crystal z-line for this element: spacing + order from the unit cell.
        line = None
        if z_spacing is None:
            line = _element_column_line(unit_cell, symbol, zone_axis, max_count)
        if line is None:
            # Explicit override, or no crystal available: uniform grid.
            spacing = z_spacing if z_spacing is not None else _infer_z_spacing(
                self, analyzer, zone_axis
            )
            line = np.arange(-(max_count + 2), max_count + 3) * spacing

        reference_idx = int(np.argmin(np.abs(line)))
        if len(line) > 1:
            spacings.append(float(np.median(np.diff(line))))

        for (x, y, n_atoms) in cols:
            lo = max(reference_idx - (n_atoms - 1) // 2, 0)
            hi = min(reference_idx + n_atoms // 2 + 1, len(line))
            z_layers = line[lo:hi]
            per_element_atoms[symbol] = per_element_atoms.get(symbol, 0) + len(z_layers)
            per_element_columns[symbol] = per_element_columns.get(symbol, 0) + 1
            for z in z_layers:
                symbols.append(symbol)
                positions.append((x, y, float(z)))

    model = Atoms(symbols=symbols, positions=np.asarray(positions))
    region.columns_symmetric = model
    eff_spacing = float(np.median(spacings)) if spacings else 0.0

    summary = ", ".join(
        f"{sym}: {per_element_atoms[sym]} atoms / {per_element_columns[sym]} columns"
        for sym in sorted(per_element_atoms)
    )
    logging.info(
        f"Built Z-symmetric 3D model for region {region_index}: "
        f"{len(model)} atoms ({summary}); crystal layer spacing {eff_spacing:.3f} A."
    )

    if plot:
        _plot_symmetric_3d_model(self, model, region_index, eff_spacing, elev, azim)
    if ase_view:
        from ase.visualize import view

        view(model)

    return model


def _plot_symmetric_3d_model(self, model, region_index, z_spacing, elev, azim):
    """Render a Z-symmetric 3D atomic model as a matplotlib 3D scatter.

    Atoms are coloured by element (Jmol colours) and sized by covalent radius,
    with the ``z = 0`` mirror plane drawn for reference.
    """
    import matplotlib.pyplot as plt
    from ase.data import atomic_numbers as Z_OF
    from ase.data import covalent_radii
    from ase.data.colors import jmol_colors

    pos = model.positions
    syms = np.asarray(model.get_chemical_symbols())

    fig = plt.figure(figsize=(8, 7))
    ax = fig.add_subplot(111, projection="3d")

    for symbol in sorted(set(syms), key=lambda s: Z_OF[s]):
        mask = syms == symbol
        z_num = Z_OF[symbol]
        ax.scatter(
            pos[mask, 0],
            pos[mask, 1],
            pos[mask, 2],
            s=40 + 45 * covalent_radii[z_num],
            color=jmol_colors[z_num],
            edgecolor="k",
            linewidth=0.3,
            depthshade=True,
            label=f"{symbol} ({int(mask.sum())} atoms)",
        )

    # Draw the z = 0 mirror plane the atoms are symmetric about.
    x_lo, x_hi = pos[:, 0].min(), pos[:, 0].max()
    y_lo, y_hi = pos[:, 1].min(), pos[:, 1].max()
    xx, yy = np.meshgrid([x_lo, x_hi], [y_lo, y_hi])
    ax.plot_surface(xx, yy, np.zeros_like(xx), alpha=0.12, color="gray")

    z_abs = float(np.abs(pos[:, 2]).max())
    z_lim = max(z_abs * 1.15, z_spacing)
    ax.set_zlim(-z_lim, z_lim)
    ax.set_xlabel("x (Å)")
    ax.set_ylabel("y (Å)")
    ax.set_zlabel("z (Å)  —  symmetric about z = 0")
    ax.set_title(
        f"Region {region_index}: Z-symmetric 3D model\n"
        f"{len(model)} atoms · layer spacing {z_spacing:.3f} Å"
    )
    ax.legend(loc="upper left", fontsize=8)
    ax.view_init(elev=elev, azim=azim)
    fig.tight_layout()
    plt.show()
    return fig


class FitterGMMMixin:
    """Gaussian-mixture atom counting for :class:`Fitter`."""

    estimate_atom_counts_with_gmm = estimate_atom_counts_with_gmm
    integrate_gmm_with_crystal_analyzer = integrate_gmm_with_crystal_analyzer
    update_all_regions_with_gmm = update_all_regions_with_gmm
    export_gmm_updated_structure = export_gmm_updated_structure
    build_symmetric_3d_model = build_symmetric_3d_model


__all__ = [
    "FitterGMMMixin",
    "estimate_atom_counts_with_gmm",
    "integrate_gmm_with_crystal_analyzer",
    "update_all_regions_with_gmm",
    "export_gmm_updated_structure",
    "build_symmetric_3d_model",
]
