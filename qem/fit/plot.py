"""Fitter plotting helpers — extracted from qem.fit.fitter (Linus #9).

These functions live here as module-level methods (`self` first) and are
bound back onto the Fitter class via `_bind(Fitter)` at the bottom of
`qem.fit.fitter`. So `fitter.plot_fitting()` keeps working from
notebooks while the bodies physically live in this module.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.path import Path
from matplotlib_scalebar.scalebar import ScaleBar

from qem.utils.tensors import to_numpy

if TYPE_CHECKING:
    from qem.fit.fitter import Fitter  # noqa: F401


def _plot_progress(self, params, index, select_params):
    """Helper function to keep plotting logic separate."""
    global_prediction = to_numpy(self.predict(params))
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Original Image with selected atoms
    axes[0].imshow(self.image, cmap="gray")
    axes[0].set_title("Original + Selected Atoms")
    axes[0].scatter(to_numpy(params["pos_x"][index]), to_numpy(params["pos_y"][index]), color="r", s=5)
    axes[0].set_aspect("equal")

    # Full Prediction
    axes[1].imshow(global_prediction, cmap="gray")
    axes[1].set_title("Current Full Prediction")
    axes[1].set_aspect("equal")

    # Residual
    axes[2].imshow(self.image - global_prediction, cmap="gray")
    axes[2].set_title("Residual")
    axes[2].set_aspect("equal")

    plt.tight_layout()
    plt.show()
    

def plot_coordinates(self, s=1):
    """
    Plot the coordinates of the atomic columns.

    Args:
        color (str, optional): The color of the atomic columns. Defaults to "red".
        s (int, optional): The size of the atomic columns. Defaults to 1.
    """
    plt.figure()
    plt.imshow(self.image, cmap="gray")
    for atom_type in np.unique(self.atom_types):
        mask = self.atom_types == atom_type
        elements = self.elements[atom_type]
        plt.scatter(
            self.coordinates[mask][:, 0],
            self.coordinates[mask][:, 1],
            s=s,
            label=elements,
        )
    plt.legend()

def plot_fitting(self,save = False):
    plt.figure(figsize=(15, 5))
    vmin = self.image.min()
    vmax = self.image.max()
    plt.subplot(1, 3, 1)
    im = plt.imshow(self.image, cmap="gray", vmin=vmin, vmax=vmax)
    plt.colorbar(im, fraction=0.046, pad=0.04)
    plt.gca().set_aspect("equal", adjustable="box")
    plt.title("Original Image")
    plt.tight_layout()
    plt.subplot(1, 3, 2)
    im = plt.imshow(self.prediction, cmap="gray", vmin=vmin, vmax=vmax)
    plt.gca().set_aspect("equal", adjustable="box")
    plt.colorbar(im, fraction=0.046, pad=0.04)
    plt.title("Model")
    plt.tight_layout()
    plt.subplot(1, 3, 3)
    im = plt.imshow(self.image - self.prediction, cmap="gray")
    plt.colorbar(im, fraction=0.046, pad=0.04)
    plt.gca().set_aspect("equal", adjustable="box")
    plt.title("Residual")
    plt.tight_layout()
    if save:
        plt.savefig("fitting.png", dpi=300)
        plt.savefig("fitting.svg")

def plot_scs(
    self,
    layout="horizontal",
    per_element=False,
    s=1,
    save=False,
    has_units=True,
    half: str = None,
    figsize=(10, 5),
):
    assert layout in {
        "horizontal",
        "vertical",
    }, "Layout should be horizontal or vertical"
    if layout == "horizontal":
        row, col = 1, 2
        if per_element:
            col += len(np.unique(self.atom_types)) - 1
    else:
        row, col = 2, 1
        if per_element:
            row += len(np.unique(self.atom_types)) - 1
    plt.figure(figsize=figsize)
    plt.subplot(row, col, 1)
    plt.imshow(self.image, cmap="gray")
    for atom_type in np.unique(self.atom_types):
        mask = self.atom_types == atom_type
        element = self.elements[int(atom_type)]
        if half is not None:
            if half == "top":
                mask = mask & (self.coordinates[:, 1] < self.ny / 2)
            elif half == "bottom":
                mask = mask & (self.coordinates[:, 1] > self.ny / 2)
            elif half == "left":
                mask = mask & (self.coordinates[:, 0] < self.nx / 2)
            elif half == "right":
                mask = mask & (self.coordinates[:, 0] > self.nx / 2)
        plt.scatter(
            self.coordinates[mask, 0],
            self.coordinates[mask, 1],
            s=s,
            label=element,
        )
    plt.legend(loc="upper right")
    plt.gca().set_aspect("equal", adjustable="box")
    plt.axis("off")
    scalebar = self.scalebar
    plt.gca().add_artist(scalebar)
    plt.title("Image")
    plt.tight_layout()

    # plot the scs
    pos_x = self.params["pos_x"] * self.dx
    pos_y = self.params["pos_y"] * self.dx
    pos_x = to_numpy(pos_x)
    pos_y = to_numpy(pos_y)
    if per_element:
        plt_idx = 1
        col = len(np.unique(self.atom_types)) + 1
        for atom_type in np.unique(self.atom_types):
            plt_idx += 1
            plt.subplot(row, col, plt_idx)
            mask = self.atom_types == atom_type
            element = self.elements[atom_type]
            im = plt.scatter(
                pos_x[mask],
                pos_y[mask],
                c=self.volume[mask],
                s=s,
                label=element,
            )
            cbar = plt.colorbar(im, fraction=0.046, pad=0.04)
            plt.gca().set_aspect("equal", adjustable="box")
            # plt.axis("off")
            plt.xlim(0, self.nx * self.dx)
            plt.ylim(0, self.ny * self.dx)
            plt.xlabel(r"X (A)")
            plt.ylabel(r"Y (A)")
            plt.gca().invert_yaxis()
            plt.title(f"{element}")
            if atom_type == self.atom_types.max():
                if has_units:
                    cbar.set_label(r"SCS (A^2)")
                else:
                    cbar.set_label("Integrated intensities")
            plt.tight_layout()
    else:
        plt.subplot(row, col, 2)
        im = plt.scatter(pos_x, pos_y, c=self.volume, s=2)
        cbar = plt.colorbar(im, fraction=0.046, pad=0.04)
        # plt.axis("off")
        plt.xlim(0, self.nx * self.dx)
        plt.ylim(0, self.ny * self.dx)
        plt.xlabel(r"X (A)")
        plt.ylabel(r"Y (A)")
        plt.gca().invert_yaxis()
        plt.gca().set_aspect("equal", adjustable="box")
        if has_units:
            cbar.set_label(r"SCS (A^2)")
        else:
            cbar.set_label("Integrated intensities")
        plt.tight_layout()
    if save:
        plt.savefig("scs.svg")
        plt.savefig("scs.png", dpi=300)

def plot_scs_voronoi(
    self,
    layout="horizontal",
    s=1,
    per_element=False,
    save=False,
    has_units=True,
    half: str = None,
    figsize=(10, 5),
):
    assert self.voronoi_volume is not None, "Please run the voronoi analysis first"
    if per_element:
        row, col = 1, 2
        col += len(np.unique(self.atom_types)) - 1
        plt.figure(figsize=figsize)
        plt.subplot(row, col, 1)
        plt.imshow(self.image, cmap="gray")
        for atom_type in np.unique(self.atom_types):
            mask = self.atom_types == atom_type
            element = self.elements[atom_type]
            if half is not None:
                if half == "top":
                    mask = mask & (self.coordinates[:, 1] < self.ny / 2)
                elif half == "bottom":
                    mask = mask & (self.coordinates[:, 1] > self.ny / 2)
                elif half == "left":
                    mask = mask & (self.coordinates[:, 0] < self.nx / 2)
                elif half == "right":
                    mask = mask & (self.coordinates[:, 0] > self.nx / 2)
            plt.scatter(
                self.coordinates[mask, 0],
                self.coordinates[mask, 1],
                s=1,
                label=element,
            )
        plt.legend(loc="upper right")
        plt.gca().add_artist(self.scalebar)
        plot_idx = 2
        for atom_type in np.unique(self.atom_types):
            mask = self.atom_types == atom_type
            plt.subplot(row, col, plot_idx)
            element = self.elements[atom_type]
            pos_x = self.params["pos_x"][mask] * self.dx
            pos_y = self.params["pos_y"][mask] * self.dx
            pos_x = to_numpy(pos_x)
            pos_y = to_numpy(pos_y) 
            im = plt.scatter(
                pos_x, pos_y, c=self.voronoi_volume[mask], s=s, label=element
            )
            plt.gca().set_aspect("equal", adjustable="box")
            cbar = plt.colorbar(im, fraction=0.046, pad=0.04)
            # plt.axis("off")
            plt.xlim(0, self.nx * self.dx)
            plt.ylim(0, self.ny * self.dx)
            plt.xlabel(r"X (A)")
            plt.ylabel(r"Y (A)")
            plt.gca().invert_yaxis()
            plt.title(f"{element}")
            if atom_type == self.atom_types.max():
                if has_units:
                    cbar.set_label(r"Voronoi SCS (A^2)")
                else:
                    cbar.set_label("Voronoi integrated intensities")
            plot_idx += 1
    else:
        row, col = (1, 2) if layout == "horizontal" else (2, 1)
        plt.figure()
        plt.subplot(row, col, 1)
        plt.imshow(self.image, cmap="gray")
        for atom_type in np.unique(self.atom_types):
            mask = self.atom_types == atom_type
            element = self.elements[atom_type]
            plt.scatter(
                self.coordinates[mask, 0],
                self.coordinates[mask, 1],
                s=1,
                label=element,
            )
        plt.legend()
        plt.gca().set_aspect("equal", adjustable="box")
        plt.title("Image")

        plt.subplot(row, col, 2)
        pos_x = self.params["pos_x"] * self.dx
        pos_y = self.params["pos_y"] * self.dx
        im = plt.scatter(pos_x, pos_y, c=self.voronoi_volume, s=s)
        # make aspect ratio equal
        plt.gca().invert_yaxis()
        plt.gca().set_aspect("equal", adjustable="box")
        plt.colorbar(im, fraction=0.046, pad=0.04)
        if has_units:
            plt.title(r"Voronoi scs (A^2)")
        else:
            plt.title("Voronoi integrated intensities")
    plt.tight_layout()

    if save:
        plt.savefig("voronoi_scs.svg")
        plt.savefig("voronoi_scs.png", dpi=300)

def plot_voronoi_integration_intensity(self,plot = False, save=False):
    if plot:
        plt.imshow(self._voronoi_map, cmap="viridis")
        plt.colorbar(label="Voronoi Integrated Intensity")
    if save:
        plt.savefig("Voronoi Integrated Intensity.png", dpi=300)
        plt.savefig("Voronoi Integrated Intensity.svg")

def _plot_gmm_results(self, cross_sections, gmm_model, element_name, save_results=False):
    """Legacy method - redirects to GMM module plotting for compatibility."""
    return gmm_model.plot_interactive_gmm_selection(
        cross_sections, element_name, save_results, interactive_selection=False
    )

def plot_scs_histogram(self, save=False, has_units=True):
    """Plot histogram of refined scattering cross-sections."""
    plt.figure()
    for atom_type in np.unique(self.atom_types):
        mask = self.atom_types == atom_type
        element = self.elements[atom_type]
        plt.hist(self.volume[mask], bins=100, alpha=0.5, label=element)
    plt.legend()
    if has_units:
        plt.xlabel(r"Refined SCS (A^2)")
    else:
        plt.xlabel("Integrated intensities")
    plt.ylabel("Frequency")
    plt.title("Histogram of QEM refined SCS")
    if save:
        plt.savefig("scs_histogram.svg")
        plt.savefig("scs_histogram.png", dpi=300)

def plot_atom_count_map(self, element_name=None, save=False, figsize=(12, 8)):
    """Plot spatial map of estimated atom counts with proper colorbar.
    
    Args:
        element_name: Specific element to plot, or None for all elements
        save: Whether to save the plot
        figsize: Figure size tuple
    """
    if not hasattr(self, 'atom_count_estimates'):
        raise ValueError("Please run estimate_atom_counts_with_gmm first")
    
    fig, ax = plt.subplots(figsize=figsize)
    
    if element_name is None:
        # Plot all elements with different symbols/colors
        all_counts = []
        all_pos_x = []
        all_pos_y = []
        scatter = None  # Initialize scatter variable
        
        for atom_type in np.unique(self.atom_types):
            element = self.elements[atom_type]
            if element in self.atom_count_estimates:
                mask = self.atom_types == atom_type
                counts = self.atom_count_estimates[element]
                
                pos_x = self.params["pos_x"][mask] * self.dx
                pos_y = self.params["pos_y"][mask] * self.dx
                
                pos_x_np = to_numpy(pos_x)
                pos_y_np = to_numpy(pos_y)
                
                all_counts.extend(counts)
                all_pos_x.extend(pos_x_np)
                all_pos_y.extend(pos_y_np)
                
                # Plot each element with different marker
                markers = ['o', 's', '^', 'D', 'v', '<', '>', 'p', 'h', 'H']
                marker = markers[atom_type % len(markers)]
                
                scatter = ax.scatter(
                    pos_x_np, pos_y_np,
                    c=counts, s=80, alpha=0.8, 
                    marker=marker, label=f'{element}',
                    cmap='viridis', vmin=1, vmax=max(all_counts) if all_counts else 5
                )
        
        # Create colorbar for all elements
        if all_counts and scatter is not None:
            cbar = plt.colorbar(scatter, ax=ax, shrink=0.8)
            cbar.set_label('Number of Atoms', fontsize=14, fontweight='bold')
            # Set integer ticks on colorbar
            max_count = max(all_counts)
            cbar.set_ticks(range(1, max_count + 1))
            
        ax.legend(loc='upper right', bbox_to_anchor=(1.15, 1))
        
    else:
        # Plot specific element
        if element_name not in self.atom_count_estimates:
            raise ValueError(f"No atom count estimates found for {element_name}")
        
        atom_type = list(self.elements).index(element_name)
        mask = self.atom_types == atom_type
        counts = self.atom_count_estimates[element_name]
        
        pos_x = self.params["pos_x"][mask] * self.dx
        pos_y = self.params["pos_y"][mask] * self.dx
        
        pos_x_np = to_numpy(pos_x)
        pos_y_np = to_numpy(pos_y)
        
        scatter = ax.scatter(
            pos_x_np, pos_y_np,
            c=counts, s=100, alpha=0.8, cmap='viridis',
            edgecolors='black', linewidth=0.5
        )
        
        # Create colorbar with proper title
        cbar = plt.colorbar(scatter, ax=ax, shrink=0.8)
        cbar.set_label('Number of Atoms', fontsize=14, fontweight='bold')
        # Set integer ticks on colorbar
        unique_counts = np.unique(counts)
        cbar.set_ticks(unique_counts)
        
        ax.set_title(f'Atom Count Map - {element_name}', fontsize=16, fontweight='bold')
    
    ax.set_xlabel('X (Å)', fontsize=12)
    ax.set_ylabel('Y (Å)', fontsize=12)
    if element_name is None:
        ax.set_title('Spatial Map of Estimated Atom Counts', fontsize=16, fontweight='bold')
    
    ax.set_aspect('equal', adjustable='box')
    ax.invert_yaxis()
    ax.grid(True, alpha=0.3)
    
    # Add summary text
    if hasattr(self, 'gmm_results'):
        summary_info = []
        for elem, results in self.gmm_results.items():
            if 'selected_components' in results:
                selected = results['selected_components']
                recommended = results.get('recommended_components', 'N/A')
                summary_info.append(f"{elem}: {selected} components (rec: {recommended})")
        
        if summary_info:
            summary_text = "GMM Selection: " + ", ".join(summary_info)
            ax.text(0.02, 0.02, summary_text, transform=ax.transAxes,
                   bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
                   fontsize=10, verticalalignment='bottom')
    
    plt.tight_layout()
    
    if save:
        filename = f'atom_count_map_{element_name or "all"}.png'
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        logging.info(f"Atom count map saved as {filename}")
    
    plt.show()

def plot_region(self):
    plt.figure()
    plt.imshow(self.image, cmap="gray")
    plt.imshow(self.regions.region_map, alpha=0.5)
    scalebar = self.scalebar
    plt.gca().add_artist(scalebar)
    plt.axis("off")
    cbar = plt.colorbar()
    cbar.set_ticks(np.arange(self.regions.num_regions))  # type: ignore
    plt.title("Region Map")

# domain analysis

def _plot_domain_analysis(
    self, vacuum_mask, boundary_strength, polygon_data, domain_label
):
    """
    Enhanced plotting with polygon boundaries and region indices.
    """
    fig, axes = plt.subplots(1, 3, figsize=(24, 12))
    
    # Original image
    axes[0].imshow(self.image, cmap='gray')
    axes[0].set_title('Original Image')
    axes[0].axis('off')
    
    # Boundary strength
    im1 = axes[1].imshow(boundary_strength, cmap='viridis')
    axes[1].set_title('Boundary Strength')
    axes[1].axis('off')
    plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)
    
    # Continuous domain separation
    domain_map = domain_label.copy()
    domain_map[vacuum_mask] = -1  # Background
    axes[2].imshow(self.image, cmap='gray')
    im2 = axes[2].imshow(domain_map, vmin=-1, vmax=domain_label.max(),alpha=0.3)
    axes[2].set_title('Domain Map\n(-1=Background, 0=Bulk, >1=Domains)')
    axes[2].axis('off')
    plt.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)
    
    # Polygon boundaries
    if polygon_data:
        for region_id, region_info in polygon_data.items():
            vertices = region_info['vertices']
            axes[2].plot(vertices[:, 1], vertices[:, 0], linewidth=2)
            centroid = region_info['centroid']
            axes[2].text(centroid[1], centroid[0], str(region_id), 
                          color='white', fontsize=8, ha='center', va='center')
    axes[2].set_title('Polygon Boundaries')
    axes[2].axis('off')
    
    plt.tight_layout()
    plt.show()

# Properties



def _bind(cls) -> None:
    """Attach extracted methods back onto Fitter at class-load time."""
    cls._plot_progress = _plot_progress
    cls.plot_coordinates = plot_coordinates
    cls.plot_fitting = plot_fitting
    cls.plot_scs = plot_scs
    cls.plot_scs_voronoi = plot_scs_voronoi
    cls.plot_voronoi_integration_intensity = plot_voronoi_integration_intensity
    cls._plot_gmm_results = _plot_gmm_results
    cls.plot_scs_histogram = plot_scs_histogram
    cls.plot_atom_count_map = plot_atom_count_map
    cls.plot_region = plot_region
    cls._plot_domain_analysis = _plot_domain_analysis


__all__ = [
    "_plot_progress",
    "plot_coordinates",
    "plot_fitting",
    "plot_scs",
    "plot_scs_voronoi",
    "plot_voronoi_integration_intensity",
    "_plot_gmm_results",
    "plot_scs_histogram",
    "plot_atom_count_map",
    "plot_region",
    "_plot_domain_analysis",
    "_bind",
]
