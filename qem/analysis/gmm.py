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
    """Integrate GMM atom count estimates with crystal analyzer atomic model.
    
    This method combines the statistical atom counting from GMM with the 
    crystal structure analysis to create a 3D atomic model with realistic
    atom counts in each column. Z-spacing is automatically determined from
    the supercell structure.
    
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
        except Exception as e:
            logging.warning(f"Could not update region {region_index}: {str(e)}")
            
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



def _bind(cls) -> None:
    """Attach extracted methods back onto Fitter at class-load time."""
    cls.estimate_atom_counts_with_gmm = estimate_atom_counts_with_gmm
    cls.integrate_gmm_with_crystal_analyzer = integrate_gmm_with_crystal_analyzer
    cls.update_all_regions_with_gmm = update_all_regions_with_gmm
    cls.export_gmm_updated_structure = export_gmm_updated_structure


__all__ = [
    "estimate_atom_counts_with_gmm",
    "integrate_gmm_with_crystal_analyzer",
    "update_all_regions_with_gmm",
    "export_gmm_updated_structure",
    "_bind",
]
