"""
Example script demonstrating sophisticated peak position estimation 
for complex STO domains with antiphase boundaries and fixed interfaces.

This script shows how to use the new `estimate_initial_peaks_for_complex_domains` method
to separate bulk phase regions from interface regions and apply different peak detection
strategies to each.
"""

import hyperspy.api as hs
import numpy as np
import matplotlib.pyplot as plt
from qem.fit.image_fitting import ImageFitting

def analyze_complex_sto_domain(image_file_path, dx_value=None):
    """
    Analyze a complex STO domain image with antiphase boundaries.
    
    Args:
        image_file_path: Path to the HAADF-STEM image
        dx_value: Pixel size in Angstroms (if None, will be extracted from metadata)
    """
    
    # Load the image
    print(f"Loading image: {image_file_path}")
    s = hs.load(image_file_path)
    image = s.data
    
    # Extract or set pixel size
    if dx_value is None:
        dx = s.axes_manager[1].scale * 10  # convert nm to Angstrom
    else:
        dx = dx_value
    
    print(f"Pixel size: {dx:.3f} Å")
    print(f"Image shape: {image.shape}")
    
    # Initialize the ImageFitting model
    model = ImageFitting(
        image, 
        dx=dx, 
        units='A',
        elements=['Sr', 'Ti']  # STO elements
    )
    
    # Apply the sophisticated domain analysis
    print("\nApplying sophisticated domain analysis...")
    results = model.estimate_initial_peaks_for_complex_domains(
        domain_separation_method="intensity_gradient",  # or "laplacian", "sobel"
        interface_width=3.0,  # Expected interface width in Angstroms
        bulk_detection_sensitivity=0.3,  # Standard sensitivity for bulk regions
        interface_detection_sensitivity=0.1,  # Higher sensitivity for interfaces
        antiphase_detection=True,  # Enable specialized antiphase boundary detection
        min_bulk_region_size=100,  # Minimum pixels for bulk region
        plot_analysis=True,  # Show the analysis plots
        sigma_bulk=3.0,  # Gaussian filter for bulk regions
        sigma_interface=1.5  # Gaussian filter for interface regions
    )
    
    # Print results summary
    print(f"\nResults Summary:")
    print(f"Total peaks detected: {len(results['all_peaks'])}")
    print(f"Bulk peaks: {len(results['bulk_peaks'])}")
    print(f"Interface peaks: {len(results['interface_peaks'])}")
    
    # Analyze peak distribution
    bulk_fraction = len(results['bulk_peaks']) / len(results['all_peaks']) if len(results['all_peaks']) > 0 else 0
    interface_fraction = len(results['interface_peaks']) / len(results['all_peaks']) if len(results['all_peaks']) > 0 else 0
    
    print(f"Bulk peak fraction: {bulk_fraction:.2%}")
    print(f"Interface peak fraction: {interface_fraction:.2%}")
    
    return model, results

def compare_detection_methods(image_file_path, dx_value=None):
    """
    Compare different domain separation methods on the same image.
    """
    s = hs.load(image_file_path)
    image = s.data
    dx = dx_value if dx_value is not None else s.axes_manager[1].scale * 10
    
    methods = ["intensity_gradient", "laplacian", "sobel"]
    results = {}
    
    fig, axes = plt.subplots(2, len(methods), figsize=(15, 8))
    
    for i, method in enumerate(methods):
        print(f"\nTesting method: {method}")
        
        model = ImageFitting(image, dx=dx, units='A', elements=['Sr', 'Ti'])
        
        result = model.estimate_initial_peaks_for_complex_domains(
            domain_separation_method=method,
            interface_width=2.0,
            bulk_detection_sensitivity=0.3,
            interface_detection_sensitivity=0.1,
            antiphase_detection=True,
            min_bulk_region_size=100,
            plot_analysis=False  # Don't show plots for comparison
        )
        
        results[method] = result
        
        # Plot domain separation
        domain_map = result['bulk_mask'].astype(int) + result['interface_mask'].astype(int) * 2
        axes[0, i].imshow(domain_map, cmap='Set1')
        axes[0, i].set_title(f'{method}\nDomain Separation')
        axes[0, i].axis('off')
        
        # Plot detected peaks
        axes[1, i].imshow(image, cmap='gray')
        if len(result['bulk_peaks']) > 0:
            axes[1, i].scatter(result['bulk_peaks'][:, 0], result['bulk_peaks'][:, 1], 
                             c='red', s=5, alpha=0.8, label='Bulk')
        if len(result['interface_peaks']) > 0:
            axes[1, i].scatter(result['interface_peaks'][:, 0], result['interface_peaks'][:, 1], 
                             c='blue', s=5, alpha=0.8, label='Interface')
        axes[1, i].set_title(f'{method}\nPeaks: B={len(result["bulk_peaks"])}, I={len(result["interface_peaks"])}')
        axes[1, i].legend()
        axes[1, i].axis('off')
    
    plt.tight_layout()
    plt.show()
    
    # Print comparison summary
    print("\nMethod Comparison Summary:")
    print("Method                | Bulk Peaks | Interface Peaks | Total | Bulk %")
    print("-" * 70)
    for method, result in results.items():
        bulk_count = len(result['bulk_peaks'])
        interface_count = len(result['interface_peaks'])
        total_count = bulk_count + interface_count
        bulk_pct = bulk_count / total_count * 100 if total_count > 0 else 0
        print(f"{method:20s} | {bulk_count:10d} | {interface_count:15d} | {total_count:5d} | {bulk_pct:5.1f}%")
    
    return results

if __name__ == "__main__":
    # Example usage with the STO dataset
    import os
    
    # Path to your STO image file
    sto_file = 'data/STOonSTO/83-100/aligned_hdf5/1848 HAADF 33.7 nm.emd.hspy'
    
    if os.path.exists(sto_file):
        print("=== Complex STO Domain Analysis ===")
        model, results = analyze_complex_sto_domain(sto_file)
        
        print("\n=== Method Comparison ===")
        comparison_results = compare_detection_methods(sto_file)
        
        # You can now proceed with further analysis using the detected peaks
        print("\n=== Next Steps ===")
        print("1. Use model.init_params() to initialize fitting parameters")
        print("2. Use model.fit_global() or model.fit_stochastic() for refinement")
        print("3. Analyze results with model.plot_scs() and other visualization methods")
        
    else:
        print(f"Image file not found: {sto_file}")
        print("Please update the file path to point to your STO image data.")