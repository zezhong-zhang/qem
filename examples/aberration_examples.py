"""Example: Using Aberrations with ADF and Ptychography Fitting

This example demonstrates how to properly use aberrations with the new
CTF/PSF calculation system, which integrates tightly with the existing
probe.py infrastructure.
"""

import numpy as np
import matplotlib.pyplot as plt

from qem.instruments import (
    SSB_CTF,
    ADF_CTF,
    create_aberration_list,
    aberration_starter_pack,
    demonstrate_aberration_effects,
)
from qem.fit import PtychographyOptimizer


def example_1_basic_aberrations():
    """Example 1: Basic aberration usage with SSB CTF."""
    print("\n" + "="*60)
    print("Example 1: Basic Aberration Usage with SSB CTF")
    print("="*60)

    # Create aberrations with common parameters
    aberrations = create_aberration_list(
        defocus=50,              # 50 Angstrom defocus
        spherical_aberration=0.5e7,  # 0.5 mm Cs (typical for uncorrected STEM)
    )

    # Create SSB CTF with aberrations
    ctf = SSB_CTF(
        alpha=20,              # 20 mrad convergence angle
        eV=60e3,               # 60 kV
        df=50,                 # Additional defocus parameter
        aberrations=aberrations,
    )

    # Get PSF
    psf = ctf.get_psf((64, 64), (64, 64))

    print(f"Created SSB CTF with {len(aberrations)} aberrations")
    print(f"PSF shape: {psf.shape}")
    print(f"PSF range: [{np.min(psf):.4f}, {np.max(psf):.4f}]")
    print(f"PSF has negative halo: {np.min(psf) < 0}")


def example_2_multiple_aberrations():
    """Example 2: Multiple aberrations including astigmatism and coma."""
    print("\n" + "="*60)
    print("Example 2: Multiple Aberrations")
    print("="*60)

    # Create aberrations with multiple effects
    aberrations = create_aberration_list(
        defocus=50,                    # 50 A defocus
        two_fold_astigmatism=15,       # 15 A 2-fold astigmatism
        two_fold_angle=np.pi/4,        # at 45 degrees
        three_fold_astigmatism=10,     # 10 A 3-fold astigmatism
        three_fold_angle=0,            # at 0 degrees
        coma=500,                      # 500 A coma
        coma_angle=np.pi/3,            # at 60 degrees
    )

    print(f"Created {len(aberrations)} aberrations:")
    for ab in aberrations:
        print(f"  - {ab.Description}: {ab.amplitude:.1f} A")

    # Create CTF
    ctf = SSB_CTF(alpha=20, eV=60e3, aberrations=aberrations)
    psf = ctf.get_psf((64, 64), (64, 64))

    print(f"\nPSF range: [{np.min(psf):.4f}, {np.max(psf):.4f}]")


def example_3_full_aberration_set():
    """Example 3: Using the full aberration starter pack."""
    print("\n" + "="*60)
    print("Example 3: Full Aberration Starter Pack")
    print("="*60)

    # Get the full aberration set (up to 5th order)
    aberrations = aberration_starter_pack()

    # Set some typical non-zero values for a microscope
    # that hasn't been fully corrected
    for ab in aberrations:
        if ab.Krivanek == "C10":  # Defocus
            ab.amplitude = 50
        elif ab.Krivanek == "C12":  # 2-fold astigmatism
            ab.amplitude = 20
            ab.angle = np.pi/6  # 30 degrees
        elif ab.Krivanek == "C30":  # 3rd order spherical
            ab.amplitude = 1e7  # 1 mm Cs
        elif ab.Krivanek == "C23":  # 3-fold astigmatism
            ab.amplitude = 10
            ab.angle = 0

    print(f"Full aberration set contains {len(aberrations)} aberrations")

    # Count non-zero aberrations
    non_zero = [ab for ab in aberrations if ab.amplitude != 0]
    print(f"Non-zero aberrations: {len(non_zero)}")
    for ab in non_zero:
        if ab.m > 0:
            print(f"  - {ab.Description}: {ab.amplitude:.1f} A at {np.rad2deg(ab.angle):.1f}°")
        else:
            print(f"  - {ab.Description}: {ab.amplitude:.1f} A")

    # Create CTF
    ctf = SSB_CTF(alpha=20, eV=60e3, aberrations=aberrations)
    psf = ctf.get_psf((128, 128), (128, 128))

    print(f"\nPSF shape: {psf.shape}")
    print(f"PSF range: [{np.min(psf):.4f}, {np.max(psf):.4f}]")

    return ctf, psf


def example_4_adf_with_aberrations():
    """Example 4: ADF CTF with aberrations."""
    print("\n" + "="*60)
    print("Example 4: ADF CTF with Aberrations")
    print("="*60)

    # ADF also supports aberrations
    aberrations = create_aberration_list(
        defocus=30,
        spherical_aberration=0.3e7,
    )

    ctf = ADF_CTF(
        alpha=20,
        eV=60e3,
        detector_inner=50,    # mrad
        detector_outer=200,   # mrad
        df=30,
        aberrations=aberrations,
    )

    psf = ctf.get_psf((64, 64), (64, 64))

    print(f"ADF CTF with aberrations")
    print(f"PSF range: [{np.min(psf):.4f}, {np.max(psf):.4f}]")
    print(f"PSF is all positive: {np.min(psf) >= 0}")


def example_5_optimization_with_aberrations():
    """Example 5: Ptychography optimization with aberrations."""
    print("\n" + "="*60)
    print("Example 5: Optimization with Aberrations")
    print("="*60)

    # Create synthetic SSB image (in practice, load from experiment)
    ny, nx = 64, 64
    y, x = np.indices((ny, nx))

    # Create a simple test image with two atoms
    test_image = np.zeros((ny, nx))
    test_image[30:34, 30:34] = 1.0  # Atom 1
    test_image[30:34, 34:38] = 0.8  # Atom 2

    # Add some noise
    test_image += np.random.normal(0, 0.01, (ny, nx))

    # Create aberrations
    aberrations = create_aberration_list(
        defocus=20,
        two_fold_astigmatism=5,
        two_fold_angle=np.pi/4,
    )

    # Initialize optimizer with aberrations
    optimizer = PtychographyOptimizer(
        target_image=test_image,
        ctf_type='SSB',
        alpha=20,
        eV=60e3,
        aberrations=aberrations,  # Pass aberrations here
    )

    # Initial positions (from peak finding)
    initial_positions = np.array([[32, 32], [36, 32]])
    initial_phases = np.ones(2)

    # Run optimization
    result = optimizer.optimize(
        initial_positions=initial_positions,
        initial_phases=initial_phases,
        max_iterations=50,
        step_size=0.01,
        verbose=False,
    )

    print(f"Optimization completed")
    print(f"Final correlation: {result.correlation:.4f}")
    print(f"Optimized positions:\n{result.positions}")
    print(f"Optimized phases: {result.phases}")


def example_6_visualize_aberration_effects():
    """Example 6: Visualize aberration effects."""
    print("\n" + "="*60)
    print("Example 6: Visualizing Aberration Effects")
    print("="*60)

    # Use the built-in demonstration function
    fig = demonstrate_aberration_effects()

    print("Created aberration effects visualization")
    print("Saving to aberration_effects.png...")
    fig.savefig('aberration_effects.png', dpi=150, bbox_inches='tight')
    print("Done!")

    # Also create a comparison plot
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Parameters
    alpha = 20
    eV = 60e3

    # No aberrations
    ctf1 = SSB_CTF(alpha, eV)
    psf1 = ctf1.get_psf((64, 64), (64, 64))
    im1 = axes[0].imshow(psf1, cmap='RdBu')
    axes[0].set_title('No Aberrations')
    plt.colorbar(im1, ax=axes[0])

    # With defocus
    aberrations2 = create_aberration_list(defocus=50)
    ctf2 = SSB_CTF(alpha, eV, aberrations=aberrations2)
    psf2 = ctf2.get_psf((64, 64), (64, 64))
    im2 = axes[1].imshow(psf2, cmap='RdBu')
    axes[1].set_title('With Defocus (50 Å)')
    plt.colorbar(im2, ax=axes[1])

    # With defocus + Cs
    aberrations3 = create_aberration_list(
        defocus=50,
        spherical_aberration=1e7,  # 1 mm
    )
    ctf3 = SSB_CTF(alpha, eV, aberrations=aberrations3)
    psf3 = ctf3.get_psf((64, 64), (64, 64))
    im3 = axes[2].imshow(psf3, cmap='RdBu')
    axes[2].set_title('With Defocus + Cs (1 mm)')
    plt.colorbar(im3, ax=axes[2])

    plt.suptitle('Effect of Aberrations on SSB PSF', fontsize=14)
    plt.tight_layout()
    plt.savefig('aberration_comparison.png', dpi=150, bbox_inches='tight')
    print("Also saved aberration_comparison.png")


def example_7_practical_workflow():
    """Example 7: Practical workflow for experimental data."""
    print("\n" + "="*60)
    print("Example 7: Practical Workflow for Experimental Data")
    print("="*60)

    print("""
Practical workflow for analyzing experimental ptychography data:

1. Characterize microscope aberrations:
   - Use known standards (gold nanoparticles, amorphous carbon)
   - Fit aberration parameters from known structures
   - Or use manufacturer specifications

2. Create aberration list:
   ```python
   from qem.instruments import create_aberration_list

   # Example: Typical partially-corrected microscope
   aberrations = create_aberration_list(
       defocus=30,              # Calibrated defocus
       spherical_aberration=0.2e7,  # 0.2 mm residual Cs
       two_fold_astigmatism=8,    # Residual 2-fold astig
       two_fold_angle=np.pi/5,   # Measured angle
   )
   ```

3. Initialize optimizer with aberrations:
   ```python
   from qem.fit import PtychographyOptimizer

   optimizer = PtychographyOptimizer(
       target_image=experimental_ssb,
       ctf_type='SSB',
       alpha=20,          # From microscope settings
       eV=60e3,           # Acceleration voltage
       aberrations=aberrations,  # From characterization
   )
   ```

4. Run optimization:
   ```python
   result = optimizer.optimize(
       initial_positions=peak_positions,
       initial_phases=np.ones(n_atoms),
       optimize_tilt=True,  # Account for sample mistilt
       max_iterations=100,
   )
   ```

5. Validate results:
   - Check correlation coefficient (>0.9 is good)
   - Examine residual image
   - Compare with known structures

6. Refine if needed:
   - Adjust aberration parameters if fit is poor
   - Try different CTF types (SSB, ePIE, iCoM)
   - Check for sample drift or damage
    """)


def main():
    """Run all examples."""
    print("\n" + "="*70)
    print(" ADF AND PTYCHOGRAPHY FITTING - ABERRATION EXAMPLES")
    print("="*70)

    # Run examples
    example_1_basic_aberrations()
    example_2_multiple_aberrations()
    example_3_full_aberration_set()
    example_4_adf_with_aberrations()
    example_5_optimization_with_aberrations()
    example_6_visualize_aberration_effects()
    example_7_practical_workflow()

    print("\n" + "="*70)
    print(" All examples completed successfully!")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
