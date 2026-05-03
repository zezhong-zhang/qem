# Example Usage: ADF and Ptychography Fitting Methods

This document demonstrates how to use the new ADF convolution-based fitting and ptychography phase quantification methods implemented in QEM.

## Table of Contents
1. [SSB Ptychography Phase Quantification](#ssb-ptychography-phase-quantification)
2. [ADF Convolution Fitting](#adf-convolution-fitting)
3. [Sample Tilt Optimization](#sample-tilt-optimization)
4. [Working with Different CTF Types](#working-with-different-ctf-types)
5. [Complete Workflow Example](#complete-workflow-example)

---

## SSB Ptychography Phase Quantification

This example shows how to quantify phases from an SSB ptychography reconstruction.

### Basic Usage

```python
import numpy as np
from qem.fit import PtychographyOptimizer

# Load experimental SSB ptychography image
# Shape: (ny, nx) in pixels
ssb_image = np.load('ssb_reconstruction.npy')

# Initialize optimizer with microscope parameters
optimizer = PtychographyOptimizer(
    target_image=ssb_image,
    ctf_type='SSB',
    alpha=20,      # 20 mrad convergence angle
    eV=60e3,       # 60 kV acceleration voltage
    df=0,          # Defocus in Angstroms
)

# Initial atomic model (from peak finding or reference structure)
# For example, from WS2 with known lattice:
n_atoms = 10
initial_positions = np.array([
    [10.5, 20.3],  # W atom 1
    [15.2, 25.1],  # S atom 1
    # ... more positions
])  # Shape: (N, 2) in pixels

# Initial phase estimates (all 1.0 for start)
initial_phases = np.ones(n_atoms)

# Run optimization
result = optimizer.optimize(
    initial_positions=initial_positions,
    initial_phases=initial_phases,
    max_iterations=100,
    step_size=0.01,
    optimizer='adam',  # Options: 'adam', 'adamw', 'sgd'
    verbose=True,
)

# Results
print(f"Final correlation: {result.correlation:.4f}")
print(f"Final NRMSE: {result.nrmse:.6f}")
print(f"Optimized phases: {result.phases}")
print(f"Number of iterations: {result.n_iterations}")
print(f"Converged: {result.converged}")

# Access optimized atomic positions
optimized_positions = result.positions  # Shape: (N, 2)
```

### Optimization with Sample Tilt

```python
# Enable tilt optimization to account for sample mistilt
result = optimizer.optimize(
    initial_positions=initial_positions,
    initial_phases=initial_phases,
    optimize_tilt=True,
    max_iterations=150,
    step_size=0.005,
    optimizer='adam',
)

print(f"Sample tilt: {result.tilt_x:.2f}, {result.tilt_y:.2f} mrad")
```

### Optimization with PSF Width Scaling

```python
# Enable PSF width optimization (accounts for convergence angle variations)
result = optimizer.optimize(
    initial_positions=initial_positions,
    initial_phases=initial_phases,
    optimize_psf_width=True,
    max_iterations=150,
    step_size=0.005,
)

print(f"Optimized PSF width: {result.psf_width:.2f} pixels")
```

### Visualization

```python
import matplotlib.pyplot as plt

# Plot optimization history
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

# Correlation progress
axes[0].plot(result.history['correlation'])
axes[0].set_xlabel('Iteration')
axes[0].set_ylabel('Correlation Coefficient')
axes[0].set_title('Optimization Progress')
axes[0].grid(True)

# NRMSE progress
axes[1].plot(result.history['nrmse'])
axes[1].set_xlabel('Iteration')
axes[1].set_ylabel('NRMSE')
axes[1].set_title('Error Reduction')
axes[1].grid(True)

plt.tight_layout()
plt.show()

# Compare original and optimized
from qem.fit.point_potential import PointPotentialModel

potential_model = PointPotentialModel()
simulated = potential_model.simulate_from_positions(
    result.positions[:, 0],
    result.positions[:, 1],
    result.phases,
    optimizer.psf,
    ssb_image.shape,
)

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
im1 = axes[0].imshow(ssb_image, cmap='viridis')
axes[0].set_title('Experimental SSB')
plt.colorbar(im1, ax=axes[0])

im2 = axes[1].imshow(simulated, cmap='viridis')
axes[1].set_title('Simulated from Fit')
plt.colorbar(im2, ax=axes[1])

im3 = axes[2].imshow(ssb_image - simulated, cmap='RdBu')
axes[2].set_title('Residual')
plt.colorbar(im3, ax=axes[2])

plt.tight_layout()
plt.show()
```

---

## ADF Convolution Fitting

This example shows how to fit ADF images using a convolution model.

### Basic Usage

```python
from qem.fit import ADFConvolutionFitting

# Load ADF image
adf_image = np.load('adf_image.npy')

# Initialize ADF fitter with microscope parameters
fitter = ADFConvolutionFitting(
    image=adf_image,
    eV=60e3,                    # 60 kV
    alpha=20,                    # 20 mrad convergence angle
    detector_inner=50,           # 50 mrad inner detector angle
    detector_outer=200,          # 200 mrad outer detector angle
    df=0,                       # Defocus
)

# Initial atomic model
initial_positions = np.array([
    [10.5, 20.3],
    [15.2, 25.1],
    # ... more positions
])

# Initial intensity estimates
initial_intensities = np.ones(len(initial_positions))

# Fit the image
result = fitter.fit(
    initial_positions=initial_positions,
    initial_intensities=initial_intensities,
    max_iterations=100,
    step_size=0.01,
    optimizer='adam',
)

# Results
print(f"Final correlation: {result.correlation:.4f}")
print(f"Optimized intensities: {result.phases}")  # Called 'phases' but these are intensities
```

---

## Sample Tilt Optimization

The `SampleTilt` class provides utilities for working with sample tilt.

### Apply Tilt to Positions

```python
from qem.instruments import SampleTilt

# Atomic positions (in pixels or Angstroms)
positions = np.array([
    [10.0, 20.0],
    [15.0, 25.0],
    [20.0, 30.0],
])

# Apply 5 mrad tilt around x-axis and 3 mrad around y-axis
tilted_positions = SampleTilt.apply_tilt(
    positions,
    tilt_x=5.0,   # mrad
    tilt_y=3.0,   # mrad
    thickness=6.0,  # Angstroms (sample thickness)
)

print("Original positions:", positions)
print("Tilted positions:", tilted_positions)
```

### Find Tilt from Reference Structure

```python
# Reference positions (untilted structure)
reference_positions = np.array([
    [10.0, 20.0],
    [15.0, 25.0],
    [20.0, 30.0],
])

# Observed positions (from experiment, possibly tilted)
observed_positions = reference_positions + np.array([0.5, -0.3])  # Shifted

# Estimate the tilt that best explains the shift
tilt_x, tilt_y, residual = SampleTilt.optimize_tilt(
    positions_ref=reference_positions,
    positions_observed=observed_positions,
    thickness=6.0,
)

print(f"Estimated tilt: {tilt_x:.2f}, {tilt_y:.2f} mrad")
print(f"Residual error: {residual:.6f}")
```

---

## Working with Different CTF Types

The optimizer supports multiple imaging modes.

### SSB Ptychography

```python
optimizer = PtychographyOptimizer(
    target_image=ssb_image,
    ctf_type='SSB',
    alpha=20,
    eV=60e3,
)
```

### ePIE Ptychography

```python
optimizer = PtychographyOptimizer(
    target_image=epie_image,
    ctf_type='ePIE',
    alpha=20,
    eV=60e3,
    defocus=1.0,  # Additional ePIE parameter
)
```

### iCoM Imaging

```python
# Without filter
optimizer = PtychographyOptimizer(
    target_image=icom_image,
    ctf_type='iCoM',
    alpha=20,
    eV=60e3,
)

# With high-pass filter
optimizer = PtychographyOptimizer(
    target_image=icom_image,
    ctf_type='iCoM',
    alpha=20,
    eV=60e3,
    high_pass_cutoff=5.0,  # mrad
)
```

### Custom PSF from Simulation

```python
# Use a PSF extracted from a single atom simulation
psf_from_simulation = np.load('single_atom_psf.npy')

optimizer = PtychographyOptimizer(
    target_image=experimental_image,
    ctf_type='SSB',
    alpha=20,
    eV=60e3,
    psf_kernel=psf_from_simulation,  # Override CTF calculation
)
```

---

## Complete Workflow Example

This example demonstrates a complete analysis workflow.

```python
import numpy as np
import matplotlib.pyplot as plt
from qem.fit import PtychographyOptimizer
from qem.instruments import SampleTilt
from skimage.feature import peak_local_max

# 1. Load experimental data
ssb_image = np.load('experimental_ssb.npy')
print(f"Image shape: {ssb_image.shape}")

# 2. Find initial atomic positions using peak detection
coordinates = peak_local_max(
    ssb_image,
    min_distance=5,  # Minimum distance between peaks (pixels)
    threshold_abs=np.percentile(ssb_image, 95),  # Top 5% intensity
)
print(f"Found {len(coordinates)} atomic sites")

# 3. Initialize optimizer
optimizer = PtychographyOptimizer(
    target_image=ssb_image,
    ctf_type='SSB',
    alpha=20,
    eV=60e3,
    df=0,
)

# 4. Set initial parameters
initial_positions = coordinates[:, [1, 0]]  # Swap to [x, y] order
initial_phases = np.ones(len(initial_positions))

# 5. Run optimization with tilt and PSF width optimization
result = optimizer.optimize(
    initial_positions=initial_positions,
    initial_phases=initial_phases,
    optimize_tilt=True,
    optimize_psf_width=True,
    max_iterations=200,
    step_size=0.005,
    optimizer='adam',
    verbose=True,
)

# 6. Print results
print("\n" + "="*50)
print("OPTIMIZATION RESULTS")
print("="*50)
print(f"Final correlation: {result.correlation:.4f}")
print(f"Final NRMSE: {result.nrmse:.6f}")
print(f"Converged: {result.converged}")
print(f"Iterations: {result.n_iterations}")
print(f"Sample tilt: {result.tilt_x:.2f}, {result.tilt_y:.2f} mrad")
print(f"PSF width: {result.psf_width:.2f} pixels")
print(f"\nOptimized phases (first 10): {result.phases[:10]}")

# 7. Simulate final image and visualize
from qem.fit.point_potential import PointPotentialModel

potential_model = PointPotentialModel()
simulated = potential_model.simulate_from_positions(
    result.positions[:, 0],
    result.positions[:, 1],
    result.phases,
    optimizer.psf,
    ssb_image.shape,
)

# Create visualization
fig = plt.figure(figsize=(15, 10))
gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

# Original image
ax1 = fig.add_subplot(gs[0, 0])
im1 = ax1.imshow(ssb_image, cmap='viridis')
ax1.set_title('Experimental SSB')
ax1.set_xlabel('x (pixels)')
ax1.set_ylabel('y (pixels)')
plt.colorbar(im1, ax=ax1, label='Phase (rad)')

# Simulated image
ax2 = fig.add_subplot(gs[0, 1])
im2 = ax2.imshow(simulated, cmap='viridis')
ax2.set_title('Simulated from Fit')
ax2.set_xlabel('x (pixels)')
ax2.set_ylabel('y (pixels)')
plt.colorbar(im2, ax=ax2, label='Phase (rad)')

# Residual
ax3 = fig.add_subplot(gs[0, 2])
residual = ssb_image - simulated
im3 = ax3.imshow(residual, cmap='RdBu', vmin=-np.max(np.abs(residual)), vmax=np.max(np.abs(residual)))
ax3.set_title('Residual')
ax3.set_xlabel('x (pixels)')
ax3.set_ylabel('y (pixels)')
plt.colorbar(im3, ax=ax3, label='Phase (rad)')

# Optimization progress
ax4 = fig.add_subplot(gs[1, :])
ax4.plot(result.history['correlation'], label='Correlation')
ax4.set_xlabel('Iteration')
ax4.set_ylabel('Correlation Coefficient')
ax4.set_title('Optimization Progress')
ax4.legend()
ax4.grid(True)

# Phase histogram
ax5 = fig.add_subplot(gs[2, 0])
ax5.hist(result.phases, bins=20, edgecolor='black')
ax5.set_xlabel('Phase (rad)')
ax5.set_ylabel('Count')
ax5.set_title('Phase Distribution')
ax5.grid(True)

# Phase vs position
ax6 = fig.add_subplot(gs[2, 1])
scatter = ax6.scatter(result.positions[:, 0], result.positions[:, 1],
                     c=result.phases, cmap='viridis', s=50)
ax6.set_xlabel('x (pixels)')
ax6.set_ylabel('y (pixels)')
ax6.set_title('Spatial Phase Map')
plt.colorbar(scatter, ax=ax6, label='Phase (rad)')

# Residual histogram
ax7 = fig.add_subplot(gs[2, 2])
ax7.hist(residual.ravel(), bins=50, edgecolor='black')
ax7.set_xlabel('Residual Phase (rad)')
ax7.set_ylabel('Count')
ax7.set_title('Residual Distribution')
ax7.grid(True)

plt.suptitle('SSB Ptychography Phase Quantification', y=0.995, fontsize=16)
plt.show()

# 8. Save results
np.savez(
    'optimization_results.npz',
    positions=result.positions,
    phases=result.phases,
    tilt_x=result.tilt_x,
    tilt_y=result.tilt_y,
    psf_width=result.psf_width,
    correlation=result.correlation,
    nrmse=result.nrmse,
    history=result.history,
)

print("\nResults saved to 'optimization_results.npz'")
```

---

## Advanced Usage

### Custom Optimizer Parameters

```python
# Using AdamW optimizer with custom learning rate
result = optimizer.optimize(
    initial_positions=initial_positions,
    initial_phases=initial_phases,
    optimizer='adamw',
    step_size=0.001,  # Lower learning rate for finer convergence
    max_iterations=300,
    tolerance=1e-7,  # Stricter convergence criterion
)

# Using SGD with momentum (requires Keras 3.x+)
from qem.optimizers.lbfgs import LBFGSOptimizer

# For very precise optimization, use L-BFGS (PyTorch backend only)
if keras.backend.backend() == "torch":
    result = optimizer.optimize(
        initial_positions=initial_positions,
        initial_phases=initial_phases,
        optimizer='lbfgs',
        max_iterations=50,  # L-BFGS needs fewer iterations
        tolerance_grad=1e-7,
        tolerance_change=1e-9,
    )
```

### Working with Aberrations

```python
from qem.instruments import aberration, aberration_starter_pack

# Create aberration list
aberrations = aberration_starter_pack(eV=60e3)

# Add custom aberration if needed
custom_aberration = aberration(
    amplitude=100,  # Angstroms
    angle=0,        # radians
    n=3,           # radial order
    m=0,           # azimuthal order
)
aberrations.append(custom_aberration)

# Use in optimizer
optimizer = PtychographyOptimizer(
    target_image=ssb_image,
    ctf_type='SSB',
    alpha=20,
    eV=60e3,
    df=50,  # 50 Angstrom defocus
    aberrations=aberrations,
)
```

### Batch Processing Multiple Images

```python
# Process multiple regions
results = []
for i, image in enumerate(image_list):
    print(f"Processing region {i+1}/{len(image_list)}")

    optimizer = PtychographyOptimizer(
        target_image=image,
        ctf_type='SSB',
        alpha=20,
        eV=60e3,
    )

    result = optimizer.optimize(
        initial_positions=initial_positions_list[i],
        initial_phases=initial_phases_list[i],
        max_iterations=100,
        verbose=False,
    )

    results.append(result)

# Analyze trends across regions
mean_phase = np.mean([r.phases.mean() for r in results])
std_phase = np.std([r.phases.mean() for r in results])

print(f"Mean phase across regions: {mean_phase:.4f} ± {std_phase:.4f} rad")
```

---

## Troubleshooting

### Optimization Doesn't Converge

**Problem**: Correlation plateaus at a low value.

**Solutions**:
1. Increase `max_iterations`
2. Try different optimizer (`adamw` instead of `adam`)
3. Adjust `step_size` (try 0.001 or 0.02)
4. Check initial positions are reasonable
5. Enable `optimize_tilt` if sample might be tilted

### Negative Phases After Optimization

**Problem**: Some phase values are negative.

**Solutions**:
1. This is expected for SSB ptychography with negative halos
2. The relative values matter more than absolute values
3. Consider using phase differences rather than absolute values
4. Check if the correlation coefficient is reasonable (>0.9)

### Memory Issues

**Problem**: Out of memory errors with large images.

**Solutions**:
1. Crop image to region of interest
2. Downsample image slightly
3. Process in smaller patches
4. Reduce `max_iterations` for initial testing

---

## References

- Hofer & Pennycook, "Reliable phase quantification in focused probe electron ptychography of thin materials", Ultramicroscopy 254 (2023) 113829.
- QEM documentation: https://github.com/your-repo/qem

For more examples and updates, see the QEM repository.
