Basic Analysis Tutorial
======================

This tutorial introduces the core QEM workflow using the **ImageFitting** class and **Model** classes for analyzing atomic-resolution STEM images.

Learning Objectives
-------------------

By the end of this tutorial, you will:

- Understand QEM's core **ImageFitting** class
- Use different **Model** types (Gaussian, Lorentzian, Voigt)
- Load and visualize STEM data
- Find atomic column positions automatically
- Fit models to experimental data
- Analyze and interpret results

Core QEM Components
-------------------

**ImageFitting Class**
   The main analysis engine that handles:
   - Peak detection
   - Parameter initialization  
   - Model fitting and optimization
   - Results extraction and visualization

**Model Classes**
   Mathematical models for atomic peaks:
   - ``GaussianModel``: Standard 2D Gaussian peaks
   - ``LorentzianModel``: Lorentzian peak shapes
   - ``VoigtModel``: Convolution of Gaussian and Lorentzian

Prerequisites
-------------

- QEM installed
- Basic Python/NumPy knowledge
- Sample STEM image (or use provided example)

Step 1: Import Libraries and Load Data
--------------------------------------

.. code-block:: python

   import numpy as np
   import matplotlib.pyplot as plt
   from qem.fit.image_fitting import ImageFitting
   from qem import io
   
   # Load example STO data
   image, metadata = io.load_example_data('STO')
   dx = metadata.get('pixel_size', 0.01)  # Angstroms per pixel
   
   print(f"Image shape: {image.shape}")
   print(f"Pixel size: {dx} Å")

Step 2: Visualize the Raw Data
------------------------------

.. code-block:: python

   plt.figure(figsize=(10, 8))
   plt.imshow(image, cmap='gray')
   plt.colorbar(label='Intensity')
   plt.title('Raw STEM Image')
   plt.xlabel('x (pixels)')
   plt.ylabel('y (pixels)')
   plt.show()

Step 3: Initialize ImageFitting
-------------------------------

.. code-block:: python

   # Create ImageFitting instance
   fitter = ImageFitting(
       image=image,
       dx=dx,
       model_type="gaussian"  # Can be 'gaussian', 'lorentzian', or 'voigt'
   )
   
   print(f"Using backend: {fitter.backend}")
   print(f"Image size: {fitter.image.shape}")

Step 4: Find Atomic Column Positions
------------------------------------

.. code-block:: python

   # Automatic peak finding
   coordinates = fitter.find_peaks(
       min_distance=8,      # Minimum distance between peaks (pixels)
       threshold_abs=0.3,   # Absolute intensity threshold
       threshold_rel=0.1    # Relative threshold (fraction of max intensity)
   )
   
   print(f"Found {len(coordinates)} atomic columns")
   
   # Visualize detected peaks
   plt.figure(figsize=(10, 8))
   plt.imshow(image, cmap='gray')
   plt.scatter(coordinates[:, 1], coordinates[:, 0], 
               c='red', s=30, marker='+', linewidth=2)
   plt.title(f'Detected Atomic Columns ({len(coordinates)} peaks)')
   plt.colorbar()
   plt.show()

Step 5: Initialize Model Parameters
-----------------------------------

.. code-block:: python

   # Set coordinates and initialize parameters
   fitter.coordinates = coordinates
   
   # Initialize parameters with reasonable starting values
   params = fitter.init_params(
       atom_size=2.0,       # Initial Gaussian width in pixels
       intensity_guess=1.0,  # Initial intensity
       background=None      # Auto-estimate background
   )
   
   print("Initial parameters set")
   print(f"Number of atomic columns: {fitter.num_coordinates}")

Step 6: Perform the Fit
-----------------------

.. code-block:: python

   # Fit the model
   result = fitter.fit(
       max_iterations=200,   # Maximum optimization steps
       learning_rate=0.01,   # Optimization step size
       tolerance=1e-6,       # Convergence criterion
       verbose=True          # Show progress
   )
   
   print(f"\nFitting completed!")
   print(f"Iterations: {result['iterations']}")
   print(f"Final loss: {result['final_loss']:.8f}")
   print(f"Converged: {result['converged']}")

Step 7: Analyze Results
-----------------------

.. code-block:: python

   # Get fitted parameters
   fitted_positions = fitter.get_positions()
   fitted_intensities = fitter.get_intensities()
   fitted_widths = fitter.get_widths()
   
   print("Summary of fitted parameters:")
   print(f"Position precision: {np.std(fitted_positions, axis=0)} pixels")
   print(f"Intensity range: {np.min(fitted_intensities):.3f} - {np.max(fitted_intensities):.3f}")
   print(f"Width range: {np.min(fitted_widths):.3f} - {np.max(fitted_widths):.3f} pixels")

Step 8: Visualize Results
------------------------

.. code-block:: python

   # Generate model prediction
   prediction = fitter.predict()
   residuals = image - prediction
   
   # Create comprehensive plot
   fig, axes = plt.subplots(2, 3, figsize=(18, 12))
   
   # Original image
   im1 = axes[0, 0].imshow(image, cmap='gray')
   axes[0, 0].set_title('Original Image')
   axes[0, 0].scatter(fitted_positions[:, 1], fitted_positions[:, 0], 
                      c='red', s=20, marker='+')
   plt.colorbar(im1, ax=axes[0, 0])
   
   # Model prediction
   im2 = axes[0, 1].imshow(prediction, cmap='gray')
   axes[0, 1].set_title('Model Fit')
   plt.colorbar(im2, ax=axes[0, 1])
   
   # Residuals
   im3 = axes[0, 2].imshow(residuals, cmap='RdBu_r')
   axes[0, 2].set_title('Residuals (Data - Model)')
   plt.colorbar(im3, ax=axes[0, 2])
   
   # Intensity histogram
   axes[1, 0].hist(fitted_intensities, bins=20, alpha=0.7)
   axes[1, 0].set_xlabel('Fitted Intensity')
   axes[1, 0].set_ylabel('Count')
   axes[1, 0].set_title('Intensity Distribution')
   
   # Width histogram
   axes[1, 1].hist(fitted_widths, bins=20, alpha=0.7)
   axes[1, 1].set_xlabel('Fitted Width (pixels)')
   axes[1, 1].set_ylabel('Count')
   axes[1, 1].set_title('Width Distribution')
   
   # Convergence plot
   if 'loss_history' in result:
       axes[1, 2].plot(result['loss_history'])
       axes[1, 2].set_xlabel('Iteration')
       axes[1, 2].set_ylabel('Loss')
       axes[1, 2].set_title('Convergence')
       axes[1, 2].set_yscale('log')
   
   plt.tight_layout()
   plt.show()

Step 9: Quality Assessment
--------------------------

.. code-block:: python

   # Calculate quality metrics
   r_squared = fitter.calculate_r_squared()
   rmse = fitter.calculate_rmse()
   
   print(f"\nFit Quality Metrics:")
   print(f"R² (coefficient of determination): {r_squared:.4f}")
   print(f"RMSE (root mean square error): {rmse:.6f}")
   
   # Residual statistics
   residual_std = np.std(residuals)
   residual_mean = np.mean(residuals)
   
   print(f"Residual mean: {residual_mean:.6f}")
   print(f"Residual std: {residual_std:.6f}")

Step 10: Save Results
--------------------

.. code-block:: python

   # Save fitted parameters
   results_dict = {
       'positions': fitted_positions,
       'intensities': fitted_intensities,
       'widths': fitted_widths,
       'model_prediction': prediction,
       'residuals': residuals,
       'fit_metrics': {
           'r_squared': r_squared,
           'rmse': rmse,
           'iterations': result['iterations'],
           'final_loss': result['final_loss']
       }
   }
   
   # Save to file (optional)
   # np.save('analysis_results.npy', results_dict)
   
   print("Analysis complete!")

Advanced Tips
-------------

**Improving Peak Detection:**

.. code-block:: python

   # Use preprocessing for better peak detection
   from qem.processing import butterworth_filter
   
   filtered_image = butterworth_filter(image, high_cutoff=0.8)
   fitter_filtered = ImageFitting(filtered_image, dx, "gaussian")
   coordinates_improved = fitter_filtered.find_peaks(min_distance=8)

**Manual Peak Selection:**

.. code-block:: python

   # Interactive peak selection (in Jupyter)
   from qem.select import InteractivePlot
   
   interactive = InteractivePlot(image)
   manual_coordinates = interactive.get_coordinates()

**Different Model Types:**

.. code-block:: python

   # Try different fitting models
   for model_type in ['gaussian', 'lorentzian', 'voigt']:
       fitter_test = ImageFitting(image, dx, model_type)
       fitter_test.coordinates = coordinates
       fitter_test.init_params(atom_size=2.0)
       result_test = fitter_test.fit(max_iterations=100)
       print(f"{model_type}: R² = {fitter_test.calculate_r_squared():.4f}")

Next Steps
----------

- Try :doc:`advanced_fitting` for optimization techniques
- Learn about :doc:`multi_element` analysis
- Explore :doc:`strain_analysis` for displacement mapping

Common Issues
-------------

**Poor Peak Detection:**
- Adjust ``threshold_abs`` and ``threshold_rel`` parameters
- Try image preprocessing (filtering, denoising)
- Check image contrast and quality

**Fitting Not Converging:**
- Increase ``max_iterations``
- Adjust ``learning_rate`` (try 0.001 to 0.1)
- Check initial parameter estimates
- Ensure adequate peak separation

**Memory Issues:**
- Use smaller image regions for testing
- Consider using JAX backend for large images
- Process in batches for very large datasets