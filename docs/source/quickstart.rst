Quick Start Guide
=================

This guide will get you up and running with QEM's core functionality in just a few minutes.

Core Concepts
-------------

QEM's analysis workflow centers around two main components:

1. **ImageFitting**: The primary analysis class for processing STEM images
2. **Models**: Mathematical models (Gaussian, Lorentzian, Voigt) for fitting atomic peaks

Basic Usage
-----------

Import Core Classes
^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   import numpy as np
   import matplotlib.pyplot as plt
   
   # Import the main analysis class
   from qem.fit.image_fitting import ImageFitting
   
   # Models are automatically selected, but you can import specific ones
   from qem.fit.model import GaussianModel, LorentzianModel, VoigtModel

Load and Analyze an Image
^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   # Load your STEM image (replace with your data)
   # For this example, we'll create synthetic data
   size = 100
   image = np.random.random((size, size)) + 0.1
   
   # Add some Gaussian peaks to simulate atomic columns
   from qem.fit.model import gaussian_2d_single
   x, y = np.meshgrid(np.arange(size), np.arange(size))
   
   # Add peaks at specific positions
   peaks = [(25, 25), (75, 25), (50, 75)]
   for px, py in peaks:
       # Args: xy meshgrid, pos_x, pos_y, height, width, background
       image += gaussian_2d_single((x, y), px, py, 1.0, 2.0, 0.0).reshape(size, size)

Initialize Image Fitting
^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   # Create ImageFitting instance
   fitter = ImageFitting(
       image=image,
       dx=0.1,  # pixel size in Angstroms
       model_type="gaussian"
   )

Find Atomic Columns
^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   # Automatically find peaks
   coordinates = fitter.find_peaks(
       min_distance=10,
       threshold_abs=0.3
   )
   
   print(f"Found {len(coordinates)} atomic columns")

Fit the Model
^^^^^^^^^^^^^

.. code-block:: python

   # Set coordinates and initialize parameters
   fitter.coordinates = coordinates
   params = fitter.init_params(atom_size=2.0)
   
   # Perform the fit
   result = fitter.fit(
       max_iterations=100,
       learning_rate=0.01
   )
   
   print(f"Fit completed in {result['iterations']} iterations")
   print(f"Final loss: {result['final_loss']:.6f}")

Visualize Results
^^^^^^^^^^^^^^^^^

.. code-block:: python

   # Plot original image and fit
   fig, axes = plt.subplots(1, 3, figsize=(15, 5))
   
   # Original image
   axes[0].imshow(image, cmap='gray')
   axes[0].set_title('Original Image')
   axes[0].scatter(coordinates[:, 1], coordinates[:, 0], 
                   c='red', s=20, marker='+')
   
   # Model prediction
   prediction = fitter.predict()
   axes[1].imshow(prediction, cmap='gray')
   axes[1].set_title('Model Fit')
   
   # Residuals
   residuals = image - prediction
   axes[2].imshow(residuals, cmap='RdBu_r')
   axes[2].set_title('Residuals')
   
   plt.tight_layout()
   plt.show()

Extract Results
^^^^^^^^^^^^^^^

.. code-block:: python

   # Get fitted parameters
   fitted_positions = fitter.get_positions()
   fitted_intensities = fitter.get_intensities()
   fitted_widths = fitter.get_widths()
   
   print("Fitted atomic positions:")
   for i, (x, y) in enumerate(fitted_positions):
       print(f"Atom {i+1}: ({x:.2f}, {y:.2f}) Å")

Working with Real Data
----------------------

Loading STEM Data
^^^^^^^^^^^^^^^^^^

QEM supports various microscopy data formats through HyperSpy:

.. code-block:: python

   import hyperspy.api as hs
   
   # Load your STEM data
   signal = hs.load('your_stem_data.dm3')  # or .hspy, .msa, etc.
   image = signal.data
   
   # Get pixel size
   dx = signal.axes_manager[0].scale  # in units from metadata

Advanced Features
-----------------

Multi-element Analysis
^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   # For multi-element samples
   fitter = ImageFitting(
       image=image,
       dx=0.1,
       model_type="gaussian"
   )
   
   # Set different atom types
   fitter.set_atom_types(['Sr', 'Ti', 'O'])
   
   # Assign atom types to positions
   fitter.assign_atom_types_by_intensity()

Strain Analysis
^^^^^^^^^^^^^^^

.. code-block:: python

   # Calculate strain maps
   strain_maps = fitter.calculate_strain(
       reference_lattice='cubic',
       lattice_parameter=3.9  # Angstroms
   )

Next Steps
----------

- Explore the :doc:`tutorials/index` for detailed workflows
- Check the :doc:`api/index` for complete function reference
- See :doc:`tutorials/examples` for real-world applications

Common Parameters
-----------------

**ImageFitting Parameters:**

- ``dx``: Pixel size in Angstroms
- ``model_type``: 'gaussian', 'lorentzian', or 'voigt'
- ``backend``: 'jax', 'tensorflow', or 'torch' (auto-detected)

**Fitting Parameters:**

- ``learning_rate``: Step size for optimization (0.001-0.1)
- ``max_iterations``: Maximum fitting iterations (50-1000)
- ``tolerance``: Convergence criterion (1e-6 to 1e-3)

**Peak Finding Parameters:**

- ``min_distance``: Minimum separation between peaks (pixels)
- ``threshold_abs``: Absolute intensity threshold
- ``threshold_rel``: Relative intensity threshold (0-1)