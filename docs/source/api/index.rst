API Reference
=============

This section provides detailed documentation for all QEM modules, classes, and functions.

Core Analysis Classes
---------------------

These are the main classes you'll use for electron microscopy analysis:

ImageFitting
^^^^^^^^^^^^

The primary class for analyzing STEM images and fitting atomic models.

.. autoclass:: qem.image_fitting.ImageFitting
   :members:
   :undoc-members:
   :show-inheritance:

Model Classes
^^^^^^^^^^^^^

Core model implementations for different peak shapes:

.. autoclass:: qem.model.ImageModel
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: qem.model.GaussianModel
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: qem.model.LorentzianModel
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: qem.model.VoigtModel
   :members:
   :undoc-members:
   :show-inheritance:

Essential Modules
-----------------

Core functionality modules:

.. autosummary::
   :toctree: generated/
   :nosignatures:

   qem.image_fitting
   qem.model

Analysis and Processing Modules
-------------------------------

Specialized analysis tools:

.. autosummary::
   :toctree: generated/
   :nosignatures:

   qem.processing
   qem.refine
   qem.voronoi
   qem.region
   qem.stats
   qem.atomic_column
   qem.crystal_analyzer

Utility Modules
---------------

Supporting utilities and tools:

.. autosummary::
   :toctree: generated/
   :nosignatures:

   qem.backend
   qem.utils
   qem.io
   qem.periodic_table
   qem.color
   qem.zoom

Specialized Modules
-------------------

Advanced features and specialized functionality:

.. autosummary::
   :toctree: generated/
   :nosignatures:

   qem.probe
   qem.detector
   qem.abberration
   qem.gui_classes

Key Functions
-------------

Backend Utilities
^^^^^^^^^^^^^^^^^

Backend detection and configuration utilities for JAX/TensorFlow/PyTorch support.

.. automodule:: qem.backend
   :members:
   :undoc-members:
   :show-inheritance:

Utility Functions
^^^^^^^^^^^^^^^^^

General utility functions for data processing and analysis.

.. automodule:: qem.utils
   :members:
   :undoc-members:
   :show-inheritance:

Processing Functions
^^^^^^^^^^^^^^^^^^^^

Image processing and filtering functions.

.. automodule:: qem.processing
   :members:
   :undoc-members:
   :show-inheritance: