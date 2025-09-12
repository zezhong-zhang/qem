Installation
============

System Requirements
-------------------

QEM requires Python 3.8 or higher and is compatible with:

- **Operating Systems**: Linux, macOS, Windows
- **Python**: 3.8, 3.9, 3.10, 3.11
- **Hardware**: CPU (minimum), GPU (recommended for large datasets)

Dependencies
------------

QEM depends on several scientific Python packages:

**Core Dependencies:**
- numpy >= 1.11.1
- scipy >= 1.5
- matplotlib >= 1.5.1
- scikit-image >= 0.14.2
- keras (latest)

**Backend Support:**
- JAX (recommended for performance)
- TensorFlow
- PyTorch

**Optional Dependencies:**
- numba (for JIT compilation)
- hyperspy (for microscopy data)
- ase (for atomic structures)

Installation Methods
--------------------

Via pip (Recommended)
^^^^^^^^^^^^^^^^^^^^^

The easiest way to install QEM is using pip:

.. code-block:: bash

   pip install qem

This will install QEM and all required dependencies.

Development Installation
^^^^^^^^^^^^^^^^^^^^^^^^

For developers or users who want the latest features:

.. code-block:: bash

   git clone https://github.com/zezhong-zhang/qem.git
   cd qem
   pip install -e .

This installs QEM in "editable" mode, allowing you to modify the source code.

Conda Installation
^^^^^^^^^^^^^^^^^^

If you prefer conda:

.. code-block:: bash

   conda create -n qem python=3.11
   conda activate qem
   pip install qem

Backend Configuration
---------------------

QEM automatically detects and uses the best available backend. You can also specify a backend:

.. code-block:: python

   import os
   os.environ['KERAS_BACKEND'] = 'jax'  # or 'tensorflow', 'torch'
   
   import qem

Verification
------------

To verify your installation:

.. code-block:: python

   import qem
   from qem.utils.backend import detect_available_backends
   
   print(f"QEM version: {qem.__version__}")
   print(f"Available backends: {detect_available_backends()}")

Troubleshooting
---------------

Common Issues
^^^^^^^^^^^^^

**ImportError: No module named 'keras'**
   Install a backend first: ``pip install tensorflow`` or ``pip install torch``

**CUDA/GPU Issues**
   Ensure you have the correct GPU drivers and CUDA toolkit installed for your backend.

**Memory Issues**
   For large datasets, consider using JAX backend with GPU support.

Getting Help
^^^^^^^^^^^^

If you encounter installation issues:

1. Check the `GitHub Issues <https://github.com/zezhong-zhang/qem/issues>`_
2. Create a new issue with your system information
3. Contact the developers: zhangzz@aisi.ac.cn