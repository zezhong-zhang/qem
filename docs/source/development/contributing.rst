Contributing to QEM
===================

We welcome contributions to QEM! This guide explains how to contribute effectively.

Types of Contributions
----------------------

**Bug Reports**
   Report bugs using GitHub issues with detailed information about the problem.

**Feature Requests**
   Suggest new features or improvements via GitHub issues.

**Code Contributions**
   Submit pull requests for bug fixes, new features, or improvements.

**Documentation**
   Help improve documentation, tutorials, and examples.

**Testing**
   Add tests for existing functionality or help improve test coverage.

Development Workflow
--------------------

1. **Fork and Clone**

.. code-block:: bash

   git clone https://github.com/yourusername/qem.git
   cd qem

2. **Set Up Environment** (``uv`` recommended)

.. code-block:: bash

   # Install uv if you don't have it: https://docs.astral.sh/uv/
   curl -LsSf https://astral.sh/uv/install.sh | sh

   # Create the dev venv. QEM requires Python 3.11+.
   uv venv .venv --python 3.11
   source .venv/bin/activate    # or: .venv\Scripts\activate on Windows

   # Install qem in editable mode with all dev + GUI extras.
   uv pip install -e ".[dev,gui,cv]"

   # Optional: install io / notebook extras as you need them.
   uv pip install -e ".[io,notebook]"

   # Run the test suite to verify the install.
   pytest -q

   # Launch the desktop GUI to check napari + magicgui work.
   qem-app

If you prefer ``pip`` directly:

.. code-block:: bash

   python3.11 -m venv .venv
   source .venv/bin/activate
   pip install -e ".[dev,gui,cv]"

If you still use conda:

.. code-block:: bash

   conda create -n qem-dev python=3.11
   conda activate qem-dev
   pip install -e ".[dev,gui,cv]"

The CI runs ``pytest tests/`` against Python 3.11 and 3.12. ``uv`` is
~10× faster for the heavy installs (PyTorch + napari + magicgui +
matplotlib pulls in ~1.5 GB), which matters for iterating locally.

3. **Create Feature Branch**

.. code-block:: bash

   git checkout -b feature/your-feature-name

4. **Make Changes**
   - Write clean, documented code
   - Add tests for new functionality
   - Update documentation as needed

5. **Run Tests**

.. code-block:: bash

   pytest
   ruff check qem/
   ruff format qem/

6. **Submit Pull Request**
   - Push to your fork
   - Create pull request on GitHub
   - Describe your changes clearly

Code Standards
--------------

**Style Guide**
   - Follow PEP 8
   - Use Ruff for formatting and linting
   - Maximum line length: 160 characters

**Type Hints**
   - Add type hints to all public functions
   - Use `numpy.typing.NDArray` for arrays

**Documentation**
   - Write NumPy-style docstrings
   - Include examples in docstrings
   - Update user guides for new features

**Testing**
   - Write tests for all new functionality
   - Maintain >80% test coverage
   - Use pytest fixtures for setup

Example Contribution
--------------------

Here's an example of a well-structured contribution:

.. code-block:: python

   def new_analysis_function(
       image: NDArray[np.floating],
       threshold: float = 0.5
   ) -> NDArray[np.floating]:
       """
       Perform new type of analysis on STEM image.
       
       Parameters
       ----------
       image : NDArray[np.floating]
           Input STEM image
       threshold : float, default=0.5
           Analysis threshold value
           
       Returns
       -------
       NDArray[np.floating]
           Analysis result
           
       Examples
       --------
       >>> import numpy as np
       >>> image = np.random.random((100, 100))
       >>> result = new_analysis_function(image)
       >>> result.shape
       (100, 100)
       """
       # Implementation here
       return processed_image

Pull Request Guidelines
-----------------------

**Before Submitting**
   - Ensure all tests pass
   - Update documentation
   - Add entry to changelog
   - Rebase on latest main branch

**PR Description**
   - Clear title describing the change
   - Detailed description of what was changed
   - Link to related issues
   - Screenshots for UI changes

**Review Process**
   - Maintainers will review your PR
   - Address feedback promptly
   - Keep discussion focused and constructive

Reporting Issues
----------------

When reporting bugs, please include:

- QEM version
- Python version
- Operating system
- Backend being used (JAX/TensorFlow/PyTorch)
- Minimal code example reproducing the issue
- Error message and traceback

Community Guidelines
--------------------

- Be respectful and inclusive
- Focus on constructive feedback
- Help others learn and improve
- Follow the code of conduct

Getting Help
------------

If you need help contributing:

- Ask questions in GitHub discussions
- Check existing issues and PRs
- Contact maintainers: zhangzz@aisi.ac.cn

Recognition
-----------

All contributors are recognized in:

- AUTHORS file
- GitHub contributors list
- Release notes for significant contributions

Thank you for contributing to QEM!