from setuptools import setup, find_packages

setup(
    name="QEM",
    version="0.0.1",
    author="EMAT, University of Antwerp",
    author_email="zezhong.zhang@uantwerpen.be",
    description="Package for quantification of STEM data",
    packages=find_packages(exclude=["tests", "tests.*"]),
    install_requires=[
        "pyparsing==2.4.7",
        "numpy>=1.11.1",
        "matplotlib>=1.5.1",
        "Pillow>=8.0",
        "scipy>=1.5",
        "jupyter>=1.0",
        "h5py>=2.0",
        "tqdm>=4.19.4",
        "scikit-image>=0.14.2",
        "optax",
        "numba",
        "shapely",
        "matscipy",
        "ase",
        "hyperspy",
        "jaxopt",
        "optax",
    ],
    python_requires=">=3.6",
    extras_require={
        "dev": [
            "build",
            "black",
            "pytest",
            "pytest-cov"
            "sphinx",
            "sphinx_rtd_theme",
            "myst-parser",
            "jupyterlab",
        ],
    },
    tests_require=["pytest", "pytest-cov"],
    test_suite="tests",
)
