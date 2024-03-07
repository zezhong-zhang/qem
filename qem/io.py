import scipy.io as sio
from PIL import Image
from os.path import exists as file_exists
import numpy as np


def read_legacyInputStatSTEM(filename):
    """
    This function is used to read legacy StatSTEM input files in from .mat files.

    StatSTEM Input class objects from matlab must be converted to `struct` objects in matlab first.
    (e.g. `StatSTEM_Input = struct(StatSTEM_Input)`))
    This is intended to be a temporary solution until the new pyStatSTEM classes are implemented.
    Parameters
    ----------
    filename : string
        path to .mat file containing StatSTEM input data
    Returns
    -------
    dict
        dictionary containing StatSTEM input data mirroring the legacy matlab class structure
    Raises
    ------
    FileNotFoundError
        Provided filename does not exist
    Examples
    --------
    >>> from pyStatSTEM.io import read_legacyInputStatSTEM
    >>> legacyData = read_legacyInputStatSTEM('Examples/Example_PtIr.mat')
    >>> inputStatSTEM = legacyData["input"]
    >>> plt.imshow(inputStatSTEM['obs'])
    """
    if not file_exists(filename):
        raise FileNotFoundError(f"{filename} not found")

    mat = sio.loadmat(filename, simplify_cells=True)
    datasets = {}
    for key in mat.keys():
        if key[0] != "_":
            datasets[key] = mat[key]
    return datasets


def read_image(filename):
    """
    This function is used to read images in common image file formats like .tif, .png etc.

    Parameters
    ----------
    filename : string
        path to image file
    Returns
    -------
    array_like
        image data as numpy array
    Raises
    ------
    FileNotFoundError
        Provided filename does not exist
    Examples
    --------
    >>> img = pyStatSTEM.io.read_image('Examples/det.tif')
    >>> plt.imshow(img)
    >>> plt.show()
    """
    if not file_exists(filename):
        raise FileNotFoundError(f"{filename} not found")

    im = Image.open(filename)
    data = np.asarray(im)
    if data.ndim == 3:
        data = np.mean(data, axis=2)

    return data


def read_delimited_text(filename, delimiter=None):
    """
    This function is used to read delimited text files.

    Parameters
    ----------
    filename : string
        path to delimited text file
    delimiter : string
        delimiter used in file
    Returns
    -------
    array_like
        delimited text data as numpy array
    Raises
    ------
    FileNotFoundError
        Provided filename does not exist
    Examples
    --------
    >>> data = pyStatSTEM.io.read_delimited_text('Examples/det.txt')
    >>> plt.imshow(img)
    >>> plt.show()
    """
    if not file_exists(filename):
        raise FileNotFoundError(f"{filename} not found")

    with open(filename, "r") as f:
        data = np.loadtxt(f, delimiter=delimiter)
    return data
