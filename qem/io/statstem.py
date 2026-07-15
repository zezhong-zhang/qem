from os.path import exists as file_exists

import scipy.io as sio


def read_statstem(filename):
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
    >>> from pyStatSTEM.io import read_statstem
    >>> legacyData = read_statstem('Examples/Example_PtIr.mat')
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
