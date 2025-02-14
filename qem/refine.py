import matplotlib.pyplot as plt
import numpy as np
from numba import jit
from scipy.optimize import curve_fit


def calculate_center_of_mass(arr):
    """Find the center of mass of a NumPy array.

    Find the center of mass of a single 2D array, or a
    list (or multidimensional array) of 2D arrays.

    Parameters
    ----------
    arr : Numpy 2D Array (or list/nd-array of such)

    Returns
    -------
    cy, cx: array of floats (or nd-array of floats)
        Giving centre coordinates with sub-pixel accuracy

    Examples
    --------
    >>> import atomap.atom_finding_refining as afr
    >>> arr = np.random.randint(100, size=(10, 10))
    >>> data = afr.calculate_center_of_mass(arr)

    Notes
    -----
    This is a much simpler center of mass approach than the one from scipy.
    Gotten from stackoverflow:
    https://stackoverflow.com/questions/37519238/python-find-center-of-object-in-an-image

    """
    # Can consider subtracting minimum value
    # this gives the center of mass higher "contrast"
    # arr -= arr.min()
    if len(arr.shape) > 2:
        arr = (arr.T / np.sum(arr, axis=(-1, -2)).T).T
    else:
        arr = arr / np.sum(arr, axis=(-1, -2))

    dy = np.sum(arr, -1)
    dx = np.sum(arr, -2)

    (Y, X) = arr.shape[-2:]
    cx = np.sum(dx * np.arange(X), axis=-1).T
    cy = np.sum(dy * np.arange(Y), axis=-1).T
    return np.array([cy, cx]).T


@jit(nopython=True)
def gauss2d(xy_meshgrid: np.ndarray, amplitude: float, x0: float, y0: float, 
            sigma_x: float, sigma_y: float, theta: float, offset: float) -> np.ndarray:
    """2D Gaussian function for fitting.
    
    Parameters
    ----------
    xy_meshgrid : tuple of np.ndarray
        Tuple containing the X and Y meshgrids
    amplitude : float
        Peak amplitude
    x0, y0 : float
        Peak center coordinates
    sigma_x, sigma_y : float
        Standard deviations in x and y directions
    theta : float
        Rotation angle in radians
    offset : float
        Background offset
        
    Returns
    -------
    np.ndarray
        2D Gaussian evaluated on the meshgrid
    """
    x, y = xy_meshgrid[0], xy_meshgrid[1]  # Explicitly get arrays from tuple
    # Calculate rotated coordinates
    x_rot = (x - x0) * np.cos(theta) + (y - y0) * np.sin(theta)
    y_rot = -(x - x0) * np.sin(theta) + (y - y0) * np.cos(theta)
    # Calculate Gaussian
    gauss = amplitude * np.exp(-(x_rot**2 / (2*sigma_x**2) + y_rot**2 / (2*sigma_y**2))) + offset
    return gauss.ravel()


def fit_gaussian(
    x0, y0, sigma_x, sigma_y, theta, offset, data, plot=False, verbose=True
):
    # Define mesh for input values and initial guess
    xy_meshgrid = np.meshgrid(range(np.shape(data)[1]), range(np.shape(data)[0]))
    initial_guess = (data[int(y0), int(x0)], x0, y0, sigma_x, sigma_y, theta, offset)
    # Perform fit and pull out centers
    try:
        popt, pcov = curve_fit(
            gauss2d, xy_meshgrid, data.ravel(), p0=initial_guess
        )  # , ftol=1.49012e-10, xtol=1.49012e-10)
    except RuntimeError:
        if verbose:
            print(
                "Particle could not be fit to a 2D gaussian.  Returning guess parameters."
            )
        return np.array(initial_guess), None, False
    # Plotting for troubleshooting
    if plot:
        data_fitted = gauss2d(xy_meshgrid, *popt)
        fig, ax = plt.subplots(1, 1)
        ax.matshow(data, cmap="gray")
        ax.contour(
            data_fitted.reshape(np.shape(data)[0], np.shape(data)[1]),
            8,
            colors="w",
        )
        plt.show()
    return popt, pcov, True
