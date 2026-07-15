"""
Signal processing utilities for QEM.
"""

import sys

import numpy as np


def safe_ln(x):
    """
    Natural logarithm function, avoiding division by zero warnings.

    Parameters
    ----------
    x : float
        The value to take the logarithm of.

    Returns
    -------
    float
        The natural logarithm of x.
    """
    x[x < sys.float_info.min] = sys.float_info.min
    return np.log(x)


def fft2d(array):
    """
    2D FFT of a numpy array.

    Parameters
    ----------
    array : numpy.ndarray
        The array to transform.

    Returns
    -------
    numpy.ndarray
        The transformed array.
    """
    return np.fft.ifftshift(np.fft.fft2(np.fft.fftshift(array)))


def ifft2d(array):
    """
    2D inverse FFT of a numpy array.

    Parameters
    ----------
    array : numpy.ndarray
        The array to transform.

    Returns
    -------
    numpy.ndarray
        The transformed array.
    """
    return np.fft.ifftshift(np.fft.ifft2(np.fft.fftshift(array)))


def remove_freq(image, low, high):
    """Remove frequency components in a specified range."""
    nx, ny = image.shape[1:]
    x = np.linspace(-nx / 2, nx / 2, nx) / nx
    y = np.linspace(-ny / 2, ny / 2, ny) / ny
    yv, xv = np.meshgrid(y, x)
    mask = np.where(
        (np.sqrt(xv**2 + yv**2) >= low) * (np.sqrt(xv**2 + yv**2) < high),
        1,
        0,
    )
    return np.real(ifft2d(fft2d(image) * mask))


def apply_threshold(image, image_ref, threshold):
    """Apply threshold to image based on reference image."""
    nz = np.size(image, 0)
    if type(threshold) != list:
        threshold = [threshold]
    img = np.zeros(image.shape)
    for i in range(nz):
        m = np.amax(image_ref[i])
        img[i, :, :] = np.where(image_ref[i] < threshold[i] * m, 0, image[i])
    return img


def q_space_array(pixels, gridsize, meshed=True):
    """
    Return the appropriately scaled 2D reciprocal space coordinates.

    Parameters
    -----------
    pixels : (N,) array_like
        Pixels in each dimension of a ND array
    gridsize : (N,) array_like
        Dimensions of the array in real space units
    meshed : bool, optional
        Option to output dense meshed grid (True) or output unbroadcasted
        arrays (False)

    Parameters
    -----------
    pixels : (N,) array_like
        Pixels in each dimension of a 2D array
    gridsize : (N,) array_like
        Dimensions of the array in real space units
    """
    # N is the dimensionality of grid
    N = len(pixels)

    qspace = [np.fft.fftfreq(pixels[i], d=gridsize[i] / pixels[i]) for i in range(N)]
    # At this point we can return the arrays without broadcasting
    if meshed:
        return broadcast_from_unmeshed(qspace)
    else:
        return qspace


def broadcast_from_unmeshed(coords):
    """
    For an unmeshed set of coordinates broadcast to a meshed ND array.

    Examples
    --------
    >>> broadcast_from_unmeshed([np.arange(5),np.arange(6)])
    [array([[0, 0, 0, 0, 0, 0],
       [1, 1, 1, 1, 1, 1],
       [2, 2, 2, 2, 2, 2],
       [3, 3, 3, 3, 3, 3],
       [4, 4, 4, 4, 4, 4]]), array([[0, 1, 2, 3, 4, 5],
       [0, 1, 2, 3, 4, 5],
       [0, 1, 2, 3, 4, 5],
       [0, 1, 2, 3, 4, 5],
       [0, 1, 2, 3, 4, 5]])]
    """

    N = len(coords)
    pixels = [a.shape[0] for a in coords]

    # Broadcasting patterns
    R = np.ones((N, N), dtype=np.int16) + np.diag(pixels) - np.eye(N, dtype=np.int16)

    # Broadcast unmeshed grids
    return [np.broadcast_to(a.reshape(rr), pixels) for a, rr in zip(coords, R, strict=False)]

def butterworth_window(shape, cutoff_radius_ftr, order):
    """
    Generate a 2D Butterworth window.

    Parameters:
    - shape: tuple of ints, the shape of the window (height, width).
    - cutoff_radius_ftr: float, the cutoff frequency as a fraction of the radius (0, 0.5].
    - order: int, the order of the Butterworth filter.

    Returns:
    - window: 2D numpy array, the Butterworth window.
    """
    assert len(shape) == 2, "Shape must be a tuple of length 2 (height, width)"
    assert (
        0 < cutoff_radius_ftr <= 0.5
    ), "Cutoff frequency must be in the range (0, 0.5]"

    def butterworth_1d(length, cutoff_radius_ftr, order):
        n = np.arange(-np.floor(length / 2), length - np.floor(length / 2))
        return 1 / (1 + (n / (cutoff_radius_ftr * length)) ** (2 * order))

    window_y = butterworth_1d(shape[0], cutoff_radius_ftr, order)
    window_x = butterworth_1d(shape[1], cutoff_radius_ftr, order)

    window = np.outer(window_y, window_x)

    return window
