import logging
import sys

import matplotlib.pyplot as plt
import numpy as np
from numba import jit

logging.basicConfig(level=logging.INFO)


def plot_image(image, x_labels, y_labels, colormap="gray", colorbar=True):
    """
    Plot an image. with x and y labels.

    Parameters
    ----------
    image : numpy.ndarray
        The image to plot.
    x_labels : list
        The labels for the x-axis.
    y_labels : list
        The labels for the y-axis.
    colormap : str, optional
        The colormap to use.
    colorbar : bool, optional
        Whether to show a colorbar.

    Returns
    -------
    matplotlib.pyplot.figure
        The figure containing the plot.
    """
    fig, ax = plt.subplots(1)
    ax.imshow(
        image,
        cmap=colormap,
        extent=[x_labels[0], x_labels[-1], y_labels[0], y_labels[-1]],
        origin="lower",
    )
    if colorbar:
        fig.colorbar(ax.images[0])
    return fig


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
    nz = np.size(image, 0)
    if type(threshold) != list:
        threshold = [threshold]
    img = np.zeros(image.shape)
    for i in range(nz):
        m = np.amax(image_ref[i])
        img[i, :, :] = np.where(image_ref[i] < threshold[i] * m, 0, image[i])
    return img


@jit(nopython=True)
def make_mask_circle_centre(arr, radius):
    """Create a circular mask with same shape as arr

    The circle is centered on the center of the array,
    with the circle having True values.

    Similar to _make_circular_mask, but simpler and potentially
    faster.

    Numba jit compatible.

    Parameters
    ----------
    arr : NumPy array
        Must be 2 dimensions
    radius : scalar
        Radius of the circle

    Returns
    -------
    mask : NumPy array
        Boolean array

    Example
    -------
    >>> import atomap.atom_finding_refining as afr
    >>> arr = np.random.randint(100, size=(20, 20))
    >>> mask = afr._make_mask_circle_centre(arr, 10)

    """
    # if len(arr.shape) != 2:
    #     raise ValueError("arr must be 2D, not {0}".format(len(arr.shape)))
    imageSizeX, imageSizeY = arr.shape
    centerX = (arr.shape[0] - 1) / 2
    centerY = (arr.shape[1] - 1) / 2

    x = np.expand_dims(np.arange(-centerX, imageSizeX - centerX), axis=1)
    y = np.arange(-centerY, imageSizeY - centerY)
    mask = x**2 + y**2 < radius**2
    return mask


def find_duplicate_row_indices(array):
    """
    Find the indices of duplicate rows in a NumPy array.

    Parameters:
    - array: NumPy array to check for duplicate rows.

    Returns:
    - idx_duplicates: NumPy array of indices of the duplicate rows.
    """
    # Step 1: Find indices of unique rows
    _, idx_unique = np.unique(array, axis=0, return_index=True)

    # Step 2: Initialize a mask assuming all are duplicates
    mask = np.ones(len(array), dtype=bool)

    # Step 3: Mark unique rows as not duplicates
    mask[idx_unique] = False

    # Step 4: Find indices of duplicates
    idx_duplicates = np.where(mask)[0]

    return idx_duplicates


def find_row_indices(array1, array2):
    """
    Efficiently find the indices of rows of array1 in array2.

    :param array1: A NumPy array of shape (M, N).
    :param array2: A NumPy array of shape (K, N).
    :return: A NumPy array of indices indicating the position of each row of array1 in array2.
             If a row from array1 is not found in array2, the corresponding index is -1.
    """
    # Create a dictionary mapping from row tuple to index for array2
    row_to_index = {tuple(row): i for i, row in enumerate(array2)}

    # Find indices for rows in array1 using the dictionary
    indices = np.array([row_to_index.get(tuple(row), -1) for row in array1])

    return indices


def find_element_indices(array1, array2):
    """
    Find indices of elements of array1 in array2.

    :param array1: A 1D NumPy array of elements to find.
    :param array2: A 1D NumPy array where to search for the elements.
    :return: A list of indices indicating the position of each element of array1 in array2.
             If an element from array1 is not found in array2, the corresponding index is -1.
    """
    indices = [
        (
            np.where((array2 == element).all(axis=1))[0][0]
            if (array2 == element).all(axis=1).any()
            else -1
        )
        for element in array1
    ]
    indices = np.array(indices)
    # drop the -1s
    indices = indices[indices != -1]
    return indices


# @jit(nopython=True, parallel=True)
def remove_close_coordinates(coordinates, threshold):
    """
    Remove coordinates that are within a specified threshold distance of each other.

    Parameters:
    - threshold: The distance below which coordinates are considered too close and should be removed.
    """

    # Create a boolean mask initialized to keep all coordinates
    keep_mask = np.ones(len(coordinates), dtype=bool)

    # Calculate the absolute differences in x and y coordinates
    for i, coord in enumerate(coordinates):
        if not keep_mask[i]:
            # Already marked for removal
            continue

        # Compute absolute differences for all coordinates with the current one
        diffs = np.abs(coordinates - coord)

        # Find coordinates too close in either x or y (excluding the current one)
        too_close_mask = (
            (diffs[:, 0] < threshold)
            & (diffs[:, 1] < threshold)
            & (np.arange(len(coordinates)) != i)
        )

        # Update the keep mask
        keep_mask[too_close_mask] = False

    # Filter coordinates based on the keep mask
    return coordinates[keep_mask], keep_mask


def export_params(params, filename):
    """
    Export the parameters to a file.

    Parameters:
    - params: Dictionary of parameters to export.
    - filename: Name of the file to export to.
    """
    with open(filename, "w") as f:
        for key, value in params.items():
            f.create_dataset(key, data=value)
    f.close()


def get_random_indices_in_batches(total_examples, batch_size):
    all_indices = np.random.permutation(total_examples)
    batches = [
        all_indices[i : i + batch_size] for i in range(0, total_examples, batch_size)
    ]
    return batches


def is_point_in_polygon(point, polygon):
    """
    Determine if a point is inside a polygon using the ray casting algorithm.

    Parameters:
    - point: A 2D point as a tuple or numpy array (x, y).
    - polygon: A list of tuples or numpy arrays [(x1, y1), (x2, y2), ..., (xn, yn)] representing the polygon vertices.

    Returns:
    - Boolean indicating whether the point is inside the polygon.
    """
    x, y = point
    inside = False
    n = len(polygon)
    px, py = polygon[0]
    for i in range(1, n + 1):
        qx, qy = polygon[i % n]
        if y > min(py, qy) and y <= max(py, qy) and x <= max(px, qx):
            if py != qy:
                intercept = px + (y - py) * (qx - px) / (qy - py)
            if px == qx or x <= intercept:
                inside = not inside
        px, py = qx, qy
    return inside


def find_peaks_in_rectangle(peaks, origin, a, b):
    """
    Find all peaks that lie within the rectangle defined by origin, origin+a, origin+b, and origin+a+b.

    Parameters:
    - peaks: A list of peak positions as tuples or numpy arrays (x, y).
    - origin: The origin point as a tuple or numpy array (x, y).
    - a: The vector a as a tuple or numpy array (x, y).
    - b: The vector b as a tuple or numpy array (x, y).

    Returns:
    - A list of peaks within the defined rectangle.
    """
    origin = np.array(origin)
    a = np.array(a)
    b = np.array(b)
    peaks = np.array(peaks)
    # Define the rectangle's vertices
    vertices = [origin, origin + a, origin + a + b, origin + b]

    # Initialize a list to hold indices of peaks within the rectangle
    indices_inside = []

    # Check each peak to see if it's inside the rectangle
    for idx, peak in enumerate(peaks):
        if is_point_in_polygon(peak, vertices):
            indices_inside.append(idx)

    # Extract the peaks that are inside using the indices
    peaks_inside = peaks[indices_inside]

    return peaks_inside, np.array(indices_inside)


def rotate_vector(vector, axis, angle):
    # Rotate a vector around a specified axis by a given angle
    axis = axis / np.linalg.norm(axis)
    rot_matrix = np.array(
        [
            [
                np.cos(angle) + axis[0] ** 2 * (1 - np.cos(angle)),
                axis[0] * axis[1] * (1 - np.cos(angle)) - axis[2] * np.sin(angle),
                axis[0] * axis[2] * (1 - np.cos(angle)) + axis[1] * np.sin(angle),
            ],
            [
                axis[1] * axis[0] * (1 - np.cos(angle)) + axis[2] * np.sin(angle),
                np.cos(angle) + axis[1] ** 2 * (1 - np.cos(angle)),
                axis[1] * axis[2] * (1 - np.cos(angle)) - axis[0] * np.sin(angle),
            ],
            [
                axis[2] * axis[0] * (1 - np.cos(angle)) - axis[1] * np.sin(angle),
                axis[2] * axis[1] * (1 - np.cos(angle)) + axis[0] * np.sin(angle),
                np.cos(angle) + axis[2] ** 2 * (1 - np.cos(angle)),
            ],
        ]
    )

    return np.dot(rot_matrix, vector.T).T


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
    return [np.broadcast_to(a.reshape(rr), pixels) for a, rr in zip(coords, R)]
