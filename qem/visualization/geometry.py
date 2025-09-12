"""
Geometric utilities for QEM analysis.
"""

import numpy as np
from numba import jit


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
    """Rotate a vector around a specified axis by a given angle"""
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