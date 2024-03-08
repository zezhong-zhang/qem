import matplotlib.pyplot as plt
import numpy as np
import sys
from functools import partial
from numba import jit

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


def onclick(event, peaks_locations, tolerance):
    # Check if the click is a double click
    if event.dblclick:
        y = event.ydata
        x = event.xdata

        distance = np.sqrt(
            (peaks_locations[:, 0] - x) ** 2 + (peaks_locations[:, 1] - y) ** 2
        )
        # Check if the click is on a peak
        if distance.min() < tolerance:
            # find the index of the peak
            i = np.argmin(distance)
            # If the nearest peak is within the tolerance, remove it
            peaks_locations = np.delete(peaks_locations, i, axis=0)
        else:
            # If the click was not on a peak, add a new peak
            peaks_locations = np.vstack((peaks_locations, (x, y)))

        # Redraw the plot
        plt.scatter(peaks_locations[:, 0], peaks_locations[:, 1], color="red")
        plt.draw()
        return peaks_locations


def interactive_plot(peaks_locations, image, tolerance):
    # Initial plot
    fig = plt.figure()
    plt.imshow(image)
    # for (px, py) in peaks_locations:
    plt.scatter(peaks_locations[:, 0], peaks_locations[:, 1], color="red")

    # Connect the click event to the handler function
    plt.gcf().canvas.mpl_connect(
        "button_press_event", partial(onclick, tolerance=tolerance)
    )

    # Show the plot
    plt.show()

    # Wait for the user to close the plot window
    while plt.fignum_exists(fig.number):
        plt.pause(0.1)  # Pause for a short period to allow other processing to occur

    # Now you can use the updated peaks_locations
    print("Updated peak locations.")
    return peaks_locations


class InteractivePlot:
    def __init__(self, peaks_locations, image, tolerance):
        self.pos_x = peaks_locations[:, 0]
        self.pos_y = peaks_locations[:, 1]
        self.image = image
        self.tolerance = tolerance
        self.scatter_plot = None

    def onclick(self, event):
        if event.dblclick:
            x, y = event.xdata, event.ydata
            distance = np.sqrt((self.pos_x - x) ** 2 + (self.pos_y - y) ** 2)
            if distance.min() < self.tolerance:
                i = np.argmin(distance)
                self.pos_x = np.delete(self.pos_x, i, axis=0)
                self.pos_y = np.delete(self.pos_y, i, axis=0)
            else:
                self.pos_x = np.append(self.pos_x, x)
                self.pos_y = np.append(self.pos_y, y)

            self.update_plot()

    def update_plot(self):
        plt.clf()
        plt.imshow(self.image)
        self.scatter_plot = plt.scatter(self.pos_x, self.pos_y, color="red", s=1)
        plt.draw()

    def show(self):
        fig = plt.figure()
        plt.imshow(self.image)
        self.scatter_plot = plt.scatter(self.pos_x, self.pos_y, color="red", s=1)

        fig.canvas.mpl_connect("button_press_event", self.onclick)
        plt.show()

        while plt.fignum_exists(fig.number):
            plt.pause(0.1)

        print("Updated peak locations.")


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
    mask = x**2 + y **2 < radius**2
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
        too_close_mask = (diffs[:, 0] < threshold) & (diffs[:, 1] < threshold) & (np.arange(len(coordinates)) != i)

        # Update the keep mask
        keep_mask[too_close_mask] = False

    # Filter coordinates based on the keep mask
    return coordinates[keep_mask]