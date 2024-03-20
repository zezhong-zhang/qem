import matplotlib.pyplot as plt
import numpy as np
import sys
from functools import partial
from numba import jit
import logging

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
        # Added attributes for vector selection
        self.origin = None
        self.vector_a = None
        self.vector_b = None
        self.selection_stage = 0  # 0: Select origin, 1: Select vector a, 2: Select vector b

    def onclick_add_or_remove(self, event):
        if event.dblclick:
            x, y = event.xdata, event.ydata
            distance = np.sqrt((self.pos_x - x) ** 2 + (self.pos_y - y) ** 2)
            if distance.min() < self.tolerance:
                i = np.argmin(distance)
                logging.info(f"Removing peak at ({self.pos_x[i]}, {self.pos_y[i]}).")
                self.pos_x = np.delete(self.pos_x, i, axis=0)
                self.pos_y = np.delete(self.pos_y, i, axis=0)
            else:
                self.pos_x = np.append(self.pos_x, x)
                self.pos_y = np.append(self.pos_y, y)
                logging.info(f"Adding peak at ({x}, {y}).")
            
            self.update_plot()

    def update_plot(self):
        plt.clf()
        plt.imshow(self.image)
        self.scatter_plot = plt.scatter(self.pos_x, self.pos_y, color="red", s=1)
        plt.draw()

    def add_or_remove(self):
        fig = plt.figure()
        plt.imshow(self.image)
        self.scatter_plot = plt.scatter(self.pos_x, self.pos_y, color="red", s=1)
        fig.canvas.mpl_connect("button_press_event", self.onclick_add_or_remove)
        plt.show()

        while plt.fignum_exists(fig.number):
            plt.pause(0.1)

        print("Updated peak locations.")

    def onclick_select(self, event):
        if event.dblclick:
            x, y = event.xdata, event.ydata
            distance = np.sqrt((self.pos_x - x) ** 2 + (self.pos_y - y) ** 2)
            if distance.min() < self.tolerance:
                i = np.argmin(distance)
                self.selected_point = (self.pos_x[i], self.pos_y[i])
                plt.scatter(self.pos_x[i], self.pos_y[i], color="red", s=5)
                plt.draw()

    def select(self):
        fig = plt.figure()
        plt.imshow(self.image)
        self.scatter_plot = plt.scatter(self.pos_x, self.pos_y, color="blue", s=1)
        fig.canvas.mpl_connect("button_press_event", self.onclick_select)
        plt.show()

        while plt.fignum_exists(fig.number):
            plt.pause(0.1)

        print(f"Selected peak location: {self.selected_point}.")
        return self.selected_point

    def onclick_select_vectors(self, event):
        if event.dblclick:
            x, y = event.xdata, event.ydata
            distance = np.sqrt((self.pos_x - x) ** 2 + (self.pos_y - y) ** 2)
            if distance.min() < self.tolerance:
                i = np.argmin(distance)
                point = np.array([self.pos_x[i], self.pos_y[i]])
                plt.scatter(self.pos_x[i], self.pos_y[i], color="red", s=10)
                plt.draw()

                if self.selection_stage == 0:
                    self.origin = point
                    self.selection_stage += 1
                    print("Origin selected:", self.origin)
                elif self.selection_stage == 1:
                    self.vector_a = point - self.origin
                    self.selection_stage += 1
                    print("Vector a selected:", self.vector_a)
                    self.draw_arrow(self.origin, point, 'a')
                elif self.selection_stage == 2:
                    self.vector_b = point - self.origin
                    print("Vector b selected:", self.vector_b)
                    self.draw_arrow(self.origin, point, 'b')
                    self.selection_stage = 0  # Reset the selection stage to allow new selections
                    return self.origin, self.vector_a, self.vector_b
            plt.draw()

    def draw_arrow(self, start, end, label):
        dx = end[0] - start[0]
        dy = end[1] - start[1]
        plt.arrow(start[0], start[1], dx, dy, head_width=10, head_length=10, fc='black', ec='black')
        plt.text((start[0] + end[0])/2, (start[1] + end[1])/2, label, color='black', fontsize=14)

    def select_vectors(self):
        fig = plt.figure()
        plt.imshow(self.image)
        self.scatter_plot = plt.scatter(self.pos_x, self.pos_y, color="blue", s=1)
        fig.canvas.mpl_connect("button_press_event", self.onclick_select_vectors)
        plt.show()

        while plt.fignum_exists(fig.number):
            plt.pause(0.1)

        if self.origin.any() and self.vector_a.any() and self.vector_b.any():
            print(f"Origin: {self.origin}, Vector a: {self.vector_a}, Vector b: {self.vector_b}")
            return self.origin, self.vector_a, self.vector_b
        else:
            print("Selection incomplete.")
            return None, None, None

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
    indices = [np.where((array2 == element).all(axis=1))[0][0] if (array2 == element).all(axis=1).any() else -1 for element in array1]
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
        too_close_mask = (diffs[:, 0] < threshold) & (diffs[:, 1] < threshold) & (np.arange(len(coordinates)) != i)

        # Update the keep mask
        keep_mask[too_close_mask] = False

    # Filter coordinates based on the keep mask
    return coordinates[keep_mask]

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
    batches = [all_indices[i:i + batch_size] for i in range(0, total_examples, batch_size)]
    return batches
