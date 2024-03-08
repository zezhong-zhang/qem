import jax
import jax.numpy as jnp
import numpy as np
from jax import jit, lax, vmap
from jax.scipy.signal import convolve2d
from numba import jit as njit


@jit
def gaussian_local(X, Y, pos_x, pos_y, height, width):
    # Unpack the parameters
    gauss = height * jnp.exp(
        -(
            (X[:, :, None] - pos_x[None, None, :] % 1) ** 2
            + (Y[:, :, None] - pos_y[None, None, :] % 1) ** 2
        )
        / (2 * width**2)
    )
    return gauss


@njit(nopython=True, parallel=True)
def add_gauss_to_total_sum(total_sum, pos_x, pos_y, gaussian_local, windows_size):
    pos_x = pos_x.astype(np.int32)
    pos_y = pos_y.astype(np.int32)
    for i in range(len(pos_x)):
        left = max(pos_y[i] - windows_size, 0)
        right = min(pos_y[i] + windows_size + 1, total_sum.shape[0])
        top = max(pos_x[i] - windows_size, 0)
        bottom = min(pos_x[i] + windows_size + 1, total_sum.shape[1])
        local_left = left - pos_y[i] + windows_size
        local_right = right - pos_y[i] + windows_size
        local_top = top - pos_x[i] + windows_size
        local_bottom = bottom - pos_x[i] + windows_size
        total_sum[left:right, top:bottom] += gaussian_local[
            local_left:local_right, local_top:local_bottom, i
        ]
    return total_sum


def add_gauss_to_total_sum_jax(total_sum, pos_x, pos_y, gaussian_local, window_size):
    # Correct type casting
    pos_x = pos_x.astype(jnp.int32)
    pos_y = pos_y.astype(jnp.int32)

    def update_total_sum(i, total_sum):
        left = max(pos_y[i] - window_size, 0)
        right = min(pos_y[i] + window_size + 1, total_sum.shape[0])
        top = max(pos_x[i] - window_size, 0)
        bottom = min(pos_x[i] + window_size + 1, total_sum.shape[1])

        local_left = left - pos_y[i] + window_size
        local_right = right - pos_y[i] + window_size
        local_top = top - pos_x[i] + window_size
        local_bottom = bottom - pos_x[i] + window_size

        update_slice = gaussian_local[local_left:local_right, local_top:local_bottom, i]

        # Define the indices where updates are applied, ensuring they are within the bounds of total_sum
        ix, iy = jnp.meshgrid(
            jnp.arange(left, right), jnp.arange(top, bottom), indexing="ij"
        )
        indices = jnp.stack([ix.flatten(), iy.flatten()], axis=-1)

        # Flatten the update_slice to match the indices
        updates = update_slice.flatten()

        # Use JAX's functional update to add values at specified indices
        updated_sum = total_sum.at[indices[:, 0], indices[:, 1]].add(updates)
        return updated_sum

    # Loop through each position to update the total_sum
    # Note: This loop is for illustration. In practice, consider vectorizing your computations with vmap or similar.
    for i in range(len(pos_x)):
        total_sum = update_total_sum(i, total_sum)

    return total_sum


@jit
def gaussian_sum(X, Y, pos_x, pos_y, height, width, background):
    # Unpack the parameters
    sum = (
        jnp.sum(
            height
            * jnp.exp(
                -(
                    (X[:, :, None] - pos_x[None, None, :]) ** 2
                    + (Y[:, :, None] - pos_y[None, None, :]) ** 2
                )
                / (2 * width**2)
            ),
            axis=2,
        )
        + background
    )
    return sum


@jit
def gaussian_sum_batched(X, Y, pos_x, pos_y, height, width, background):
    """
    Computes the sum of Gaussian functions on a grid in batches to save memory.
    Each batch processes up to 100 peaks at a time.

    Parameters:
    - X, Y: Meshgrids of x and y coordinates.
    - pos_x, pos_y: Arrays containing the x and y positions of the Gaussian peaks.
    - height: Heights of the Gaussian peaks.
    - width: Width (standard deviation) of the Gaussian peaks, assumed constant for all peaks for simplicity.
    - background: Background intensity level.

    Returns:
    - A 2D array of the summed intensity values with background added.
    """
    # Initialize the sum with the background level
    gaussian_sum = jnp.zeros(X.shape) + background

    # Number of peaks
    num_peaks = len(pos_x)

    # Process in batches of 100
    for i in range(0, num_peaks, 100):
        # Indices for the current batch
        end_idx = i + 100

        # Select the batch of parameters
        batch_pos_x = pos_x[i:end_idx]
        batch_pos_y = pos_y[i:end_idx]
        batch_height = height[i:end_idx]

        # Assuming width is a scalar or has the same value for each peak for simplicity
        # If width varies per peak, it should be indexed similar to pos_x, pos_y, and height

        # Calculate the Gaussian contributions for the current batch
        batch_contributions = jnp.sum(
            batch_height[:, None, None]
            * jnp.exp(
                -(
                    (X[None, :, :] - batch_pos_x[:, None, None]) ** 2
                    + (Y[None, :, :] - batch_pos_y[:, None, None]) ** 2
                )
                / (2 * width**2)
            ),
            axis=0,
        )

        # Update the total sum with contributions from this batch
        gaussian_sum += batch_contributions

    return gaussian_sum


@jit
def voigt_sum(X, Y, pos_x, pos_y, height, sigma, gamma, ratio, background):
    R2 = (X[:, :, None] - pos_x[None, None, :]) ** 2 + (
        Y[:, :, None] - pos_y[None, None, :]
    ) ** 2

    sum = (
        jnp.sum(
            height
            * (
                ratio * jnp.exp(-(R2) / (2 * sigma**2))
                + (1 - ratio) * gamma**2 / (R2 + gamma**2) ** (3 / 2)
            ),
            axis=2,
        )
        + background
    )
    return sum


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

@jax.jit
def gaussian_kernel(sigma: float) -> jnp.ndarray:
    """
    Creates a 2D Gaussian kernel with the given size and sigma.
    """
    x = jnp.arange(-20 // 2, 20 // 2 + 1.)
    y = jnp.arange(-20 // 2, 20 // 2 + 1.)
    xx, yy = jnp.meshgrid(x, y)
    kernel = jnp.exp(-(xx**2 + yy**2) / (2 * sigma**2))
    kernel = kernel / jnp.sum(kernel)
    return kernel

@jax.jit
def gaussian_filter_jax(image: jnp.ndarray, sigma: float) -> jnp.ndarray:
    """
    Applies Gaussian filter to a 2D image.
    """
    # x = jnp.linspace(-2, 2, 10)
    # window = jax.scipy.stats.norm.pdf(x) * jax.scipy.stats.norm.pdf(x[:, None])
    # diff = jax.scipy.signal.convolve2d(diff, window, mode="same")
    kernel = gaussian_kernel(sigma)
    # Convolve the image with the Gaussian kernel
    filtered_image = convolve2d(image, kernel, mode='same')
    return filtered_image