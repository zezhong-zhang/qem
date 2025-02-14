import jax
import jax.numpy as jnp
import numpy as np
from jax import jit
from jax.scipy.signal import convolve2d
from numba import jit as njit
from functools import partial


@njit(nopython=True)
def gaussian_2d_numba(X, Y, pos_x, pos_y, height, width):
    # Unpack the parameters
    gauss = height * np.exp(
        -(
            (X[:, :, None] - pos_x[None, None, :]) ** 2
            + (Y[:, :, None] - pos_y[None, None, :]) ** 2
        )
        / (2 * width**2)
    )
    return gauss


@njit(nopython=True)
def lorentzian_2d_numba(X, Y, pos_x, pos_y, height, gamma):
    # Unpack the parameters
    lorentz = height * gamma**2 / (
        (X[:, :, None] - pos_x[None, None, :]) ** 2
        + (Y[:, :, None] - pos_y[None, None, :]) ** 2
        + gamma**2
    )
    return lorentz


@njit(nopython=True)
def voigt_2d_numba(X, Y, pos_x, pos_y, height, sigma, gamma, ratio):
    # Unpack the parameters
    R2 = (X[:, :, None] - pos_x[None, None, :]) ** 2 + (
        Y[:, :, None] - pos_y[None, None, :]
    ) ** 2

    voigt = height * (
        ratio * np.exp(-(R2) / (2 * sigma**2))
        + (1 - ratio) * gamma**3 / (R2 + gamma**2) ** (3 / 2)
    )
    return voigt


@jax.jit
def gaussian_2d_jax(X, Y, pos_x, pos_y, height, width):
    # Unpack the parameters
    gauss = height * jnp.exp(
        -(
            (X[:, :, None] - pos_x[None, None, :]) ** 2
            + (Y[:, :, None] - pos_y[None, None, :]) ** 2
        )
        / (2 * width**2)
    )
    return gauss


@njit(nopython=True, parallel=True)
def add_peak_at_positions(total_sum, pos_x, pos_y, gaussian_local, windows_size):
    pos_x = pos_x.astype(np.int32)
    pos_y = pos_y.astype(np.int32)
    for i in range(len(pos_x)):
        left = max(pos_x[i] - windows_size, 0)
        right = min(pos_x[i] + windows_size + 1, total_sum.shape[1])
        top = max(pos_y[i] - windows_size, 0)
        bottom = min(pos_y[i] + windows_size + 1, total_sum.shape[0])
        local_left = left - pos_x[i] + windows_size
        local_right = right - pos_x[i] + windows_size
        local_top = top - pos_y[i] + windows_size
        local_bottom = bottom - pos_y[i] + windows_size
        total_sum[top:bottom, left:right] += gaussian_local[
            local_top:local_bottom, local_left:local_right, i
        ]
    return total_sum


@jax.jit
def gaussian_sum_parallel(X, Y, pos_x, pos_y, height, width, background):
    # Unpack the parameters
    total = (
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
    return total


@jax.jit
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


@jax.jit
def voigt_sum_parallel(X, Y, pos_x, pos_y, height, sigma, gamma, ratio, background):
    R2 = (X[:, :, None] - pos_x[None, None, :]) ** 2 + (
        Y[:, :, None] - pos_y[None, None, :]
    ) ** 2

    total = (
        jnp.sum(
            height
            * (
                ratio * jnp.exp(-(R2) / (2 * sigma**2))
                + (1 - ratio) * gamma**3 / (R2 + gamma**2) ** (3 / 2)
            ),
            axis=2,
        )
        + background
    )
    return total


@jax.jit
def lorentzian_sum_parallel(X, Y, pos_x, pos_y, height, gamma, background):
    R2 = (X[:, :, None] - pos_x[None, None, :]) ** 2 + (
        Y[:, :, None] - pos_y[None, None, :]
    ) ** 2

    total = (
        jnp.sum(
            height * gamma**4 / (R2 + gamma**2) ** (2),
            axis=2,
        )
        + background
    )
    return total


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
    x = jnp.arange(-20 // 2, 20 // 2 + 1.0)
    y = jnp.arange(-20 // 2, 20 // 2 + 1.0)
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
    filtered_image = convolve2d(image, kernel, mode="same")
    return filtered_image


def mask_grads(grads, keys_to_mask):
    """A pre-update function that masks out gradients for specified keys."""
    # Extract gradients from the optimizer state
    # Zero out gradients for the keys to mask
    masked_grads = {
        k: np.zeros_like(grads[k]) if k in keys_to_mask else grads[k] for k in grads
    }

    # Update the state with masked gradients

    return masked_grads


@jax.jit
def get_window_size(width, threshold=1e-6):
    """
    Calculate the window size needed to capture most of the Gaussian's intensity.
    
    Args:
        width (float): Width (sigma) of the Gaussian
        threshold (float): Minimum relative intensity to consider (e.g., 1e-6 means 0.0001% of peak height)
    
    Returns:
        int: Window size that captures the Gaussian intensity above the threshold
    """
    return jnp.ceil(width * jnp.sqrt(-2 * jnp.log(threshold))).astype(jnp.int32)


@jax.jit
def gaussian_2d_window(x, y, pos_x, pos_y, height, width):
    """
    Compute a single Gaussian contribution in a window.
    """
    return height * jnp.exp(-((x - pos_x) ** 2 + (y - pos_y) ** 2) / (2 * width ** 2))


def get_static_window_size(width_max, threshold=1e-6):
    """
    Calculate a static window size that will work for all Gaussians.
    
    Args:
        width_max: Maximum width (sigma) of all Gaussians
        threshold: Minimum relative intensity to consider
    """
    return int(np.ceil(width_max * 5))  # 5 sigma covers >99.99% of the Gaussian


@jax.jit
def get_window_indices(pos_x, pos_y, window_size, ny, nx):
    """
    Calculate window indices for a Gaussian peak using static window size.
    """
    x_start = jnp.maximum(0, jnp.floor(pos_x - window_size)).astype(jnp.int32)
    x_end = jnp.minimum(nx, jnp.ceil(pos_x + window_size + 1)).astype(jnp.int32)
    y_start = jnp.maximum(0, jnp.floor(pos_y - window_size)).astype(jnp.int32)
    y_end = jnp.minimum(ny, jnp.ceil(pos_y + window_size + 1)).astype(jnp.int32)
    return x_start, x_end, y_start, y_end


@jax.jit
def add_gaussian_to_window(result, x_start, x_end, y_start, y_end, pos_x, pos_y, height, width):
    """
    Add a Gaussian contribution to a specific window in the result array.
    """
    y_coords = jnp.arange(y_start, y_end)[:, jnp.newaxis]
    x_coords = jnp.arange(x_start, x_end)[jnp.newaxis, :]
    window = gaussian_2d_window(x_coords, y_coords, pos_x, pos_y, height, width)
    return result.at[y_start:y_end, x_start:x_end].add(window)


def create_gaussian_sum_local(ny: int, nx: int):
    """
    Create a JIT-compiled function for specific image dimensions.
    
    Args:
        ny, nx: Static image dimensions
    Returns:
        JIT-compiled function that takes (pos_x, pos_y, height, width, background, window_size)
    """
    @jax.jit
    def gaussian_sum_local(pos_x, pos_y, height, width, background, window_size):
        result = jnp.full((ny, nx), background)
        
        def scan_body(carry, x):
            result, i = carry
            p_x, p_y, h, w = pos_x[i], pos_y[i], height[i], width[i]
            x_start, x_end, y_start, y_end = get_window_indices(p_x, p_y, window_size, ny, nx)
            result = add_gaussian_to_window(result, x_start, x_end, y_start, y_end, p_x, p_y, h, w)
            return (result, i + 1), None
        
        (result, _), _ = jax.lax.scan(scan_body, (result, 0), None, length=len(pos_x))
        return result
    
    return gaussian_sum_local
