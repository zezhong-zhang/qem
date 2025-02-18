import jax
import jax.numpy as jnp
import numpy as np
from jax import jit
from jax.scipy.signal import convolve2d
from numba import jit as njit
from functools import partial
from abc import ABC, abstractmethod




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

def model_sum_local(model_fn, ny: int, nx: int):
    """Create a JIT-compiled function for local window-based computation."""    
    @jax.jit
    def add_to_window(result, x_start, x_end, y_start, y_end, pos_x, pos_y, height, width, *args):
        """Add a model's contribution to a window in the result array."""
        y_coords = jnp.arange(y_start, y_end)[:, jnp.newaxis]
        x_coords = jnp.arange(x_start, x_end)[jnp.newaxis, :]
        window = model_fn(x_coords, y_coords, pos_x, pos_y, height, width, *args)
        return result.at[y_start:y_end, x_start:x_end].add(window)

    @jax.jit
    def sum_local(pos_x, pos_y, height, width, *args, background=0.0, window_size=None):
        # Initialize result with explicit float32 dtype
        result = jnp.full((ny, nx), background, dtype=jnp.float32)
        
        def scan_body(carry, x):
            result, i = carry
            p_x, p_y, h, w = pos_x[i], pos_y[i], height[i], width[i]
            x_start, x_end, y_start, y_end = get_window_indices(p_x, p_y, window_size, ny, nx)
            result = add_to_window(result, x_start, x_end, y_start, y_end, p_x, p_y, h, w)
            return (result, i + 1), None
        
        (result, _), _ = jax.lax.scan(scan_body, (result, 0), None, length=len(pos_x))
        return result

    return sum_local
class ImageModel(ABC):
    """Base class for all image models."""

    def __init__(self, dx=1.0, background=0.0):
        """Initialize the model.
        
        Args:
            dx (float, optional): Pixel size. Defaults to 1.0.
            background (float, optional): Background level. Defaults to 0.0.
        """
        self.dx = dx
        self.background = background

    @abstractmethod
    def model_fn(self, x, y, pos_x, pos_y, height, width, *args):
        """Core model function that defines the peak shape."""
        pass

    @staticmethod
    @abstractmethod
    def model_fn_numba(x, y, pos_x, pos_y, height, width, *args):
        """Numba version of the model function."""
        pass

    @abstractmethod
    def volume(self, params: dict) -> jnp.ndarray:
        """Calculate the volume of each peak."""
        pass

    @partial(jax.jit, static_argnums=(0,))
    def sum(self, X, Y, pos_x, pos_y, height, width, *args):
        """Calculate sum of peaks using JAX."""
        total = (
            jnp.sum(
                self.model_fn(
                    X[:, :, None], Y[:, :, None],
                    pos_x[None, None, :], pos_y[None, None, :],
                    height, width, *args
                ),
                axis=2,
            )
            + self.background
        )
        return total

    @staticmethod
    @njit(nopython=True)
    def sum_numba(X, Y, pos_x, pos_y, height, width, *args):
        """Calculate sum of peaks using numba."""
        return ImageModel.model_fn_numba(
            X[:, :, None], Y[:, :, None],
            pos_x[None, None, :], pos_y[None, None, :],
            height, width, *args
        )

    def model_local(self, x, y, pos_x, pos_y, height, width, *args):
        """Calculate the 2D window contribution for a single peak."""
        return self.model_fn(x, y, pos_x, pos_y, height, width, *args)

    @partial(jax.jit, static_argnums=(0,))
    def add_model_to_window(self, result, x_start, x_end, y_start, y_end, pos_x, pos_y, height, width, *args):
        """Add a model's contribution to a window in the result array."""
        y_coords = jnp.arange(y_start, y_end)[:, jnp.newaxis]
        x_coords = jnp.arange(x_start, x_end)[jnp.newaxis, :]
        window = self.model_fn(x_coords, y_coords, pos_x, pos_y, height, width, *args)
        return result.at[y_start:y_end, x_start:x_end].add(window)

    def sum_local(self, ny: int, nx: int):
        return model_sum_local(self.model_fn,ny=ny,nx=nx)


class GaussianModel(ImageModel):
    """Gaussian peak model."""

    def volume(self, params: dict) -> jnp.ndarray:
        """Calculate the volume of each Gaussian peak.
        
        For a 2D Gaussian, the volume is: height * 2π * width²
        """
        height = params["height"]
        width = params["width"]
        return height * 2 * jnp.pi * width**2 * self.dx**2

    def model_fn(self, x, y, pos_x, pos_y, height, width, *args):
        """Core computation for Gaussian model using JAX."""
        return height * jnp.exp(-((x - pos_x) ** 2 + (y - pos_y) ** 2) / (2 * width**2))

    @staticmethod
    @njit(nopython=True)
    def model_fn_numba(x, y, pos_x, pos_y, height, width, *args):
        """Core computation for Gaussian model using numba."""
        return height * np.exp(-((x - pos_x) ** 2 + (y - pos_y) ** 2) / (2 * width**2))


class LorentzianModel(ImageModel):
    """Lorentzian peak model."""

    def volume(self, params: dict) -> jnp.ndarray:
        """Calculate the volume of each Lorentzian peak.
        
        For a 2D Lorentzian, the volume is: height * π * width²
        """
        height = params["height"]
        width = params["width"]
        return height * jnp.pi * width**2 * self.dx**2

    def model_fn(self, x, y, pos_x, pos_y, height, width, *args):
        """Core computation for Lorentzian model using JAX."""
        return height * width**2 / ((x - pos_x) ** 2 + (y - pos_y) ** 2 + width**2)

    @staticmethod
    @njit(nopython=True)
    def model_fn_numba(x, y, pos_x, pos_y, height, width, *args):
        """Core computation for Lorentzian model using numba."""
        return height * width**2 / ((x - pos_x) ** 2 + (y - pos_y) ** 2 + width**2)


class VoigtModel(ImageModel):
    """Voigt peak model."""

    def volume(self, params: dict) -> jnp.ndarray:
        """Calculate the volume of each Voigt peak.
        
        For a 2D Voigt profile, the volume is a weighted sum of Gaussian and Lorentzian volumes:
        V = ratio * (height * 2π * width²) + (1-ratio) * (height * π * width²)
        """
        height = params["height"]
        width = params["width"]
        ratio = params["ratio"]
        
        gaussian_vol = height * 2 * jnp.pi * width**2 * self.dx**2
        lorentzian_vol = height * jnp.pi * width**2 * self.dx**2
        
        return ratio * gaussian_vol + (1 - ratio) * lorentzian_vol

    def model_fn(self, x, y, pos_x, pos_y, height, width, ratio):
        """Core computation for Voigt model using JAX."""
        # Convert width to sigma and gamma
        sigma = width
        gamma = width / jnp.sqrt(2 * jnp.log(2))
        
        # Calculate squared distance
        R2 = (x - pos_x) ** 2 + (y - pos_y) ** 2
        
        # Compute Gaussian and Lorentzian parts
        gaussian_part = jnp.exp(-R2 / (2 * sigma**2))
        lorentzian_part = gamma**3 / (R2 + gamma**2) ** (3 / 2)
        
        # Return weighted sum
        return height * (ratio * gaussian_part + (1 - ratio) * lorentzian_part)

    @staticmethod
    @njit(nopython=True)
    def model_fn_numba(x, y, pos_x, pos_y, height, width, ratio):
        """Core computation for Voigt model using numba."""
        # Convert width to sigma and gamma
        sigma = width
        gamma = width / np.sqrt(2 * np.log(2))
        
        # Calculate squared distance
        R2 = (x - pos_x) ** 2 + (y - pos_y) ** 2
        
        # Compute Gaussian and Lorentzian parts
        gaussian_part = np.exp(-R2 / (2 * sigma**2))
        lorentzian_part = gamma**3 / (R2 + gamma**2) ** (3 / 2)
        
        # Return weighted sum
        return height * (ratio * gaussian_part + (1 - ratio) * lorentzian_part)

class GaussianKernel:
    """Gaussian kernel implementation."""

    @staticmethod
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

    @staticmethod
    def gaussian_filter_jax(image: jnp.ndarray, sigma: float) -> jnp.ndarray:
        """
        Applies Gaussian filter to a 2D image.
        """
        kernel = GaussianKernel.gaussian_kernel(sigma)
        filtered_image = convolve2d(image, kernel, mode="same")
        return filtered_image
