import jax
import jax.numpy as jnp
import numpy as np
from numba import jit as njit
from functools import partial
from abc import ABC, abstractmethod
import os
import logging


def set_backend(backend: str):
    """Set the Keras backend.
    
    Args:
        backend (str): Backend to use ('tensorflow', 'pytorch', or 'jax')
        
    Returns:
        str: The actually used backend
    """
    backend = backend.lower()
    if backend not in ['tensorflow', 'pytorch', 'jax']:
        raise ValueError(f"Backend {backend} not supported. Use 'tensorflow', 'pytorch', or 'jax'")
    
    # Try the requested backend first
    try:
        os.environ['KERAS_BACKEND'] = backend
        # Import keras to test if backend works
        import keras
        return backend
    except Exception as e:
        logging.warning(f"Failed to initialize {backend} backend: {str(e)}")
        
        # Try alternative backends in order of preference
        alternatives = ['jax', 'pytorch', 'tensorflow']
        alternatives.remove(backend)  # Remove the one that just failed
        
        for alt_backend in alternatives:
            try:
                os.environ['KERAS_BACKEND'] = alt_backend
                import keras
                logging.info(f"Successfully switched to {alt_backend} backend")
                return alt_backend
            except Exception as e:
                logging.warning(f"Failed to initialize {alt_backend} backend: {str(e)}")
        
        # If all backends fail, try JAX with CPU
        if 'jax' in alternatives:
            try:
                os.environ['JAX_PLATFORMS'] = 'cpu'
                os.environ['KERAS_BACKEND'] = 'jax'
                import keras
                logging.info("Successfully switched to JAX CPU backend")
                return 'jax'
            except Exception as e:
                logging.warning(f"Failed to initialize JAX CPU backend: {str(e)}")
        
        raise RuntimeError("Failed to initialize any backend. Please ensure at least one of TensorFlow, PyTorch, or JAX is properly installed.")


class ImageModel(ABC):
    """Base class for all image models."""

    def __init__(self, dx=1.0, background=0.0, backend='jax'):
        """Initialize the model.
        
        Args:
            dx (float, optional): Pixel size. Defaults to 1.0.
            background (float, optional): Background level. Defaults to 0.0.
            backend (str, optional): Backend to use ('tensorflow', 'pytorch', or 'jax'). Defaults to 'jax'.
        """
        self.dx = dx
        self.background = background
        
        # Set and import backend
        self.backend = set_backend(backend)
        import keras.backend as K
        from keras import ops
        
        self.K = K
        self.ops = ops

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
    def volume(self, params: dict) -> np.ndarray:
        """Calculate the volume of each peak."""
        pass

    def sum(self, X, Y, pos_x, pos_y, height, width, *args):
        """Calculate sum of peaks using Keras."""
        total = (
            self.ops.sum(
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


class GaussianModel(ImageModel):
    """Gaussian peak model."""

    def volume(self, params: dict) -> np.ndarray:
        """Calculate the volume of each Gaussian peak.
        
        For a 2D Gaussian, the volume is: height * 2π * width²
        """
        height = params["height"]
        width = params["width"]
        return height * 2 * np.pi * width**2 * self.dx**2

    def model_fn(self, x, y, pos_x, pos_y, height, width, *args):
        """Core computation for Gaussian model using Keras."""
        return height * self.ops.exp(
            -(self.ops.square(x - pos_x) + self.ops.square(y - pos_y)) / (2 * self.ops.square(width))
        )

    @staticmethod
    @njit(nopython=True)
    def model_fn_numba(x, y, pos_x, pos_y, height, width, *args):
        """Core computation for Gaussian model using numba."""
        return height * np.exp(-((x - pos_x) ** 2 + (y - pos_y) ** 2) / (2 * width**2))


class LorentzianModel(ImageModel):
    """Lorentzian peak model."""

    def volume(self, params: dict) -> np.ndarray:
        """Calculate the volume of each Lorentzian peak.
        
        For a 2D Lorentzian, the volume is: height * π * width²
        """
        height = params["height"]
        width = params["width"]
        return height * np.pi * width**2 * self.dx**2

    def model_fn(self, x, y, pos_x, pos_y, height, width, *args):
        """Core computation for Lorentzian model using Keras."""
        return height / (
            1 + (self.ops.square(x - pos_x) + self.ops.square(y - pos_y)) / self.ops.square(width)
        )

    @staticmethod
    @njit(nopython=True)
    def model_fn_numba(x, y, pos_x, pos_y, height, width, *args):
        """Core computation for Lorentzian model using numba."""
        return height / (1 + ((x - pos_x) ** 2 + (y - pos_y) ** 2) / width**2)


class VoigtModel(ImageModel):
    """Voigt peak model."""

    def volume(self, params: dict) -> np.ndarray:
        """Calculate the volume of each Voigt peak.
        
        For a 2D Voigt profile, the volume is a weighted sum of Gaussian and Lorentzian volumes:
        V = ratio * (height * 2π * width²) + (1-ratio) * (height * π * width²)
        """
        height = params["height"]
        width = params["width"]
        ratio = params["ratio"]
        
        gaussian_vol = height * 2 * np.pi * width**2 * self.dx**2
        lorentzian_vol = height * np.pi * width**2 * self.dx**2
        
        return ratio * gaussian_vol + (1 - ratio) * lorentzian_vol

    def model_fn(self, x, y, pos_x, pos_y, height, width, ratio):
        """Core computation for Voigt model using Keras."""
        # Convert width to sigma and gamma
        sigma = width
        gamma = width / self.ops.sqrt(2 * self.ops.log(2.0))
        
        # Calculate squared distance
        R2 = self.ops.square(x - pos_x) + self.ops.square(y - pos_y)
        
        # Compute Gaussian and Lorentzian parts
        gaussian_part = self.ops.exp(-R2 / (2 * sigma**2))
        lorentzian_part = gamma**3 / self.ops.power(R2 + gamma**2, 3/2)
        
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
        lorentzian_part = gamma**3 / (R2 + gamma**2) ** (3/2)
        
        # Return weighted sum
        return height * (ratio * gaussian_part + (1 - ratio) * lorentzian_part)


class GaussianKernel:
    """Gaussian kernel implementation."""
    
    def __init__(self, backend='jax'):
        """Initialize the kernel.
        
        Args:
            backend (str, optional): Backend to use ('tensorflow', 'pytorch', or 'jax'). Defaults to 'jax'.
        """
        self.backend = set_backend(backend)
        import keras.backend as K
        from keras import ops
        
        self.K = K
        self.ops = ops

    def gaussian_kernel(self, sigma):
        """Creates a 2D Gaussian kernel with the given sigma."""
        size = int(4 * sigma + 0.5) * 2 + 1  # Odd size
        x = self.ops.arange(-(size // 2), (size // 2) + 1, dtype='float32')
        X, Y = self.ops.meshgrid(x, x)
        kernel = self.ops.exp(-(X**2 + Y**2) / (2 * sigma**2))
        return kernel / self.ops.sum(kernel)

    def gaussian_filter(self, image, sigma):
        """Applies Gaussian filter to a 2D image."""
        kernel = self.gaussian_kernel(sigma)
        kernel = self.ops.reshape(kernel, (kernel.shape[0], kernel.shape[1], 1, 1))
        image = self.ops.expand_dims(self.ops.expand_dims(image, 0), -1)
        filtered = self.K.conv2d(image, kernel, padding='same')
        return self.ops.squeeze(filtered, 0)


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