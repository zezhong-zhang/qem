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

    def _sum(self, X, Y, pos_x, pos_y, height, width, *kargs, local=False):
        """Calculate all peaks either globally or locally.
        
        Args:
            X (array): X coordinates mesh
            Y (array): Y coordinates mesh
            pos_x (array): X positions of peaks in the same coordinate space as X
            pos_y (array): Y positions of peaks in the same coordinate space as Y
            height (array): Heights of peaks
            width (array): Widths of peaks
            *args: Additional arguments for specific peak models
            local (bool, optional): If True, calculate peaks locally within a fixed window. Defaults to False.
            
        Returns:
            array: Sum of all peaks plus background
        """
        if not local:
            # Calculate all peaks at once and sum them
            peaks = self.model_fn(
                X[:, :, None], Y[:, :, None],
                pos_x[None, None, :], pos_y[None, None, :],
                height, width, *kargs
            )
            return self.ops.sum(peaks, axis=-1) + self.background
        else:
            # Local calculation with parallel processing
            width_max = width.max() if isinstance(width, (list, tuple, np.ndarray)) else width            
            # Window size in pixels
            window_size = int(5 * width_max)  # Fixed window size of 5*width in grid units
            if window_size % 2 == 0:
                window_size += 1  # Ensure odd window size for centered peak
            
            half_size = window_size // 2
            
            # Create fixed-size local window coordinates
            window_x = self.ops.arange(-half_size, half_size + 1)
            window_y = self.ops.arange(-half_size, half_size + 1)
            window_X, window_Y = self.ops.meshgrid(window_x, window_y)
            
            # Calculate all local peaks in parallel
            # Reshape window coordinates to (window_size^2, 2) for broadcasting
            window_coords = self.ops.stack([window_X.flatten(), window_Y.flatten()], axis=-1)
            
            # Calculate global indices for each peak
            x_indices = self.ops.cast(
                self.ops.round((pos_x - X[0, 0])),
                dtype='int32'
            )
            y_indices = self.ops.cast(
                self.ops.round((pos_y - Y[0, 0])),
                dtype='int32'
            )
            
            # Calculate actual coordinates for each window point relative to peak centers
            peak_coords = self.ops.stack([x_indices, y_indices], axis=-1)  # Shape: (n_peaks, 2)
            window_offsets = window_coords[None, :, :]  # Shape: (1, window_size^2, 2)
            peak_centers = peak_coords[:, None, :]  # Shape: (n_peaks, 1, 2)
            
            # Global coordinates for all window points for all peaks
            # Shape: (n_peaks, window_size^2, 2)
            global_coords = peak_centers + window_offsets
            
            # Calculate valid mask for points within image bounds
            valid_x = (global_coords[..., 0] >= 0) & (global_coords[..., 0] < X.shape[1])
            valid_y = (global_coords[..., 1] >= 0) & (global_coords[..., 1] < X.shape[0])
            valid_mask = valid_x & valid_y
            
            # Calculate local peaks using model_fn on window coordinates
            local_peaks = self.model_fn(
                window_X[None, :, :],  # Shape: (1, window_size, window_size)
                window_Y[None, :, :],  # Shape: (1, window_size, window_size)
                self.ops.zeros_like(pos_x)[:, None, None],  # Center each peak at (0,0)
                self.ops.zeros_like(pos_y)[:, None, None],
                height[:, None, None],
                width if isinstance(width, (float, int)) else width[:, None, None],
                *[arg if isinstance(arg, (float, int)) else arg[:, None, None] for arg in kargs]
            )
            
            # Initialize output array with background
            total = self.ops.zeros_like(X) + self.background
            
            # Add each peak's contribution to the total at the correct positions
            local_peaks_flat = local_peaks.reshape(len(pos_x), -1)  # Flatten window dimensions
            for i in range(len(pos_x)):
                valid_points = valid_mask[i]
                valid_coords = global_coords[i][valid_points]
                valid_values = local_peaks_flat[i][valid_points]
                
                # Use scatter_add to accumulate values at valid coordinates
                total = total.at[valid_coords[:, 1], valid_coords[:, 0]].add(valid_values)
            
            return total

    def sum(self, X, Y, pos_x, pos_y, height, width, *kargs, local=False):
        """Calculate sum of peaks using Keras.
        
        Args:
            X (array): X coordinates mesh
            Y (array): Y coordinates mesh
            pos_x (array): X positions of peaks
            pos_y (array): Y positions of peaks
            height (array): Heights of peaks
            width (array): Widths of peaks
            *args: Additional arguments for specific peak models
            local (bool, optional): If True, calculate peaks locally within a fixed window. Defaults to False.
            
        Returns:
            array: Sum of all peaks plus background
        """
        return self._sum(X, Y, pos_x, pos_y, height, width, *kargs, local=local)

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

    def sum(self, X, Y, pos_x, pos_y, height, width, ratio, local=False):
        """Calculate sum of peaks using Keras.
        
        Args:
            X (array): X coordinates mesh
            Y (array): Y coordinates mesh
            pos_x (array): X positions of peaks
            pos_y (array): Y positions of peaks
            height (array): Heights of peaks
            width (array): Widths of peaks
            ratio (array): Ratios of peaks
            local (bool, optional): If True, calculate peaks locally within a fixed window. Defaults to False.
        
        Returns:
            array: Sum of all peaks plus background
        """
        return self._sum(X, Y, pos_x, pos_y, height, width, ratio, local=local)


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
        # Ensure both image and kernel are float32
        image = self.ops.cast(image, 'float32')
        kernel = self.gaussian_kernel(sigma)
        # Add channel dimensions for input and kernel
        image = self.ops.expand_dims(self.ops.expand_dims(image, 0), -1)  # [1, H, W, 1]
        kernel = self.ops.expand_dims(self.ops.expand_dims(kernel, -1), -1)  # [H, W, 1, 1]
        filtered = self.ops.conv(image, kernel, padding='same')
        return self.ops.squeeze(filtered)  # Remove extra dimensions