from abc import abstractmethod

import numpy as np
from dotenv import load_dotenv
from numba import jit as njit

load_dotenv()
from qem.utils import torch_compat as keras

from qem.utils.params import safe_convert_to_numpy, safe_convert_to_tensor

class ImageModel(keras.Model):
    """Base class for all image models."""

    def __init__(self, dx: float=1.0):
        """Initialize the model.
        
        Args:
            dx (float, optional): Pixel size. Defaults to 1.0.
        """
        super().__init__()
        self.dx = dx
        self.input_params = None
            
    def set_params(self, params):
        # Set params as tensors, but do not build variables yet
        self.input_params = {k: keras.ops.convert_to_tensor(v) for k, v in params.items()}
        # If already built and shapes match, update values
        if self.built:
            self.update_params(self.input_params)

    def update_params(self, params):
        """Update the model parameters (values only, not shapes)."""
        # Configuration parameters that are not model weights
        config_params = {'same_width', 'atom_types'}
        
        for key, value in params.items():
            if key in config_params:
                # Skip configuration parameters - they're handled in input_params
                continue
            elif hasattr(self, key):
                current_value = getattr(self, key)
                current_value.assign(value)
            else:
                raise ValueError(f"Parameter {key} does not exist in the model.")

    def build(self, input_shape=(1,)):
        if self.input_params is None:
            raise ValueError("initial_params must be set before building the model.")
        # If already built and shapes match, do nothing
        if self.built:
            return
        # Otherwise, create new variables
        self.pos_x = self.add_weight(shape=(self.input_params['pos_x'].shape[0],), initializer=keras.initializers.Constant(self.input_params['pos_x']), name="pos_x")
        self.pos_y = self.add_weight(shape=(self.input_params['pos_y'].shape[0],), initializer=keras.initializers.Constant(self.input_params['pos_y']), name="pos_y")
        self.height = self.add_weight(shape=(self.input_params['height'].shape[0],), initializer=keras.initializers.Constant(self.input_params['height']), name="height")
        self.width = self.add_weight(shape=(self.input_params['width'].shape[0],), initializer=keras.initializers.Constant(self.input_params['width']), name="width")
        self.background = self.add_weight(shape=(), initializer=keras.initializers.Constant(self.input_params['background']), name="background")
        super().build(input_shape)
        
    def get_params(self):
        dict_params = {
            "pos_x": keras.ops.convert_to_tensor(self.pos_x),
            "pos_y": keras.ops.convert_to_tensor(self.pos_y),
            "height": keras.ops.convert_to_tensor(self.height),
            "width": keras.ops.convert_to_tensor(self.width),
            "background": keras.ops.convert_to_tensor(self.background),
            "same_width": self.input_params.get('same_width', False),
        }
        # atom_types may not be present in all models (e.g., convolution model)
        if 'atom_types' in self.input_params:
            dict_params['atom_types'] = keras.ops.convert_to_tensor(self.input_params['atom_types'])
        else:
            # Default to zeros if not present
            dict_params['atom_types'] = keras.ops.convert_to_tensor(
                keras.ops.zeros_like(self.height, dtype='int32')
            )
        if hasattr(self, 'ratio'):
            dict_params['ratio'] = keras.ops.convert_to_tensor(self.ratio)
        return dict_params

    def call(self, inputs):
        """Forward pass of the model."""
        x_grid, y_grid = inputs
        return self.sum(x_grid, y_grid)

    @abstractmethod
    def model_fn(self, x, y, pos_x, pos_y, height, width, *args):
        """Core model function that defines the peak shape."""
        pass


    @abstractmethod
    def volume(self, params: dict) -> np.ndarray:
        """Calculate the volume of each peak."""
        pass

    # The revised method starts here.
    def sum(self, x_grid: np.ndarray, y_grid: np.ndarray, local: bool = True):
        """Calculates the sum of all peaks on a grid, with an optional background.

        This method supports both a global calculation and a memory-efficient local
        calculation suitable for JIT compilation.

        Args:
            x_grid (array): A 2D array of X coordinates (from meshgrid).
            y_grid (array): A 2D array of Y coordinates (from meshgrid).
            local (bool, optional): If True, calculates peaks in local windows to
                conserve memory. This approach is JIT-compatible. Defaults to True.

        Returns:
            array: A 2D array representing the rendered image of peaks plus background.
        """
        x_grid = keras.ops.convert_to_tensor(x_grid, dtype="float32")
        y_grid = keras.ops.convert_to_tensor(y_grid, dtype="float32")

        # Squeeze batch dimension for processing, if it exists.
        has_batch_dim = len(x_grid.shape) > 2
        if has_batch_dim:
            x_grid = keras.ops.squeeze(x_grid, axis=0)
            y_grid = keras.ops.squeeze(y_grid, axis=0)

        # Prepare arguments for the peak model function.
        # This is refactored to avoid code duplication.
        width = self.width
        ratio = self.ratio if hasattr(self, 'ratio') else None
        if self.input_params.get('same_width', False):
            atom_types = keras.ops.cast(
                safe_convert_to_tensor(self.input_params['atom_types']), dtype='int32'
            )
            width = keras.ops.take(self.width, atom_types)
            if ratio is not None:
                ratio = keras.ops.take(self.ratio, atom_types)
        
        width_ratio_args = (width, ratio) if ratio is not None else (width,)

        if not local:
            # Global calculation: simpler but uses more memory.
            peaks = self.model_fn(
                x_grid[..., None], y_grid[..., None],  # Broadcasting to match peak dimensions
                self.pos_x, self.pos_y,
                self.height, *width_ratio_args
            )
            result = keras.ops.sum(peaks, axis=-1) + self.background
        else:
            # --- JIT-Compatible Local Calculation ---
            
            # 1. Define a static window size. For JIT compilation, this value
            # cannot be a traced tensor; it must be a concrete Python integer.
            # We assume the width parameter used for this is a static NumPy array.
            max_width = np.max(safe_convert_to_numpy(self.input_params['width']))
            window_size = int(max_width * 4)

            # 2. Create a local coordinate grid for the window.
            window_coords = keras.ops.arange(-window_size, window_size + 1, dtype=x_grid.dtype)
            local_x_grid, local_y_grid = keras.ops.meshgrid(window_coords, window_coords)

            # 3. Calculate all local peak shapes centered at (0,0).
            peak_params = (
                keras.ops.mod(self.pos_x, 1), keras.ops.mod(self.pos_y, 1),
                self.height, *width_ratio_args
            )
            local_peaks = self.model_fn(local_x_grid[..., None], local_y_grid[..., None], *peak_params)

            # 4. Calculate the global coordinates where each local peak point should go.
            pos_x_int = keras.ops.floor(self.pos_x)
            pos_y_int = keras.ops.floor(self.pos_y)
            global_x = keras.ops.expand_dims(local_x_grid, -1) + pos_x_int
            global_y = keras.ops.expand_dims(local_y_grid, -1) + pos_y_int

            # 5. Create a mask to identify points that fall within the image boundaries.
            mask = (global_x >= 0) & (global_x < x_grid.shape[1]) & (global_y >= 0) & (global_y < y_grid.shape[0])

            # 6. Apply the mask to zero out contributions from out-of-bounds points.
            # This is the key to JIT compatibility, as it preserves the static shape of the array.
            masked_peaks = keras.ops.where(mask, local_peaks, 0.0)

            # 7. Clip coordinates to prevent scatter operations from failing on out-of-bounds indices.
            # The values for these points are already zeroed out by the mask.
            h, w = x_grid.shape
            global_x_safe = keras.ops.clip(global_x, 0, w - 1)
            global_y_safe = keras.ops.clip(global_y, 0, h - 1)

            # 8. Scatter the masked peaks onto the canvas with PyTorch's scatter_add.
            total = keras.ops.zeros_like(x_grid, dtype='float32')
            masked_peaks_flat = keras.ops.reshape(masked_peaks, [-1])
            g_y_flat = keras.ops.cast(keras.ops.reshape(global_y_safe, [-1]), 'int64')
            g_x_flat = keras.ops.cast(keras.ops.reshape(global_x_safe, [-1]), 'int64')
            indices = g_y_flat * w + g_x_flat

            total_flat = keras.ops.reshape(total, (-1,))
            total = total_flat.scatter_add(0, indices, masked_peaks_flat).reshape(total.shape)

            result = total + self.background

        # Add batch dimension back if it was originally present.
        if has_batch_dim:
            result = keras.ops.expand_dims(result, axis=0)

        return result



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
        """Core computation for Gaussian model using PyTorch."""
        return height * keras.ops.exp(
            -(keras.ops.square(x - pos_x) + keras.ops.square(y - pos_y)) / (2 * keras.ops.square(width))
        )

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
        """Core computation for Lorentzian model using PyTorch."""
        return height / (
            1 + (keras.ops.square(x - pos_x) + keras.ops.square(y - pos_y)) / keras.ops.square(width)
        )


class VoigtModel(ImageModel):
    """Voigt peak model."""
    def __init__(self, dx: float=1.0):
        """Initialize the model.

        Args:
            dx (float, optional): Pixel size. Defaults to 1.0.
        """
        super().__init__(dx)
        self.ratio = None

    def set_params(self, params):
        super().set_params(params)
        # The ratio parameter will be handled by the base class update_params method

    def build(self, input_shape=None):
        if self.input_params is None:
            raise ValueError("initial_params must be set before building the model.")
        # If already built and shapes match, do nothing
        if self.built:
            return
            
        # Handle both scalar and array ratio parameters
        ratio_param = self.input_params['ratio']
        if hasattr(ratio_param, 'shape') and len(ratio_param.shape) > 0:
            # Array case (different ratios for each peak)
            ratio_shape = (ratio_param.shape[0],)
        else:
            # Scalar case (same ratio for all peaks)
            ratio_shape = ()
        
        self.ratio = self.add_weight(
            shape=ratio_shape, 
            initializer=keras.initializers.Constant(ratio_param), 
            name="ratio"
        )
        # Call parent build method
        super().build(input_shape)


    def get_params(self):
        params = super().get_params()
        params['ratio'] = keras.ops.convert_to_tensor(self.ratio)
        return params

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
        """Core computation for Voigt model using PyTorch."""
        # Convert width to sigma and gamma
        sigma = width
        gamma = width / keras.ops.sqrt(2 * keras.ops.log(2.0))
        
        # Calculate squared distance
        r2 = keras.ops.square(x - pos_x) + keras.ops.square(y - pos_y)
        
        # Compute Gaussian and Lorentzian parts
        gaussian_part = keras.ops.exp(-r2 / (2 * sigma**2))
        lorentzian_part = gamma**3 / keras.ops.power(r2 + gamma**2, 3/2)
        
        # Return weighted sum
        return height * (ratio * gaussian_part + (1 - ratio) * lorentzian_part)


class GaussianKernel:
    """Gaussian kernel implementation."""

    def __init__(self):
        """Initialize the kernel."""

    def gaussian_kernel(self, sigma):
        """Creates a 2D Gaussian kernel with the given sigma."""
        size = int(4 * sigma + 0.5) * 2 + 1  # Odd size
        x = keras.ops.arange(-(size // 2), (size // 2) + 1, dtype='float32')
        x_grid, y_grid = keras.ops.meshgrid(x, x)
        kernel = keras.ops.exp(-(x_grid**2 + y_grid**2) / (2 * sigma**2))
        return kernel / keras.ops.sum(kernel)

    def gaussian_filter(self, image, sigma):
        """Applies Gaussian filter to a 2D image."""
        # Ensure both image and kernel are float32
        image = keras.ops.cast(image, 'float32')
        kernel = self.gaussian_kernel(sigma)
        # Add channel dimensions for input and kernel
        image = keras.ops.expand_dims(keras.ops.expand_dims(image, 0), -1)  # [1, H, W, 1]
        kernel = keras.ops.expand_dims(keras.ops.expand_dims(kernel, -1), -1)  # [H, W, 1, 1]
        filtered = keras.ops.conv(image, kernel, padding='same')
        return keras.ops.squeeze(filtered)  # Remove extra dimensions

@njit
def gaussian_2d_single(xy, pos_x, pos_y, height, width, background):
    """2D Gaussian function for single atom."""
    x_grid, y_grid = xy
    return (
        height
        * np.exp(
            -((x_grid[:,:,None] - pos_x) ** 2 + (y_grid[:,:,None] - pos_y) ** 2) / (2 * width**2)
        ) + background
    ).ravel()
