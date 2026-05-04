import torch
# Standard library imports
import copy
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import nullcontext
from typing import Any

# Third-party library imports
import matplotlib.pyplot as plt
import numpy as np
from matplotlib_scalebar.scalebar import ScaleBar
from numpy.typing import NDArray
from scipy.optimize import curve_fit
from skimage.feature import peak_local_max
from scipy.ndimage import sobel, binary_erosion, binary_dilation, gaussian_filter,laplace
from skimage.morphology import remove_small_objects, label
from skimage.measure import find_contours
from matplotlib.path import Path
from tqdm import tqdm

# Application-specific imports
from qem.analysis.crystal_analyzer import CrystalAnalyzer
from qem.analysis.region import Regions, Region
from qem.viz.select import (
    GetAtomSelection,
    GetRegionSelection,
    InteractivePlot,
)
from qem.fit.model import (
    ImageModel,
    GaussianKernel,
    GaussianModel,
    LorentzianModel,
    VoigtModel,
    gaussian_2d_single,
)
from qem.processing import butterworth_window
from qem.fit.refine import calculate_center_of_mass
from qem.utils.params import (
    safe_convert_to_numpy,
    safe_convert_to_tensor,
    safe_deepcopy_params,
    safe_stop_gradient,
)
from qem.utils.backend import release_backend_memory
from qem.utils.arrays import get_random_indices_in_batches
from qem.viz.geometry import remove_close_coordinates
from qem.fit.voronoi import voronoi_integrate, voronoi_point_record
from qem.fit.background import Background, estimate_background
from qem.fit.solver import (
    ParameterValidator,
    DesignMatrixBuilder,
    LinearSystemSolver,
    SolutionProcessor,
)
from qem.fit.validation import FitterValidator, FitParamsValidator
from qem.utils.memory import MemoryMonitor

import h5py

# Only configure logging if not already configured
if not logging.getLogger().handlers:
    logging.basicConfig(level=logging.INFO, format='%(levelname)s:%(name)s:%(message)s')


class Fitter:
    def __init__(
        self,
        image: np.ndarray,
        dx: float = 1.0,
        units: str = "A",
        elements: list[str] = None,  # type: ignore
        model_type: str = "gaussian",
        same_width: bool = True,
        pbc: bool = False,
        fit_background: bool = True,
        monitor_memory: bool = False,
    ):
        """Initialize the Fitter.

        Args:
            image: 2-D STEM image as a numpy array.
            dx: Pixel size in `units`.
            units: Length units (e.g. ``"A"``, ``"nm"``).
            elements: Atomic species present in the image.
            model_type: Peak model — ``"gaussian"`` / ``"lorentzian"`` / ``"voigt"``.
            same_width: Share width across atoms of the same type.
            pbc: Periodic boundary conditions on the image grid.
            fit_background: Whether to include a background term in the fit.
            monitor_memory: If True, attach a MemoryMonitor for `with monitor_operation`.
        """
        self.image = image
        self.dx = dx
        self.elements = elements
        self.model_type = model_type
        self.units = units
        self.same_width = bool(same_width)
        self.pbc = bool(pbc)
        self.fit_background = bool(fit_background)
        self.monitor_memory = monitor_memory

        self.memory_monitor = MemoryMonitor() if self.monitor_memory else None

        logging.info(
            "Initializing Fitter with %s image, dx=%s %s, model=%s",
            self.image.shape, self.dx, self.units, self.model_type,
        )

        self.model = self._select_model()
        self.kernel = GaussianKernel()
        self._window = None
        self._window_t: torch.Tensor | None = None  # cached float32 view used in loss().

        self._atom_types = np.array([])
        self._coordinates = np.array([])
        self.coordinates_history: dict = {}

        # Boundary penalty + adaptive edge loss off by default.
        self.use_boundary_penalty = False
        self.boundary_margin = 2.0
        self.boundary_strength = 0.01
        self.use_adaptive_edge_loss = False

        self.coordinates_state = 0
        self.init_background = 0.0
        self.prediction = np.zeros_like(self.image)

        self.params = None
        self.converged = False
        self.ny, self.nx = image.shape
        self.regions = Regions(image=image)
        self.initialize_grid()
        self.background_estimator = Background(self.image, self.dx)

    # I/O functions
    def save(self, filepath: str) -> None:
        """
        Save Fitter state to HDF5 file.
        
        Args:
            filepath: Path to save the HDF5 file
        """
        with h5py.File(filepath, 'w') as f:
            # Save input image and parameters
            f.create_dataset('image', data=self.image)
            f.attrs['dx'] = self.dx
            f.attrs['units'] = self.units
            f.attrs['model_type'] = self.model_type
            f.attrs['same_width'] = self.same_width
            f.attrs['pbc'] = self.pbc
            f.attrs['fit_background'] = self.fit_background
            
            # Save fitted parameters
            if self.params is not None:
                params_group = f.create_group('params')
                for key, value in self.params.items():
                    # Handle different tensor types and devices
                    if hasattr(value, 'cpu') and hasattr(value, 'numpy'):
                        # Handle PyTorch GPU tensors
                        value = value.cpu().detach().numpy()
                    elif hasattr(value, 'numpy'):
                        # Handle NumPy or CPU tensors
                        value = value.numpy()
                    elif hasattr(value, 'device') and 'GPU' in str(value.device):
                        # Additional GPU tensor handling
                        try:
                            value = np.array(value)
                        except (TypeError, RuntimeError):
                            value = np.asarray(value)
                    params_group.create_dataset(key, data=value)
            
            # Save fitted image
            if hasattr(self, 'prediction') and self.prediction is not None:
                f.create_dataset('prediction', data=self.prediction)
            
            # Save coordinates and atom types
            f.create_dataset('coordinates', data=self.coordinates)
            f.create_dataset('atom_types', data=self.atom_types)
            
            # Save elements list
            if self.elements is not None:
                f.create_dataset('elements', data=[e.encode('utf-8') for e in self.elements])
            
            # Save voronoi data if available
            if hasattr(self, '_voronoi_volume') and self._voronoi_volume is not None:
                f.create_dataset('voronoi_volume', data=self._voronoi_volume)
            if hasattr(self, '_voronoi_map') and self._voronoi_map is not None:
                f.create_dataset('voronoi_map', data=self._voronoi_map)

    def load(self, filepath: str) -> 'Fitter':
        """
        Load Fitter state from HDF5 file.
        
        Args:
            filepath: Path to the HDF5 file to load
            
        Returns:
            Fitter instance with loaded state
            
        Raises:
            FileNotFoundError: If the file doesn't exist
            KeyError: If required attributes are missing
            ValueError: If data validation fails
        """
        try:
            with h5py.File(filepath, 'r') as f:
                # Validate required datasets exist
                required_datasets = ['image']
                for dataset in required_datasets:
                    if dataset not in f:
                        raise KeyError(f"Missing required dataset: {dataset}")
                
                # Validate required attributes exist
                required_attrs = ['dx', 'units', 'model_type', 'same_width', 'pbc', 'fit_background']
                for attr in required_attrs:
                    if attr not in f.attrs:
                        raise KeyError(f"Missing required attribute: {attr}")
                
                # Load input image and parameters
                self.image = f['image'][:]
                self.image_tensor = safe_convert_to_tensor(self.image)
                self.dx = float(f.attrs['dx'])
                self.units = str(f.attrs['units'])
                self.model_type = str(f.attrs['model_type'])
                self.same_width = bool(f.attrs['same_width'])
                self.pbc = bool(f.attrs['pbc'])
                self.fit_background = bool(f.attrs['fit_background'])
                
                # Validate image data
                if not isinstance(self.image, np.ndarray) or self.image.ndim != 2:
                    raise ValueError("Image must be a 2D numpy array")
                
                # Load fitted parameters
                if 'params' in f:
                    params = {}
                    params_group = f['params']
                    for key in params_group.keys():
                        data = params_group[key]
                        if data.ndim == 0:
                            params[key] = data[()]
                        else:
                            params[key] = data[:]
                    
                    # Validate parameter shapes
                    num_coords = len(self.coordinates)
                    if 'pos_x' in params and 'pos_y' in params:
                        if len(params['pos_x']) != len(params['pos_y']):
                            raise ValueError("pos_x and pos_y must have the same length")
                        num_coords = len(params['pos_x'])
                        self.coordinates = np.stack([params['pos_x'], params['pos_y']], axis=1)
                    
                    # Validate other parameters
                    for key, value in params.items():
                        if key in ['pos_x', 'pos_y', 'height']:
                            if len(value) != num_coords:
                                raise ValueError(f"Parameter {key} length mismatch with coordinates")

                    self.params = {k: safe_convert_to_tensor(v) for k, v in params.items()}

                # Load fitted image
                if 'prediction' in f:
                    prediction = f['prediction'][:]
                    if prediction.shape != self.image.shape:
                        raise ValueError("Prediction shape must match image shape")
                    self.prediction = prediction

                # Load coordinates and atom types
                if 'coordinates' in f:
                    coords = f['coordinates'][:]
                    if coords.ndim != 2 or coords.shape[1] != 2:
                        raise ValueError("Coordinates must be Nx2 array")
                    self.coordinates = coords

                if 'atom_types' in f:
                    atom_types = f['atom_types'][:]
                    if len(atom_types) != len(self.coordinates):
                        raise ValueError("Atom types length must match coordinates length")
                    self.atom_types = atom_types

                # Load elements
                if 'elements' in f:
                    try:
                        elements_bytes = f['elements'][:]
                        self.elements = [e.decode('utf-8') if isinstance(e, bytes) else str(e) 
                                           for e in elements_bytes]
                    except (UnicodeDecodeError, AttributeError):
                        logging.warning("Could not decode elements list, using default")
                        self.elements = None

                # Load voronoi data
                if 'voronoi_volume' in f:
                    voronoi_volume = f['voronoi_volume'][:]
                    if len(voronoi_volume) != len(self.coordinates):
                        raise ValueError("Voronoi volume length must match coordinates length")
                    self._voronoi_volume = voronoi_volume
                
                if 'voronoi_map' in f:
                    voronoi_map = f['voronoi_map'][:]
                    if voronoi_map.shape != self.image.shape:
                        raise ValueError("Voronoi map shape must match image shape")
                    self._voronoi_map = voronoi_map
                
                logging.info(f"Successfully loaded Fitter state from {filepath}")
                return self
                
        except FileNotFoundError:
            raise FileNotFoundError(f"HDF5 file not found: {filepath}")
        except Exception as e:
            raise ValueError(f"Error loading HDF5 file: {str(e)}") from e

    def get_memory_usage(self) -> dict:
        """
        Get current memory usage statistics.
        
        Returns:
            Dictionary with memory usage information. Returns empty dict if monitoring is disabled.
        """
        if self.memory_monitor is None:
            return {}
        return self.memory_monitor.get_memory_info()

    def enable_memory_monitoring(self) -> None:
        """Enable memory monitoring if it was disabled."""
        if self.memory_monitor is None:
            self.memory_monitor = MemoryMonitor()
            logging.info("Memory monitoring enabled")

    def disable_memory_monitoring(self) -> None:
        """Disable memory monitoring."""
        if self.memory_monitor is not None:
            self.memory_monitor = None
            logging.info("Memory monitoring disabled")

    def detach(self) -> dict[str, Any]:
        """Snapshot every fitted parameter to numpy.

        Call this once you're done fitting and want everything in numpy
        for plotting / saving / external consumption. Internal hot-path
        state stays on-device until this point.
        """
        out: dict[str, Any] = {}
        for k, v in (self.params or {}).items():
            out[k] = v.detach().cpu().numpy() if torch.is_tensor(v) else v
        out["prediction"] = (
            self.prediction.copy()
            if isinstance(self.prediction, np.ndarray)
            else np.asarray(self.prediction)
        )
        return out

    # Init grids and models
    def initialize_grid(self):
        """Initialize the coordinate grids for the model."""
        self.image_tensor = torch.as_tensor(gaussian_filter(self.image,1), dtype=torch.float32)
        x = torch.arange(self.nx, dtype=torch.float32)
        y = torch.arange(self.ny, dtype=torch.float32)
        x_grid, y_grid = torch.meshgrid(x, y, indexing="xy")
        self.x_grid = torch.as_tensor(x_grid, dtype=torch.float32)
        self.y_grid = torch.as_tensor(y_grid, dtype=torch.float32)
        # Pre-batched views for optimize() — torch view, free.
        self.x_grid_batched = self.x_grid.unsqueeze(0)
        self.y_grid_batched = self.y_grid.unsqueeze(0)

    def _select_model(self):
        """Create a new model instance based on the model type."""
        if self.model_type == "gaussian":
            model = GaussianModel(dx=float(self.dx))
        elif self.model_type == "lorentzian":
            model = LorentzianModel(dx=float(self.dx))
        elif self.model_type == "voigt":
            model = VoigtModel(dx=float(self.dx))
        elif self.model_type == "convolution":
            # For convolution model, PSF kernel must be set before model selection
            # This is used by PtychoFit which overrides _select_model
            if not hasattr(self, '_psf_kernel') or self._psf_kernel is None:
                raise ValueError(
                    "Convolution model requires PSF kernel. "
                    "Use PtychoFit instead of Fitter directly."
                )
            from qem.fit.potential import ConvolutionImageModel
            model = ConvolutionImageModel(
                psf_kernel=self._psf_kernel,
                dx=float(self.dx),
            )
        else:
            raise ValueError(f"Model type {self.model_type} not supported.")
        return model

    def _create_fitting_model(self, params: dict):
        """
        Create a model instance for fitting with appropriate parameter trainability.
        
        Args:
            params: Parameters dictionary.
            
        Returns:
            ImageModel: Model with background trainability based on fit_background setting.
        """
        model = self._select_model()
        model.set_params(params)
        if not model.built:
            model.build()
        
        # Handle background trainability based on fit_background setting
        if not self.fit_background:
            if hasattr(model, 'background') and hasattr(model.background, 'requires_grad_'):
                model.background.requires_grad_(False)
            if hasattr(model, 'background_scale') and hasattr(model.background_scale, 'requires_grad_'):
                model.background_scale.requires_grad_(False)
            
        return model

    # init parameters
    def guess_radius(self):
        """
        Estimate the density of atomic columns in an image.

        Parameters:
        id (int): Identifier for a specific image or set of coordinates.

        Returns:
        tuple: density, influence_map, background_region
        """
        num_coordinates = self.coordinates.shape[0]
        if num_coordinates == 0:
            raise ValueError("No coordinates found for the given id.")

        rate, rate_max, n_filled, n = 1, 1, 0, 0
        nx, ny = self.image.shape

        while rate > 0.5 * rate_max:
            influence_map = np.zeros((nx, ny))
            for i in range(num_coordinates):
                i_l = np.maximum(self.coordinates[i, 0] - n, 0).astype(np.int32)
                i_r = np.minimum(self.coordinates[i, 0] + n, self.nx).astype(np.int32)
                i_u = np.maximum(self.coordinates[i, 1] - n, 0).astype(np.int32)
                i_d = np.minimum(self.coordinates[i, 1] + n, self.ny).astype(np.int32)
                influence_map[i_l : i_r + 1, i_u : i_d + 1] = 1
            if n == 0:
                rate = (np.sum(influence_map) - n_filled) / num_coordinates
            else:
                rate = (np.sum(influence_map) - n_filled) / (8 * n) / num_coordinates
            n_filled = np.sum(influence_map)
            rate_max = max(rate_max, rate)
            n += 1

        # Scaled factors
        n1 = int(np.round((n - 1) * 10))
        n2 = int(np.round((n - 1) * 1))

        influence_map = np.zeros((nx, ny))
        direct_influence_map = np.zeros((nx, ny))

        for i in range(num_coordinates):
            # Calculate the indices for the larger area (influence_map)
            i_l = np.maximum(self.coordinates[i, 0] - n1, 0).astype(np.int32)
            i_r = np.minimum(self.coordinates[i, 0] + n1, nx).astype(np.int32)
            i_u = np.maximum(self.coordinates[i, 1] - n1, 0).astype(np.int32)
            i_d = np.minimum(self.coordinates[i, 1] + n1, ny).astype(np.int32)
            influence_map[i_l : i_r + 1, i_u : i_d + 1] = 1

            # Calculate the indices for the smaller area (direct_influence_map)
            i_l = np.maximum(self.coordinates[i, 0] - n2, 0).astype(np.int32)
            i_r = np.minimum(self.coordinates[i, 0] + n2, nx).astype(np.int32)
            i_u = np.maximum(self.coordinates[i, 1] - n2, 0).astype(np.int32)
            i_d = np.minimum(self.coordinates[i, 1] + n2, ny).astype(np.int32)
            direct_influence_map[i_l : i_r + 1, i_u : i_d + 1] = 1

        radius = (np.sum(direct_influence_map) / num_coordinates) ** (1 / 2) / np.pi

        background_region = influence_map - direct_influence_map
        return radius, direct_influence_map, background_region

    def enable_2d_background(self,
                           method: str = 'photutils',
                           **kwargs) -> dict:
        """
        Enable 2D background estimation for the image fitting.
        
        Args:
            method: Background estimation method ('photutils', 'median', 'polynomial')
            **kwargs: Additional parameters for background estimation
            
        Returns:
            Dictionary with background estimation information
        """
        logging.info("Enabling 2D background estimation with method: %s", method)
        
        # Enable 2D background in the estimator
        info = self.background_estimator.enable_2d_background(method=method, **kwargs)
        
        # Update fit_background to use 2D mode
        self.fit_background = True
        
        logging.info("2D background estimation completed: scale=%.3f", info['initial_scale'])
        return info
    
    def disable_2d_background(self):
        """Disable 2D background estimation and revert to scalar background."""
        self.background_estimator.disable_2d_background()
        logging.info("2D background estimation disabled")
    
    def enable_boundary_penalty(self, margin: float = 2.0, strength: float = 0.01):
        """
        Enable soft boundary penalty to improve edge atom fitting.
        
        This adds a smooth penalty term to the loss function that gently pushes
        atoms back when they get too close to image boundaries, without hard clipping
        that would zero out gradients.
        
        Args:
            margin: Distance from edge (in pixels) where penalty starts. Default 2.0
            strength: Penalty strength multiplier. Higher = stronger constraint. Default 0.01
                     Recommended range: 0.001 to 0.1
        
        Example:
            >>> fitter.enable_boundary_penalty(margin=3.0, strength=0.01)
            >>> fitter.fit_global()  # Edge atoms will be constrained
        """
        self.use_boundary_penalty = True
        self.boundary_margin = margin
        self.boundary_strength = strength
        
        logging.info(f"Boundary penalty enabled: margin={margin}, strength={strength}")
    
    def disable_boundary_penalty(self):
        """Disable boundary penalty constraint."""
        self.use_boundary_penalty = False
        
        logging.info("Boundary penalty disabled")
    
    def enable_adaptive_edge_loss(self):
        """
        Enable adaptive gradient boosting for edge peaks.
        
        This amplifies the gradient signal for peaks with low visibility
        (near or outside image boundaries), helping the optimizer converge
        to the correct position even when most of the peak is invisible.
        
        Example:
            >>> fitter.enable_adaptive_edge_loss()
            >>> fitter.fit_global()  # Gradient boosting active for edge peaks
        """
        self.use_adaptive_edge_loss = True
        
        logging.info("Adaptive edge loss enabled (gradient boosting for edge peaks)")
    
    def disable_adaptive_edge_loss(self):
        """Disable adaptive edge loss."""
        self.use_adaptive_edge_loss = False
        
        logging.info("Adaptive edge loss disabled")
    
    def fit_with_edge_correction(self, maxiter=300, step_size=0.01, verbose=True):
        """
        Two-stage fitting optimized for edge peaks.
        
        Stage 1: Fit with positions constrained inside to get height/width estimates
        Stage 2: Boost parameters and refit with positions allowed outside
        
        This addresses the initialization bias for edge peaks where height and
        width are underestimated when initialized from clipped positions.
        
        Args:
            maxiter: Maximum iterations per stage
            step_size: Learning rate
            verbose: Whether to print progress
            
        Returns:
            Optimized parameters
            
        Example:
            >>> fitter.coordinates = edge_coordinates
            >>> fitter.disable_edge_window()
            >>> params = fitter.fit_with_edge_correction()
        """
        if verbose:
            logging.info("Starting two-stage edge-corrected fitting")
        
        # Stage 1: Constrained fit (positions stay inside)
        if verbose:
            logging.info("Stage 1: Fitting with positions constrained inside")
        
        # Temporarily disable boundary penalty
        original_boundary_state = getattr(self, 'use_boundary_penalty', False)
        self.use_boundary_penalty = False
        
        params_stage1 = self.fit_global(
            maxiter=maxiter//2,
            step_size=step_size,
            verbose=False
        )
        
        # Stage 2: Correct parameters and refit unconstrained
        if verbose:
            logging.info("Stage 2: Correcting parameters and refitting unconstrained")
        
        # Detect edge peaks (within 5 pixels of boundary)
        h, w = self.image.shape
        pos_x = safe_convert_to_numpy(params_stage1['pos_x'])
        pos_y = safe_convert_to_numpy(params_stage1['pos_y'])
        
        edge_mask = (pos_x < 5) | (pos_x > w-5) | (pos_y < 5) | (pos_y > h-5)
        
        if np.any(edge_mask):
            # Boost height and width for edge peaks
            height = safe_convert_to_numpy(params_stage1['height'])
            width = safe_convert_to_numpy(params_stage1['width'])
            
            height[edge_mask] *= 2.0  # Double height
            width[edge_mask] *= 1.8   # Increase width by 80%
            
            params_stage1['height'] = safe_convert_to_tensor(height)
            params_stage1['width'] = safe_convert_to_tensor(width)
            
            if verbose:
                n_edge = np.sum(edge_mask)
                logging.info(f"Corrected {n_edge} edge peak(s): height×2.0, width×1.8")
        
        # Enable boundary penalty for Stage 2
        self.use_boundary_penalty = True
        self.boundary_strength = 0.001
        
        params_final = self.fit_global(
            params=params_stage1,
            maxiter=maxiter,
            step_size=step_size,
            verbose=False
        )
        
        # Restore original boundary penalty state
        self.use_boundary_penalty = original_boundary_state
        
        if verbose:
            logging.info("Two-stage fitting complete")
        
        self.params = params_final
        self.prediction = safe_convert_to_numpy(self.predict(params_final, local=True))
        
        return params_final
    
    def calculate_peak_visibility(self, pos_x, pos_y, width):
        """
        Calculate what fraction of each peak is visible in the image.
        
        For a Gaussian, ~99.7% of intensity is within 3*sigma of center.
        We check how much of this region overlaps with the image.
        
        Args:
            pos_x: Peak center x positions (tensor or array)
            pos_y: Peak center y positions (tensor or array)
            width: Peak widths (sigma) (tensor or array)
            
        Returns:
            visibility: Fraction of peak visible (0.01 to 1) for each peak
        """
        h, w = self.image.shape
        
        # Define the "effective region" as 3*sigma around center
        radius = 3.0 * width
        
        # Calculate overlap with image bounds for each dimension
        x_min = torch.maximum(pos_x - radius, 0.0)
        x_max = torch.minimum(pos_x + radius, w - 1)
        y_min = torch.maximum(pos_y - radius, 0.0)
        y_max = torch.minimum(pos_y + radius, h - 1)
        
        # Visible width and height
        visible_width = torch.maximum(x_max - x_min, 0.0)
        visible_height = torch.maximum(y_max - y_min, 0.0)
        
        # Total width and height of effective region
        total_width = 2 * radius
        total_height = 2 * radius
        
        # Visibility as fraction of area
        visibility = (visible_width * visible_height) / (total_width * total_height)
        
        # Clamp to [0.01, 1.0] to avoid division by zero and extreme values
        visibility = torch.clamp(visibility, 0.01, 1.0)
        
        return visibility
    
    def calculate_boundary_penalty(self, pos_x, pos_y, width, max_distance=3.0):
        """
        Calculate soft boundary penalty for positions near or outside image edges.
        
        This penalty allows peaks to be outside the image by up to max_distance * width,
        but applies a smooth quadratic penalty for positions beyond that.
        
        Args:
            pos_x: Peak x positions (tensor or array)
            pos_y: Peak y positions (tensor or array)
            width: Peak widths (tensor or array)
            max_distance: Maximum allowed distance outside (in units of sigma). Default 3.0
            
        Returns:
            penalty: Scalar penalty value
        """
        h, w = self.image.shape
        
        # Calculate how far outside the boundary each peak is
        # Negative means inside, positive means outside
        dist_left = -pos_x
        dist_right = pos_x - (w - 1)
        dist_top = -pos_y
        dist_bottom = pos_y - (h - 1)
        
        # Maximum allowed distance for each peak
        allowed = max_distance * width
        
        # Penalty only when exceeding allowed distance
        # Use smooth quadratic penalty
        penalty_left = torch.maximum(dist_left - allowed, 0.0) ** 2
        penalty_right = torch.maximum(dist_right - allowed, 0.0) ** 2
        penalty_top = torch.maximum(dist_top - allowed, 0.0) ** 2
        penalty_bottom = torch.maximum(dist_bottom - allowed, 0.0) ** 2
        
        total_penalty = torch.sum(
            penalty_left + penalty_right + penalty_top + penalty_bottom
        )
        
        return total_penalty
    
    def get_current_background(self) -> np.ndarray:
        """
        Get the current background (2D or scalar).
        
        Returns:
            Background array (2D if enabled, otherwise scalar broadcast to 2D)
        """
        if self.background_estimator.use_2d_background:
            return self.background_estimator.get_current_background()
        else:
            # Get scalar background value
            bg_value = getattr(self, 'init_background', 0.0)
            if self.params is not None and 'background' in self.params:
                bg_value = safe_convert_to_numpy(self.params['background'])
                if np.isscalar(bg_value):
                    bg_value = float(bg_value)
                else:
                    bg_value = float(bg_value.item()) if bg_value.size == 1 else float(bg_value[0])
            return self.background_estimator.get_current_background(bg_value)
    
    def update_2d_background_scale(self, new_scale: float):
        """Update the 2D background scaling factor."""
        self.background_estimator.update_2d_background_scale(new_scale)
    
    def optimize_2d_background_scale(self) -> float:
        """
        Optimize the 2D background scaling factor for the current image.
        
        This method finds the optimal scaling factor for the 2D background
        that minimizes the residual between the scaled background and the image.
        
        Returns:
            Optimal scaling factor
        """
        if not self.background_estimator.use_2d_background or self.background_estimator.background_2d is None:
            raise ValueError("2D background not enabled or not estimated")
        
        from scipy.optimize import minimize_scalar
        
        background_2d = self.background_estimator.background_2d
        
        def objective(scale: float) -> float:
            """Objective function using robust loss."""
            scaled_bg = scale * background_2d
            residual = self.image - scaled_bg
            
            # Use robust loss (Huber loss)
            abs_residual = np.abs(residual)
            threshold = 2.0 * np.median(abs_residual)
            
            loss = np.where(
                abs_residual <= threshold,
                0.5 * residual**2,
                threshold * (abs_residual - 0.5 * threshold)
            )
            return np.mean(loss)
        
        # Get initial estimate
        initial_scale = self.background_estimator.background_scale
        
        # Optimize with reasonable bounds
        result = minimize_scalar(objective, bounds=(0.01, 100.0), method='bounded')
        optimal_scale = result.x
        
        # Update the background estimator
        self.update_2d_background_scale(optimal_scale)
        
        logging.info("2D background scale optimized: %.3f -> %.3f", initial_scale, optimal_scale)
        return float(optimal_scale)

    def init_params(
        self,
        atom_size: float = 0.7,
        guess_radius: bool = False,
        init_background: float = 0.0,
        background_2d: bool = False,
    ):
        """Initialize model parameters based on the current model type and settings.

        Args:
            atom_size (float, optional): Base size for atomic peaks. Defaults to 0.7.
            guess_radius (bool, optional): Whether to estimate peak width from data. Defaults to False.
            init_background (float, optional): Initial background value. Defaults to 0.0.
        """
        self.coordinates_history = dict()
        self.coordinates_state = 0

        # Get width parameter
        if guess_radius:
            width = self.guess_radius()[0]
        else:
            width = atom_size / self.dx

        # Handle periodic boundary conditions
        if self.pbc:
            mask = (self.coordinates[:, 0] < self.nx - 1) & (
                self.coordinates[:, 1] < self.ny - 1
            )
            self.coordinates = self.coordinates[mask]

        # Initialize position and height parameters
        pos_x = copy.deepcopy(self.coordinates[:, 0]).astype(float)
        pos_y = copy.deepcopy(self.coordinates[:, 1]).astype(float)
        
        # Note: We intentionally do NOT clip positions here to allow edge peaks
        # to be initialized at their detected positions (which may be at x=0 or y=0).
        # The fitting process will handle positions outside bounds if needed.

        # Initialize background using robust estimation
        if self.fit_background:
            if self.background_estimator.use_2d_background:
                # For 2D background, optimize the scale and use background_scale parameter
                background_scale = self.optimize_2d_background_scale()
                self.init_background = background_scale
                init_background = background_scale  # This will be the scale parameter
            else:
                # For 1D background, estimate scalar value
                init_background = self.background_estimator.estimate_scalar_background(method='robust')
                self.init_background = init_background
        else:
            self.init_background = init_background

        # Initialize heights from image values
        if self.background_estimator.use_2d_background:
            # For 2D background, subtract the scaled background at peak positions
            current_bg = self.get_current_background()
            height = (
                self.image[pos_y.astype(int), pos_x.astype(int)].ravel() - 
                current_bg[pos_y.astype(int), pos_x.astype(int)].ravel()
            )
        else:
            # For scalar background, subtract the scalar value
            height = (
                self.image[pos_y.astype(int), pos_x.astype(int)].ravel() - init_background
            )
        height[height < 0] = 0  # Ensure non-negative heights

        # Initialize width parameters based on model type
        if self.same_width:
            width = np.tile(width, self.num_atom_types).astype(float)
        else:
            width = np.tile(width, self.num_coordinates).astype(float)

        # Create parameter dictionary based on model type
        params = {
            "pos_x": pos_x,
            "pos_y": pos_y,
            "height": height,
            "width": width,
            "same_width": self.same_width,
            "atom_types": self.atom_types
        }
        
        # Add background parameter (scalar background or 2D background scale)
        if self.background_estimator.use_2d_background:
            params["background_scale"] = init_background  # This is the scaling factor
        else:
            params["background"] = init_background  # This is the scalar background value

        if self.model_type == "voigt":
            if self.same_width:
                ratio = np.tile(0.9, self.num_atom_types).astype(float)
            else:
                ratio = np.tile(0.9, self.num_coordinates).astype(float)
            params.update({"ratio": ratio})

        for key in params.keys():
            params[key] = torch.as_tensor(params[key], dtype=torch.float32)
        
        self.params = params
        self.model = self._create_fitting_model(self.params)
        return params

    # find atomic columns  
    def import_coordinates(self, coordinates: np.ndarray):
        self.coordinates = coordinates[:, :2]

    def find_peaks(
        self,
        min_distance: int = 10,
        threshold_rel: float = 0.2,
        threshold_abs=None,
        exclude_border: bool = False,
        plot: bool = True,
        region_index: int = 0,
        sigma: float = 5,
    ):
        """
        Find the peaks in the image.

        Args:
            atom_size (float, optional): The size of the atomic columns. Defaults to 1.
            threshold_rel (float, optional): The relative threshold. Defaults to 0.2.
            exclude_border (bool, optional): Whether to exclude the border. Defaults to False.
            image (np.array, optional): The input image. Defaults to None.

        Returns:
            np.array: The coordinates of the peaks.
        """
        assert (
            region_index in self.regions.keys
        ), "The region index is not in the regions."
        region_map = self.regions.region_map == region_index
        image_filtered = gaussian_filter(self.image, sigma)
        peaks_locations = peak_local_max(
            image_filtered * region_map,
            min_distance=min_distance,
            threshold_rel=threshold_rel,
            threshold_abs=threshold_abs,
            exclude_border=exclude_border,
        )
        if self.coordinates.size > 0:
            column_mask = self.region_column_labels == region_index
            coordinates = np.delete(self.coordinates, np.where(column_mask), dim=0)
            coordinates = np.vstack(
                [coordinates, peaks_locations[:, [1, 0]].astype(float)]
            )
            self.coordinates = coordinates
            atom_types = np.delete(self.atom_types, np.where(column_mask), dim=0)
            atom_types = np.append(
                atom_types, np.zeros(peaks_locations.shape[0], dtype=int)
            )
            self.atom_types = atom_types
        else:
            self.coordinates = peaks_locations[:, [1, 0]].astype(float)
            self.atom_types = np.zeros(peaks_locations.shape[0], dtype=int)
        if plot:
            self.add_or_remove_peaks(min_distance=min_distance, image=self.image)
        return self.coordinates

    def total_lattice(self, region_index: int = None):
        return self.regions.lattice(region_index)

    def view_3d(self, region_index: int = None):
        self.regions.view_3d(region_index)

    def map_lattice(
        self,
        cif_file: str,
        elements: list[str] = None,
        reciprocal: bool = False,
        region_index: int = 0,
        sigma: float = 0.8,
    ):
        """
        Map the atomic columns in the CIF file to the peaks found in the image.

        Args:
            cif_file (str): The path to the CIF file.
            elements (list[str]): The elements in the CIF file.
            unit_cell (Atoms, optional): The unit cell of the crystal. Defaults to None.
            reciprocal (bool, optional): Whether to use reciprocal space. Defaults to False.
            region_index (int, optional): The index of the region. Defaults to 0.
            sigma (float, optional): The sigma of the Gaussian filter. Defaults to 0.8.

        Returns:
            AtomicColumns: The atomic columns mapped from the CIF file.
        """
        # find the column within the region_index
        column_mask = self.region_column_labels == region_index
        region_mask = self.regions.region_map == region_index

        if elements is None:
            elements = self.elements

        crystal_analyzer = CrystalAnalyzer(
            image=self.image,
            dx=self.dx,
            peak_positions=self.coordinates[column_mask],
            atom_types=self.atom_types[column_mask],
            elements=elements,
            units="A",
            region_mask=region_mask,
        )
        # if unit_cell is not None:
        #     crystal_analyzer.unit_cell = unit_cell
        if cif_file is not None:
            crystal_analyzer.read_cif(cif_file)
        atomic_column_list = crystal_analyzer.get_atomic_columns(
            reciprocal=reciprocal, sigma=sigma
        )
        # remove the self.coordinates in the column mask and append the new coordinates find in the atomic_column_list
        coordinates = np.delete(self.coordinates, np.where(column_mask), dim=0)
        coordinates = np.vstack([coordinates, atomic_column_list.positions_pixel])
        self.coordinates = coordinates
        atom_types = np.delete(self.atom_types, np.where(column_mask), dim=0)
        atom_types = np.append(atom_types, atomic_column_list.atom_types)
        self.atom_types = atom_types
        crystal_analyzer.plot_unitcell()
        self.regions[region_index].analyzer = crystal_analyzer
        self.regions[region_index].columns = atomic_column_list
        return atomic_column_list

    def assign_region_label(
        self, region_index: int = 0, invert_selection: bool = False
    ):
        atom_select = GetRegionSelection(
            image=self.image,
            invert_selection=invert_selection,
            region_map=self.regions.region_map,
        )
        try:
            atom_select.poly.verts = self.regions[region_index].path.vertices  # type: ignore
            atom_select.path = self.regions[region_index].path
        except KeyError:
            pass
        while plt.fignum_exists(atom_select.fig.number):  # type: ignore
            plt.pause(0.1)

        region_mask = atom_select.get_region_mask()
        # self.regions.region_map[region_mask] = region_index
        region = Region(
            name=f"region_{region_index}",
            index=region_index,
            path=atom_select.path,
            image_shape=self.image.shape)
        self.regions.add_region(region)
        logging.info(
            f"Assigned label {region_index} with {region_mask.sum()} pixels to the region map."
        )

    def select_atoms(self, invert_selection: bool = False):
        atom_select = GetAtomSelection(
            image=self.image,
            atom_positions=self.coordinates,
            invert_selection=invert_selection,
        )
        while plt.fignum_exists(atom_select.fig.number):  # type: ignore
            plt.pause(0.1)
        peak_positions_selected = np.array(atom_select.atom_positions_selected)
        selection_mask = atom_select.selection_mask

        if peak_positions_selected.shape[0] == 0:
            logging.info("No atoms selected.")
            return None
        else:
            logging.info(
                f"Selected {peak_positions_selected.shape[0]} atoms out of {self.num_coordinates} atoms."
            )

            self.atom_types = self.atom_types[selection_mask]
            self.coordinates = peak_positions_selected
        return selection_mask

    def get_nearest_peak_distance(self, peak_position: np.ndarray):
        """
        Get the distance of the nearest peak for each peak.

        Args:
            peak_positions (np.array): The positions of the peaks.
            threshold (int, optional): The threshold distance. Defaults to 10.

        Returns:
            np.array: The distances of the nearest peaks.
        """
        other_peaks = np.delete(
            self.coordinates, np.where(self.coordinates == peak_position), dim=0
        )
        distances = np.linalg.norm(other_peaks - peak_position, axis=1).min()
        return distances

    def refine_center_of_mass(self, params=None, plot=False):
        # Refine center of mass for each Voronoi cell
        pre_coordinates = self.coordinates.copy()
        current_coordinates = self.coordinates.copy()
        converged = False

        if params is None and hasattr(self, "params") and len(self.params) > 0:
            params = self.params
        elif params is None:
            params = self.init_params()
        while not converged:
            # Generate Voronoi cell map
            coords = np.stack([pre_coordinates[:, 1], pre_coordinates[:, 0]])  # (y, x)
            max_radius = params["width"].max() * 5
            point_record = voronoi_point_record(self.image, coords, max_radius)

            # In refine_center_of_mass, replace the for-loop with:
            with ThreadPoolExecutor() as executor:
                futures = [
                    executor.submit(self._refine_one_center, i, point_record, plot)
                    for i in range(self.num_coordinates)
                ]
                for future in tqdm(
                    as_completed(futures),
                    total=self.num_coordinates,
                    desc="Refining center of mass",
                ):
                    result, i = future.result()
                    if result is not None:
                        current_coordinates[i] = result

            converged = np.abs(current_coordinates - pre_coordinates).mean() < 0.5
            pre_coordinates = current_coordinates.copy()
        params["pos_x"] = current_coordinates[:, 0]
        params["pos_y"] = current_coordinates[:, 1]
        self.params = params
        self.coordinates = current_coordinates
        return params

    def _refine_one_center(self, i: int, point_record: np.ndarray, plot: bool = False):
        mask = point_record == (i + 1)
        if not np.any(mask):
            return None, i

        cell_img = self.image * mask
        ys, xs = np.where(mask)
        y0, y1 = ys.min(), ys.max() + 1
        x0, x1 = xs.min(), xs.max() + 1
        cropped_img = cell_img[y0:y1, x0:x1]
        cropped_mask = mask[y0:y1, x0:x1]

        # Subtract local min (only over masked region)
        local_min = cropped_img[cropped_mask].min()
        cropped_img = cropped_img - local_min
        cropped_img[~cropped_mask] = 0

        # Normalize for center of mass
        if cropped_img[cropped_mask].max() > 0:
            norm_img = (cropped_img - cropped_img[cropped_mask].min()) / (
                cropped_img[cropped_mask].max() - cropped_img[cropped_mask].min()
            )
        else:
            norm_img = cropped_img
        norm_img[~cropped_mask] = 0

        # Compute center of mass in the cropped region
        local_y, local_x = calculate_center_of_mass(norm_img)
        assert isinstance(local_x, float), "local_x is not a float"
        assert isinstance(local_y, float), "local_y is not a float"
        result = np.array(
            [
                x0 + local_x,
                y0 + local_y,
            ],
            dtype=float,
        )

        if plot:
            plt.clf()
            plt.imshow(norm_img, cmap="gray")
            plt.scatter(local_x, local_y, color="red", s=2, label="refined")
            plt.legend()
            plt.pause(1.0)
        return result, i

    def refine_local_max(
        self,
        plot=False,
        min_distance=10,
        threshold_rel=0.3,
        threshold_abs=None,
        exclude_border=True,
    ):
        windows_size = min_distance * 2
        peak_total = np.array([], dtype=int).reshape(0, 2)
        for coordinate_index in range(self.num_coordinates):
            x, y = self.coordinates[coordinate_index]
            top = max(int(x) - windows_size, 0)
            bottom = min(int(x) + windows_size + 1, self.nx)
            left = max(int(y) - windows_size, 0)
            right = min(int(y) + windows_size + 1, self.ny)
            # calculate the mask for distance < r
            region = self.image[left:right, top:bottom]
            peaks_locations = peak_local_max(
                region,
                min_distance=int(min_distance / 4),
                threshold_rel=threshold_rel,
                threshold_abs=threshold_abs,
                exclude_border=exclude_border,
            )
            peaks_locations = peaks_locations[:, [1, 0]].astype(int)
            if peaks_locations.shape[0] > 0:
                peak_total = np.append(
                    peak_total,
                    peaks_locations
                    + np.array([int(x) - windows_size, int(y) - windows_size]),
                    axis=0,
                )
            if plot:
                plt.clf()
                plt.subplot(1, 2, 1)
                plt.imshow(self.image, cmap="gray")
                plt.scatter(
                    self.coordinates[:, 0],
                    self.coordinates[:, 1],
                    color="blue",
                    s=1,
                )
                plt.scatter(x, y, color="red", s=2)
                plt.subplot(1, 2, 2)
                plt.imshow(region, cmap="gray")
                plt.scatter(
                    x % 1 + windows_size, y % 1 + windows_size, color="red", s=2
                )
                if peaks_locations.shape[0] > 0:
                    plt.scatter(
                        peaks_locations[:, 0],
                        peaks_locations[:, 1],
                        color="green",
                        s=2,
                    )
                plt.show()
                plt.pause(1.0)
        self.coordinates = np.unique(peak_total, axis=0)
        # self.coordinates = self.refine_duplicate_peaks()
        return self.coordinates

    def remove_close_coordinates(self, threshold: int = 10):
        """
        Remove coordinates that are too close to each other, considering periodic boundary conditions (PBC) if enabled.
        Also removes the corresponding atom types from self.atom_types.

        Args:
            threshold (int): Minimum allowed distance between coordinates. Defaults to 10.

        Returns:
            np.ndarray: The filtered coordinates.
        """
        if self.pbc:
            # Remove close coordinates in the original box
            coords, atom_types, _ = remove_close_coordinates(self.coordinates.copy(), self.atom_types.copy(), threshold)
            
            # Identify coordinates near the boundary
            mask_boundary = (
                (coords[:, 0] < threshold)
                | (coords[:, 0] > self.nx - threshold)
                | (coords[:, 1] < threshold)
                | (coords[:, 1] > self.ny - threshold)
            )
            coords_boundary = coords[mask_boundary]
            atom_types_boundary = atom_types[mask_boundary]
            
            # Generate periodic images of boundary coordinates
            shifts = np.array([
                [i * self.nx, j * self.ny]
                for i, j in [(1, 0), (0, 1), (1, 1), (-1, 0), (0, -1), (-1, -1), (1, -1), (-1, 1)]
            ])
            
            # Check if any periodic image is too close to the original boundary coordinates
            to_remove = set()
            for shift in shifts:
                shifted_coords = coords_boundary + shift
                for i, coord in enumerate(coords_boundary):
                    distances = np.linalg.norm(shifted_coords - coord, axis=1)
                    if (distances < threshold).any():
                        to_remove.add(i)
            
            # Remove overlapping boundary coordinates and corresponding atom types
            coords_boundary_filtered = np.delete(coords_boundary, list(to_remove), axis=0)
            atom_types_boundary_filtered = np.delete(atom_types_boundary, list(to_remove), axis=0)
            
            # Combine non-boundary and filtered boundary coordinates and atom types
            self.coordinates = np.vstack([coords[~mask_boundary], coords_boundary_filtered])
            self.atom_types = np.concatenate([atom_types[~mask_boundary], atom_types_boundary_filtered])
        else:
            self.coordinates, self.atom_types,_ = remove_close_coordinates(self.coordinates, self.atom_types, threshold)
        
        return self.coordinates, self.atom_types

    def add_or_remove_peaks(self, min_distance: int = 2, image=None):
        if image is None:
            image = self.image
        peaks_locations = self.coordinates
        interactive_plot = InteractivePlot(
            image=image,
            peaks_locations=peaks_locations,
            atom_types=self.atom_types,
            tolerance=min_distance,
        )
        interactive_plot.add_or_remove(tolerance=min_distance)
        peaks_locations = [interactive_plot.pos_x, interactive_plot.pos_y]
        peaks_locations = np.array(peaks_locations).T.astype(float)
        self.coordinates = peaks_locations
        self.atom_types = interactive_plot.atom_types
        return peaks_locations

    def remove_peaks_outside_image(self):
        coordinates = self.coordinates
        mask = (
            (coordinates[:, 0] >= 0)
            & (coordinates[:, 0] < self.nx)
            & (coordinates[:, 1] >= 0)
            & (coordinates[:, 1] < self.ny)
        )
        self.coordinates = coordinates[mask]
        return self.coordinates

    # loss function and model prediction
    def predict(self, params: dict = None, model:ImageModel=None, local: bool = True):
        """Predict the image based on the model's current parameters.

        Args:
            params (dict, optional): Parameters to use for prediction. If None, uses current params.
            local (bool, optional): If True, calculate peaks locally. Defaults to False.

        Returns:
            array: Predicted image
        """
        
        if params is None:
            params = self.params
        if model is None:
            model = self.model
        model.set_params(params)
        
        # # Ensure model is built
        if not model.built:
            model.build()
        
        prediction = model.sum(self.x_grid, self.y_grid, local=local)

        # Handle periodic boundary conditions by rolling the image
        if self.pbc:
            for i, j in [
                (1, 0),
                (0, 1),
                (-1, 0),
                (0, -1),
                (1, 1),
                (-1, -1),
                (1, -1),
                (-1, 1),
            ]:
                # Temporarily set shifted grids for periodic boundary conditions
                prediction += model.sum(self.x_grid + i * self.nx, self.y_grid + j * self.ny, local=local)
        # self.prediction = safe_convert_to_numpy(prediction)
        return prediction

    def loss(self, y_true, y_pred, use_adaptive_edge_loss=None):
        """
        Compute the loss value between the image and the prediction.

        Parameters:
        -----------
        y_true : np.ndarray
            The original image tensor (ground truth).
        y_pred : np.ndarray
            The predicted image tensor (model output).
        use_adaptive_edge_loss : bool, optional
            If True, use adaptive gradient boosting for edge peaks.
            If None, uses self.use_adaptive_edge_loss. Default None.

        Returns:
        --------
        float
            The computed loss value.
        """
        # Use instance variable if not explicitly specified
        if use_adaptive_edge_loss is None:
            use_adaptive_edge_loss = getattr(self, 'use_adaptive_edge_loss', False)
        
        diff = y_true - y_pred
        if self._window_t is None:
            self._window_t = torch.as_tensor(self.window, dtype=torch.float32)
        diff = torch.mul(diff, self._window_t)
        
        # Base MSE loss
        mse = torch.sqrt(torch.mean(torch.square(diff)))
        
        # Optionally use adaptive edge loss for better gradient signal
        if use_adaptive_edge_loss:
            # Use the model currently being optimized (if available)
            model = getattr(self, '_optimization_model', self.model)
            if model is not None:
                # Get current parameters
                params = model.get_params()
                pos_x = params['pos_x']
                pos_y = params['pos_y']
                width = params['width']
                
                # Calculate visibility and apply gradient boost
                visibility = self.calculate_peak_visibility(pos_x, pos_y, width)
                boost_factor = 1.0 / torch.sqrt(visibility)
                avg_boost = torch.mean(boost_factor)
                mse = mse * avg_boost
        
        # Add soft boundary penalty if enabled
        if hasattr(self, 'use_boundary_penalty') and self.use_boundary_penalty:
            # Use the model currently being optimized (if available)
            model = getattr(self, '_optimization_model', self.model)
            if model is not None:
                # Get current parameters
                params = model.get_params()
                pos_x = params['pos_x']
                pos_y = params['pos_y']
                width = params['width']
                
                # Calculate soft boundary penalty
                boundary_penalty = self.calculate_boundary_penalty(
                    pos_x, pos_y, width, max_distance=3.0
                )
                
                # Apply penalty with strength factor
                penalty_weight = getattr(self, 'boundary_strength', 0.01)
                penalty_term = penalty_weight * boundary_penalty
                mse = mse + penalty_term
        
        return mse 

    def residual(self, params: dict):
        # Compute the sum of the Gaussians
        prediction = self.predict(params)
        diff = self.image_tensor - prediction
        diff = safe_convert_to_numpy(diff)
        return diff

    # fitting
    def linear_estimator(
        self,
        params: dict = None,
        non_negative: bool = False,
        device: str = 'cpu',
        best_effort: bool = False,
    ) -> dict:
        """
        Perform linear estimation of peak heights using least squares fitting.

        Builds a sparse design matrix from the current peak model and solves
        a linear system to estimate optimal height scaling factors.

        Args:
            params: Model parameters dictionary. If ``None``, uses ``self.params``.
            non_negative: Whether to enforce non-negative height constraints.
            device: Compute device hint passed through to the solver.
            best_effort: If ``True``, log and swallow exceptions and return the
                input parameters unchanged. Defaults to ``False`` so callers
                see real failures (parameter validation errors, numerical
                breakdowns, OOM) instead of getting silently stale results.

        Returns:
            Updated parameters dictionary with refined height values, or the
            original parameters when ``best_effort`` swallows a failure.

        Raises:
            ParameterError: If ``params`` fails validation.
            QEMError: For backend / memory / numerical solver failures
                (when ``best_effort=False``).
        """
        # Initialize parameters if needed
        if params is None:
            if self.params is None:
                self.init_params()
            params = self.params

        operation_context = (
            self.memory_monitor.monitor_operation("linear_estimator")
            if self.memory_monitor else nullcontext()
        )

        def _run() -> dict:
            validated_params = ParameterValidator.validate_params(params)

            matrix_builder = DesignMatrixBuilder(self.model, self.nx, self.ny)
            peak_local, global_x, global_y, mask = matrix_builder.build_local_peaks(
                validated_params, self.same_width, self.atom_types
            )

            background_2d_for_matrix = None
            if self.background_estimator.use_2d_background:
                background_2d_for_matrix = self.background_estimator.get_background_for_linear_estimation()

            design_matrix = matrix_builder.build_sparse_matrix(
                peak_local, global_x, global_y, mask,
                self.fit_background, self.num_coordinates,
                self.x_grid, self.y_grid,
                background_2d_for_matrix,
            )
            target = self._prepare_target_vector(validated_params)
            solver = LinearSystemSolver()
            solution = solver.solve_system(design_matrix, target, non_negative)
            return self._process_solution(solution, validated_params)

        with operation_context:
            if not best_effort:
                # Default path: surface real failures to the caller.
                return _run()
            try:
                return _run()
            except Exception as e:
                # Opt-in best-effort behaviour for resilient outer loops
                # (e.g. the stochastic fitter that pre-conditions params).
                logging.warning(
                    "linear_estimator failed in best_effort mode; "
                    "returning input parameters unchanged: %s", e,
                )
                return params
    
    def _prepare_target_vector(self, params: dict) -> np.ndarray:
        """
        Prepare target vector for linear system.
        
        Args:
            params: Model parameters
            
        Returns:
            Flattened target vector
        """
        # target = safe_convert_to_numpy(self.image_tensor).ravel()
        target = self.image_tensor.ravel()
        
        if not self.fit_background:
            if self.background_estimator.use_2d_background:
                # Subtract current 2D background
                current_bg = self.get_current_background()
                target = target - current_bg.ravel()
            else:
                # Subtract scalar background
                bg_key = "background_scale" if "background_scale" in params else "background"
                target = target - params[bg_key]
            
        return target

    def _process_solution(self, solution: np.ndarray, params: dict, update_threshold: float = 0.2) -> dict:
        """
        Process linear system solution and update parameters.
        
        Args:
            solution: Solution vector from linear solver (numpy array from scipy fallback)
            params: Original parameters dictionary
            
        Returns:
            Updated parameters dictionary
        """
        processor = SolutionProcessor()
        
        # Validate solution
        if not processor.validate_solution(solution):
            logging.warning("Invalid solution obtained, returning original parameters")
            return params
        
        # Extract height scaling and background
        if self.fit_background:
            if self.background_estimator.use_2d_background:
                # For 2D background, the last element is the scaling factor
                background_scale = solution[-1]
                
                # Validate and update 2D background scale
                if 0.01 < background_scale < 100.0:  # Reasonable bounds
                    self.update_2d_background_scale(float(background_scale))
                    # Update the background_scale parameter
                    params["background_scale"] = safe_convert_to_tensor(float(background_scale))
                    # Remove old background parameter if it exists
                    if "background" in params:
                        del params["background"]
                else:
                    logging.warning("2D background scale out of bounds: %.3f, keeping current scale", background_scale)
                
                height_scale = solution[:-1]
            else:
                # Scalar background processing
                background, valid = processor.process_background(
                    solution, params, self.init_background, update_threshold
                )
                if not valid:
                    logging.warning("Background update too large, skipping parameter update with linear estimator")
                    return params
                
                # Convert background to Keras tensor to match parameter types
                params["background"] = safe_convert_to_tensor(background)
                height_scale = solution[:-1]
        else:
            height_scale = solution
        
        # Process height scaling factors
        processed_scale = processor.process_height_scaling(height_scale)
        
        # Convert processed scale to Keras tensor to match parameter types
        processed_scale_tensor = safe_convert_to_tensor(processed_scale)
        
        # Update height parameters
        params["height"] *= processed_scale_tensor

        # Update instance parameters
        self.params = params
        return params

    def optimize(
        self,
        model: ImageModel,
        image_tensor: np.ndarray = None,
        params: dict = None,
        maxiter: int = 1000,
        tol: float = 1e-4,
        step_size: float = 0.01,
        verbose: bool = True,
        batch_size: int = 1024,
        optimizer: str = "adam",
        **optimizer_kwargs
    ) -> dict[str, Any]:  # actually torch.Tensor for grad params; use detach() for numpy snapshot.
        """
        Optimize model parameters using specified optimizer.
        
        Args:
            model: The image model to optimize
            image_tensor: Target image tensor (uses self.image_tensor if None)
            params: Initial parameters (uses model params if None)
            maxiter: Maximum iterations/epochs
            tol: Tolerance for convergence
            step_size: Learning rate
            verbose: Whether to print progress
            batch_size: Batch size for training
            optimizer_type: Type of optimizer ('adam', 'adamw', 'lbfgs')
            **optimizer_kwargs: Additional optimizer-specific parameters
            
        Returns:
            Dictionary containing optimized parameters
        """
        if image_tensor is None:
            image_tensor = self.image_tensor
        if params is not None:
            model.set_params(params)

        # Build the model if not already built
        if not model.built:
            model.build()
        
        # Store reference to model being optimized so loss function can access it
        self._optimization_model = model

        if verbose:
            print(f"Using {optimizer} optimizer for fitting.")
        # PyTorch expects a leading batch dimension on inputs.
        image_tensor = image_tensor.unsqueeze(0)
        model_inputs = [self.x_grid_batched, self.y_grid_batched]
        
        operation_context = (
            self.memory_monitor.monitor_operation("optimize") 
            if self.memory_monitor else nullcontext()
        )
        
        with operation_context:
            from qem.fit._loop import fit_loop, make_optimizer

            opt = make_optimizer(optimizer, model.parameters(), step_size)
            fit_loop(
                model=model,
                inputs=model_inputs,
                target=image_tensor,
                loss_fn=self.loss,
                optimizer=opt,
                epochs=maxiter,
                tol=tol,
                patience=100,
                lr_patience=10,
                lr_factor=0.1,
                min_lr=1e-6,
                verbose=verbose,
            )
        
        # Clean up model reference
        self._optimization_model = None
                
        optimized_params = model.get_params()
        return optimized_params

    def fit_global(
        self,
        params: dict = None,
        maxiter: int = 1000,
        tol: float = 1e-3,
        step_size: float = 0.01,
        optimizer: str = "adam",
        local: bool = True,
        verbose: bool = True,
        **optimizer_kwargs
    ):
        """
        Fit model parameters globally using specified optimizer.
        
        Args:
            params: Initial parameters (uses self.params or initializes if None)
            maxiter: Maximum iterations/epochs
            tol: Tolerance for convergence
            step_size: Learning rate
            optimizer_type: Type of optimizer ('adam', 'adamw', 'lbfgs')
            local: Whether to use local prediction for final result
            verbose: Whether to print optimization progress
            **optimizer_kwargs: Additional optimizer-specific parameters
            
        Returns:
            Dictionary containing optimized parameters
        """
        if params is None:
            params = self.params if self.params is not None else self.init_params()
        
        fitting_model = self._create_fitting_model(params)
        
        params = self.optimize(
            model=fitting_model,
            image_tensor=self.image_tensor,
            params=params,
            maxiter=maxiter,
            tol=tol,
            step_size=step_size,
            optimizer=optimizer,
            verbose=verbose,
            **optimizer_kwargs
        )
        
        self.params = params
        self.prediction = safe_convert_to_numpy(self.predict(params, local=local))
        return params

    def fit_stochastic(
        self,
        params: dict = None,
        num_epoch: int = 5,
        batch_size: int = 500,
        maxiter: int = 50,
        tol: float = 1e-3,
        step_size: float = 1e-2,
        optimizer: str = "adam",
        verbose: bool = True,
        local: bool = True,
        plot: bool = False,
        **optimizer_kwargs
    ):
        """
        Fit model parameters stochastically by optimizing random batches of coordinates.
        
        Args:
            params: Initial parameters (uses self.params or initializes if None)
            num_epoch: Number of training epochs
            batch_size: Size of random batches
            maxiter: Maximum iterations per batch
            tol: Tolerance for convergence
            step_size: Learning rate
            optimizer_type: Type of optimizer ('adam', 'adamw', 'lbfgs')
            local: Whether to use local prediction
            plot: Whether to plot progress
            **optimizer_kwargs: Additional optimizer-specific parameters
            
        Returns:
            Dictionary containing optimized parameters
        """
        if params is None:
            params = self.params if self.params is not None else self.init_params()
        params = {k: safe_stop_gradient(v) for k, v in params.items()}

        self.converged = False
        operation_context = (
            self.memory_monitor.monitor_operation("fit_stochastic") 
            if self.memory_monitor else nullcontext()
        )
        
        # Pre-condition heights with a least-squares pass. The stochastic
        # fitter is robust to a no-op pre-conditioning, so swallow failures
        # here rather than aborting the whole run.
        params = self.linear_estimator(params, best_effort=True)

        with operation_context:
            for epoch in tqdm(range(num_epoch), desc="Training epochs", leave=False):
                pre_params = safe_deepcopy_params(params)
                random_batches = get_random_indices_in_batches(self.num_coordinates, batch_size)

                for batch_indices in tqdm(random_batches, desc="Fitting batch", leave=False):
                    # Calculate local target (subtract other atoms' contributions)
                    if batch_size < self.num_coordinates:
                        # in cuda
                        params_without_batch = safe_deepcopy_params(params)
                        height_tensor = params_without_batch['height']
                        batch_indices_tensor = torch.as_tensor(batch_indices, dtype=torch.int64)
                        update_indices = torch.unsqueeze(batch_indices_tensor, dim=-1)
                        update_values = torch.zeros(tuple(batch_indices_tensor.shape))
                        new_height = height_tensor.clone()
                        new_height.view(-1)[batch_indices_tensor] = update_values.to(new_height.dtype)
                        params_without_batch['height'] = new_height
                        params_without_batch['background'] = torch.zeros_like(params_without_batch['background'])

                        model_others = self._create_fitting_model(params_without_batch)

                        prediction_from_others = self.predict(params_without_batch, model=model_others, local=local)
                        local_target = (self.image_tensor - prediction_from_others).detach()

                        del params_without_batch
                        del prediction_from_others
                        del height_tensor
                        del update_values
                        release_backend_memory()
                    else:
                        local_target = self.image_tensor

                    # Optimize batch using unified optimize method
                    atoms_selected_mask = np.zeros(self.num_coordinates, dtype=bool)
                    atoms_selected_mask[batch_indices] = True
                    select_params = self.select_params(params, atoms_selected_mask)
                    
                    local_model = self._create_fitting_model(select_params)
                    
                    # Use unified optimize method for batch
                    optimized_params = self.optimize(
                        model=local_model,
                        image_tensor=local_target,
                        params=select_params,
                        maxiter=maxiter,
                        tol=tol,
                        step_size=step_size,
                        optimizer=optimizer,
                        verbose=verbose,
                        **optimizer_kwargs
                    )
                    del local_target
                    release_backend_memory()
                    params = self.update_from_local_params(params, optimized_params, atoms_selected_mask)
                    if plot:
                        self._plot_progress(params, batch_indices, select_params)

                # Check convergence
                if self.convergence(params, pre_params, tol):
                    logging.info("Convergence criteria met.")
                    self.converged = True
                    break
        
        self.params = params
        self.prediction = safe_convert_to_numpy(self.predict(params, local=local))
        logging.info("Stochastic fitting complete.")
        return self.params

    def _plot_progress(self, params, index, select_params):
        """Helper function to keep plotting logic separate."""
        global_prediction = safe_convert_to_numpy(self.predict(params))
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # Original Image with selected atoms
        axes[0].imshow(self.image, cmap="gray")
        axes[0].set_title("Original + Selected Atoms")
        axes[0].scatter(safe_convert_to_numpy(params["pos_x"][index]), safe_convert_to_numpy(params["pos_y"][index]), color="r", s=5)
        axes[0].set_aspect("equal")

        # Full Prediction
        axes[1].imshow(global_prediction, cmap="gray")
        axes[1].set_title("Current Full Prediction")
        axes[1].set_aspect("equal")

        # Residual
        axes[2].imshow(self.image - global_prediction, cmap="gray")
        axes[2].set_title("Residual")
        axes[2].set_aspect("equal")

        plt.tight_layout()
        plt.show()
        
    def fit_voronoi(
        self,
        params: dict = None,  # initial params, optional
        max_radius: int = None,  # optional, for Voronoi cell size
        tol: float = 1e-3,
        border: int = 0,  # optional, exclude border pixels
    ):
        """
        Fit a Gaussian model to each Voronoi cell defined by the current coordinates.
        Each cell is fit independently and in parallel.
        The local minimum is subtracted from each cell before fitting.
        """
        if params is None:
            if self.params is not None:
                if "pos_x" in self.params and "pos_y" in self.params:
                    params = self.params
                else:
                    params = self.init_params()
            else:
                params = self.init_params()

        pos_x = params["pos_x"]
        pos_y = params["pos_y"]
        coords = torch.stack([pos_y, pos_x])
        num_coordinates = coords.shape[1]

        # Generate Voronoi cell map
        if max_radius is None:
            max_radius = params["width"].max() * 3

        image = safe_convert_to_numpy(self.image)
        max_radius = safe_convert_to_numpy(max_radius)
        coords = safe_convert_to_numpy(coords)

        point_record = voronoi_point_record(image, coords, max_radius)

        # Prepare per-cell fitting function
        def fit_cell(index, params):
            mask = point_record == index + 1
            if not np.any(mask):
                return None  # No pixels in this cell

            cell_img = image * mask
            # Crop to bounding box for efficiency
            ys, xs = np.where(mask)
            y0, y1 = ys.min(), ys.max() + 1
            x0, x1 = xs.min(), xs.max() + 1
            cropped_img = cell_img[y0:y1, x0:x1]
            cropped_mask = mask[y0:y1, x0:x1]

            # Subtract local min (only over masked region)
            local_min = cropped_img[cropped_mask].min()
            cropped_img = cropped_img - local_min
            cropped_img[~cropped_mask] = 0

            # Prepare grid for fitting
            x_c, y_c = torch.meshgrid(
                torch.arange(x0, x1), torch.arange(y0, y1), indexing="xy"
            )
            x_c = safe_convert_to_numpy(x_c)
            y_c = safe_convert_to_numpy(y_c)

            # Prepare initial params for this cell
            local_param = {}
            local_param["pos_x"] = [params["pos_x"][index]]
            local_param["pos_y"] = [params["pos_y"][index]]
            local_param["height"] = (
                params["height"][index] + params["background"] - local_min
            )
            local_param["width"] = params["width"]
            local_param["background"] = [0.0]
            self.fit_background = False

            atoms_selected = np.zeros(self.num_coordinates, dtype=bool)
            atoms_selected[index] = True

            p0 = [
                local_param["pos_x"][0],
                local_param["pos_y"][0],
                local_param["height"],
                local_param["width"][self.atom_types[index]],
                local_param["background"][0],
            ]
            if border > 0 and (
                pos_x.min() < border
                or pos_x.max() > self.nx - border
                or pos_y.min() < border
                or pos_y.max() > self.ny - border
            ):
                popt = p0
            else:
                try:
                    popt, _ = curve_fit(  # pylint: disable=unbalanced-tuple-unpacking
                        gaussian_2d_single,
                        (x_c, y_c),
                        cropped_img.ravel(),
                        p0=p0,
                        maxfev=2000,
                    )
                except Exception as _:
                    popt = p0  # fallback if fit fails

            # if popt[0] < 0 or popt[1] < 0:
            #     popt = p0
            # if popt[0] > self.image.shape[0] or popt[1] > self.image.shape[1]:
            #     popt = p0

            optimized_param = {
                "pos_x": popt[0],
                "pos_y": popt[1],
                "height": popt[2],
                "width": popt[3],
                "background": popt[4],
            }
            return optimized_param, index

        converged = False
        pre_params = safe_deepcopy_params(self.params)
        current_params = safe_deepcopy_params(self.params)


        operation_context = (
            self.memory_monitor.monitor_operation("fit_voronoi") 
            if self.memory_monitor else nullcontext()
        )
        
        with operation_context:
            while not converged:
                with ThreadPoolExecutor() as executor:
                    futures = [
                        executor.submit(fit_cell, i, current_params)
                        for i in range(num_coordinates)
                    ]
                    # Collect all updates first
                    pos_x_updates = {}
                    pos_y_updates = {}

                    for future in tqdm(
                        as_completed(futures), total=num_coordinates, desc="Fitting cells"
                    ):
                        result = future.result()
                        if result is None:
                            continue
                        optimized_param, index = result
                        pos_x_updates[index] = optimized_param["pos_x"]
                        pos_y_updates[index] = optimized_param["pos_y"]

                    # Apply updates by creating new tensors (avoid in-place operations)
                    if pos_x_updates:
                        pos_x_array = safe_convert_to_numpy(current_params["pos_x"]).copy()
                        pos_y_array = safe_convert_to_numpy(current_params["pos_y"]).copy()

                        for index, value in pos_x_updates.items():
                            pos_x_array[index] = value
                        for index, value in pos_y_updates.items():
                            pos_y_array[index] = value

                        current_params["pos_x"] = safe_convert_to_tensor(
                            pos_x_array, dtype=torch.float32
                        )
                        current_params["pos_y"] = safe_convert_to_tensor(
                            pos_y_array, dtype=torch.float32
                        )
                converged = self.convergence(current_params, pre_params, tol)
                pre_params = safe_deepcopy_params(current_params)
        self.params = current_params
        # self.model = self.predict(self.params, self.x_grid, self.y_grid)
        return self.params

    def voronoi_integration(self, max_radius: float = None, plot=False,save=False):
        """
        Compute the Voronoi integration of the atomic columns.

        Returns:
            np.array: The Voronoi integration of the atomic columns.
        """
        if self.params is None:
            raise ValueError("Please initialize the parameters first.")
        if self.fit_background:
            image = (self.image - safe_convert_to_numpy(self.params["background"]))
        else:
            image = (self.image - self.init_background)
        pos_x = self.params["pos_x"]
        pos_y = self.params["pos_y"]
        pos_x = safe_convert_to_numpy(pos_x)
        pos_y = safe_convert_to_numpy(pos_y)
        if max_radius is None:
            max_radius = self.params["width"].max() * 5
            max_radius = safe_convert_to_numpy(max_radius)
        integrated_intensity, intensity_record, point_record = voronoi_integrate(
            image, pos_x, pos_y, max_radius=max_radius, pbc=self.pbc
        )
        integrated_intensity = integrated_intensity * self.dx**2
        intensity_record = intensity_record * self.dx**2
        self._voronoi_volume = integrated_intensity
        self._voronoi_map = intensity_record
        self._voronoi_cell = point_record
        if plot:
            plt.imshow(intensity_record, cmap="viridis")
            plt.colorbar(label="Voronoi Integrated Intensity")
        if save:
            plt.savefig("Voronoi Integrated Intensity.png", dpi=300)
            plt.savefig("Voronoi Integrated Intensity.svg")

        return integrated_intensity, intensity_record, point_record

    # parameters updates and convergence
    def convergence(self, params: dict, pre_params: dict, tol: float = 1e-2):
        """
        Checks if the parameters have converged within a specified tolerance.

        This function iterates over each parameter in `params` and its corresponding
        value in `pre_params` to determine if the change (update) is within a specified
        tolerance level, `tol`. For position parameters ('pos_x', 'pos_y'), it checks if
        the absolute update exceeds 1. For other parameters ('height', 'width', 'ratio', 'background'), it checks if the relative update exceeds `tol`.

        Parameters:
            params (dict): Current values of the parameters.
            pre_params (dict): Previous values of the parameters.
            tol (float, optional): Tolerance level for convergence. Default is 1e-2.

        Returns:
            bool: True if all parameters have converged within the tolerance, False otherwise.
        """
        # logging.info(f"Checking convergence with tolerance {tol}")
        # Loop through current parameters and their previous values
        for key, value in params.items():
            if key not in pre_params:
                continue  # Skip keys that are not in pre_params

            # Calculate the update difference
            update = torch.abs(value - pre_params[key])

            # Check convergence based on parameter type
            if key in ["pos_x", "pos_y"]:
                max_update = update.max()
                logging.info(f"Convergence rate for {key} = {max_update}")
                if max_update > 1:
                    logging.info("Convergence not reached")
                    return False
            else:
                # Avoid division by zero and calculate relative update
                value_with_offset = value + 1e-10
                rate = torch.abs(update / value_with_offset).mean()
                logging.info(f"Convergence rate for {key} = {rate}")
                if rate > tol:
                    logging.info("Convergence not reached")
                    return False

        logging.info("Convergence reached")
        return True

    def select_params(self, params: dict, mask: np.ndarray):
        select_params = {}
        select_params["background"] = params["background"]
        if self.same_width:
            if "width" in params:
                select_params["width"] = params["width"]
            if "ratio" in params:
                select_params["ratio"] = params["ratio"]
            for key in ["pos_x", "pos_y", "height"]:
                select_params[key] = params[key][mask]
        else:
            for key, value in params.items():
                if key != "background":
                    select_params[key] = value[mask]
        select_params['same_width'] = params['same_width']
        select_params['atom_types'] = params['atom_types'][mask]
        return select_params

    def update_from_local_params(self, params: dict, local_params: dict, mask: np.ndarray):
        """
        Updates the main parameter set from the locally optimized batch parameters.
        This version is defensively coded to prevent JAX 'deleted array' errors.
        """
        shared_value_list = ["background"]
        if getattr(self, 'same_width', True):
            shared_value_list.extend(["width", "ratio"])
            
        const_value_list =['same_width', 'atom_types']
        for key, value in local_params.items():
            if key in const_value_list:
                pass
            elif key in shared_value_list:
                weight = mask.sum() / self.num_coordinates
                params[key] = params[key] * (1 - weight) + value * weight                
            else:
                # --- Logic for per-atom parameters ---
                # This part uses the robust scatter_update function.
                update_indices = torch.as_tensor(np.where(mask)[0], dtype=torch.int64)
                value_tensor = torch.as_tensor(value)
                new_param = params[key].clone()
                new_param.view(-1)[update_indices] = value_tensor.to(new_param.dtype)
                params[key] = new_param
                
        return params

    def update_coordinates(self):
        # check the refined coorinates is different from the current coordinates
        refined_coordinates = np.stack(
            [self.params["pos_x"], self.params["pos_y"]], dim=1
        )
        if np.allclose(refined_coordinates, self.coordinates):
            logging.info("The coordinates have converged.")
            return self.coordinates
        else:
            # create & save the initial coordinates
            self.coordinates_history[self.coordinates_state] = self.coordinates.copy()
            # update the coordinates from the params refinement
            self.coordinates = np.stack(
                [self.params["pos_x"], self.params["pos_y"]], dim=1
            )
            self.coordinates_state += 1
            logging.info(
                f"The coordinates have been updated. Current state: {self.coordinates_state}"
            )
        return self.coordinates

    def update_region_analyzers(self):
        for index, region in self.regions.items:
            region.analyzer.peak_positions = self.coordinates[
                self.region_column_labels == index
            ]
            region.analyzer.atom_types = self.atom_types[
                self.region_column_labels == index
            ]
            logging.info(f"Updated region {index} coordinates for crystal analyzer.")

    # plot functions
    def calibrate(
        self,
        cif_file: str = None,
        a: float = None,
        b: float = None,
        region_index: int = 0,
        unit_cell: list = None,
    ):
        """
        Calibrate the pixel size based on the FFT of the lattice.
        """
        if self.coordinates.size == 0:
            logging.warning("No coordinates found. Please run find_peaks first.")
            self.find_peaks()
        column_mask = self.region_column_labels == region_index
        region_mask = self.regions.region_map == region_index
        crystal_analyzer = CrystalAnalyzer(
            image=self.image,
            dx=self.dx,
            peak_positions=self.coordinates[column_mask],
            atom_types=self.atom_types[column_mask],
            elements=self.elements,
            units="A",
            region_mask=region_mask,
        )
        if unit_cell is not None:
            crystal_analyzer.unit_cell = unit_cell
        if cif_file is not None:
            crystal_analyzer.read_cif(cif_file)

        a = a if a is not None else np.linalg.norm(crystal_analyzer.unit_cell.cell[0])
        b = b if b is not None else np.linalg.norm(crystal_analyzer.unit_cell.cell[1])
        _, vec_a_pixel, vec_b_pixel = crystal_analyzer.select_lattice_vectors(
            reciprocal=True
        )
        dx_a = a / np.linalg.norm(vec_a_pixel)
        dx_b = b / np.linalg.norm(vec_b_pixel)
        self.dx = (dx_a + dx_b) / 2
        logging.info(f"Calibrated pixel size: {self.dx} A")

    def plot(self, vmin=None, vmax=None):
        if vmin is None:
            # get the bottom 5% of the image
            vmin = np.percentile(self.image, 5)

        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        im = plt.imshow(self.image, cmap="gray", vmin=vmin, vmax=vmax)
        plt.axis("off")
        scalebar = self.scalebar
        plt.gca().add_artist(scalebar)
        plt.colorbar(im, fraction=0.046, pad=0.04)
        plt.tight_layout()
        plt.gca().add_artist(scalebar)
        plt.gca().set_aspect("equal", adjustable="box")
        plt.title("Image")

        plt.subplot(1, 2, 2)
        plt.hist(self.image.ravel(), bins=256)
        plt.xlabel("Intensity")
        plt.ylabel("Counts")
        plt.title("Intensity Histogram")
        plt.tight_layout()

    def plot_coordinates(self, s=1):
        """
        Plot the coordinates of the atomic columns.

        Args:
            color (str, optional): The color of the atomic columns. Defaults to "red".
            s (int, optional): The size of the atomic columns. Defaults to 1.
        """
        plt.figure()
        plt.imshow(self.image, cmap="gray")
        for atom_type in np.unique(self.atom_types):
            mask = self.atom_types == atom_type
            elements = self.elements[atom_type]
            plt.scatter(
                self.coordinates[mask][:, 0],
                self.coordinates[mask][:, 1],
                s=s,
                label=elements,
            )
        plt.legend()

    def plot_fitting(self,save = False):
        plt.figure(figsize=(15, 5))
        vmin = self.image.min()
        vmax = self.image.max()
        plt.subplot(1, 3, 1)
        im = plt.imshow(self.image, cmap="gray", vmin=vmin, vmax=vmax)
        plt.colorbar(im, fraction=0.046, pad=0.04)
        plt.gca().set_aspect("equal", adjustable="box")
        plt.title("Original Image")
        plt.tight_layout()
        plt.subplot(1, 3, 2)
        im = plt.imshow(self.prediction, cmap="gray", vmin=vmin, vmax=vmax)
        plt.gca().set_aspect("equal", adjustable="box")
        plt.colorbar(im, fraction=0.046, pad=0.04)
        plt.title("Model")
        plt.tight_layout()
        plt.subplot(1, 3, 3)
        im = plt.imshow(self.image - self.prediction, cmap="gray")
        plt.colorbar(im, fraction=0.046, pad=0.04)
        plt.gca().set_aspect("equal", adjustable="box")
        plt.title("Residual")
        plt.tight_layout()
        if save:
            plt.savefig("fitting.png", dpi=300)
            plt.savefig("fitting.svg")

    def plot_scs(
        self,
        layout="horizontal",
        per_element=False,
        s=1,
        save=False,
        has_units=True,
        half: str = None,
        figsize=(10, 5),
    ):
        assert layout in {
            "horizontal",
            "vertical",
        }, "Layout should be horizontal or vertical"
        if layout == "horizontal":
            row, col = 1, 2
            if per_element:
                col += len(np.unique(self.atom_types)) - 1
        else:
            row, col = 2, 1
            if per_element:
                row += len(np.unique(self.atom_types)) - 1
        plt.figure(figsize=figsize)
        plt.subplot(row, col, 1)
        plt.imshow(self.image, cmap="gray")
        for atom_type in np.unique(self.atom_types):
            mask = self.atom_types == atom_type
            element = self.elements[int(atom_type)]
            if half is not None:
                if half == "top":
                    mask = mask & (self.coordinates[:, 1] < self.ny / 2)
                elif half == "bottom":
                    mask = mask & (self.coordinates[:, 1] > self.ny / 2)
                elif half == "left":
                    mask = mask & (self.coordinates[:, 0] < self.nx / 2)
                elif half == "right":
                    mask = mask & (self.coordinates[:, 0] > self.nx / 2)
            plt.scatter(
                self.coordinates[mask, 0],
                self.coordinates[mask, 1],
                s=s,
                label=element,
            )
        plt.legend(loc="upper right")
        plt.gca().set_aspect("equal", adjustable="box")
        plt.axis("off")
        scalebar = self.scalebar
        plt.gca().add_artist(scalebar)
        plt.title("Image")
        plt.tight_layout()

        # plot the scs
        pos_x = self.params["pos_x"] * self.dx
        pos_y = self.params["pos_y"] * self.dx
        pos_x = safe_convert_to_numpy(pos_x)
        pos_y = safe_convert_to_numpy(pos_y)
        if per_element:
            plt_idx = 1
            col = len(np.unique(self.atom_types)) + 1
            for atom_type in np.unique(self.atom_types):
                plt_idx += 1
                plt.subplot(row, col, plt_idx)
                mask = self.atom_types == atom_type
                element = self.elements[atom_type]
                im = plt.scatter(
                    pos_x[mask],
                    pos_y[mask],
                    c=self.volume[mask],
                    s=s,
                    label=element,
                )
                cbar = plt.colorbar(im, fraction=0.046, pad=0.04)
                plt.gca().set_aspect("equal", adjustable="box")
                # plt.axis("off")
                plt.xlim(0, self.nx * self.dx)
                plt.ylim(0, self.ny * self.dx)
                plt.xlabel(r"X (A)")
                plt.ylabel(r"Y (A)")
                plt.gca().invert_yaxis()
                plt.title(f"{element}")
                if atom_type == self.atom_types.max():
                    if has_units:
                        cbar.set_label(r"SCS (A^2)")
                    else:
                        cbar.set_label("Integrated intensities")
                plt.tight_layout()
        else:
            plt.subplot(row, col, 2)
            im = plt.scatter(pos_x, pos_y, c=self.volume, s=2)
            cbar = plt.colorbar(im, fraction=0.046, pad=0.04)
            # plt.axis("off")
            plt.xlim(0, self.nx * self.dx)
            plt.ylim(0, self.ny * self.dx)
            plt.xlabel(r"X (A)")
            plt.ylabel(r"Y (A)")
            plt.gca().invert_yaxis()
            plt.gca().set_aspect("equal", adjustable="box")
            if has_units:
                cbar.set_label(r"SCS (A^2)")
            else:
                cbar.set_label("Integrated intensities")
            plt.tight_layout()
        if save:
            plt.savefig("scs.svg")
            plt.savefig("scs.png", dpi=300)

    def plot_scs_voronoi(
        self,
        layout="horizontal",
        s=1,
        per_element=False,
        save=False,
        has_units=True,
        half: str = None,
        figsize=(10, 5),
    ):
        assert self.voronoi_volume is not None, "Please run the voronoi analysis first"
        if per_element:
            row, col = 1, 2
            col += len(np.unique(self.atom_types)) - 1
            plt.figure(figsize=figsize)
            plt.subplot(row, col, 1)
            plt.imshow(self.image, cmap="gray")
            for atom_type in np.unique(self.atom_types):
                mask = self.atom_types == atom_type
                element = self.elements[atom_type]
                if half is not None:
                    if half == "top":
                        mask = mask & (self.coordinates[:, 1] < self.ny / 2)
                    elif half == "bottom":
                        mask = mask & (self.coordinates[:, 1] > self.ny / 2)
                    elif half == "left":
                        mask = mask & (self.coordinates[:, 0] < self.nx / 2)
                    elif half == "right":
                        mask = mask & (self.coordinates[:, 0] > self.nx / 2)
                plt.scatter(
                    self.coordinates[mask, 0],
                    self.coordinates[mask, 1],
                    s=1,
                    label=element,
                )
            plt.legend(loc="upper right")
            plt.gca().add_artist(self.scalebar)
            plot_idx = 2
            for atom_type in np.unique(self.atom_types):
                mask = self.atom_types == atom_type
                plt.subplot(row, col, plot_idx)
                element = self.elements[atom_type]
                pos_x = self.params["pos_x"][mask] * self.dx
                pos_y = self.params["pos_y"][mask] * self.dx
                pos_x = safe_convert_to_numpy(pos_x)
                pos_y = safe_convert_to_numpy(pos_y) 
                im = plt.scatter(
                    pos_x, pos_y, c=self.voronoi_volume[mask], s=s, label=element
                )
                plt.gca().set_aspect("equal", adjustable="box")
                cbar = plt.colorbar(im, fraction=0.046, pad=0.04)
                # plt.axis("off")
                plt.xlim(0, self.nx * self.dx)
                plt.ylim(0, self.ny * self.dx)
                plt.xlabel(r"X (A)")
                plt.ylabel(r"Y (A)")
                plt.gca().invert_yaxis()
                plt.title(f"{element}")
                if atom_type == self.atom_types.max():
                    if has_units:
                        cbar.set_label(r"Voronoi SCS (A^2)")
                    else:
                        cbar.set_label("Voronoi integrated intensities")
                plot_idx += 1
        else:
            row, col = (1, 2) if layout == "horizontal" else (2, 1)
            plt.figure()
            plt.subplot(row, col, 1)
            plt.imshow(self.image, cmap="gray")
            for atom_type in np.unique(self.atom_types):
                mask = self.atom_types == atom_type
                element = self.elements[atom_type]
                plt.scatter(
                    self.coordinates[mask, 0],
                    self.coordinates[mask, 1],
                    s=1,
                    label=element,
                )
            plt.legend()
            plt.gca().set_aspect("equal", adjustable="box")
            plt.title("Image")

            plt.subplot(row, col, 2)
            pos_x = self.params["pos_x"] * self.dx
            pos_y = self.params["pos_y"] * self.dx
            im = plt.scatter(pos_x, pos_y, c=self.voronoi_volume, s=s)
            # make aspect ratio equal
            plt.gca().invert_yaxis()
            plt.gca().set_aspect("equal", adjustable="box")
            plt.colorbar(im, fraction=0.046, pad=0.04)
            if has_units:
                plt.title(r"Voronoi scs (A^2)")
            else:
                plt.title("Voronoi integrated intensities")
        plt.tight_layout()

        if save:
            plt.savefig("voronoi_scs.svg")
            plt.savefig("voronoi_scs.png", dpi=300)
    def plot_voronoi_integration_intensity(self,plot = False, save=False):
        if plot:
            plt.imshow(self._voronoi_map, cmap="viridis")
            plt.colorbar(label="Voronoi Integrated Intensity")
        if save:
            plt.savefig("Voronoi Integrated Intensity.png", dpi=300)
            plt.savefig("Voronoi Integrated Intensity.svg")

    def estimate_atom_counts_with_gmm(
        self,
        max_components: int = 5,
        scoring_method: str = "icl",
        initialization_method: str = "middle",
        plot_results: bool = True,
        per_element: bool = True,
        save_results: bool = False,
        interactive_selection: bool = True,
        use_first_local_minimum: bool = True,
    ):
        """Estimate atom counts using Gaussian Mixture Model on cross-section histograms.
        
        This method applies GMM to the refined cross-section histogram to statistically
        determine the number of atoms in each atomic column based on scattering cross-sections.
        
        Args:
            max_components: Maximum number of Gaussian components to test
            scoring_method: Information criterion for model selection ('icl', 'aic', 'bic')
            initialization_method: Method for initializing GMM means
            plot_results: Whether to plot the GMM fitting results
            per_element: Whether to fit GMM separately for each element type
            save_results: Whether to save plots and results
            interactive_selection: Whether to allow interactive component selection
            use_first_local_minimum: Whether to use first local minimum instead of global
            
        Returns:
            dict: Dictionary containing GMM results and atom count estimates
        """
        if not hasattr(self, 'params') or self.params is None:
            raise ValueError("Please run fitting first to obtain refined cross-sections")
        
        from qem.analysis.gaussian_mixture_model import GaussianMixtureModel
        
        # Get refined cross-sections (volumes)
        cross_sections = self.volume.reshape(-1, 1)  # Reshape for GMM input
        
        gmm_results = {}
        atom_count_estimates = {}
        
        if per_element:
            # Fit GMM separately for each element type
            for atom_type in np.unique(self.atom_types):
                element_name = self.elements[atom_type]
                mask = self.atom_types == atom_type
                element_cross_sections = cross_sections[mask]
                
                if len(element_cross_sections) < 10:  # Skip if too few data points
                    logging.warning(f"Skipping GMM for {element_name}: insufficient data points")
                    continue
                
                # Initialize and fit GMM
                gmm = GaussianMixtureModel(element_cross_sections)
                gmm.fit_gaussian_mixture_model(
                    num_components=max_components,
                    scoring_methods=[scoring_method, "nllh"],
                    initialization_method=initialization_method,
                    use_first_local_minimum=use_first_local_minimum,
                )
                
                # Plot results and allow component selection
                if plot_results:
                    selected_components = gmm.plot_interactive_gmm_selection(
                        element_cross_sections, element_name, 
                        save_results, interactive_selection
                    )
                else:
                    # Use recommendation if no plotting
                    selected_components = gmm.get_optimal_components("recommendation")
                
                # Get component parameters using user-selected components
                component_idx = selected_components - 1
                weights = gmm.fit_result.weight[component_idx]
                means = gmm.fit_result.mean[component_idx]
                widths = gmm.fit_result.width[component_idx]
                
                # Estimate atom counts based on component means
                # Assume components correspond to different atom counts (1, 2, 3, etc.)
                sorted_indices = np.argsort(means.flatten())
                atom_counts = np.arange(1, len(sorted_indices) + 1)
                
                # Assign atom counts to each atomic column
                column_assignments = gmm.fit_result.idxComponentOfScs(component_idx)
                estimated_counts = atom_counts[sorted_indices][column_assignments]
                
                gmm_results[element_name] = {
                    'gmm_model': gmm,
                    'selected_components': selected_components,  # Store user selection
                    'recommended_components': gmm.recommended_components,  # Store recommendation
                    'weights': weights,
                    'means': means[sorted_indices],
                    'widths': widths[sorted_indices],
                    'scores': gmm.fit_result.score,
                }
                
                atom_count_estimates[element_name] = estimated_counts
                
        else:
            # Fit GMM to all cross-sections together
            gmm = GaussianMixtureModel(cross_sections)
            gmm.fit_gaussian_mixture_model(
                num_components=max_components,
                scoring_methods=[scoring_method, "nllh"],
                initialization_method=initialization_method,
                use_first_local_minimum=use_first_local_minimum,
            )
            
            # Plot results and allow component selection
            if plot_results:
                selected_components = gmm.plot_interactive_gmm_selection(
                    cross_sections, 'all_elements', 
                    save_results, interactive_selection
                )
            else:
                selected_components = gmm.get_optimal_components("recommendation")
            
            component_idx = selected_components - 1
            
            weights = gmm.fit_result.weight[component_idx]
            means = gmm.fit_result.mean[component_idx]
            widths = gmm.fit_result.width[component_idx]
            
            sorted_indices = np.argsort(means.flatten())
            atom_counts = np.arange(1, len(sorted_indices) + 1)
            
            column_assignments = gmm.fit_result.idxComponentOfScs(component_idx)
            estimated_counts = atom_counts[sorted_indices][column_assignments]
            
            gmm_results['all_elements'] = {
                'gmm_model': gmm,
                'selected_components': selected_components,  # Store user selection
                'recommended_components': gmm.recommended_components,  # Store recommendation
                'weights': weights,
                'means': means[sorted_indices],
                'widths': widths[sorted_indices],
                'scores': gmm.fit_result.score,
            }
            
            atom_count_estimates['all_elements'] = estimated_counts
        
        # Store results as instance attributes
        self.gmm_results = gmm_results
        self.atom_count_estimates = atom_count_estimates
        
        return {
            'gmm_results': gmm_results,
            'atom_count_estimates': atom_count_estimates,
        }

    def _plot_gmm_results(self, cross_sections, gmm_model, element_name, save_results=False):
        """Legacy method - redirects to GMM module plotting for compatibility."""
        return gmm_model.plot_interactive_gmm_selection(
            cross_sections, element_name, save_results, interactive_selection=False
        )

    def plot_scs_histogram(self, save=False, has_units=True):
        """Plot histogram of refined scattering cross-sections."""
        plt.figure()
        for atom_type in np.unique(self.atom_types):
            mask = self.atom_types == atom_type
            element = self.elements[atom_type]
            plt.hist(self.volume[mask], bins=100, alpha=0.5, label=element)
        plt.legend()
        if has_units:
            plt.xlabel(r"Refined SCS (A^2)")
        else:
            plt.xlabel("Integrated intensities")
        plt.ylabel("Frequency")
        plt.title("Histogram of QEM refined SCS")
        if save:
            plt.savefig("scs_histogram.svg")
            plt.savefig("scs_histogram.png", dpi=300)
    
    def plot_atom_count_map(self, element_name=None, save=False, figsize=(12, 8)):
        """Plot spatial map of estimated atom counts with proper colorbar.
        
        Args:
            element_name: Specific element to plot, or None for all elements
            save: Whether to save the plot
            figsize: Figure size tuple
        """
        if not hasattr(self, 'atom_count_estimates'):
            raise ValueError("Please run estimate_atom_counts_with_gmm first")
        
        fig, ax = plt.subplots(figsize=figsize)
        
        if element_name is None:
            # Plot all elements with different symbols/colors
            all_counts = []
            all_pos_x = []
            all_pos_y = []
            scatter = None  # Initialize scatter variable
            
            for atom_type in np.unique(self.atom_types):
                element = self.elements[atom_type]
                if element in self.atom_count_estimates:
                    mask = self.atom_types == atom_type
                    counts = self.atom_count_estimates[element]
                    
                    pos_x = self.params["pos_x"][mask] * self.dx
                    pos_y = self.params["pos_y"][mask] * self.dx
                    
                    pos_x_np = safe_convert_to_numpy(pos_x)
                    pos_y_np = safe_convert_to_numpy(pos_y)
                    
                    all_counts.extend(counts)
                    all_pos_x.extend(pos_x_np)
                    all_pos_y.extend(pos_y_np)
                    
                    # Plot each element with different marker
                    markers = ['o', 's', '^', 'D', 'v', '<', '>', 'p', 'h', 'H']
                    marker = markers[atom_type % len(markers)]
                    
                    scatter = ax.scatter(
                        pos_x_np, pos_y_np,
                        c=counts, s=80, alpha=0.8, 
                        marker=marker, label=f'{element}',
                        cmap='viridis', vmin=1, vmax=max(all_counts) if all_counts else 5
                    )
            
            # Create colorbar for all elements
            if all_counts and scatter is not None:
                cbar = plt.colorbar(scatter, ax=ax, shrink=0.8)
                cbar.set_label('Number of Atoms', fontsize=14, fontweight='bold')
                # Set integer ticks on colorbar
                max_count = max(all_counts)
                cbar.set_ticks(range(1, max_count + 1))
                
            ax.legend(loc='upper right', bbox_to_anchor=(1.15, 1))
            
        else:
            # Plot specific element
            if element_name not in self.atom_count_estimates:
                raise ValueError(f"No atom count estimates found for {element_name}")
            
            atom_type = list(self.elements).index(element_name)
            mask = self.atom_types == atom_type
            counts = self.atom_count_estimates[element_name]
            
            pos_x = self.params["pos_x"][mask] * self.dx
            pos_y = self.params["pos_y"][mask] * self.dx
            
            pos_x_np = safe_convert_to_numpy(pos_x)
            pos_y_np = safe_convert_to_numpy(pos_y)
            
            scatter = ax.scatter(
                pos_x_np, pos_y_np,
                c=counts, s=100, alpha=0.8, cmap='viridis',
                edgecolors='black', linewidth=0.5
            )
            
            # Create colorbar with proper title
            cbar = plt.colorbar(scatter, ax=ax, shrink=0.8)
            cbar.set_label('Number of Atoms', fontsize=14, fontweight='bold')
            # Set integer ticks on colorbar
            unique_counts = np.unique(counts)
            cbar.set_ticks(unique_counts)
            
            ax.set_title(f'Atom Count Map - {element_name}', fontsize=16, fontweight='bold')
        
        ax.set_xlabel('X (Å)', fontsize=12)
        ax.set_ylabel('Y (Å)', fontsize=12)
        if element_name is None:
            ax.set_title('Spatial Map of Estimated Atom Counts', fontsize=16, fontweight='bold')
        
        ax.set_aspect('equal', adjustable='box')
        ax.invert_yaxis()
        ax.grid(True, alpha=0.3)
        
        # Add summary text
        if hasattr(self, 'gmm_results'):
            summary_info = []
            for elem, results in self.gmm_results.items():
                if 'selected_components' in results:
                    selected = results['selected_components']
                    recommended = results.get('recommended_components', 'N/A')
                    summary_info.append(f"{elem}: {selected} components (rec: {recommended})")
            
            if summary_info:
                summary_text = "GMM Selection: " + ", ".join(summary_info)
                ax.text(0.02, 0.02, summary_text, transform=ax.transAxes,
                       bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
                       fontsize=10, verticalalignment='bottom')
        
        plt.tight_layout()
        
        if save:
            filename = f'atom_count_map_{element_name or "all"}.png'
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            logging.info(f"Atom count map saved as {filename}")
        
        plt.show()

    def integrate_gmm_with_crystal_analyzer(self, region_index: int = 0):
        """Integrate GMM atom count estimates with crystal analyzer atomic model.
        
        This method combines the statistical atom counting from GMM with the 
        crystal structure analysis to create a 3D atomic model with realistic
        atom counts in each column. Z-spacing is automatically determined from
        the supercell structure.
        
        Args:
            region_index: Index of the region to update (default: 0)
            
        Returns:
            Updated crystal analyzer object with GMM-based atom counts
        """
        if not hasattr(self, 'atom_count_estimates'):
            raise ValueError("Please run estimate_atom_counts_with_gmm() first")
            
        if region_index not in self.regions.keys:
            raise ValueError(f"Region {region_index} not found in regions")
            
        # Get the crystal analyzer for this region
        region = self.regions[region_index]
        if not hasattr(region, 'analyzer') or region.analyzer is None:
            raise ValueError(f"No crystal analyzer found for region {region_index}. "
                           "Please run map_lattice() first.")
                           
        crystal_analyzer = region.analyzer
        
        # Filter atom count estimates for columns in this region
        column_mask = self.region_column_labels == region_index
        region_atom_counts = {}
        
        for element_name, all_counts in self.atom_count_estimates.items():
            if element_name == 'all_elements':
                # Handle case where GMM was fit to all elements together
                region_atom_counts[element_name] = all_counts[column_mask]
            else:
                # Handle per-element GMM fitting
                element_columns = column_mask & (self.atom_types == self.elements.index(element_name))
                if element_columns.any():
                    region_atom_counts[element_name] = all_counts
        
        # Update the crystal analyzer with GMM results
        updated_columns = crystal_analyzer.update_atoms_from_gmm(
            region_atom_counts
        )
        
        # Update the region's columns
        region.columns = updated_columns
        
        return crystal_analyzer
        
    def update_all_regions_with_gmm(self):
        """Update all regions with GMM atom count estimates.
        
        Z-spacing is automatically determined from the supercell structure.
            
        Returns:
            Dictionary mapping region indices to updated crystal analyzers
        """
        updated_analyzers = {}
        
        for region_index in self.regions.keys:
            try:
                analyzer = self.integrate_gmm_with_crystal_analyzer(region_index)
                updated_analyzers[region_index] = analyzer
                logging.info(f"Successfully updated region {region_index} with GMM results")
            except Exception as e:
                logging.warning(f"Could not update region {region_index}: {str(e)}")
                
        return updated_analyzers
        
    def export_gmm_updated_structure(self, region_index: int = 0, filename: str = None):
        """Export the GMM-updated atomic structure to various formats.
        
        Args:
            region_index: Index of the region to export
            filename: Output filename (without extension)
            
        Returns:
            ASE Atoms object of the updated structure
        """
        if region_index not in self.regions.keys:
            raise ValueError(f"Region {region_index} not found")
            
        region = self.regions[region_index]
        if not hasattr(region, 'columns') or region.columns is None:
            raise ValueError(f"No atomic columns found for region {region_index}. "
                           "Please run integrate_gmm_with_crystal_analyzer() first.")
        
        # Get the updated lattice
        updated_lattice = region.columns.lattice
        
        if filename:
            # Export to different formats
            from ase.io import write
            write(f"{filename}.xyz", updated_lattice)
            write(f"{filename}.cif", updated_lattice) 
            logging.info(f"Exported GMM-updated structure to {filename}.xyz and {filename}.cif")
            
        return updated_lattice

    def plot_region(self):
        plt.figure()
        plt.imshow(self.image, cmap="gray")
        plt.imshow(self.regions.region_map, alpha=0.5)
        scalebar = self.scalebar
        plt.gca().add_artist(scalebar)
        plt.axis("off")
        cbar = plt.colorbar()
        cbar.set_ticks(np.arange(self.regions.num_regions))  # type: ignore
        plt.title("Region Map")
    
    # domain analysis
    def estimate_complex_domains(
        self,
        domain_separation_method: str = "intensity_gradient",
        min_domain_size: int = 200,
        domain_threshold: int = 15,  # Percentile threshold for domain boundary detection
        plot_analysis: bool = True,
        vacuum_threshold: float = 0.05,  # Threshold for vacuum detection
        polygon_enclosure: bool = True,  # Enable polygon enclosure
    ):
        """
        Enhanced peak position estimation for complex STO domains with comprehensive
        domain analysis, polygon enclosure, and robust peak detection.
        
        This enhanced method addresses several critical issues:
        1. Separates vacuum from interface regions before gradient calculation
        2. Creates continuous mask regions instead of lattice patterns
        3. Automatically encloses each domain using polygons with indexing
        4. Implements robust multi-scale algorithm for strong/weak peak detection
        
        Args:
            domain_separation_method: Method to separate domains ('intensity_gradient', 'laplacian', 'sobel')
            min_domain_size: Minimum size for a domain to be considered bulk
            plot_analysis: Whether to plot the analysis results
            vacuum_threshold: Threshold for vacuum region detection
            polygon_enclosure: Whether to use automatic polygon enclosure
            
        Returns:
            dict: Dictionary containing enhanced peak coordinates, region classifications, and polygon data
        """
        
        # Convert interface width from Angstroms to pixels
        
        # Step 1: Vacuum separation and preprocessing
        vacuum_mask, clean_image = self._separate_vacuum_and_sample(
            vacuum_threshold=vacuum_threshold
        )
        
        
        # Step 2: Enhanced domain boundary identification
        sample_mask, boundary_strength, domain_regions, domain_label = self._identify_domain_boundaries(
            method=domain_separation_method,
            min_domain_size=min_domain_size,
            domain_threshold=domain_threshold,
            vacuum_mask=vacuum_mask,
            clean_image=clean_image
        )
        
        # Step 3: Automatic polygon enclosure with indexing
        polygon_data = {}
        if polygon_enclosure:
            polygon_data = self._create_polygon_enclosures(domain_regions)        

        # Step 4: plotting
        if plot_analysis:
            self._plot_domain_analysis(vacuum_mask,  boundary_strength, polygon_data, domain_label)
        
        results = {
            'bulk_mask': sample_mask,
            'boundary_strength': boundary_strength,
            'domain_regions': domain_regions,
            'polygon_data': polygon_data,
            'vacuum_mask': vacuum_mask
        }
        
        
        return results

    def _separate_vacuum_and_sample(self, vacuum_threshold: float = 0.05):
        """
        Separate vacuum regions from interface regions using intensity-based thresholding.
        
        Args:
            vacuum_threshold: Threshold for identifying vacuum regions (low intensity)
            
        Returns:
            tuple: (vacuum_mask, clean_image) where vacuum_mask identifies vacuum regions
                   and clean_image has vacuum regions masked out
        """

        
        # Create intensity histogram to identify vacuum threshold
        image_flat = self.image.flatten()
        # Use median absolute deviation for robust threshold estimation
        median_intensity = np.median(image_flat)
        mad = np.median(np.abs(image_flat - median_intensity))
        
        vacuum_threshold_abs = np.percentile(image_flat, vacuum_threshold*100)
        # Adaptive vacuum threshold based on image statistics
        adaptive_threshold = min(vacuum_threshold_abs, median_intensity - 2 * mad)
        
        # Detect vacuum regions
        vacuum_mask = self.image < adaptive_threshold
        
        # Clean up vacuum mask to remove noise
        vacuum_mask = gaussian_filter(vacuum_mask.astype(float), 10) > 0.95
        vacuum_mask = remove_small_objects(vacuum_mask)
        # vacuum_mask = binary_dilation(vacuum_mask, iterations=5)

        # Create clean image with vacuum masked out
        clean_image = self.image.copy()
        clean_image[vacuum_mask] = np.median(self.image[~vacuum_mask])
        
        return vacuum_mask, clean_image

    def _identify_domain_boundaries(self, method="intensity_gradient", min_domain_size=50, domain_threshold = 15, vacuum_mask=None, clean_image=None):
        """
        Enhanced domain boundary identification with continuous regions and vacuum separation.
        
        Args:
            method: Method for boundary detection
            min_domain_size: Minimum size for bulk regions
            vacuum_mask: Mask identifying vacuum regions
            clean_image: Pre-processed image with vacuum removed
            
        Returns:
            tuple: (bulk_mask, interface_mask, boundary_strength, domain_regions)
        """

        
        if clean_image is None:
            clean_image = self.image
        
        # Apply different boundary detection methods on clean image
        if method == "intensity_gradient":
            # Use gradient magnitude to identify boundaries
            grad_x = sobel(gaussian_filter(clean_image, 2), axis=1)
            grad_y = sobel(gaussian_filter(clean_image, 2), axis=0)
            boundary_strength = np.sqrt(grad_x**2 + grad_y**2)
            
        elif method == "laplacian":
            # Use Laplacian to identify rapid intensity changes
            boundary_strength = np.abs(laplace(gaussian_filter(clean_image, 1.5)))
            
        elif method == "sobel":
            # Use Sobel operator for edge detection
            boundary_strength = sobel(gaussian_filter(clean_image, 2))
            
        else:
            raise ValueError(f"Unknown boundary detection method: {method}")
        
        # Normalize boundary strength
        boundary_strength = boundary_strength / boundary_strength.max()
        boundary_strength = gaussian_filter(boundary_strength, sigma=20.0)


        sample_threshold = np.percentile(gaussian_filter(self.image, 5), 5)
        sample_mask = gaussian_filter(self.image, 5) > sample_threshold
        sample_mask = gaussian_filter(remove_small_objects(sample_mask), 5) > 0.5
        # # Create boundary mask using adaptive threshold
        domain_threshold_abs = np.percentile(boundary_strength, domain_threshold)  
        
        domain_mask = (boundary_strength < domain_threshold_abs) & (~vacuum_mask) & sample_mask
        domain_mask = remove_small_objects(domain_mask, min_size=min_domain_size)
        
        # Remove small bulk regions
        domain_label = label(domain_mask)

        # Identify continuous bulk regions
        unique_regions = np.unique(domain_label)
        unique_regions = unique_regions[unique_regions != 0]  # Remove background
        
        domain_regions = {}
        
        for region_id in unique_regions:
            region_mask = domain_label == region_id
            region_size = np.sum(region_mask)
            
            if region_size >= min_domain_size:
                domain_regions[region_id] = {
                    'mask': region_mask,
                    'size': region_size,
                    'centroid': np.array(np.where(region_mask)).mean(axis=1)
                }
        return sample_mask, boundary_strength, domain_regions, domain_label

    def _create_polygon_enclosures(self, domain_regions):
        """
        Automatically create polygon enclosures for each identified domain.
        
        Args:
            domain_regions: Dictionary of domain regions
            interface_mask: Mask of interface regions
            
        Returns:
            dict: Polygon data with indices and boundaries
        """

        
        polygon_data = {}
        
        # Create polygon for each domain region
        for region_id, region_info in domain_regions.items():
            mask = region_info['mask']
            
            # Find contours for this region
            contours = find_contours(mask.astype(float), 0.5)
            
            if len(contours) > 0:
                # Use the largest contour
                largest_contour = max(contours, key=len)
                
                # Create polygon path
                polygon_path = Path(largest_contour)
                
                polygon_data[region_id] = {
                    'vertices': largest_contour,
                    'path': polygon_path,
                    'centroid': region_info['centroid'],
                    'area': region_info['size'],
                }

        return polygon_data

    def _plot_domain_analysis(
        self, vacuum_mask, boundary_strength, polygon_data, domain_label
    ):
        """
        Enhanced plotting with polygon boundaries and region indices.
        """
        fig, axes = plt.subplots(1, 3, figsize=(24, 12))
        
        # Original image
        axes[0].imshow(self.image, cmap='gray')
        axes[0].set_title('Original Image')
        axes[0].axis('off')
        
        # Boundary strength
        im1 = axes[1].imshow(boundary_strength, cmap='viridis')
        axes[1].set_title('Boundary Strength')
        axes[1].axis('off')
        plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)
        
        # Continuous domain separation
        domain_map = domain_label.copy()
        domain_map[vacuum_mask] = -1  # Background
        axes[2].imshow(self.image, cmap='gray')
        im2 = axes[2].imshow(domain_map, vmin=-1, vmax=domain_label.max(),alpha=0.3)
        axes[2].set_title('Domain Map\n(-1=Background, 0=Bulk, >1=Domains)')
        axes[2].axis('off')
        plt.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)
        
        # Polygon boundaries
        if polygon_data:
            for region_id, region_info in polygon_data.items():
                vertices = region_info['vertices']
                axes[2].plot(vertices[:, 1], vertices[:, 0], linewidth=2)
                centroid = region_info['centroid']
                axes[2].text(centroid[1], centroid[0], str(region_id), 
                              color='white', fontsize=8, ha='center', va='center')
        axes[2].set_title('Polygon Boundaries')
        axes[2].axis('off')
        
        plt.tight_layout()
        plt.show()

    # Properties

    @property
    def atom_types(self):
        if len(self._atom_types) == 0 or self._atom_types is None:
            self._atom_types = np.zeros(self.num_coordinates, dtype=int)
        return self._atom_types

    @atom_types.setter
    def atom_types(self, atom_types: np.ndarray):
        self._atom_types = atom_types

    @property
    def coordinates(self):
        return self._coordinates

    @coordinates.setter
    def coordinates(self, coordinates: np.ndarray):
        self._coordinates = coordinates

    @property
    def num_coordinates(self):
        return len(self._coordinates) if len(self._coordinates.shape) > 0 else 0

    @property
    def num_atom_types(self):
        assert self.atom_types is not None, "Atom types are not set."
        assert len(self.atom_types) > 0, "Atom types are empty."
        return len(np.unique(self.atom_types))

    @property
    def region_column_labels(self):
        coordinates = self.coordinates
        atom_types = self.atom_types
        mask = (
            (coordinates[:, 0] >= 0)
            & (coordinates[:, 0] < self.nx)
            & (coordinates[:, 1] >= 0)
            & (coordinates[:, 1] < self.ny)
        )
        self.coordinates = coordinates[mask]
        self.atom_types = atom_types[mask]
        return self.regions.region_map[
            self.coordinates[:, 1].astype(int), self.coordinates[:, 0].astype(int)
        ]

    @property
    def voronoi_volume(self):
        return self._voronoi_volume


    @property
    def window(self):
        """
        Returns the window used for fitting.

        Returns:
            numpy.ndarray: A Butterworth-style window used for fitting.
        """
        if self._window is None:
            window = butterworth_window(self.image.shape, 0.5, 10)
            self._window = window
        return self._window
    
    def disable_edge_window(self):
        """
        Disable edge dampening window for better edge peak fitting.
        
        The default Butterworth window dampens edge pixels to reduce
        Fourier artifacts, but this makes fitting edge peaks harder.
        Call this method to use uniform weighting across the image.
        
        Example:
            >>> fitter.disable_edge_window()
            >>> fitter.enable_boundary_penalty()
            >>> fitter.fit_global()  # Better edge peak fitting
        """
        self._window = np.ones_like(self.image)
        self._window_t = None  # invalidate cached torch view
        logging.info("Edge window dampening disabled (uniform weighting)")

    @property
    def volume(self):
        """Calculate the volume of each peak in the model.

        Returns:
            numpy.ndarray: Array of volumes for each peak.
        """
        if not self.params:
            raise ValueError("Parameters not initialized. Call init_params first.")

        # Update the model's pixel size
        self.model.dx = self.dx

        # Create parameters dict for volume calculation
        params = self.params.copy()
        if self.same_width:
            params["width"] = params["width"][self.atom_types]
            if "ratio" in params:
                params["ratio"] = params["ratio"][self.atom_types]
        volume = self.model.volume(params)
        return safe_convert_to_numpy(volume)

    @property
    def scalebar(self):
        from qem.utils.scalebar import to_scalebar_units

        scale, units = to_scalebar_units(self.dx, self.units)
        scalebar = ScaleBar(
            scale,
            units=units,
            location="lower right",
            length_fraction=0.2,
            font_properties={"size": 20},
        )
        return scalebar
