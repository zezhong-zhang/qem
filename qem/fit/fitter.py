import copy
import logging
from contextlib import nullcontext
from typing import Any

import h5py
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib_scalebar.scalebar import ScaleBar
from scipy.ndimage import gaussian_filter
from tqdm import tqdm

from qem.analysis.crystal_analyzer import CrystalAnalyzer
from qem.analysis.region import Region, Regions
from qem.fit.background import Background
from qem.viz.select import GetAtomSelection, GetRegionSelection
from qem.fit.model import (
    GaussianKernel,
    GaussianModel,
    ImageModel,
    LorentzianModel,
    VoigtModel,
)
from qem.processing import butterworth_window
from qem.utils.arrays import get_random_indices_in_batches
from qem.utils.memory import MemoryMonitor
from qem.utils.tensors import (
    best_device,
    clone_params,
    release_memory,
    stop_grad,
    to_numpy,
    to_tensor,
)

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
                self.image_tensor = to_tensor(self.image)
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

                    self.params = {k: to_tensor(v) for k, v in params.items()}

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

    def _params_to_device(self, params: dict[str, Any]) -> dict[str, Any]:
        """Coerce a parameter dict to torch tensors on ``self.device``.

        Bool / config entries (``same_width``) pass through unchanged.
        """
        out: dict[str, Any] = {}
        for k, v in params.items():
            if isinstance(v, bool):
                out[k] = v
                continue
            tensor = v if torch.is_tensor(v) else torch.as_tensor(v)
            if k in ("atom_types",):
                tensor = tensor.to(dtype=torch.int64, device=self.device)
            else:
                tensor = tensor.to(dtype=torch.float32, device=self.device)
            out[k] = tensor
        return out

    # Init grids and models
    def initialize_grid(self):
        """Initialize the coordinate grids for the model on the best device."""
        device = best_device()
        self.device = device
        self.image_tensor = torch.as_tensor(
            gaussian_filter(self.image, 1), dtype=torch.float32, device=device,
        )
        x = torch.arange(self.nx, dtype=torch.float32, device=device)
        y = torch.arange(self.ny, dtype=torch.float32, device=device)
        x_grid, y_grid = torch.meshgrid(x, y, indexing="xy")
        self.x_grid = x_grid.to(dtype=torch.float32)
        self.y_grid = y_grid.to(dtype=torch.float32)
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
        pos_x = to_numpy(params_stage1['pos_x'])
        pos_y = to_numpy(params_stage1['pos_y'])
        
        edge_mask = (pos_x < 5) | (pos_x > w-5) | (pos_y < 5) | (pos_y > h-5)
        
        if np.any(edge_mask):
            # Boost height and width for edge peaks
            height = to_numpy(params_stage1['height'])
            width = to_numpy(params_stage1['width'])
            
            height[edge_mask] *= 2.0  # Double height
            width[edge_mask] *= 1.8   # Increase width by 80%
            
            params_stage1['height'] = to_tensor(height)
            params_stage1['width'] = to_tensor(width)
            
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
        self.prediction = to_numpy(self.predict(params_final, local=True))
        
        return params_final
    
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
        coordinates = np.delete(self.coordinates, np.where(column_mask), axis=0)
        coordinates = np.vstack([coordinates, atomic_column_list.positions_pixel])
        self.coordinates = coordinates
        atom_types = np.delete(self.atom_types, np.where(column_mask), axis=0)
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

        # Place parameters on the active device (CUDA / MPS / CPU).
        model.to(self.device)

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
        # self.prediction = to_numpy(prediction)
        return prediction

    def residual(self, params: dict):
        # Compute the sum of the Gaussians
        prediction = self.predict(params)
        diff = self.image_tensor - prediction
        diff = to_numpy(diff)
        return diff

    # fitting
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

        # Move all model parameters to the chosen accelerator (CUDA / MPS / CPU).
        # initialize_grid() set self.device; the model parameters were created
        # on CPU by nn.Parameter(...) and must follow.
        model.to(self.device)

        # Store reference to model being optimized so loss function can access it
        self._optimization_model = model

        if verbose:
            print(f"Using {optimizer} optimizer for fitting on {self.device}.")
        # PyTorch expects a leading batch dimension on inputs.
        image_tensor = image_tensor.to(self.device).unsqueeze(0)
        model_inputs = [self.x_grid_batched, self.y_grid_batched]
        
        operation_context = (
            self.memory_monitor.monitor_operation("optimize") 
            if self.memory_monitor else nullcontext()
        )
        
        with operation_context:
            from qem.fit.loop import fit_loop, make_optimizer

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
        self.prediction = to_numpy(self.predict(params, local=local))
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
        params = {k: stop_grad(v) for k, v in params.items()}

        self.converged = False
        operation_context = (
            self.memory_monitor.monitor_operation("fit_stochastic")
            if self.memory_monitor else nullcontext()
        )

        # Pre-condition heights with a least-squares pass. The stochastic
        # fitter is robust to a no-op pre-conditioning, so swallow failures
        # here rather than aborting the whole run.
        params = self.linear_estimator(params, best_effort=True)
        # Move everything to the active device once; subsequent batch
        # operations rely on params already being torch tensors there.
        params = self._params_to_device(params)

        with operation_context:
            for epoch in tqdm(range(num_epoch), desc="Training epochs", leave=False):
                pre_params = clone_params(params)
                random_batches = get_random_indices_in_batches(self.num_coordinates, batch_size)

                for batch_indices in tqdm(random_batches, desc="Fitting batch", leave=False):
                    # Calculate local target (subtract other atoms' contributions)
                    if batch_size < self.num_coordinates:
                        params_without_batch = clone_params(params)
                        height_tensor = params_without_batch['height']
                        batch_indices_tensor = torch.as_tensor(
                            batch_indices, dtype=torch.int64, device=height_tensor.device,
                        )
                        new_height = height_tensor.clone()
                        new_height.view(-1)[batch_indices_tensor] = 0.0
                        params_without_batch['height'] = new_height
                        params_without_batch['background'] = torch.zeros_like(params_without_batch['background'])

                        model_others = self._create_fitting_model(params_without_batch)

                        prediction_from_others = self.predict(params_without_batch, model=model_others, local=local)
                        local_target = (self.image_tensor - prediction_from_others).detach()

                        del params_without_batch
                        del prediction_from_others
                        del height_tensor
                        release_memory()
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
                    release_memory()
                    params = self.update_from_local_params(params, optimized_params, atoms_selected_mask)
                    if plot:
                        self._plot_progress(params, batch_indices, select_params)

                # Check convergence
                if self.convergence(params, pre_params, tol):
                    logging.info("Convergence criteria met.")
                    self.converged = True
                    break
        
        self.params = params
        self.prediction = to_numpy(self.predict(params, local=local))
        logging.info("Stochastic fitting complete.")
        return self.params

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
            if not torch.is_tensor(value):
                continue
            other = pre_params[key]
            if torch.is_tensor(other) and other.device != value.device:
                other = other.to(value.device)

            # Calculate the update difference
            update = torch.abs(value - other)

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
                new_param = params[key].clone()
                update_indices = torch.as_tensor(
                    np.where(mask)[0], dtype=torch.int64, device=new_param.device,
                )
                value_tensor = torch.as_tensor(
                    value, dtype=new_param.dtype, device=new_param.device,
                )
                new_param.view(-1)[update_indices] = value_tensor
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
        return to_numpy(volume)

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


# Hook the extracted methods back onto Fitter so call sites like
# fitter.plot_fitting() / fitter.estimate_complex_domains() keep working —
# the bodies live in their own sibling modules.
from qem.fit.plot import _bind as _bind_plot  # noqa: E402
from qem.fit.loss import _bind as _bind_loss  # noqa: E402
from qem.fit.peaks import _bind as _bind_peaks  # noqa: E402
from qem.fit.background import _bind as _bind_background  # noqa: E402
from qem.analysis.domains import _bind as _bind_domains  # noqa: E402
from qem.analysis.gmm import _bind as _bind_gmm  # noqa: E402

from qem.fit.voronoi import _bind as _bind_voronoi  # noqa: E402
from qem.fit.solver import _bind as _bind_solver  # noqa: E402

_bind_plot(Fitter)
_bind_loss(Fitter)
_bind_peaks(Fitter)
_bind_background(Fitter)
_bind_voronoi(Fitter)
_bind_solver(Fitter)
_bind_domains(Fitter)
_bind_gmm(Fitter)
