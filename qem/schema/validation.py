"""
Input validation utilities for QEM image fitting.
Provides comprehensive validation for user inputs and model parameters.
"""

import logging
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import keras


class ValidationError(ValueError):
    """Custom validation error with better context and user guidance."""
    
    def __init__(self, parameter: str, value: any, message: str, suggestion: str = None):
        full_message = f"Parameter '{parameter}' validation failed: {value} - {message}"
        if suggestion:
            full_message += f". Suggestion: {suggestion}"
        super().__init__(full_message)
        self.parameter = parameter
        self.value = value


class ImageFittingValidator:
    """Comprehensive validation for ImageFitting class parameters."""
    
    @staticmethod
    def validate_image(image: np.ndarray, max_size: int = 5000, max_memory_mb: int = 1000) -> np.ndarray:
        """
        Validate input image array with comprehensive edge case handling.
        
        Args:
            image: Input image array
            max_size: Maximum allowed dimension size
            max_memory_mb: Maximum allowed memory usage in MB
            
        Returns:
            Validated image array (C-contiguous)
            
        Raises:
            ValidationError: If image is invalid
        """
        if not isinstance(image, np.ndarray):
            raise ValidationError(
                "image", type(image).__name__, 
                "must be a numpy array",
                "Convert your data to numpy.ndarray using np.array()"
            )
        
        if image.ndim != 2:
            raise ValidationError(
                "image_shape", image.shape,
                f"must be 2D array, got {image.ndim}D",
                f"Expected shape (height, width), got {image.shape}"
            )
        
        if image.size == 0:
            raise ValidationError(
                "image", image.shape,
                "cannot be empty",
                "Provide a non-empty 2D array"
            )
        
        # Check for minimum size
        if image.shape[0] < 10 or image.shape[1] < 10:
            raise ValidationError(
                "image_size", image.shape,
                f"too small: {image.shape}. Minimum size is 10x10 pixels",
                "Use larger images or resize your input"
            )
        
        # Check for maximum reasonable size
        max_pixels = max_size * max_size
        if image.shape[0] * image.shape[1] > max_pixels:
            raise ValidationError(
                "image_size", image.shape,
                f"too large: {image.shape} ({image.shape[0]*image.shape[1]:,} pixels)",
                f"Maximum allowed: {max_pixels:,} pixels ({max_size}x{max_size})"
            )
        
        # Check memory usage
        estimated_memory_mb = image.nbytes / (1024 * 1024)
        if estimated_memory_mb > max_memory_mb:
            raise ValidationError(
                "image_memory", f"{estimated_memory_mb:.1f}MB",
                f"exceeds memory limit of {max_memory_mb}MB",
                "Use smaller images or downsample your data"
            )
        
        # Check data type and range
        supported_dtypes = {np.float32, np.float64, np.int16, np.int32, np.int64, np.uint8, np.uint16}
        if image.dtype.type not in supported_dtypes:
            raise ValidationError(
                "image_dtype", image.dtype,
                f"unsupported dtype. Supported: {supported_dtypes}",
                f"Convert using image.astype(np.float32)"
            )
        
        if np.any(np.isnan(image)):
            raise ValidationError(
                "image_data", "NaN detected",
                "contains NaN values",
                "Clean your data using np.nan_to_num() or remove NaN pixels"
            )
        
        if np.any(np.isinf(image)):
            raise ValidationError(
                "image_data", "Inf detected",
                "contains infinite values",
                "Clean your data using np.isfinite() mask"
            )
        
        # Ensure image has reasonable dynamic range
        if image.max() == image.min():
            raise ValidationError(
                "image_dynamic_range", f"min={image.min()}, max={image.max()}",
                "has no intensity variation (constant values)",
                "Check if your image contains actual data"
            )
        
        # Ensure C-contiguous array for performance
        if not image.flags['C_CONTIGUOUS']:
            image = np.ascontiguousarray(image)
        
        return image
    
    @staticmethod
    def validate_dx(dx: float) -> float:
        """
        Validate pixel size parameter with numerical stability checks.
        
        Args:
            dx: Pixel size in specified units
            
        Returns:
            Validated dx value
            
        Raises:
            ValidationError: If dx is invalid
        """
        if not isinstance(dx, (int, float)):
            raise ValidationError(
                "dx", type(dx).__name__,
                "must be a number",
                "Provide a numeric value (int or float)"
            )
        
        if np.isnan(dx) or np.isinf(dx):
            raise ValidationError(
                "dx", dx,
                "must be finite (not NaN or Inf)",
                "Check your input data for invalid values"
            )
        
        if dx <= 0:
            raise ValidationError(
                "dx", dx,
                "must be positive",
                "Provide a positive value greater than 0"
            )
        
        # Add reasonable bounds for numerical stability
        if dx < 1e-6:
            raise ValidationError(
                "dx", dx,
                "too small for numerical stability",
                "Use dx >= 1e-6 to avoid precision issues"
            )
        
        if dx > 100:
            logging.warning(
                f"Large pixel size detected: {dx}. Typical values are 0.1-5.0 Å. "
                f"Are you sure this is correct?"
            )
        
        return float(dx)
    
    @staticmethod
    def validate_elements(elements: Optional[List[str]], required_length: Optional[int] = None) -> List[str]:
        """
        Validate elements list.
        
        Args:
            elements: List of element symbols
            required_length: Expected length if known
            
        Returns:
            Validated elements list
            
        Raises:
            ValueError: If elements list is invalid
        """
        if elements is None:
            return ["A", "B", "C"]  # Default elements
        
        if not isinstance(elements, list):
            raise ValueError("Elements must be a list")
        
        if len(elements) == 0:
            raise ValueError("Elements list cannot be empty")
        
        if required_length is not None and len(elements) != required_length:
            raise ValueError(f"Expected {required_length} elements, got {len(elements)}")
        
        # Validate each element string
        for i, element in enumerate(elements):
            if not isinstance(element, str):
                raise ValueError(f"Element at index {i} must be a string, got {type(element)}")
            
            if len(element) == 0:
                raise ValueError(f"Element at index {i} cannot be empty string")
        
        return elements
    
    @staticmethod
    def validate_model_type(model_type: str) -> str:
        """
        Validate model type parameter.
        
        Args:
            model_type: Model type string
            
        Returns:
            Validated model type
            
        Raises:
            ValueError: If model type is invalid
        """
        if not isinstance(model_type, str):
            raise ValueError("Model type must be a string")
        
        valid_types = {"gaussian", "lorentzian", "voigt"}
        model_type_lower = model_type.lower()
        
        if model_type_lower not in valid_types:
            raise ValueError(f"Invalid model type: {model_type}. Valid types are: {valid_types}")
        
        return model_type_lower
    
    @staticmethod
    def validate_coordinates(coordinates: np.ndarray, image_shape: Tuple[int, int], 
                           strict_bounds: bool = True, epsilon: float = 1e-10) -> np.ndarray:
        """
        Validate atomic coordinate array with comprehensive edge case handling.
        
        Args:
            coordinates: Array of (x, y) coordinates
            image_shape: Shape of the image for boundary checking
            strict_bounds: Whether to enforce strict boundary checking
            epsilon: Tolerance for floating point comparisons
            
        Returns:
            Validated coordinates array (C-contiguous, clipped if needed)
            
        Raises:
            ValidationError: If coordinates are invalid
        """
        if not isinstance(coordinates, np.ndarray):
            raise ValidationError(
                "coordinates", type(coordinates).__name__,
                "must be a numpy array",
                "Convert your data to numpy.ndarray using np.array()"
            )
        
        if coordinates.size == 0:
            logging.warning("No coordinates provided")
            return coordinates
        
        if coordinates.ndim != 2:
            raise ValidationError(
                "coordinates_shape", coordinates.shape,
                f"must be 2D array, got {coordinates.ndim}D",
                f"Expected shape (N, 2), got {coordinates.shape}"
            )
        
        if coordinates.shape[1] != 2:
            raise ValidationError(
                "coordinates_columns", coordinates.shape[1],
                "must have exactly 2 columns (x, y)",
                f"Reshape your array to (N, 2) where N is the number of atoms"
            )
        
        # Check for valid coordinate values
        if np.any(np.isnan(coordinates)):
            raise ValidationError(
                "coordinates_data", "NaN detected",
                "contains NaN values",
                "Clean your data using np.nan_to_num() or remove NaN coordinates"
            )
        
        if np.any(np.isinf(coordinates)):
            raise ValidationError(
                "coordinates_data", "Inf detected",
                "contains infinite values",
                "Clean your data using np.isfinite() mask"
            )
        
        # Check for duplicate coordinates
        unique_coords = np.unique(coordinates, axis=0)
        if len(unique_coords) < len(coordinates):
            num_duplicates = len(coordinates) - len(unique_coords)
            raise ValidationError(
                "coordinates_duplicates", f"{num_duplicates} duplicates",
                "contains duplicate coordinates",
                "Remove duplicate coordinates using np.unique(..., axis=0)"
            )
        
        # Check bounds with epsilon tolerance
        height, width = image_shape
        x_coords = coordinates[:, 0]
        y_coords = coordinates[:, 1]
        
        out_of_bounds_x = (x_coords < -epsilon) | (x_coords >= width + epsilon)
        out_of_bounds_y = (y_coords < -epsilon) | (y_coords >= height + epsilon)
        
        if np.any(out_of_bounds_x) or np.any(out_of_bounds_y):
            invalid_coords = coordinates[out_of_bounds_x | out_of_bounds_y]
            if strict_bounds:
                raise ValidationError(
                    "coordinates_bounds", invalid_coords.tolist(),
                    f"coordinates outside image bounds {image_shape}",
                    "Use strict_bounds=False to clip coordinates to bounds"
                )
            else:
                # Clip coordinates to bounds
                coordinates = coordinates.copy()
                coordinates[:, 0] = np.clip(x_coords, 0, width - 1)
                coordinates[:, 1] = np.clip(y_coords, 0, height - 1)
                logging.warning(
                    f"Clipped {len(invalid_coords)} coordinates to image bounds"
                )
        
        # Check array memory layout
        if not coordinates.flags['C_CONTIGUOUS']:
            coordinates = np.ascontiguousarray(coordinates)
        
        # Check for reasonable coordinate density
        if len(coordinates) > 1:
            distances = np.sqrt(np.sum(np.diff(coordinates, axis=0)**2, axis=1))
            min_distance = np.min(distances)
            if min_distance < 1.0:
                logging.warning(
                    f"Minimum distance between atoms is {min_distance:.2f} pixels. "
                    f"This may cause fitting issues."
                )
        
        return coordinates
    
    @staticmethod
    def validate_atom_types(atom_types: np.ndarray, num_coordinates: int, num_elements: int) -> np.ndarray:
        """
        Validate atom types array.
        
        Args:
            atom_types: Array of atom type indices
            num_coordinates: Expected number of coordinates
            num_elements: Number of available element types
            
        Returns:
            Validated atom types array
            
        Raises:
            ValueError: If atom types are invalid
        """
        if not isinstance(atom_types, np.ndarray):
            raise ValueError("Atom types must be a numpy array")
        
        if len(atom_types) != num_coordinates:
            raise ValueError(f"Atom types length ({len(atom_types)}) must match coordinates ({num_coordinates})")
        
        if not np.issubdtype(atom_types.dtype, np.integer):
            raise ValueError("Atom types must be integers")
        
        if np.any(atom_types < 0):
            raise ValueError("Atom types must be non-negative")
        
        if np.any(atom_types >= num_elements):
            raise ValueError(f"Atom type indices must be less than number of elements ({num_elements})")
        
        return atom_types


class FittingParameterValidator:
    """Validation for fitting-specific parameters."""
    
    @staticmethod
    def validate_fitting_params(maxiter: int, tol: float, step_size: float) -> Tuple[int, float, float]:
        """
        Validate parameters for fitting methods.
        
        Args:
            maxiter: Maximum iterations
            tol: Convergence tolerance
            step_size: Optimization step size
            
        Returns:
            Validated parameters tuple
            
        Raises:
            ValueError: If parameters are invalid
        """
        # Validate maxiter
        if not isinstance(maxiter, int):
            raise ValueError("maxiter must be an integer")
        
        if maxiter <= 0:
            raise ValueError("maxiter must be positive")
        
        if maxiter > 100000:
            logging.warning(f"Very large maxiter: {maxiter}. This may take a long time.")
        
        # Validate tolerance
        if not isinstance(tol, (int, float)):
            raise ValueError("Tolerance must be a number")
        
        if tol <= 0:
            raise ValueError("Tolerance must be positive")
        
        if tol > 0.1:
            logging.warning(f"Large tolerance: {tol}. Convergence may be too loose.")
        
        # Validate step size
        if not isinstance(step_size, (int, float)):
            raise ValueError("Step size must be a number")
        
        if step_size <= 0:
            raise ValueError("Step size must be positive")
        
        if step_size > 1.0:
            logging.warning(f"Large step size: {step_size}. Optimization may be unstable.")
        
        return maxiter, float(tol), float(step_size)
    
    @staticmethod
    def validate_batch_size(batch_size: int, total_coordinates: int) -> int:
        """
        Validate batch size for stochastic fitting.
        
        Args:
            batch_size: Size of batches for processing
            total_coordinates: Total number of coordinates
            
        Returns:
            Validated batch size
            
        Raises:
            ValueError: If batch size is invalid
        """
        if not isinstance(batch_size, int):
            raise ValueError("Batch size must be an integer")
        
        if batch_size <= 0:
            raise ValueError("Batch size must be positive")
        
        if batch_size > total_coordinates:
            logging.warning(f"Batch size ({batch_size}) larger than total coordinates ({total_coordinates}). Using full batch.")
            return total_coordinates
        
        if batch_size > 10000:
            logging.warning(f"Large batch size: {batch_size}. This may cause memory issues.")
        
        return batch_size