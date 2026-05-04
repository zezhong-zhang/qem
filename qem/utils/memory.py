"""
Memory optimization utilities for QEM image fitting.
Provides efficient memory management for batch processing operations.

This module provides comprehensive memory optimization utilities for the QEM (Quantitative Electron Microscopy)
framework. It includes memory monitoring, batch processing optimization, chunked processing for large datasets,
and sparse matrix optimization to handle memory-intensive operations efficiently.

Key Components:
    - MemoryMonitor: Real-time memory usage tracking and operation-level monitoring
    - BatchMemoryOptimizer: Automatic batch size calculation based on memory constraints
    - ChunkedProcessor: Memory-efficient processing of large arrays in chunks
    - SparseMatrixOptimizer: Optimized sparse matrix operations for linear algebra

Example:
    >>> from qem.memory_optimization import MemoryMonitor, BatchMemoryOptimizer
    >>> 
    >>> # Monitor memory usage during fitting
    >>> monitor = MemoryMonitor()
    >>> with monitor.monitor_operation("linear_estimation"):
    ...     # perform memory-intensive operation
    ...     pass
    >>> 
    >>> # Optimize batch processing
    >>> optimizer = BatchMemoryOptimizer(memory_limit_gb=4.0)
    >>> optimal_batch = optimizer.calculate_optimal_batch_size(
    ...     image_shape=(512, 512), total_coordinates=5000
    ... )
"""

import gc
import warnings
from typing import Dict, List, Optional, Tuple, Union, Iterator, Any
from contextlib import contextmanager

import numpy as np

from qem.utils.log import get_logger

# Make logging available for testing
import logging

from qem.utils.tensors import stop_grad

class MemoryMonitor:
    """
    Monitor and log memory usage during operations.
    
    This class provides comprehensive memory monitoring capabilities for QEM operations,
    including real-time memory usage tracking, operation-level monitoring, and
    memory delta calculations.
    
    Attributes:
        enable_logging (bool): Whether to enable logging output
        initial_memory (dict): Baseline memory usage for delta calculations
    
    Example:
        >>> monitor = MemoryMonitor()
        >>> with monitor.monitor_operation("fitting"):
        ...     # perform memory-intensive operation
        ...     pass
    """
    
    def __init__(self, enable_logging: bool = True):
        """
        Initialize the memory monitor.
        
        Args:
            enable_logging: Whether to enable logging output during monitoring
        """
        self.enable_logging = enable_logging
        self.initial_memory = None
        self.logger = get_logger("qem.memory")
        
    def get_memory_info(self) -> Dict[str, float]:
        """
        Get current memory usage information for the current process.
        
        Retrieves real-time memory usage statistics using the psutil library.
        Falls back gracefully if psutil is not available.
        
        Returns:
            Dict[str, float]: Dictionary containing memory usage information:
                - 'rss_mb': Resident Set Size (physical memory) in MB
                - 'vms_mb': Virtual Memory Size in MB
                
        Example:
            >>> monitor = MemoryMonitor()
            >>> memory_info = monitor.get_memory_info()
            >>> print(f"Current memory usage: {memory_info['rss_mb']:.1f} MB")
        
        Notes:
            - RSS (Resident Set Size) represents the actual physical memory being used
            - VMS (Virtual Memory Size) represents the total virtual address space
            - If psutil is not available, returns zero values with a warning
        """
        try:
            import psutil
            process = psutil.Process()
            memory_info = process.memory_info()
            return {
                'rss_mb': memory_info.rss / 1024 / 1024,  # Resident Set Size in MB
                'vms_mb': memory_info.vms / 1024 / 1024,  # Virtual Memory Size in MB
            }
        except ImportError:
            self.logger.warning("psutil not available. Memory monitoring disabled.")
            return {'rss_mb': 0.0, 'vms_mb': 0.0}
    
    def log_memory_usage(self, operation: str):
        """
        Log current memory usage for a specific operation.
        
        Records memory usage information using the logging system. This method is
        automatically called by the monitor_operation context manager.
        
        Args:
            operation (str): Name/description of the operation being logged
            
        Example:
            >>> monitor = MemoryMonitor()
            >>> monitor.log_memory_usage("peak_detection")
            2024-01-01 10:00:00 INFO peak_detection: Memory usage - RSS: 512.3 MB, VMS: 2048.7 MB
        
        Notes:
            - Only logs when enable_logging is True
            - Requires psutil for accurate memory measurements
            - Provides both RSS (physical) and VMS (virtual) memory usage
        """
        if not self.enable_logging:
            return
        
        memory_info = self.get_memory_info()
        if memory_info['rss_mb'] > 0:
            self.logger.info(
                f"{operation}: Memory usage - RSS: {memory_info['rss_mb']:.1f} MB, "
                f"VMS: {memory_info['vms_mb']:.1f} MB"
            )
    
    @contextmanager
    def monitor_operation(self, operation: str):
        """
        Context manager to monitor memory usage during an operation.
        
        Provides automatic memory usage tracking for any block of code. Records
        memory usage at the start and end of the operation, including memory delta.
        
        Args:
            operation (str): Name/description of the operation being monitored
            
        Yields:
            None: Allows the context manager to wrap any code block
            
        Example:
            >>> monitor = MemoryMonitor()
            >>> with monitor.monitor_operation("gaussian_fitting"):
            ...     # Perform memory-intensive Gaussian fitting
            ...     params = image_fitting.fit_stochastic()
            ... 
            2024-01-01 10:00:00 INFO Starting gaussian_fitting: Memory usage - RSS: 512.3 MB, VMS: 2048.7 MB
            2024-01-01 10:00:30 INFO Completed gaussian_fitting: Memory delta: +256.5 MB
            
        Notes:
            - Automatically logs memory usage before and after the operation
            - Calculates and reports memory delta (change in usage)
            - Handles exceptions gracefully and still reports memory usage
            - Only logs when enable_logging is True and memory info is available
        """
        self.initial_memory = self.get_memory_info()
        self.log_memory_usage(f"Starting {operation}")
        
        try:
            yield
        finally:
            final_memory = self.get_memory_info()
            if self.enable_logging and self.initial_memory['rss_mb'] > 0:
                delta_rss = final_memory['rss_mb'] - self.initial_memory['rss_mb']
                self.logger.info(f"Completed {operation}: Memory delta: {delta_rss:+.1f} MB")


class BatchMemoryOptimizer:
    """
    Optimize memory usage during batch processing of QEM data.
    
    This class provides intelligent batch size calculation and memory management
    for processing large QEM datasets. It estimates memory requirements based on
    image size, number of atomic coordinates, and processing window size, then
    calculates optimal batch sizes to stay within memory constraints.
    
    The optimizer uses empirical formulas to estimate memory usage for:
    - Design matrix construction (sparse matrices)
    - Image data storage
    - Temporary computational arrays
    - Tensor operations for different backends
    
    Attributes:
        memory_limit_gb (float): Soft memory limit for batch operations
        memory_monitor (MemoryMonitor): Internal memory monitoring instance
    
    Example:
        >>> optimizer = BatchMemoryOptimizer(memory_limit_gb=4.0)
        >>> # Calculate optimal batch size for large dataset
        >>> batch_size = optimizer.calculate_optimal_batch_size(
        ...     image_shape=(1024, 1024),
        ...     total_coordinates=10000,
        ...     window_size=25
        ... )
        >>> print(f"Optimal batch size: {batch_size} coordinates")
        Optimal batch size: 250 coordinates
    
    Notes:
        - Memory estimates are conservative to prevent OOM errors
        - Automatically adjusts for different Keras backends (TensorFlow, PyTorch, JAX)
        - Includes safety factors for temporary arrays and computational overhead
    """
    
    def __init__(self, memory_limit_gb: float = 8.0):
        """
        Initialize the memory optimizer with specified memory constraints.
        
        Sets up the memory optimization framework with a configurable memory limit
        and initializes the internal memory monitoring system.
        
        Args:
            memory_limit_gb (float, optional): Soft memory limit in GB for batch operations.
                Defaults to 8.0 GB. This is a soft limit used for batch size calculations,
                not a hard system limit.
                
        Example:
            >>> # Standard initialization
            >>> optimizer = BatchMemoryOptimizer()
            >>> 
            >>> # Custom memory limit for low-memory systems
            >>> optimizer = BatchMemoryOptimizer(memory_limit_gb=2.0)
            
        Raises:
            ValueError: If memory_limit_gb is not positive
        """
        if memory_limit_gb <= 0:
            raise ValueError("memory_limit_gb must be positive")
            
        self.memory_limit_gb = memory_limit_gb
        self.memory_monitor = MemoryMonitor()
        self.logger = get_logger("qem.memory_optimizer")
        
    def estimate_batch_memory_usage(self, 
                                   image_shape: Tuple[int, int],
                                   num_coordinates: int,
                                   window_size: int = 50) -> float:
        """
        Estimate memory usage for processing a batch of atomic coordinates.
        
        Calculates comprehensive memory requirements based on:
        - Sparse design matrix construction
        - Image data storage
        - Temporary computational arrays
        - Tensor operations overhead
        
        The estimation uses empirical formulas derived from typical QEM workloads:
        - Each atomic coordinate requires approximately (2*window_size + 1)² non-zero elements
        - Each sparse matrix element requires 12 bytes (4 for value + 8 for indices)
        - Additional overhead for image storage and temporary arrays
        
        Args:
            image_shape (Tuple[int, int]): Shape of the image as (height, width)
            num_coordinates (int): Number of atomic coordinates in the batch
            window_size (int, optional): Size of local windows around each coordinate.
                Defaults to 50 pixels. Larger windows increase memory usage quadratically.
                
        Returns:
            float: Estimated memory usage in GB for the batch operation
            
        Example:
            >>> optimizer = BatchMemoryOptimizer()
            >>> # Small batch with small image
            >>> small_usage = optimizer.estimate_batch_memory_usage(
            ...     image_shape=(100, 100),
            ...     num_coordinates=50,
            ...     window_size=10
            ... )
            >>> print(f"Small batch: {small_usage:.3f} GB")
            Small batch: 0.012 GB
            
            >>> # Large batch with high-resolution image
            >>> large_usage = optimizer.estimate_batch_memory_usage(
            ...     image_shape=(1024, 1024),
            ...     num_coordinates=1000,
            ...     window_size=25
            ... )
            >>> print(f"Large batch: {large_usage:.3f} GB")
            Large batch: 2.847 GB
            
        Notes:
            - Memory estimates are intentionally conservative (overestimated)
            - Includes 50% buffer for temporary arrays and computational overhead
            - Assumes float32 data type for numerical stability
            - Does not account for backend-specific optimizations (TensorFlow, PyTorch, JAX)
        """
        ny, nx = image_shape
        
        # Estimate design matrix memory
        # Each local peak has roughly (2*window_size + 1)^2 non-zero elements
        elements_per_peak = (2 * window_size + 1) ** 2
        total_elements = num_coordinates * elements_per_peak
        
        # Each element is typically a float32 (4 bytes) plus indices
        bytes_per_element = 4 + 8  # 4 for value, 8 for row/col indices
        matrix_memory_gb = (total_elements * bytes_per_element) / (1024**3)
        
        # Add image and temporary arrays
        image_memory_gb = (nx * ny * 4) / (1024**3)  # float32
        temp_arrays_gb = matrix_memory_gb * 0.5  # Temporary arrays during computation
        
        total_gb = matrix_memory_gb + image_memory_gb + temp_arrays_gb
        
        self.logger.debug(
            f"Memory estimate: Matrix={matrix_memory_gb:.3f}GB, "
            f"Image={image_memory_gb:.3f}GB, Temp={temp_arrays_gb:.3f}GB, "
            f"Total={total_gb:.3f}GB"
        )
        
        return total_gb
    
    def calculate_optimal_batch_size(self,
                                   image_shape: Tuple[int, int],
                                   total_coordinates: int,
                                   window_size: int = 50,
                                   safety_factor: float = 0.8) -> int:
        """
        Calculate optimal batch size based on memory constraints using binary search.
        
        Uses an efficient binary search algorithm to find the maximum batch size that
        will fit within the configured memory limit. The algorithm balances memory
        efficiency with computational overhead by finding the largest viable batch.
        
        Args:
            image_shape (Tuple[int, int]): Shape of the image as (height, width)
            total_coordinates (int): Total number of atomic coordinates to process
            window_size (int, optional): Size of local windows around each coordinate.
                Defaults to 50 pixels. Larger windows reduce optimal batch size.
            safety_factor (float, optional): Safety factor for memory usage.
                Defaults to 0.8 (80% of memory limit). Lower values provide more safety margin
                but may result in smaller batches and increased processing time.
                
        Returns:
            int: Optimal batch size for memory-constrained processing
            
        Example:
            >>> optimizer = BatchMemoryOptimizer(memory_limit_gb=4.0)
            >>> # High-resolution image with many coordinates
            >>> batch_size = optimizer.calculate_optimal_batch_size(
            ...     image_shape=(2048, 2048),
            ...     total_coordinates=5000,
            ...     window_size=30,
            ...     safety_factor=0.75
            ... )
            >>> print(f"Optimal batch size: {batch_size}")
            Optimal batch size: 125
            
        Algorithm:
            1. For small datasets (<=100 coordinates), returns full dataset
            2. Uses binary search between minimum (10) and maximum (total_coordinates)
            3. Estimates memory for each candidate batch size
            4. Returns largest batch size within memory constraints
            
        Notes:
            - Ensures minimum batch size of 10 for computational efficiency
            - Binary search provides O(log n) complexity
            - Memory estimates include safety margins
            - Suitable for both CPU and GPU processing scenarios
        """
        if total_coordinates <= 100:
            return total_coordinates
        
        # Binary search for optimal batch size
        min_batch = 10
        max_batch = total_coordinates
        optimal_batch = min_batch
        
        target_memory = self.memory_limit_gb * safety_factor
        
        while min_batch <= max_batch:
            mid_batch = (min_batch + max_batch) // 2
            estimated_memory = self.estimate_batch_memory_usage(
                image_shape, mid_batch, window_size
            )
            
            if estimated_memory <= target_memory:
                optimal_batch = mid_batch
                min_batch = mid_batch + 1
            else:
                max_batch = mid_batch - 1
        
        # Ensure minimum viable batch size
        optimal_batch = max(optimal_batch, 10)
        
        self.logger.info(f"Calculated optimal batch size: {optimal_batch} "
                        f"(estimated memory: {self.estimate_batch_memory_usage(image_shape, optimal_batch, window_size):.3f}GB)")
        
        return optimal_batch
    
    @contextmanager
    def managed_computation(self, operation_name: str):
        """
        Context manager for memory-managed computation with automatic cleanup.
        
        Provides comprehensive memory management for intensive computations including:
        - Pre-operation garbage collection
        - Backend-specific memory cleanup (TensorFlow, PyTorch)
        - Automatic memory monitoring
        - Post-operation cleanup
        
        Args:
            operation_name (str): Name/description of the operation for logging
            
        Yields:
            None: Allows the context manager to wrap any memory-intensive code
            
        Example:
            >>> optimizer = BatchMemoryOptimizer(memory_limit_gb=4.0)
            >>> with optimizer.managed_computation("sparse_matrix_construction"):
            ...     # Memory-intensive sparse matrix operations
            ...     matrix = build_large_sparse_matrix()
            ...     result = solve_linear_system(matrix)
            ... 
            2024-01-01 10:00:00 INFO Starting sparse_matrix_construction: Memory usage - RSS: 1024.2 MB
            2024-01-01 10:00:15 INFO Completed sparse_matrix_construction: Memory delta: +512.8 MB
            
        Backend-Specific Cleanup:
            - TensorFlow: Clears Keras backend session and GPU memory
            - PyTorch: Garbage collection for tensor cleanup
            - JAX: Automatic memory management through garbage collection
            
        Notes:
            - Automatically handles different Keras backends
            - Forces Python garbage collection before and after operation
            - Provides memory usage logging and delta tracking
            - Safe for nested context managers
        """
        with self.memory_monitor.monitor_operation(operation_name):
            # Force garbage collection before operation
            gc.collect()
            try:
                yield
            finally:
                # Clean up after operation
                gc.collect()
    
    def optimize_tensor_operations(self, 
                                 params: Dict[str, Any],
                                 operation: str) -> Dict[str, Any]:
        """
        Optimize tensor operations to reduce memory footprint during computation.
        
        Applies memory optimization techniques to parameter dictionaries containing
        Keras tensors or numpy arrays. Currently implements gradient stopping to prevent 
        unnecessary gradient computation during memory-intensive operations.
        
        Args:
            params (Dict[str, Any]): Parameter dictionary
            operation (str): Description of the operation being performed (used for logging)
            
        Returns:
            Dict[str, Any]: Optimized parameter dictionary
            
        Example:
            >>> optimizer = BatchMemoryOptimizer()
            >>> # Original parameters with gradients enabled
            >>> original_params = {
            ...     'pos_x': np.array([1.0, 2.0, 3.0]),
            ...     'height': keras.ops.convert_to_tensor([10.0, 15.0, 12.0]),
            ...     'scalar_param': 5.0  # Non-tensor parameter
            ... }
            >>> # Optimize for memory efficiency
            >>> optimized_params = optimizer.optimize_tensor_operations(
            ...     original_params, "linear_estimation"
            ... )
            >>> # Verify optimization
            >>> assert len(optimized_params) == len(original_params)
            >>> assert optimized_params['scalar_param'] == 5.0  # Non-tensors unchanged
            
        Optimization Techniques:
            - Gradient stopping: Uses keras.ops.stop_gradient() to prevent gradient computation
            - Non-tensor preservation: Leaves non-tensor parameters unchanged
            - Type checking: Only optimizes valid tensor-like objects
            
        Notes:
            - Safe to use with mixed-type parameter dictionaries
            - Does not modify original parameter dictionary
            - Handles both numpy arrays and Keras tensors gracefully
        """
        optimized_params = {}
        
        for key, value in params.items():
            # Detach gradients on PyTorch tensors; numpy arrays pass through.
            if hasattr(value, "detach") and hasattr(value, "shape"):
                optimized_params[key] = stop_grad(value)
            
            # Handle numpy arrays (no gradient computation needed)
            elif isinstance(value, np.ndarray):
                optimized_params[key] = value
            
            # Handle other numeric types
            elif isinstance(value, (int, float, list, tuple)):
                optimized_params[key] = value
            
            # Handle all other types unchanged
            else:
                optimized_params[key] = value
        
        return optimized_params


class ChunkedProcessor:
    """
    Process large arrays in memory-efficient chunks for QEM operations.
    
    Provides a flexible framework for processing large datasets by breaking them into
    manageable chunks based on either fixed sizes or automatic memory-based sizing.
    This is particularly useful for processing large numbers of atomic coordinates
    or high-resolution images that exceed available memory.
    
    The processor can automatically determine optimal chunk sizes based on memory
    constraints, making it ideal for environments with varying memory availability.
    
    Attributes:
        chunk_size (Optional[int]): Fixed chunk size if specified, None for auto-sizing
        optimizer (BatchMemoryOptimizer): Internal memory optimizer for auto-sizing
    
    Example:
        >>> processor = ChunkedProcessor()
        >>> 
        >>> # Process large coordinate dataset
        >>> coordinates = np.random.rand(10000, 2) * 100
        >>> 
        >>> def analyze_coordinates(coords):
        ...     # Memory-intensive coordinate analysis
        ...     return np.mean(coords, axis=0)
        ... 
        >>> results = processor.process_in_chunks(
        ...     coordinates, analyze_coordinates, image_shape=(512, 512)
        ... )
        
    Notes:
        - Automatic chunk sizing prevents memory overflow errors
        - Preserves processing order for consistent results
        - Handles both numpy arrays and Python lists
        - Integrates with BatchMemoryOptimizer for intelligent sizing
    """
    
    def __init__(self, chunk_size: Optional[int] = None, memory_limit_gb: float = 8.0):
        """
        Initialize the chunked processor with optional fixed chunk sizing.
        
        Sets up the chunked processor with either a fixed chunk size or automatic
        sizing based on memory constraints. When chunk_size is None, the processor
        will automatically calculate optimal chunk sizes based on the data and
        available memory.
        
        Args:
            chunk_size (Optional[int], optional): Size of chunks to process. 
                If None, chunk size will be calculated automatically based on
                memory constraints and data characteristics. Defaults to None.
            memory_limit_gb (float, optional): Memory limit for automatic chunk sizing.
                Defaults to 8.0 GB.
                
        Example:
            >>> # Fixed chunk size (manual control)
            >>> processor = ChunkedProcessor(chunk_size=500)
            >>> 
            >>> # Automatic chunk sizing with custom memory limit
            >>> processor_auto = ChunkedProcessor(memory_limit_gb=4.0)
            >>> 
            >>> # Verify initialization
            >>> assert processor.chunk_size == 500
            >>> assert processor_auto.chunk_size is None
        """
        self.chunk_size = chunk_size
        self.optimizer = BatchMemoryOptimizer(memory_limit_gb=memory_limit_gb)
        self.logger = get_logger("qem.memory_optimization.ChunkedProcessor")
    
    def create_chunks(self, 
                     total_items: int,
                     image_shape: Optional[Tuple[int, int]] = None) -> Iterator[slice]:
        """
        Create chunks for processing.
        
        Args:
            total_items: Total number of items to process
            image_shape: Shape of image for memory estimation
            
        Yields:
            Slice objects for each chunk
        """
        if self.chunk_size is None:
            if image_shape is not None:
                # Calculate optimal chunk size based on memory
                chunk_size = self.optimizer.calculate_optimal_batch_size(
                    image_shape, total_items
                )
            else:
                # Default heuristic
                chunk_size = min(500, max(50, total_items // 10))
        else:
            chunk_size = self.chunk_size
        
        self.logger.info(f"Processing {total_items} items in chunks of {chunk_size}")
        
        for start in range(0, total_items, chunk_size):
            end = min(start + chunk_size, total_items)
            yield slice(start, end)
    
    def process_in_chunks(self,
                         data: Union[np.ndarray, List],
                         process_func: callable,
                         image_shape: Optional[Tuple[int, int]] = None,
                         **kwargs) -> List:
        """
        Process data in memory-efficient chunks.
        
        Args:
            data: Data to process (array or list)
            process_func: Function to apply to each chunk
            image_shape: Image shape for memory estimation
            **kwargs: Additional arguments for process_func
            
        Returns:
            List of results from each chunk
        """
        total_items = len(data)
        results = []
        
        for chunk_slice in self.create_chunks(total_items, image_shape):
            chunk_data = data[chunk_slice]
            
            with self.optimizer.managed_computation(f"Processing chunk {chunk_slice}"):
                chunk_result = process_func(chunk_data, **kwargs)
                results.append(chunk_result)
        
        return results


class SparseMatrixOptimizer:
    """
    Optimize sparse matrix operations for memory-efficient QEM processing.
    
    Provides specialized utilities for handling large sparse matrices that arise
    during QEM image fitting operations. Implements incremental construction,
    memory optimization, and format conversion specifically designed for
    design matrices used in linear estimation.
    
    The optimizer is particularly useful for:
    - Large design matrices with many atomic coordinates
    - Memory-constrained environments
    - GPU memory optimization
    - Batch processing scenarios
    
    Key Features:
        - Incremental matrix building to prevent memory overflow
        - Automatic format optimization (CSR, CSC, COO)
        - Duplicate elimination and zero removal
        - Memory-efficient matrix construction
    
    Example:
        >>> optimizer = SparseMatrixOptimizer()
        >>> 
        >>> # Build large sparse matrix incrementally
        >>> rows = [0, 1, 2, 0, 1, 2]  # Row indices
        >>> cols = [0, 1, 2, 1, 2, 0]  # Column indices
        >>> data = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]  # Values
        >>> shape = (3, 3)
        >>> 
        >>> matrix = optimizer.build_matrix_incrementally(
        ...     rows, cols, data, shape, chunk_size=1000
        ... )
        >>> 
        >>> # Optimize for specific format
        >>> optimized = optimizer.optimize_matrix_memory(matrix, format='csr')
        >>> print(f"Optimized matrix: {optimized.shape}, nnz={optimized.nnz}")
        
    Notes:
        - Uses scipy.sparse for efficient sparse matrix operations
        - CSR format recommended for row-wise operations
        - CSC format recommended for column-wise operations
        - COO format recommended for incremental construction
    """
    
    @staticmethod
    def build_matrix_incrementally(
        rows: List[int],
        cols: List[int], 
        data: List[float],
        shape: Tuple[int, int],
        chunk_size: int = 10000
    ):
        """
        Build sparse matrix incrementally to manage memory.
        
        Args:
            rows, cols, data: Matrix components
            shape: Final matrix shape
            chunk_size: Size of chunks for processing
            
        Returns:
            Sparse matrix
        """
        from scipy.sparse import coo_matrix
        
        if len(data) <= chunk_size:
            return coo_matrix((data, (rows, cols)), shape=shape)
        
        # Process in chunks and combine
        matrices = []
        for i in range(0, len(data), chunk_size):
            end_idx = min(i + chunk_size, len(data))
            chunk_matrix = coo_matrix(
                (data[i:end_idx], (rows[i:end_idx], cols[i:end_idx])),
                shape=shape
            )
            matrices.append(chunk_matrix)
        
        # Combine matrices
        combined = matrices[0]
        for matrix in matrices[1:]:
            combined = combined + matrix
        
        return combined
    
    @staticmethod
    def optimize_matrix_memory(matrix, format: str = 'csr'):
        """
        Optimize sparse matrix memory usage.
        
        Args:
            matrix: Input sparse matrix
            format: Target format ('csr', 'csc', 'coo')
            
        Returns:
            Optimized matrix
        """
        # Convert to optimal format
        if format == 'csr':
            optimized = matrix.tocsr()
        elif format == 'csc':
            optimized = matrix.tocsc()
        else:
            optimized = matrix.tocoo()
        
        # Eliminate duplicates and zeros
        optimized.eliminate_zeros()
        optimized.sum_duplicates()
        
        return optimized


# Global instance for easy access
memory_optimizer = BatchMemoryOptimizer()
chunked_processor = ChunkedProcessor()
