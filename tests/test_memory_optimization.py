"""
Tests for the memory optimization module.
"""

import numpy as np
import pytest
from unittest.mock import patch, MagicMock

from qem.utils.memory_optimization import (
    MemoryMonitor,
    BatchMemoryOptimizer,
    ChunkedProcessor,
    SparseMatrixOptimizer,
    memory_optimizer,
    chunked_processor,
)


class TestMemoryMonitor:
    """Test memory monitoring functionality."""
    
    def test_initialization(self):
        """Test memory monitor initialization."""
        monitor = MemoryMonitor(enable_logging=True)
        assert monitor.enable_logging is True
        assert monitor.initial_memory is None
        
        monitor_disabled = MemoryMonitor(enable_logging=False)
        assert monitor_disabled.enable_logging is False
    
    def test_get_memory_info_no_psutil(self):
        """Test memory info retrieval when psutil is not available."""
        monitor = MemoryMonitor()
        
        with patch('qem.utils.memory_optimization.logging'):
            # Mock psutil import to raise ImportError
            with patch('builtins.__import__', side_effect=lambda name, *args, **kwargs: 
                      __import__(name, *args, **kwargs) if name != 'psutil' 
                      else exec('raise ImportError("No module named psutil")')):
                info = monitor.get_memory_info()
                assert info['rss_mb'] == 0.0
                assert info['vms_mb'] == 0.0
    
    def test_log_memory_usage_disabled(self):
        """Test memory logging when disabled."""
        monitor = MemoryMonitor(enable_logging=False)
        
        with patch('qem.utils.memory_optimization.logging') as mock_logging:
            monitor.log_memory_usage("test")
            mock_logging.info.assert_not_called()
    
    def test_monitor_operation_context(self):
        """Test the context manager for memory monitoring."""
        monitor = MemoryMonitor(enable_logging=False)
        
        with monitor.monitor_operation("test_operation"):
            # Just test that context manager works
            pass


class TestBatchMemoryOptimizer:
    """Test batch memory optimization functionality."""
    
    def test_initialization(self):
        """Test optimizer initialization."""
        optimizer = BatchMemoryOptimizer(memory_limit_gb=4.0)
        assert optimizer.memory_limit_gb == 4.0
        assert optimizer.memory_monitor is not None
    
    def test_estimate_batch_memory_usage(self):
        """Test memory usage estimation."""
        optimizer = BatchMemoryOptimizer()
        
        # Small image
        image_shape = (100, 100)
        num_coordinates = 50
        
        memory_gb = optimizer.estimate_batch_memory_usage(
            image_shape, num_coordinates, window_size=10
        )
        
        assert memory_gb > 0
        assert isinstance(memory_gb, float)
        assert memory_gb < 1.0  # Should be small for this test case
    
    def test_calculate_optimal_batch_size_small_dataset(self):
        """Test batch size calculation for small datasets."""
        optimizer = BatchMemoryOptimizer(memory_limit_gb=8.0)
        
        image_shape = (50, 50)
        total_coordinates = 50
        
        batch_size = optimizer.calculate_optimal_batch_size(
            image_shape, total_coordinates
        )
        
        assert batch_size == 50  # Should return all for small dataset
    
    def test_calculate_optimal_batch_size_large_dataset(self):
        """Test batch size calculation for large datasets."""
        optimizer = BatchMemoryOptimizer(memory_limit_gb=1.0)
        
        image_shape = (1000, 1000)
        total_coordinates = 10000
        
        batch_size = optimizer.calculate_optimal_batch_size(
            image_shape, total_coordinates, window_size=20
        )
        
        assert batch_size >= 10  # Should respect minimum
        assert batch_size <= 10000
    
    def test_managed_computation_context(self):
        """Test the managed computation context manager."""
        optimizer = BatchMemoryOptimizer()
        
        with optimizer.managed_computation("test_operation"):
            # Test basic functionality
            pass
    
    def test_optimize_tensor_operations(self):
        """Test tensor operation optimization."""
        optimizer = BatchMemoryOptimizer()
        
        # Create mock tensor data
        mock_tensor = MagicMock()
        mock_tensor.__class__.__name__ = 'KerasTensor'
        
        params = {
            'param1': mock_tensor,
            'param2': 42,  # Non-tensor
            'param3': np.array([1, 2, 3])
        }
        
        optimized = optimizer.optimize_tensor_operations(params, "test")
        assert len(optimized) == 3
        assert optimized['param2'] == 42


class TestChunkedProcessor:
    """Test chunked processing functionality."""
    
    def test_initialization(self):
        """Test processor initialization."""
        processor = ChunkedProcessor(chunk_size=100)
        assert processor.chunk_size == 100
        
        processor_auto = ChunkedProcessor()
        assert processor_auto.chunk_size is None
    
    def test_create_chunks_fixed_size(self):
        """Test chunk creation with fixed size."""
        processor = ChunkedProcessor(chunk_size=25)
        
        chunks = list(processor.create_chunks(100))
        assert len(chunks) == 4
        assert chunks[0] == slice(0, 25)
        assert chunks[1] == slice(25, 50)
        assert chunks[2] == slice(50, 75)
        assert chunks[3] == slice(75, 100)
    
    def test_create_chunks_auto_size(self):
        """Test chunk creation with automatic sizing."""
        processor = ChunkedProcessor()
        
        chunks = list(processor.create_chunks(100, image_shape=(100, 100)))
        
        # Should create reasonable chunks
        assert len(chunks) > 0
        assert all(isinstance(chunk, slice) for chunk in chunks)
    
    def test_create_chunks_small_dataset(self):
        """Test chunk creation for small datasets."""
        processor = ChunkedProcessor()
        
        chunks = list(processor.create_chunks(5))
        assert len(chunks) == 1
        assert chunks[0] == slice(0, 5)
    
    def test_process_in_chunks(self):
        """Test chunked processing functionality."""
        processor = ChunkedProcessor(chunk_size=10)
        
        data = np.arange(50)
        
        def mock_process(chunk):
            return chunk.sum()
        
        results = processor.process_in_chunks(data, mock_process)
        
        assert len(results) == 5  # 50 items / 10 per chunk
        assert all(isinstance(result, (int, np.integer)) for result in results)
        assert sum(results) == data.sum()


class TestSparseMatrixOptimizer:
    """Test sparse matrix optimization functionality."""
    
    def test_build_matrix_incrementally_small(self):
        """Test incremental matrix building for small matrices."""
        optimizer = SparseMatrixOptimizer()
        
        rows = [0, 1, 2]
        cols = [0, 1, 2]
        data = [1.0, 2.0, 3.0]
        shape = (3, 3)
        
        matrix = optimizer.build_matrix_incrementally(rows, cols, data, shape)
        
        assert matrix.shape == shape
        assert matrix.nnz == 3
        np.testing.assert_array_equal(matrix.toarray().diagonal(), [1, 2, 3])
    
    def test_build_matrix_incrementally_large(self):
        """Test incremental matrix building for large matrices."""
        optimizer = SparseMatrixOptimizer()
        
        # Create larger test data
        n = 1000
        rows = list(range(n))
        cols = list(range(n))
        data = [1.0] * n
        shape = (n, n)
        
        matrix = optimizer.build_matrix_incrementally(
            rows, cols, data, shape, chunk_size=100
        )
        
        assert matrix.shape == shape
        assert matrix.nnz == n
    
    def test_optimize_matrix_memory(self):
        """Test matrix memory optimization."""
        optimizer = SparseMatrixOptimizer()
        
        # Create test matrix
        from scipy.sparse import coo_matrix
        rows = [0, 1, 2, 2]
        cols = [0, 1, 2, 2]  # Duplicate for testing
        data = [1.0, 2.0, 3.0, 0.0]  # Include zero for testing
        shape = (3, 3)
        
        matrix = coo_matrix((data, (rows, cols)), shape=shape)
        
        # Test CSR format
        optimized_csr = optimizer.optimize_matrix_memory(matrix, format='csr')
        assert optimized_csr.format == 'csr'
        assert optimized_csr.nnz <= 3  # Should eliminate zero
        
        # Test CSC format
        optimized_csc = optimizer.optimize_matrix_memory(matrix, format='csc')
        assert optimized_csc.format == 'csc'
        
        # Test COO format
        optimized_coo = optimizer.optimize_matrix_memory(matrix, format='coo')
        assert optimized_coo.format == 'coo'


class TestGlobalInstances:
    """Test global instances for easy access."""
    
    def test_memory_optimizer_instance(self):
        """Test the global memory optimizer instance."""
        assert memory_optimizer is not None
        assert isinstance(memory_optimizer, BatchMemoryOptimizer)
        assert memory_optimizer.memory_limit_gb == 8.0
    
    def test_chunked_processor_instance(self):
        """Test the global chunked processor instance."""
        assert chunked_processor is not None
        assert isinstance(chunked_processor, ChunkedProcessor)


class TestIntegration:
    """Integration tests combining multiple components."""
    
    def test_full_memory_optimization_workflow(self):
        """Test complete memory optimization workflow."""
        optimizer = BatchMemoryOptimizer(memory_limit_gb=0.1)
        processor = ChunkedProcessor(memory_limit_gb=0.1)
        
        # Simulate a large dataset
        image_shape = (500, 500)
        total_coordinates = 5000
        
        # Calculate optimal batch size
        batch_size = optimizer.calculate_optimal_batch_size(
            image_shape, total_coordinates
        )
        
        assert batch_size <= total_coordinates
        assert batch_size >= 10
        
        # Test processing in chunks
        data = np.arange(total_coordinates)
        
        def process_chunk(chunk):
            return chunk ** 2
        
        results = processor.process_in_chunks(
            data, process_chunk, image_shape=image_shape
        )
        
        assert len(results) > 1  # Should be chunked
        assert sum(len(r) if hasattr(r, '__len__') else 1 for r in results) == total_coordinates