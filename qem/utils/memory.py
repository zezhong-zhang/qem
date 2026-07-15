"""Memory monitoring utilities for QEM.

Thin wrapper around psutil for measuring per-operation memory deltas.
The historical batch-/chunked-/sparse- "optimiser" classes that lived
here had zero call sites and were deleted (Linus review #1).
"""

from contextlib import contextmanager

from qem.utils.log import get_logger


class MemoryMonitor:
    """Monitor and log memory usage during operations."""

    def __init__(self, enable_logging: bool = True):
        self.enable_logging = enable_logging
        self.initial_memory = None
        self.logger = get_logger("qem.memory")

    def get_memory_info(self) -> dict[str, float]:
        """Return current process RSS / VMS in MB. Zero values if psutil missing."""
        try:
            import psutil
            memory_info = psutil.Process().memory_info()
            return {
                "rss_mb": memory_info.rss / 1024 / 1024,
                "vms_mb": memory_info.vms / 1024 / 1024,
            }
        except ImportError:
            self.logger.warning("psutil not available. Memory monitoring disabled.")
            return {"rss_mb": 0.0, "vms_mb": 0.0}

    def log_memory_usage(self, operation: str) -> None:
        if not self.enable_logging:
            return
        memory_info = self.get_memory_info()
        if memory_info["rss_mb"] > 0:
            self.logger.info(
                f"{operation}: Memory usage - RSS: {memory_info['rss_mb']:.1f} MB, "
                f"VMS: {memory_info['vms_mb']:.1f} MB"
            )

    @contextmanager
    def monitor_operation(self, operation: str):
        """Context manager — log memory at entry, exit, and the delta."""
        self.initial_memory = self.get_memory_info()
        self.log_memory_usage(f"Starting {operation}")
        try:
            yield
        finally:
            final_memory = self.get_memory_info()
            if self.enable_logging and self.initial_memory["rss_mb"] > 0:
                delta_rss = final_memory["rss_mb"] - self.initial_memory["rss_mb"]
                self.logger.info(f"Completed {operation}: Memory delta: {delta_rss:+.1f} MB")
