"""
Comprehensive logging configuration for QEM with structured output and debugging support.

This module provides:
- Structured logging with JSON format support
- Log level management for different components
- Performance timing utilities
- Debug information capture
- User-friendly log formatting
"""

import json
import logging
import logging.config
import os
import time
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Union


class StructuredFormatter(logging.Formatter):
    """JSON-formatted logging for structured output."""
    
    def format(self, record):
        log_entry = {
            'timestamp': datetime.fromtimestamp(record.created).isoformat(),
            'level': record.levelname,
            'logger': record.name,
            'message': record.getMessage(),
            'module': record.module,
            'function': record.funcName,
            'line': record.lineno,
        }
        
        # Add extra fields
        if hasattr(record, 'component'):
            log_entry['component'] = record.component
        if hasattr(record, 'operation'):
            log_entry['operation'] = record.operation
        if hasattr(record, 'duration'):
            log_entry['duration_ms'] = record.duration
        if hasattr(record, 'memory_usage'):
            log_entry['memory_mb'] = record.memory_usage
            
        # Add exception info if present
        if record.exc_info:
            log_entry['exception'] = self.formatException(record.exc_info)
            
        return json.dumps(log_entry)


class UserFriendlyFormatter(logging.Formatter):
    """User-friendly log formatting with colors and clear messages."""
    
    COLORS = {
        'DEBUG': '\033[36m',    # Cyan
        'INFO': '\033[32m',     # Green
        'WARNING': '\033[33m',  # Yellow
        'ERROR': '\033[31m',    # Red
        'CRITICAL': '\033[35m', # Magenta
        'RESET': '\033[0m',
    }
    
    def format(self, record):
        if not record.args:
            record.msg = str(record.msg)
            
        # Add emoji based on level
        emoji_map = {
            'DEBUG': '🐛',
            'INFO': 'ℹ️',
            'WARNING': '⚠️',
            'ERROR': '❌',
            'CRITICAL': '🚨',
        }
        
        level = record.levelname
        emoji = emoji_map.get(level, '📝')
        
        # Color the level name
        color = self.COLORS.get(level, '')
        reset = self.COLORS['RESET']
        
        formatted_level = f"{color}[{level}]{reset}"
        
        # Format the message
        if hasattr(record, 'component'):
            prefix = f"{emoji} {formatted_level} [{record.component}]"
        else:
            prefix = f"{emoji} {formatted_level}"
            
        message = super().format(record)
        return f"{prefix} {message}"


class QEMLogger:
    """Enhanced logger with performance tracking and user-friendly output."""
    
    def __init__(self, name: str):
        self.logger = logging.getLogger(name)
        self.name = name
        
    def debug_operation(self, operation: str, **kwargs):
        """Log debug information for operations."""
        self.logger.debug(
            f"{operation}",
            extra={'operation': operation, 'component': self.name, **kwargs}
        )
    
    def info_operation(self, operation: str, details: Optional[str] = None, **kwargs):
        """Log informational messages with context."""
        message = f"{operation}"
        if details:
            message += f" - {details}"
            
        self.logger.info(
            message,
            extra={'operation': operation, 'component': self.name, **kwargs}
        )
    
    def warning_operation(self, operation: str, warning: str, **kwargs):
        """Log warnings with operation context."""
        self.logger.warning(
            f"{operation}: {warning}",
            extra={'operation': operation, 'component': self.name, **kwargs}
        )
    
    def error_operation(self, operation: str, error: str, **kwargs):
        """Log errors with operation context."""
        self.logger.error(
            f"{operation} failed: {error}",
            extra={'operation': operation, 'component': self.name, **kwargs}
        )

    def debug(self, message: str, **kwargs):
        """Direct debug logging."""
        self.logger.debug(message, extra={'component': self.name, **kwargs})

    def info(self, message: str, **kwargs):
        """Direct info logging."""
        self.logger.info(message, extra={'component': self.name, **kwargs})

    def warning(self, message: str, **kwargs):
        """Direct warning logging."""
        self.logger.warning(message, extra={'component': self.name, **kwargs})

    def error(self, message: str, **kwargs):
        """Direct error logging."""
        self.logger.error(message, extra={'component': self.name, **kwargs})

    def critical(self, message: str, **kwargs):
        """Direct critical logging."""
        self.logger.critical(message, extra={'component': self.name, **kwargs})


class PerformanceTracker:
    """Utility for tracking operation performance and memory usage."""
    
    def __init__(self, logger: QEMLogger):
        self.logger = logger
        
    @contextmanager
    def track_operation(self, operation_name: str, track_memory: bool = True):
        """Context manager for tracking operation performance."""
        start_time = time.time()
        start_memory = self._get_memory_usage() if track_memory else None
        
        self.logger.info_operation(f"Starting {operation_name}")
        
        try:
            yield
            
            duration = (time.time() - start_time) * 1000  # Convert to ms
            
            extra = {'duration': duration}
            if track_memory and start_memory is not None:
                end_memory = self._get_memory_usage()
                if end_memory:
                    memory_diff = end_memory - start_memory
                    extra['memory_usage'] = memory_diff
                    
            self.logger.info_operation(
                f"Completed {operation_name}",
                f"took {duration:.2f}ms",
                **extra
            )
            
        except Exception as e:
            duration = (time.time() - start_time) * 1000
            self.logger.error_operation(
                operation_name,
                str(e),
                duration=duration
            )
            raise
    
    def _get_memory_usage(self) -> Optional[float]:
        """Get current memory usage in MB."""
        try:
            import psutil
            process = psutil.Process()
            return process.memory_info().rss / 1024 / 1024
        except ImportError:
            return None


def setup_logging(
    level: Union[str, int] = "INFO",
    log_file: Optional[str] = None,
    structured: bool = False,
    user_friendly: bool = True,
    component_level: Optional[Dict[str, str]] = None
) -> None:
    """
    Configure logging for QEM.
    
    Args:
        level: Overall log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Path to log file (optional)
        structured: Use JSON structured logging
        user_friendly: Use user-friendly formatting with colors
        component_level: Dict mapping component names to log levels
    """
    
    # Prevent duplicate configuration
    if logging.getLogger().handlers and not log_file:
        return
        
    # Convert string level to int
    if isinstance(level, str):
        level = getattr(logging, level.upper())
    
    # Base configuration
    config = {
        'version': 1,
        'disable_existing_loggers': False,
        'formatters': {
            'standard': {
                'format': '%(asctime)s [%(levelname)s] %(name)s: %(message)s',
                'datefmt': '%Y-%m-%d %H:%M:%S'
            },
            'structured': {
                '()': StructuredFormatter
            },
            'user_friendly': {
                '()': UserFriendlyFormatter,
                'format': '%(message)s'
            }
        },
        'handlers': {
            'console': {
                'class': 'logging.StreamHandler',
                'level': level,
                'formatter': 'structured' if structured else ('user_friendly' if user_friendly else 'standard'),
                'stream': 'ext://sys.stdout'
            }
        },
        'loggers': {
            'qem': {
                'level': level,
                'handlers': ['console'],
                'propagate': False
            }
        }
    }
    
    # Add file handler if specified
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        config['handlers']['file'] = {
            'class': 'logging.handlers.RotatingFileHandler',
            'level': level,
            'formatter': 'structured',
            'filename': str(log_path),
            'maxBytes': 10 * 1024 * 1024,  # 10MB
            'backupCount': 5
        }
        config['loggers']['qem']['handlers'].append('file')
    
    # Set component-specific levels
    if component_level:
        for component, comp_level in component_level.items():
            config['loggers'][f'qem.{component}'] = {
                'level': getattr(logging, comp_level.upper()),
                'handlers': ['console'] + (['file'] if log_file else []),
                'propagate': False
            }
    
    logging.config.dictConfig(config)


def get_logger(name: str) -> QEMLogger:
    """Get an enhanced logger instance."""
    return QEMLogger(name)


# Quick setup functions
def setup_debug_logging():
    """Setup debugging-friendly logging."""
    setup_logging("DEBUG", user_friendly=True, structured=False)


def setup_performance_logging(log_file: str = "qem_performance.log"):
    """Setup performance-focused logging."""
    setup_logging(
        "INFO",
        log_file=log_file,
        structured=True,
        component_level={
            'memory_optimization': 'DEBUG',
            'image_fitting': 'DEBUG'
        }
    )


# Context manager for temporary log level changes
@contextmanager
def temporary_log_level(logger_name: str, level: str):
    """Temporarily change log level for debugging."""
    logger = logging.getLogger(logger_name)
    old_level = logger.level
    logger.setLevel(getattr(logging, level.upper()))
    try:
        yield
    finally:
        logger.setLevel(old_level)