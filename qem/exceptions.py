"""
Enhanced exception handling for QEM with user-friendly error messages.

This module provides comprehensive error handling utilities that offer:
- User-friendly error messages with actionable guidance
- Detailed technical context for debugging
- Hierarchical exception classes for better error categorization
- Context preservation for complex operations
"""

import inspect
import logging
import traceback
from typing import Any, Dict, List, Optional, Union


class QEMError(Exception):
    """
    Base exception class for QEM with enhanced error reporting.
    
    Provides user-friendly messages with actionable guidance while
    preserving technical details for debugging.
    """
    
    def __init__(
        self,
        message: str,
        user_message: Optional[str] = None,
        technical_details: Optional[Dict[str, Any]] = None,
        suggestion: Optional[str] = None,
        code_context: Optional[Dict[str, Any]] = None
    ):
        super().__init__(message)
        self.message = message
        self.user_message = user_message or message
        self.technical_details = technical_details or {}
        self.suggestion = suggestion
        self.code_context = code_context or self._get_context()
        
    def _get_context(self) -> Dict[str, Any]:
        """Extract code context for debugging."""
        try:
            frame = inspect.currentframe().f_back.f_back
            return {
                'filename': frame.f_code.co_filename,
                'line_number': frame.f_lineno,
                'function_name': frame.f_code.co_name,
                'locals': {k: str(v)[:200] for k, v in frame.f_locals.items()}
            }
        except Exception:
            return {}
    
    def __str__(self) -> str:
        return self.format_error()
    
    def format_error(self, verbose: bool = False) -> str:
        """Format error message for display."""
        lines = [f"QEM Error: {self.user_message}"]
        
        if self.suggestion:
            lines.append(f"💡 Suggestion: {self.suggestion}")
            
        if verbose and self.technical_details:
            lines.append("\nTechnical Details:")
            for key, value in self.technical_details.items():
                lines.append(f"  {key}: {value}")
                
        if verbose and self.code_context:
            lines.append(f"\nLocation: {self.code_context.get('filename', 'Unknown')}")
            lines.append(f"Line: {self.code_context.get('line_number', 'Unknown')}")
            
        return "\n".join(lines)


class ParameterError(QEMError):
    """Exception for parameter validation errors."""
    
    def __init__(self, message: str, param_name: Optional[str] = None, **kwargs):
        user_msg = f"Invalid parameter: {message}"
        if param_name:
            user_msg = f"Invalid parameter '{param_name}': {message}"
            
        suggestion = kwargs.pop('suggestion', None)
        if not suggestion:
            suggestion = "Please check the parameter documentation and ensure correct types/values."
            
        super().__init__(
            message=message,
            user_message=user_msg,
            suggestion=suggestion,
            **kwargs
        )


class DataError(QEMError):
    """Exception for data-related errors."""
    
    def __init__(self, message: str, data_shape: Optional[tuple] = None, **kwargs):
        user_msg = f"Data issue: {message}"
        suggestion = kwargs.pop('suggestion', None)
        if not suggestion:
            suggestion = "Please check your input data format, dimensions, and ensure no NaN/inf values."
            
        tech_details = kwargs.pop('technical_details', {})
        if data_shape:
            tech_details['data_shape'] = data_shape
            
        super().__init__(
            message=message,
            user_message=user_msg,
            suggestion=suggestion,
            technical_details=tech_details,
            **kwargs
        )


class MemoryError(QEMError):
    """Exception for memory-related errors."""
    
    def __init__(self, message: str, memory_usage: Optional[float] = None, **kwargs):
        user_msg = f"Memory issue: {message}"
        suggestion = kwargs.pop('suggestion', None)
        if not suggestion:
            suggestion = (
                "Try reducing batch size, using chunked processing, or increasing system memory. "
                "Consider using the memory optimization utilities."
            )
            
        tech_details = kwargs.pop('technical_details', {})
        if memory_usage:
            tech_details['memory_usage_gb'] = f"{memory_usage:.2f}"
            
        super().__init__(
            message=message,
            user_message=user_msg,
            suggestion=suggestion,
            technical_details=tech_details,
            **kwargs
        )


class BackendError(QEMError):
    """Exception for backend-related errors."""
    
    def __init__(self, message: str, backend: Optional[str] = None, **kwargs):
        user_msg = f"Backend issue: {message}"
        suggestion = kwargs.pop('suggestion', None)
        if not suggestion:
            suggestion = "Try switching to a different backend (numpy, torch, jax) or check your installation."
            
        tech_details = kwargs.pop('technical_details', {})
        if backend:
            tech_details['backend'] = backend
            
        super().__init__(
            message=message,
            user_message=user_msg,
            suggestion=suggestion,
            technical_details=tech_details,
            **kwargs
        )


class ValidationError(QEMError):
    """Exception for validation errors."""
    
    def __init__(self, message: str, validation_rules: Optional[List[str]] = None, **kwargs):
        user_msg = f"Validation failed: {message}"
        suggestion = kwargs.pop('suggestion', None)
        if not suggestion:
            suggestion = "Please ensure all input parameters meet the required validation criteria."
            
        tech_details = kwargs.pop('technical_details', {})
        if validation_rules:
            tech_details['validation_rules'] = validation_rules
            
        super().__init__(
            message=message,
            user_message=user_msg,
            suggestion=suggestion,
            technical_details=tech_details,
            **kwargs
        )


class ErrorHandler:
    """Centralized error handling with logging and context."""
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        self.logger = logger or logging.getLogger(__name__)
    
    def handle_error(self, error: Exception, context: Optional[Dict[str, Any]] = None) -> None:
        """Handle errors with logging and user guidance."""
        if isinstance(error, QEMError):
            self.logger.error(error.format_error(verbose=True))
            if context:
                self.logger.error(f"Context: {context}")
        else:
            # Wrap non-QEM errors
            wrapped_error = QEMError(
                message=str(error),
                user_message=f"An unexpected error occurred: {str(error)}",
                technical_details={'original_error': str(error), 'type': type(error).__name__},
                suggestion="Please check the logs for more details or report this issue.",
                code_context={'traceback': traceback.format_exc()}
            )
            self.logger.error(wrapped_error.format_error(verbose=True))
    
    def validate_and_raise(
        self,
        condition: bool,
        error_type: type,
        message: str,
        **kwargs
    ) -> None:
        """Validate condition and raise appropriate error."""
        if not condition:
            raise error_type(message, **kwargs)


# Global error handler instance
error_handler = ErrorHandler()