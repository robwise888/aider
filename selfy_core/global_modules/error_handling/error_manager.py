"""
Error manager for the Selfy agent.

This module provides a centralized error management system with categorization,
severity assessment, and recovery strategies.
"""

import time
import json
import logging
import traceback
from datetime import datetime
from enum import Enum
from typing import Dict, Any, Optional, Callable, List, Type, Union

from selfy_core.global_modules.config import get as config_get

# Set up logger
logger = logging.getLogger(__name__)


class ErrorSeverity(Enum):
    """Enum for error severity levels"""
    LOW = "low"                  # Minor errors that don't affect functionality
    MEDIUM = "medium"            # Errors that affect some functionality
    HIGH = "high"                # Errors that affect major functionality
    CRITICAL = "critical"        # Errors that prevent the system from functioning


class ErrorCategory(Enum):
    """Enum for error categories"""
    SYSTEM = "system"              # System-level errors (file system, memory, etc.)
    NETWORK = "network"            # Network-related errors
    API = "api"                    # API-related errors (rate limits, authentication, etc.)
    DATA = "data"                  # Data-related errors (parsing, validation, etc.)
    LOGIC = "logic"                # Logic errors in the code
    RESOURCE = "resource"          # Resource-related errors (missing files, etc.)
    PERMISSION = "permission"      # Permission-related errors
    TIMEOUT = "timeout"            # Timeout errors
    DEPENDENCY = "dependency"      # Dependency-related errors
    INTEGRATION = "integration"    # Integration-related errors
    LLM = "llm"                    # LLM-related errors
    RAG = "rag"                    # RAG-related errors
    UNKNOWN = "unknown"            # Unknown errors


class ErrorRecoveryStrategy(Enum):
    """Enum for error recovery strategies"""
    RETRY = "retry"                # Retry the operation
    FALLBACK = "fallback"          # Use a fallback mechanism
    SKIP = "skip"                  # Skip the operation
    ABORT = "abort"                # Abort the process
    MANUAL = "manual"              # Requires manual intervention
    IGNORE = "ignore"              # Ignore the error and continue


class Error:
    """
    Class representing an error with classification and recovery information.
    """

    def __init__(self,
                 message: str,
                 category: ErrorCategory = ErrorCategory.UNKNOWN,
                 severity: ErrorSeverity = ErrorSeverity.MEDIUM,
                 recovery_strategy: ErrorRecoveryStrategy = ErrorRecoveryStrategy.ABORT,
                 exception: Optional[Exception] = None,
                 context: Optional[Dict[str, Any]] = None,
                 source: Optional[str] = None):
        """
        Initialize a new error.

        Args:
            message: Error message
            category: Error category
            severity: Error severity
            recovery_strategy: Recovery strategy
            exception: Original exception if available
            context: Additional context information
            source: Source of the error (module, function, etc.)
        """
        self.id = f"err_{int(time.time())}_{hash(message) % 10000:04d}"
        self.message = message
        self.category = category
        self.severity = severity
        self.recovery_strategy = recovery_strategy
        self.exception = exception
        self.context = context or {}
        self.source = source
        self.timestamp = datetime.now().isoformat()
        self.traceback = traceback.format_exc() if exception else None
        self.handled = False
        self.recovery_attempts = 0
        self.resolved = False
        self.resolution_message = None
        self.resolution_timestamp = None

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the error to a dictionary.

        Returns:
            Dictionary representation of the error
        """
        return {
            "id": self.id,
            "message": self.message,
            "category": self.category.value,
            "severity": self.severity.value,
            "recovery_strategy": self.recovery_strategy.value,
            "exception": str(self.exception) if self.exception else None,
            "context": self.context,
            "source": self.source,
            "timestamp": self.timestamp,
            "traceback": self.traceback,
            "handled": self.handled,
            "recovery_attempts": self.recovery_attempts,
            "resolved": self.resolved,
            "resolution_message": self.resolution_message,
            "resolution_timestamp": self.resolution_timestamp
        }

    def log(self) -> None:
        """Log the error with appropriate severity."""
        log_level = logging.ERROR
        if self.severity == ErrorSeverity.LOW:
            log_level = logging.WARNING
        elif self.severity == ErrorSeverity.CRITICAL:
            log_level = logging.CRITICAL

        logger.log(log_level, f"Error {self.id}: {self.message} "
                             f"[{self.category.value}] [{self.severity.value}]")

    def mark_handled(self) -> None:
        """Mark the error as handled."""
        self.handled = True

    def increment_recovery_attempts(self) -> None:
        """Increment the recovery attempts counter."""
        self.recovery_attempts += 1

    def mark_resolved(self, resolution_message: str) -> None:
        """
        Mark the error as resolved.

        Args:
            resolution_message: Message describing how the error was resolved
        """
        self.resolved = True
        self.resolution_message = resolution_message
        self.resolution_timestamp = datetime.now().isoformat()
        logger.info(f"Error {self.id} resolved: {resolution_message}")

    def should_retry(self) -> bool:
        """
        Check if the error should be retried.

        Returns:
            True if the error should be retried, False otherwise
        """
        return (self.recovery_strategy == ErrorRecoveryStrategy.RETRY and
                self.recovery_attempts < 3)


class ErrorManager:
    """
    Manages errors, including classification, logging, and recovery.
    """

    def __init__(self):
        """Initialize the error manager."""
        self.errors: List[Error] = []
        self.error_handlers: Dict[ErrorCategory, Callable[[Error], None]] = {}
        self.recovery_strategies: Dict[ErrorRecoveryStrategy, Callable[[Error], bool]] = {}
        self.error_log_path = config_get('error_manager.log_path', 'logs/errors.json')

        # Register default handlers
        self._register_default_handlers()

    def _register_default_handlers(self):
        """Register default error handlers."""
        # Register handlers for common error categories
        self.register_error_handler(
            ErrorCategory.NETWORK,
            self._handle_network_error
        )

        self.register_error_handler(
            ErrorCategory.API,
            self._handle_api_error
        )

        self.register_error_handler(
            ErrorCategory.TIMEOUT,
            self._handle_timeout_error
        )

        self.register_error_handler(
            ErrorCategory.LLM,
            self._handle_llm_error
        )

        self.register_error_handler(
            ErrorCategory.RAG,
            self._handle_rag_error
        )

        # Register default recovery strategies
        self.register_recovery_strategy(
            ErrorRecoveryStrategy.RETRY,
            self._retry_strategy
        )

        self.register_recovery_strategy(
            ErrorRecoveryStrategy.FALLBACK,
            self._fallback_strategy
        )

        self.register_recovery_strategy(
            ErrorRecoveryStrategy.SKIP,
            self._skip_strategy
        )

    def register_error_handler(self, category: ErrorCategory, handler: Callable[[Error], None]) -> None:
        """
        Register an error handler for a specific category.

        Args:
            category: The error category to handle
            handler: The handler function
        """
        self.error_handlers[category] = handler
        logger.debug(f"Registered error handler for category: {category.value}")

    def register_recovery_strategy(self, strategy: ErrorRecoveryStrategy,
                                  handler: Callable[[Error], bool]) -> None:
        """
        Register a recovery strategy.

        Args:
            strategy: The recovery strategy
            handler: The handler function that implements the strategy
        """
        self.recovery_strategies[strategy] = handler
        logger.debug(f"Registered recovery strategy: {strategy.value}")

    def handle_error(self, error: Union[Error, Exception],
                    context: Optional[Dict[str, Any]] = None,
                    source: Optional[str] = None) -> Error:
        """
        Handle an error.

        Args:
            error: The error to handle
            context: Additional context information
            source: Source of the error

        Returns:
            The handled error
        """
        # Convert Exception to Error if needed
        if isinstance(error, Exception) and not isinstance(error, Error):
            error = self.create_error_from_exception(error, context, source)

        # Add to error list
        self.errors.append(error)

        # Log the error
        error.log()

        # Try to handle the error with a registered handler
        if error.category in self.error_handlers:
            try:
                self.error_handlers[error.category](error)
                error.mark_handled()
            except Exception as e:
                logger.error(f"Error in handler for {error.category.value}: {e}")

        # Try to recover from the error
        if error.recovery_strategy in self.recovery_strategies:
            try:
                error.increment_recovery_attempts()
                success = self.recovery_strategies[error.recovery_strategy](error)
                if success:
                    error.mark_resolved(f"Recovered using {error.recovery_strategy.value} strategy")
            except Exception as e:
                logger.error(f"Error in recovery strategy {error.recovery_strategy.value}: {e}")

        # Save error to log
        self._save_error_to_log(error)

        return error

    def create_error_from_exception(self, exception: Exception,
                                   context: Optional[Dict[str, Any]] = None,
                                   source: Optional[str] = None) -> Error:
        """
        Create an Error object from an Exception.

        Args:
            exception: The exception to convert
            context: Additional context information
            source: Source of the error

        Returns:
            The created Error object
        """
        # Determine error category and severity based on exception type
        category = self._categorize_exception(exception)
        severity = self._assess_severity(exception, category)
        recovery_strategy = self._determine_recovery_strategy(exception, category, severity)

        return Error(
            message=str(exception),
            category=category,
            severity=severity,
            recovery_strategy=recovery_strategy,
            exception=exception,
            context=context,
            source=source
        )

    def _categorize_exception(self, exception: Exception) -> ErrorCategory:
        """
        Categorize an exception.

        Args:
            exception: The exception to categorize

        Returns:
            The error category
        """
        exception_type = type(exception).__name__

        # Network errors
        if any(name in exception_type for name in ["ConnectionError", "HTTPError", "Timeout"]):
            return ErrorCategory.NETWORK

        # API errors
        if any(name in exception_type for name in ["APIError", "RateLimitError", "AuthError"]):
            return ErrorCategory.API

        # Data errors
        if any(name in exception_type for name in ["JSONDecodeError", "ValueError", "TypeError"]):
            return ErrorCategory.DATA

        # Resource errors
        if any(name in exception_type for name in ["FileNotFoundError", "IOError"]):
            return ErrorCategory.RESOURCE

        # Permission errors
        if "PermissionError" in exception_type:
            return ErrorCategory.PERMISSION

        # Timeout errors
        if "TimeoutError" in exception_type:
            return ErrorCategory.TIMEOUT

        # LLM errors
        if any(name in exception_type for name in ["LLMError", "ModelError"]):
            return ErrorCategory.LLM

        return ErrorCategory.UNKNOWN

    def _assess_severity(self, exception: Exception, category: ErrorCategory) -> ErrorSeverity:
        """
        Assess the severity of an exception.

        Args:
            exception: The exception to assess
            category: The error category

        Returns:
            The error severity
        """
        # Critical errors
        if category in [ErrorCategory.SYSTEM, ErrorCategory.PERMISSION]:
            return ErrorSeverity.CRITICAL

        # High severity errors
        if category in [ErrorCategory.RESOURCE, ErrorCategory.DEPENDENCY]:
            return ErrorSeverity.HIGH

        # Medium severity errors
        if category in [ErrorCategory.NETWORK, ErrorCategory.API, ErrorCategory.DATA,
                       ErrorCategory.TIMEOUT, ErrorCategory.LLM]:
            return ErrorSeverity.MEDIUM

        # Default to medium severity
        return ErrorSeverity.MEDIUM

    def _determine_recovery_strategy(self, exception: Exception,
                                    category: ErrorCategory,
                                    severity: ErrorSeverity) -> ErrorRecoveryStrategy:
        """
        Determine the recovery strategy for an exception.

        Args:
            exception: The exception
            category: The error category
            severity: The error severity

        Returns:
            The recovery strategy
        """
        # Retry for network, timeout, and some API errors
        if category in [ErrorCategory.NETWORK, ErrorCategory.TIMEOUT]:
            return ErrorRecoveryStrategy.RETRY

        # Fallback for LLM and RAG errors
        if category in [ErrorCategory.LLM, ErrorCategory.RAG]:
            return ErrorRecoveryStrategy.FALLBACK

        # Skip for low severity errors
        if severity == ErrorSeverity.LOW:
            return ErrorRecoveryStrategy.SKIP

        # Abort for critical errors
        if severity == ErrorSeverity.CRITICAL:
            return ErrorRecoveryStrategy.ABORT

        # Default to manual intervention
        return ErrorRecoveryStrategy.MANUAL

    def _save_error_to_log(self, error: Error) -> None:
        """
        Save an error to the error log.

        Args:
            error: The error to save
        """
        try:
            # Convert error to dictionary
            error_dict = error.to_dict()

            # Load existing errors
            try:
                with open(self.error_log_path, 'r') as f:
                    errors = json.load(f)
            except (FileNotFoundError, json.JSONDecodeError):
                errors = []

            # Add new error
            errors.append(error_dict)

            # Save errors
            with open(self.error_log_path, 'w') as f:
                json.dump(errors, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save error to log: {e}")

    # Default error handlers
    def _handle_network_error(self, error: Error) -> None:
        """
        Handle a network error.

        Args:
            error: The error to handle
        """
        logger.info(f"Handling network error: {error.message}")
        # Implement network error handling logic

    def _handle_api_error(self, error: Error) -> None:
        """
        Handle an API error.

        Args:
            error: The error to handle
        """
        logger.info(f"Handling API error: {error.message}")
        # Implement API error handling logic

    def _handle_timeout_error(self, error: Error) -> None:
        """
        Handle a timeout error.

        Args:
            error: The error to handle
        """
        logger.info(f"Handling timeout error: {error.message}")
        # Implement timeout error handling logic

    def _handle_llm_error(self, error: Error) -> None:
        """
        Handle an LLM error.

        Args:
            error: The error to handle
        """
        logger.info(f"Handling LLM error: {error.message}")
        # Implement LLM error handling logic

    def _handle_rag_error(self, error: Error) -> None:
        """
        Handle a RAG error.

        Args:
            error: The error to handle
        """
        logger.info(f"Handling RAG error: {error.message}")
        # Implement RAG error handling logic

    # Default recovery strategies
    def _retry_strategy(self, error: Error) -> bool:
        """
        Retry strategy for errors.

        Args:
            error: The error to recover from

        Returns:
            True if recovery was successful, False otherwise
        """
        logger.info(f"Applying retry strategy for error: {error.id}")

        # Implement retry logic with exponential backoff
        max_retries = 3
        if error.recovery_attempts > max_retries:
            logger.warning(f"Maximum retry attempts ({max_retries}) reached for error: {error.id}")
            return False

        # Calculate backoff time
        backoff_time = 2 ** error.recovery_attempts  # Exponential backoff
        logger.info(f"Waiting {backoff_time} seconds before retry")
        time.sleep(backoff_time)

        # The actual retry logic would be implemented by the caller
        return False  # Return False to indicate that the caller should handle the retry

    def _fallback_strategy(self, error: Error) -> bool:
        """
        Fallback strategy for errors.

        Args:
            error: The error to recover from

        Returns:
            True if recovery was successful, False otherwise
        """
        logger.info(f"Applying fallback strategy for error: {error.id}")

        # Implement fallback logic
        # The actual fallback logic would be implemented by the caller
        return False  # Return False to indicate that the caller should handle the fallback

    def _skip_strategy(self, error: Error) -> bool:
        """
        Skip strategy for errors.

        Args:
            error: The error to recover from

        Returns:
            True if recovery was successful, False otherwise
        """
        logger.info(f"Applying skip strategy for error: {error.id}")

        # Skip the operation that caused the error
        # The actual skip logic would be implemented by the caller
        return True  # Return True to indicate that skipping is considered a successful recovery


# Global instance
_error_manager_instance = None


def setup_error_manager() -> bool:
    """
    Set up the error manager.

    Returns:
        True if setup was successful, False otherwise
    """
    global _error_manager_instance

    try:
        if _error_manager_instance is None:
            _error_manager_instance = ErrorManager()

        return True
    except Exception as e:
        logger.error(f"Failed to set up error manager: {e}")
        return False


def get_error_manager() -> Optional[ErrorManager]:
    """
    Get the error manager instance.

    Returns:
        The error manager instance, or None if not set up
    """
    global _error_manager_instance

    if _error_manager_instance is None:
        logger.warning("Error manager not initialized")

    return _error_manager_instance
