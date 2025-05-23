"""
Execution functions for the capability manifest module.

This module provides functions for executing capabilities.
"""

import time
import importlib
import inspect
from typing import Dict, List, Any, Optional, Union, Callable

from selfy_core.global_modules.logging import get_logger
from .data_structures import Capability, CapabilityResult
from .exceptions import CapabilityError, CapabilityNotFoundError, ParameterError, ExecutionError
from .setup import get_registry

# Set up logger
logger = get_logger(__name__)


def execute_capability(name: str, parameters: Dict[str, Any] = None) -> CapabilityResult:
    """
    Execute a capability.

    Args:
        name: The name of the capability to execute
        parameters: The parameters to pass to the capability

    Returns:
        A CapabilityResult object containing the result

    Raises:
        CapabilityNotFoundError: If the capability is not found
        ParameterError: If the parameters are invalid
        ExecutionError: If there's an error during execution
    """
    start_time = time.time()
    logger.info(f"Executing capability: {name}")

    # Get the capability
    capability = get_registry().get_capability(name)
    if not capability:
        error_msg = f"Capability not found: {name}"
        logger.error(error_msg)
        raise CapabilityNotFoundError(error_msg)

    # Check if the capability is enabled
    if not capability.metadata.get('enabled', True):
        error_msg = f"Capability is disabled: {name}"
        logger.error(error_msg)
        return CapabilityResult.error(
            capability=name,
            error_message=error_msg,
            error_type="CapabilityDisabled",
            execution_time=time.time() - start_time
        )

    # Initialize parameters if not provided
    if parameters is None:
        parameters = {}

    try:
        # Validate parameters
        validate_parameters(capability, parameters)

        # Execute the capability
        if capability.type == 'library':
            result = execute_library_capability(capability, parameters)
        else:
            result = execute_builtin_capability(capability, parameters)

        # Calculate execution time
        execution_time = time.time() - start_time

        # Log the result
        logger.info(f"Capability {name} executed successfully in {execution_time:.4f} seconds")

        # Return the result
        return CapabilityResult.success(
            capability=name,
            result=result,
            execution_time=execution_time
        )
    except ParameterError as e:
        logger.error(f"Parameter error executing capability {name}: {e}")
        return CapabilityResult.error(
            capability=name,
            error_message=str(e),
            error_type="ParameterError",
            execution_time=time.time() - start_time
        )
    except ExecutionError as e:
        logger.error(f"Execution error executing capability {name}: {e}")
        return CapabilityResult.error(
            capability=name,
            error_message=str(e),
            error_type="ExecutionError",
            execution_time=time.time() - start_time
        )
    except Exception as e:
        logger.error(f"Unexpected error executing capability {name}: {e}", exc_info=True)
        return CapabilityResult.error(
            capability=name,
            error_message=str(e),
            error_type=type(e).__name__,
            execution_time=time.time() - start_time
        )


def execute_capability_with_backward_compatibility(name: str, parameters: Dict[str, Any] = None, function_name: str = None) -> CapabilityResult:
    """
    Execute a capability with backward compatibility.

    This function supports the old capability format with multiple functions.

    Args:
        name: The name of the capability to execute
        parameters: The parameters to pass to the capability
        function_name: The name of the function to execute (optional)

    Returns:
        A CapabilityResult object containing the result

    Raises:
        CapabilityNotFoundError: If the capability is not found
        ParameterError: If the parameters are invalid
        ExecutionError: If there's an error during execution
    """
    start_time = time.time()
    logger.info(f"Executing capability: {name} (function: {function_name})")

    # Get the capability
    capability = get_registry().get_capability(name)
    if not capability:
        error_msg = f"Capability not found: {name}"
        logger.error(error_msg)
        raise CapabilityNotFoundError(error_msg)

    # Check if the capability is enabled
    if not capability.metadata.get('enabled', True):
        error_msg = f"Capability is disabled: {name}"
        logger.error(error_msg)
        return CapabilityResult.error(
            capability=name,
            error_message=error_msg,
            error_type="CapabilityDisabled",
            execution_time=time.time() - start_time
        )

    # Initialize parameters if not provided
    if parameters is None:
        parameters = {}

    try:
        # Check if the capability has functions
        functions = capability.metadata.get('functions', [])
        if functions and function_name:
            # Find the function
            function = next((f for f in functions if f.get('name') == function_name), None)
            if not function:
                error_msg = f"Function not found: {function_name}"
                logger.error(error_msg)
                raise CapabilityNotFoundError(error_msg)

            # Validate parameters
            validate_parameters_for_function(function, parameters)

            # Execute the function
            if capability.type == 'library':
                result = execute_library_function(capability, function, parameters)
            else:
                result = execute_builtin_function(capability, function, parameters)
        else:
            # Execute the capability directly
            if capability.type == 'library':
                result = execute_library_capability(capability, parameters)
            else:
                result = execute_builtin_capability(capability, parameters)

        # Calculate execution time
        execution_time = time.time() - start_time

        # Log the result
        logger.info(f"Capability {name} executed successfully in {execution_time:.4f} seconds")

        # Return the result
        return CapabilityResult.success(
            capability=name,
            result=result,
            execution_time=execution_time
        )
    except ParameterError as e:
        logger.error(f"Parameter error executing capability {name}: {e}")
        return CapabilityResult.error(
            capability=name,
            error_message=str(e),
            error_type="ParameterError",
            execution_time=time.time() - start_time
        )
    except ExecutionError as e:
        logger.error(f"Execution error executing capability {name}: {e}")
        return CapabilityResult.error(
            capability=name,
            error_message=str(e),
            error_type="ExecutionError",
            execution_time=time.time() - start_time
        )
    except Exception as e:
        logger.error(f"Unexpected error executing capability {name}: {e}", exc_info=True)
        return CapabilityResult.error(
            capability=name,
            error_message=str(e),
            error_type=type(e).__name__,
            execution_time=time.time() - start_time
        )


def validate_parameters(capability: Capability, parameters: Dict[str, Any]) -> None:
    """
    Validate parameters for a capability.

    Args:
        capability: The capability to validate parameters for
        parameters: The parameters to validate

    Raises:
        ParameterError: If the parameters are invalid
    """
    # Get the parameter definitions
    param_defs = capability.parameters

    # Check for required parameters
    for param_def in param_defs:
        name = param_def.get('name')
        required = param_def.get('required', False)

        if required and name not in parameters:
            raise ParameterError(f"Required parameter '{name}' not provided")

    # Check for unknown parameters
    param_names = [p.get('name') for p in param_defs]
    for name in parameters:
        if name not in param_names:
            logger.warning(f"Unknown parameter '{name}' provided")

    # Validate parameter types
    for param_def in param_defs:
        name = param_def.get('name')
        param_type = param_def.get('type')

        if name in parameters:
            value = parameters[name]

            # Skip validation for None values
            if value is None:
                continue

            # Validate type
            if param_type == 'string' and not isinstance(value, str):
                raise ParameterError(f"Parameter '{name}' must be a string")
            elif param_type == 'number' and not isinstance(value, (int, float)):
                raise ParameterError(f"Parameter '{name}' must be a number")
            elif param_type == 'integer' and not isinstance(value, int):
                raise ParameterError(f"Parameter '{name}' must be an integer")
            elif param_type == 'boolean' and not isinstance(value, bool):
                raise ParameterError(f"Parameter '{name}' must be a boolean")
            elif param_type == 'array' and not isinstance(value, list):
                raise ParameterError(f"Parameter '{name}' must be an array")
            elif param_type == 'object' and not isinstance(value, dict):
                raise ParameterError(f"Parameter '{name}' must be an object")


def validate_parameters_for_function(function: Dict[str, Any], parameters: Dict[str, Any]) -> None:
    """
    Validate parameters for a function.

    Args:
        function: The function to validate parameters for
        parameters: The parameters to validate

    Raises:
        ParameterError: If the parameters are invalid
    """
    # Get the parameter definitions
    param_defs = function.get('parameters', [])

    # Check for required parameters
    for param_def in param_defs:
        name = param_def.get('name')
        required = param_def.get('required', False)

        if required and name not in parameters:
            raise ParameterError(f"Required parameter '{name}' not provided")

    # Check for unknown parameters
    param_names = [p.get('name') for p in param_defs]
    for name in parameters:
        if name not in param_names:
            logger.warning(f"Unknown parameter '{name}' provided")

    # Validate parameter types
    for param_def in param_defs:
        name = param_def.get('name')
        param_type = param_def.get('type')

        if name in parameters:
            value = parameters[name]

            # Skip validation for None values
            if value is None:
                continue

            # Validate type
            if param_type == 'string' and not isinstance(value, str):
                raise ParameterError(f"Parameter '{name}' must be a string")
            elif param_type == 'number' and not isinstance(value, (int, float)):
                raise ParameterError(f"Parameter '{name}' must be a number")
            elif param_type == 'integer' and not isinstance(value, int):
                raise ParameterError(f"Parameter '{name}' must be an integer")
            elif param_type == 'boolean' and not isinstance(value, bool):
                raise ParameterError(f"Parameter '{name}' must be a boolean")
            elif param_type == 'array' and not isinstance(value, list):
                raise ParameterError(f"Parameter '{name}' must be an array")
            elif param_type == 'object' and not isinstance(value, dict):
                raise ParameterError(f"Parameter '{name}' must be an object")


def execute_library_capability(capability: Capability, parameters: Dict[str, Any]) -> Any:
    """
    Execute a library capability.

    Args:
        capability: The capability to execute
        parameters: The parameters to pass to the capability

    Returns:
        The result of the execution

    Raises:
        ExecutionError: If there's an error during execution
    """
    try:
        # Get the source
        source = capability.source

        # Split the source into module and function
        if '.' in source:
            module_name, function_name = source.rsplit('.', 1)
        else:
            raise ExecutionError(f"Invalid source format: {source}")

        # Import the module
        try:
            module = importlib.import_module(module_name)
        except ImportError as e:
            raise ExecutionError(f"Failed to import module {module_name}: {e}")

        # Get the function
        try:
            function = getattr(module, function_name)
        except AttributeError as e:
            raise ExecutionError(f"Failed to get function {function_name} from module {module_name}: {e}")

        # Execute the function
        return function(**parameters)
    except Exception as e:
        raise ExecutionError(f"Failed to execute library capability: {e}")


def execute_builtin_capability(capability: Capability, parameters: Dict[str, Any]) -> Any:
    """
    Execute a builtin capability.

    Args:
        capability: The capability to execute
        parameters: The parameters to pass to the capability

    Returns:
        The result of the execution

    Raises:
        ExecutionError: If there's an error during execution
    """
    # For now, just raise an error
    raise ExecutionError("Builtin capabilities not implemented yet")


def execute_library_function(capability: Capability, function: Dict[str, Any], parameters: Dict[str, Any]) -> Any:
    """
    Execute a library function.

    Args:
        capability: The capability containing the function
        function: The function to execute
        parameters: The parameters to pass to the function

    Returns:
        The result of the execution

    Raises:
        ExecutionError: If there's an error during execution
    """
    try:
        # Get the source
        source = function.get('source')
        if not source:
            source = capability.source

        # Split the source into module and function
        if '.' in source:
            module_name, function_name = source.rsplit('.', 1)
        else:
            raise ExecutionError(f"Invalid source format: {source}")

        # Import the module
        try:
            module = importlib.import_module(module_name)
        except ImportError as e:
            raise ExecutionError(f"Failed to import module {module_name}: {e}")

        # Get the function
        try:
            func = getattr(module, function_name)
        except AttributeError as e:
            raise ExecutionError(f"Failed to get function {function_name} from module {module_name}: {e}")

        # Execute the function
        return func(**parameters)
    except Exception as e:
        raise ExecutionError(f"Failed to execute library function: {e}")


def execute_builtin_function(capability: Capability, function: Dict[str, Any], parameters: Dict[str, Any]) -> Any:
    """
    Execute a builtin function.

    Args:
        capability: The capability containing the function
        function: The function to execute
        parameters: The parameters to pass to the function

    Returns:
        The result of the execution

    Raises:
        ExecutionError: If there's an error during execution
    """
    # For now, just raise an error
    raise ExecutionError("Builtin functions not implemented yet")
