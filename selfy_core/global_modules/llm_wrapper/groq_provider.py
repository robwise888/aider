"""
Groq Provider for the LLM Wrapper module.

This module provides the GroqProvider class, which implements the BaseLLMProvider
interface for the Groq cloud LLM service.
"""

import os
import time
import logging
import json
from typing import Dict, List, Any, Optional, Union

from selfy_core.global_modules.config import get as config_get
from .base_llm_provider import BaseLLMProvider
from .response import LLMResponse
from .exceptions import (
    LLMError, LLMConfigurationError, LLMConnectionError, LLMAPIError,
    LLMRateLimitError, LLMAuthenticationError, LLMTimeoutError, LLMTokenLimitError
)
from .token_utils import count_tokens, count_message_tokens
from .logging_helpers import log_llm_call

# Set up logging
logger = logging.getLogger(__name__)

# Try to import the Groq client
try:
    import groq
    from groq import Groq
    # The error classes might be in different locations depending on the version
    try:
        # Try the newer structure
        from groq.errors import (
            GroqError, APIError, APIConnectionError, AuthenticationError,
            RateLimitError, APITimeoutError, InvalidRequestError
        )
    except ImportError:
        # Try the older structure or define our own
        GroqError = Exception
        APIError = Exception
        APIConnectionError = Exception
        AuthenticationError = Exception
        RateLimitError = Exception
        APITimeoutError = Exception
        InvalidRequestError = Exception

    logger.info(f"Successfully imported groq library version: {getattr(groq, '__version__', 'unknown')}")
    GROQ_CLIENT_AVAILABLE = True
except ImportError as e:
    logger.warning(f"'groq' library not installed or import error: {e}. Groq LLM features disabled.")
    GROQ_CLIENT_AVAILABLE = False
    # Define placeholder classes for type checking
    class GroqError(Exception): pass
    class APIError(GroqError): pass
    class APIConnectionError(GroqError): pass
    class AuthenticationError(GroqError): pass
    class RateLimitError(GroqError): pass
    class APITimeoutError(GroqError): pass
    class InvalidRequestError(GroqError): pass


class GroqProvider(BaseLLMProvider):
    """
    Groq LLM provider implementation.

    This class implements the BaseLLMProvider interface for the Groq cloud LLM service.
    It provides methods for generating text and responses using Groq's API.
    """

    def __init__(self,
                api_key: str = None,
                model_name: str = None,
                max_retries: int = 2,
                retry_delay_seconds: float = 1.0,
                timeout_seconds: int = 30,
                **kwargs):
        """
        Initialize the Groq provider.

        Args:
            api_key: The Groq API key
            model_name: The name of the model to use
            max_retries: Maximum number of retries for API calls
            retry_delay_seconds: Delay between retries in seconds
            timeout_seconds: Timeout for API calls in seconds
            **kwargs: Additional provider-specific arguments
        """
        super().__init__(model_name)

        # Get API key from multiple sources with priority:
        # 1. Explicitly passed api_key parameter
        # 2. Environment variable GROQ_API_KEY
        # 3. Config file
        if api_key is None:
            # Try to get from environment variable directly
            env_api_key = os.environ.get('GROQ_API_KEY')
            if env_api_key:
                logger.debug("Using Groq API key from environment variable")
                api_key = env_api_key
            else:
                # Try to get from config
                config_api_key = config_get('llm.groq.api_key', '')
                if config_api_key:
                    logger.debug("Using Groq API key from config file")
                    api_key = config_api_key
                else:
                    logger.warning("No Groq API key found in environment variable or config file")

        # Store configuration
        self.api_key = api_key
        self.max_retries = max_retries
        self.retry_delay_seconds = retry_delay_seconds
        self.timeout_seconds = timeout_seconds

        # Use default model if not provided
        if not self.model_name:
            self.model_name = config_get('llm.groq.default_model', 'llama3-70b-8192')

        # Initialize client
        self.client = None
        self.is_available = False

        # Try to initialize the client
        if GROQ_CLIENT_AVAILABLE:
            if not self.api_key:
                logger.error("Cannot initialize Groq client: No API key provided")
                return

            # Validate API key format
            if not self.api_key.strip():
                logger.error("Cannot initialize Groq client: API key is empty or whitespace")
                return

            try:
                self.client = Groq(api_key=self.api_key)
                self.is_available = True
                # Mask API key for logging
                masked_key = self.api_key[:4] + '*' * (len(self.api_key) - 8) + self.api_key[-4:] if len(self.api_key) > 8 else '****'
                logger.info(f"Groq provider initialized with model {self.model_name} and API key {masked_key}")
            except Exception as e:
                logger.error(f"Failed to initialize Groq client: {e}")
        else:
            logger.warning("Groq client not available. Install the 'groq' package to use Groq.")

    def _parse_error(self, error: Exception) -> tuple:
        """
        Parse an error from the Groq API.

        Args:
            error: The error to parse

        Returns:
            Tuple of (error_type, status_code, error_message)
        """
        error_type = type(error).__name__
        status_code = None
        error_message = str(error)

        # Extract status code if available
        if hasattr(error, 'status_code'):
            status_code = error.status_code

        return error_type, status_code, error_message

    def _process_messages(self, messages: List[Dict[str, str]], system_prompt: Optional[str] = None) -> List[Dict[str, str]]:
        """
        Process messages to avoid duplication and ensure proper formatting.

        Args:
            messages: List of message dictionaries
            system_prompt: Optional system instructions

        Returns:
            Processed list of message dictionaries
        """
        # Make a copy to avoid modifying the original
        processed_messages = messages.copy()

        # Check if there's already a system message
        has_system = any(msg.get('role') == 'system' for msg in processed_messages)

        # If system_prompt is provided and there's no system message, add it
        if system_prompt and not has_system:
            processed_messages = [{"role": "system", "content": system_prompt}] + processed_messages

        # Check for content duplication in user messages
        for i, msg in enumerate(processed_messages):
            if msg.get('role') == 'user' and 'content' in msg:
                # Import the validation function
                from selfy_core.user_pipeline.context_engine.utils.prompt_utils import validate_and_deduplicate_prompt

                # Deduplicate the content
                cleaned_content, stats = validate_and_deduplicate_prompt(msg['content'])

                # If duplications were found, update the message
                if stats['duplications_found'] > 0:
                    logger.info(f"Removed {stats['duplications_found']} duplications from message content")
                    processed_messages[i]['content'] = cleaned_content

        return processed_messages

    def _safe_llm_call(self,
                      messages: List[Dict[str, str]],
                      system_prompt: Optional[str] = None,
                      tools: Optional[List[Dict[str, Any]]] = None,
                      tool_choice: str = "auto",
                      task_description: str = "LLM Call",
                      temperature: float = 0.7,
                      max_tokens: Optional[int] = None,
                      stop_sequences: Optional[List[str]] = None,
                      top_p: Optional[float] = None,
                      top_k: Optional[int] = None) -> LLMResponse:
        """
        Makes a call to the Groq API using messages list, handles tools, retries, errors.

        Args:
            messages: List of message dictionaries
            system_prompt: Optional system instructions
            tools: Optional list of tools
            tool_choice: Tool choice strategy
            task_description: Description of the task for logging
            temperature: Temperature for response generation
            max_tokens: Maximum number of tokens to generate
            stop_sequences: Optional list of strings that will stop generation
            top_p: Optional nucleus sampling parameter
            top_k: Optional top-k sampling parameter

        Returns:
            An LLMResponse object containing the result and usage statistics

        Raises:
            LLMError: If there's an issue during generation
        """
        start_time = time.time()

        # Log the request
        logger.info(f"Groq call ({task_description}, Temp:{temperature}, Model:{self.model_name}, Tools:{tools is not None}, MaxTokens:{max_tokens})")

        # Check if client is available
        if not self.is_available or not self.client:
            msg = "Groq call abort: Client unavailable/uninitialized."
            logger.error(msg)
            raise LLMConnectionError(msg)

        # Check if messages are provided
        if not messages:
            msg = "Groq call abort: No messages provided to send."
            logger.error(msg)
            raise LLMConfigurationError(msg)

        # Process messages to avoid duplication
        processed_messages = self._process_messages(messages, system_prompt)

        # Detailed flow logging
        logger.debug(f"CONTEXT ENGINE FLOW - GROQ INPUT MESSAGES: {json.dumps(processed_messages, indent=2)}")

        # Log the first message content for context
        if processed_messages and len(processed_messages) > 0 and 'content' in processed_messages[-1]:
            content = processed_messages[-1]['content']
            logger.debug(f"CONTEXT ENGINE FLOW - GROQ PROMPT: {content[:500]}...")

        if tools:
            logger.debug(f"Tools passed: {[t['function']['name'] for t in tools]}")
            logger.debug(f"CONTEXT ENGINE FLOW - GROQ TOOLS: {json.dumps([t['function']['name'] for t in tools], indent=2)}")

        # Count prompt tokens
        prompt_tokens = count_message_tokens(processed_messages, self.model_name)

        # Prepare API call parameters
        api_params = {
            "model": self.model_name,
            "messages": processed_messages,
            "temperature": temperature,
        }

        # Add optional parameters if provided
        if max_tokens is not None:
            api_params["max_tokens"] = max_tokens
        if stop_sequences:
            api_params["stop"] = stop_sequences
        if top_p is not None:
            api_params["top_p"] = top_p
        if tools:
            api_params["tools"] = tools
            api_params["tool_choice"] = tool_choice

        # Make the API call with retries
        retries = 0
        wait_time = self.retry_delay_seconds
        last_exception = None

        while True:
            try:
                # Make the API call
                response = self.client.chat.completions.create(**api_params)

                # Extract response content
                if hasattr(response, 'choices') and len(response.choices) > 0:
                    content = response.choices[0].message.content or ""
                    finish_reason = response.choices[0].finish_reason

                    # Extract tool calls if present
                    tool_calls = None
                    if hasattr(response.choices[0].message, 'tool_calls') and response.choices[0].message.tool_calls:
                        tool_calls = [
                            {
                                "id": tc.id,
                                "type": tc.type,
                                "function": {
                                    "name": tc.function.name,
                                    "arguments": tc.function.arguments
                                }
                            }
                            for tc in response.choices[0].message.tool_calls
                        ]
                else:
                    content = ""
                    finish_reason = "unknown"
                    tool_calls = None

                # Extract token usage
                if hasattr(response, 'usage'):
                    completion_tokens = response.usage.completion_tokens
                    total_tokens = response.usage.total_tokens
                else:
                    # Estimate if not provided
                    completion_tokens = count_tokens(content, self.model_name)
                    total_tokens = prompt_tokens + completion_tokens

                # Calculate duration
                duration = time.time() - start_time

                # Log the call
                log_llm_call(
                    logger=logger,
                    provider_name="groq",
                    model_name=self.model_name,
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                    duration=duration
                )

                # Log the response for detailed flow tracking
                logger.debug(f"CONTEXT ENGINE FLOW - GROQ RESPONSE: {content[:500]}...")

                # Create response object
                response_obj = LLMResponse(
                    content=content,
                    provider_name="groq",
                    model_name=self.model_name,
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                    total_tokens=total_tokens,
                    finish_reason=finish_reason,
                    tool_calls=tool_calls,
                    metadata={
                        "duration": duration,
                        "task_description": task_description
                    }
                )

                # Log token usage
                logger.debug(f"CONTEXT ENGINE FLOW - GROQ TOKEN USAGE: Prompt: {prompt_tokens}, Completion: {completion_tokens}, Total: {total_tokens}")

                return response_obj

            except Exception as e:
                last_exception = e
                type_e, code, msg = self._parse_error(e)

                # Handle different error types
                if isinstance(e, AuthenticationError):
                    duration = time.time() - start_time
                    log_llm_call(
                        logger=logger,
                        provider_name="groq",
                        model_name=self.model_name,
                        prompt_tokens=prompt_tokens,
                        completion_tokens=0,
                        duration=duration,
                        error=f"Authentication error: {msg}"
                    )
                    raise LLMAuthenticationError(f"Groq authentication error: {msg}", status_code=code)

                elif isinstance(e, RateLimitError):
                    if retries < self.max_retries:
                        logger.warning(f"Groq rate limit (Attempt {retries+1}/{self.max_retries+1}). Task:{task_description}. Error:{type_e}({code}): {msg}")
                        logger.warning(f"Retrying Groq in {wait_time:.1f}s...")
                        time.sleep(wait_time)
                        retries += 1
                        wait_time *= 1.5
                        continue
                    else:
                        duration = time.time() - start_time
                        log_llm_call(
                            logger=logger,
                            provider_name="groq",
                            model_name=self.model_name,
                            prompt_tokens=prompt_tokens,
                            completion_tokens=0,
                            duration=duration,
                            error=f"Rate limit exceeded: {msg}"
                        )
                        raise LLMRateLimitError(f"Groq rate limit exceeded after {self.max_retries+1} attempts: {msg}", status_code=code)

                elif isinstance(e, APITimeoutError):
                    if retries < self.max_retries:
                        logger.warning(f"Groq timeout (Attempt {retries+1}/{self.max_retries+1}). Task:{task_description}. Error:{type_e}({code}): {msg}")
                        logger.warning(f"Retrying Groq in {wait_time:.1f}s...")
                        time.sleep(wait_time)
                        retries += 1
                        wait_time *= 1.5
                        continue
                    else:
                        duration = time.time() - start_time
                        log_llm_call(
                            logger=logger,
                            provider_name="groq",
                            model_name=self.model_name,
                            prompt_tokens=prompt_tokens,
                            completion_tokens=0,
                            duration=duration,
                            error=f"Timeout: {msg}"
                        )
                        raise LLMTimeoutError(f"Groq timeout after {self.max_retries+1} attempts: {msg}", status_code=code)

                elif isinstance(e, APIConnectionError):
                    if retries < self.max_retries:
                        logger.warning(f"Groq connection error (Attempt {retries+1}/{self.max_retries+1}). Task:{task_description}. Error:{type_e}({code}): {msg}")
                        logger.warning(f"Retrying Groq in {wait_time:.1f}s...")
                        time.sleep(wait_time)
                        retries += 1
                        wait_time *= 1.5
                        continue
                    else:
                        duration = time.time() - start_time
                        log_llm_call(
                            logger=logger,
                            provider_name="groq",
                            model_name=self.model_name,
                            prompt_tokens=prompt_tokens,
                            completion_tokens=0,
                            duration=duration,
                            error=f"Connection error: {msg}"
                        )
                        raise LLMConnectionError(f"Groq connection error after {self.max_retries+1} attempts: {msg}", status_code=code)

                else:
                    # Generic error handling
                    duration = time.time() - start_time
                    log_llm_call(
                        logger=logger,
                        provider_name="groq",
                        model_name=self.model_name,
                        prompt_tokens=prompt_tokens,
                        completion_tokens=0,
                        duration=duration,
                        error=f"{type_e}: {msg}"
                    )
                    raise LLMAPIError(f"Groq API error: {msg}", status_code=code)

    def generate_text(self,
                     prompt: str,
                     system_prompt: Optional[str] = None,
                     temperature: float = 0.7,
                     max_tokens: Optional[int] = None,
                     stop_sequences: Optional[List[str]] = None,
                     top_p: Optional[float] = None,
                     top_k: Optional[int] = None,
                     **kwargs) -> LLMResponse:
        """
        Generate text from a simple prompt.

        Args:
            prompt: The text prompt to send to the LLM
            system_prompt: Optional system instructions to guide the model's behavior
            temperature: Temperature for response generation (0.0 to 2.0)
            max_tokens: Maximum number of tokens to generate (optional)
            stop_sequences: Optional list of strings that will stop generation when encountered
            top_p: Optional nucleus sampling parameter (0.0 to 1.0)
            top_k: Optional top-k sampling parameter
            **kwargs: Additional provider-specific parameters

        Returns:
            An LLMResponse object containing the result and usage statistics

        Raises:
            LLMError: If there's an issue during generation
        """
        logger.info(f"Generating text with prompt: {prompt[:50]}...")

        # Create a simple message list with a user message
        messages = [{"role": "user", "content": prompt}]

        # Get default system prompt from config or use hardcoded default
        default_system_prompt = config_get('llm.groq.default_system_prompt', "You are a helpful assistant.")

        # Use the safe LLM call
        return self._safe_llm_call(
            messages=messages,
            system_prompt=system_prompt or default_system_prompt,
            task_description="Groq Generating Text",
            temperature=temperature,
            max_tokens=max_tokens,
            stop_sequences=stop_sequences,
            top_p=top_p,
            top_k=top_k
        )

    def generate_chat_response(self,
                              messages: List[Dict[str, str]],
                              system_prompt: Optional[str] = None,
                              temperature: float = 0.7,
                              max_tokens: Optional[int] = None,
                              stop_sequences: Optional[List[str]] = None,
                              top_p: Optional[float] = None,
                              top_k: Optional[int] = None,
                              **kwargs) -> LLMResponse:
        """
        Generate a response from a chat history.

        Args:
            messages: List of message dictionaries, each with 'role' and 'content' keys
            system_prompt: Optional system instructions to guide the model's behavior
            temperature: Temperature for response generation (0.0 to 2.0)
            max_tokens: Maximum number of tokens to generate (optional)
            stop_sequences: Optional list of strings that will stop generation when encountered
            top_p: Optional nucleus sampling parameter (0.0 to 1.0)
            top_k: Optional top-k sampling parameter
            **kwargs: Additional provider-specific parameters

        Returns:
            An LLMResponse object containing the result and usage statistics

        Raises:
            LLMError: If there's an issue during generation
        """
        # Extract tools from kwargs if provided
        tools = kwargs.get('tools')
        tool_choice = kwargs.get('tool_choice', 'auto')

        # Use the safe LLM call
        return self._safe_llm_call(
            messages=messages,
            system_prompt=system_prompt,
            tools=tools,
            tool_choice=tool_choice,
            task_description="Groq Generating Response",
            temperature=temperature,
            max_tokens=max_tokens,
            stop_sequences=stop_sequences,
            top_p=top_p,
            top_k=top_k
        )

    def generate_chat_completion(self,
                               messages: List[Dict[str, str]],
                               temperature: float = 0.7,
                               max_tokens: Optional[int] = None,
                               stop_sequences: Optional[List[str]] = None,
                               system_prompt: Optional[str] = None,
                               **kwargs) -> LLMResponse:
        """
        Generate a chat completion (alias for generate_chat_response).

        This method is an alias for generate_chat_response to maintain compatibility
        with code that expects a generate_chat_completion method.

        Args:
            messages: List of message dictionaries, each with 'role' and 'content' keys
            temperature: Temperature for response generation (0.0 to 2.0)
            max_tokens: Maximum number of tokens to generate (optional)
            stop_sequences: Optional list of strings that will stop generation
            system_prompt: Optional system instructions to guide the model's behavior
            **kwargs: Additional provider-specific parameters

        Returns:
            An LLMResponse object containing the result and usage statistics

        Raises:
            LLMError: If there's an issue during generation
        """
        logger.debug("Using generate_chat_completion (alias for generate_chat_response)")
        return self.generate_chat_response(
            messages=messages,
            system_prompt=system_prompt,
            temperature=temperature,
            max_tokens=max_tokens,
            stop_sequences=stop_sequences,
            **kwargs
        )

