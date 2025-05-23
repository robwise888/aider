"""
Base LLM Provider for the Selfy agent.

This module defines the abstract base class that all LLM providers must implement.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional, Union, Callable

from .response import LLMResponse
from .exceptions import LLMError


class BaseLLMProvider(ABC):
    """
    Abstract base class for LLM providers.

    All LLM providers (Groq, Ollama, etc.) must implement this interface.
    """

    @abstractmethod
    def __init__(self, model_name: str = None, **kwargs):
        """
        Initialize the LLM provider.

        Args:
            model_name: The name of the model to use
            **kwargs: Additional provider-specific arguments
        """
        self.model_name = model_name
        self.is_available = False

    @abstractmethod
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

        This method is primarily for simple completion tasks without explicit history.
        It handles API calls, error handling, and token counting.

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
        pass

    @abstractmethod
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

        This method is for chat-based interactions with explicit message history.
        It handles API calls, error handling, and token counting.

        Args:
            messages: List of message dictionaries, each with 'role' (e.g., 'user', 'assistant', 'system') and 'content' keys
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
        pass

    def count_tokens(self, text: str) -> int:
        """
        Calculate the number of tokens for a given piece of text.

        This method uses the provider's tokenization method to count tokens.
        It's useful for prompt engineering and optimization.

        Args:
            text: The text to tokenize

        Returns:
            Integer token count

        Raises:
            LLMError: If there's an issue during tokenization
        """
        # Default implementation - should be overridden by providers
        # that have access to more accurate tokenization
        from .token_utils import count_tokens
        return count_tokens(text, model=self.model_name)

    def count_message_tokens(self, messages: List[Dict[str, str]]) -> int:
        """
        Calculate the number of tokens for a list of messages.

        This method uses the provider's tokenization method to count tokens.
        It's useful for prompt engineering and optimization.

        Args:
            messages: List of message dictionaries, each with 'role' and 'content' keys

        Returns:
            Integer token count

        Raises:
            LLMError: If there's an issue during tokenization
        """
        # Default implementation - should be overridden by providers
        # that have access to more accurate tokenization
        from .token_utils import count_message_tokens
        return count_message_tokens(messages, model=self.model_name)


