"""
HTTP Connection Manager for LLM providers.

This module provides a connection manager for HTTP requests to LLM providers,
implementing connection pooling and reuse to improve performance and reliability.
"""

import logging
import time
from typing import Dict, Any, Optional, Tuple, Union
import requests
from requests.adapters import HTTPAdapter
from requests.exceptions import RequestException, ConnectionError, Timeout
from urllib3.util.retry import Retry

from .exceptions import (
    LLMConnectionError,
    LLMTimeoutError,
    LLMAPIError
)

# Configure logger
logger = logging.getLogger(__name__)

# Default retry configuration
DEFAULT_RETRY_CONFIG = {
    'total': 3,
    'backoff_factor': 0.5,
    'status_forcelist': [429, 500, 502, 503, 504],
    'allowed_methods': ['GET', 'POST']
}

# Default timeout configuration (in seconds)
DEFAULT_TIMEOUT_CONFIG = {
    'connect': 5.0,  # Connection timeout
    'read': 60.0,    # Read timeout
}

class HTTPConnectionManager:
    """
    Manages HTTP connections to LLM providers with connection pooling and retry logic.
    
    This class provides a shared session for making HTTP requests to a specific host,
    with configurable retry logic, timeouts, and connection pooling.
    """
    
    # Class-level session cache to maintain one session per base URL
    _sessions: Dict[str, requests.Session] = {}
    
    @classmethod
    def get_session(cls, base_url: str, 
                   retry_config: Optional[Dict[str, Any]] = None,
                   timeout_config: Optional[Dict[str, float]] = None) -> requests.Session:
        """
        Get or create a session for the given base URL with appropriate configuration.
        
        Args:
            base_url: The base URL for the API
            retry_config: Configuration for retry behavior (optional)
            timeout_config: Configuration for timeouts (optional)
            
        Returns:
            A configured requests.Session object
        """
        # Normalize the base URL to ensure consistent keys
        base_url = base_url.rstrip('/')
        
        # Return existing session if available
        if base_url in cls._sessions:
            logger.debug(f"Reusing existing HTTP session for {base_url}")
            return cls._sessions[base_url]
        
        # Create a new session with connection pooling
        logger.info(f"Creating new HTTP session for {base_url}")
        session = requests.Session()
        
        # Configure retry logic
        retry_conf = retry_config or DEFAULT_RETRY_CONFIG
        retry = Retry(
            total=retry_conf.get('total', 3),
            backoff_factor=retry_conf.get('backoff_factor', 0.5),
            status_forcelist=retry_conf.get('status_forcelist', [429, 500, 502, 503, 504]),
            allowed_methods=retry_conf.get('allowed_methods', ['GET', 'POST']),
        )
        
        # Configure connection pooling
        adapter = HTTPAdapter(max_retries=retry, pool_connections=10, pool_maxsize=20)
        session.mount(base_url, adapter)
        
        # Store the session
        cls._sessions[base_url] = session
        return session
    
    @classmethod
    def close_all_sessions(cls):
        """Close all sessions to free resources."""
        for url, session in cls._sessions.items():
            logger.debug(f"Closing HTTP session for {url}")
            session.close()
        cls._sessions.clear()
    
    @classmethod
    def make_request(cls, 
                    method: str, 
                    url: str, 
                    base_url: Optional[str] = None,
                    retry_config: Optional[Dict[str, Any]] = None,
                    timeout_config: Optional[Dict[str, float]] = None,
                    **kwargs) -> Tuple[Any, Dict[str, Any]]:
        """
        Make an HTTP request with connection pooling and proper error handling.
        
        Args:
            method: HTTP method (GET, POST, etc.)
            url: Full URL for the request
            base_url: Base URL for session management (derived from url if not provided)
            retry_config: Configuration for retry behavior
            timeout_config: Configuration for timeouts
            **kwargs: Additional arguments to pass to requests
            
        Returns:
            Tuple of (response_data, response_metadata)
            
        Raises:
            LLMConnectionError: For connection issues
            LLMTimeoutError: For timeout issues
            LLMAPIError: For API errors
        """
        # Determine base URL if not provided
        if not base_url:
            from urllib.parse import urlparse
            parsed = urlparse(url)
            base_url = f"{parsed.scheme}://{parsed.netloc}"
        
        # Get session
        session = cls.get_session(base_url, retry_config, timeout_config)
        
        # Set default timeout
        timeout = kwargs.pop('timeout', None)
        if timeout is None:
            timeout_conf = timeout_config or DEFAULT_TIMEOUT_CONFIG
            timeout = (timeout_conf.get('connect', 5.0), timeout_conf.get('read', 60.0))
        
        # Track request timing
        start_time = time.time()
        response_metadata = {
            'request_time': start_time,
            'duration': 0,
            'status_code': None,
            'headers': None,
        }
        
        try:
            logger.debug(f"Making {method} request to {url}")
            response = session.request(method=method, url=url, timeout=timeout, **kwargs)
            
            # Update metadata
            response_metadata['duration'] = time.time() - start_time
            response_metadata['status_code'] = response.status_code
            response_metadata['headers'] = dict(response.headers)
            
            # Raise for HTTP errors
            response.raise_for_status()
            
            # Parse response
            if 'application/json' in response.headers.get('Content-Type', ''):
                return response.json(), response_metadata
            return response.text, response_metadata
            
        except ConnectionError as e:
            duration = time.time() - start_time
            logger.error(f"Connection error after {duration:.2f}s: {str(e)}")
            raise LLMConnectionError(f"Failed to connect to {base_url}: {str(e)}")
            
        except Timeout as e:
            duration = time.time() - start_time
            logger.error(f"Request timeout after {duration:.2f}s: {str(e)}")
            raise LLMTimeoutError(f"Request to {url} timed out after {duration:.2f}s")
            
        except RequestException as e:
            duration = time.time() - start_time
            response_metadata['duration'] = duration
            
            if hasattr(e, 'response') and e.response is not None:
                status_code = e.response.status_code
                response_metadata['status_code'] = status_code
                response_metadata['headers'] = dict(e.response.headers)
                
                # Try to get response content
                try:
                    content = e.response.json() if 'application/json' in e.response.headers.get('Content-Type', '') else e.response.text
                except:
                    content = e.response.text
                
                logger.error(f"API error ({status_code}) after {duration:.2f}s: {content}")
                raise LLMAPIError(f"API error: {content}", status_code=status_code, response=content)
            
            logger.error(f"Request error after {duration:.2f}s: {str(e)}")
            raise LLMAPIError(f"Request error: {str(e)}")
