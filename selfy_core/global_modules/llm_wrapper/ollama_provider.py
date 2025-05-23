"""
Ollama Provider for the LLM Wrapper module.

This module provides the OllamaProvider class, which implements the BaseLLMProvider
interface for the local Ollama LLM service.

It supports running Ollama as a Windows service to prevent blank terminal windows.
"""

import time
import json
import logging
import subprocess
import platform
import os
from typing import Dict, List, Any, Optional, Union

# Patch subprocess.Popen on Windows to always use CREATE_NO_WINDOW flag
if platform.system() == 'Windows':
    # Store the original Popen function
    original_popen = subprocess.Popen

    # Define Windows-specific constants for debugging
    CREATE_NO_WINDOW = 0x08000000
    DETACHED_PROCESS = 0x00000008
    CREATE_NEW_PROCESS_GROUP = 0x00000200

    # Define a patched version that always adds CREATE_NO_WINDOW flag
    def patched_popen(*args, **kwargs):
        logger = logging.getLogger(__name__)

        # Get the command being executed
        cmd = args[0] if args else kwargs.get('args', "Unknown command")
        cmd_str = cmd if isinstance(cmd, str) else " ".join(cmd)

        # Log detailed information before creating the process
        logger.debug(f"[DEBUG] Creating process: {cmd_str}")
        logger.debug(f"[DEBUG] Original creationflags: 0x{kwargs.get('creationflags', 0):08x}")

        # Log environment variables if present
        if 'env' in kwargs and kwargs['env']:
            env = kwargs['env']
            logger.debug("[DEBUG] Environment variables:")
            for key in ['OLLAMA_LOG_LEVEL', 'CUDA_VISIBLE_DEVICES', 'OLLAMA_USE_GPU', 'OLLAMA_NO_CONSOLE']:
                if key in env:
                    logger.debug(f"[DEBUG]   {key}={env[key]}")

        # Always set CREATE_NO_WINDOW flag (0x08000000) to prevent visible terminal window
        kwargs['creationflags'] = kwargs.get('creationflags', 0) | CREATE_NO_WINDOW

        # Log the final creation flags
        logger.debug(f"[DEBUG] Final creationflags: 0x{kwargs.get('creationflags', 0):08x}")
        logger.debug(f"[DEBUG] CREATE_NO_WINDOW=0x{CREATE_NO_WINDOW:08x}, DETACHED_PROCESS=0x{DETACHED_PROCESS:08x}")

        # Create the process
        logger.debug("[DEBUG] Calling original subprocess.Popen...")
        process = original_popen(*args, **kwargs)

        # Log information about the created process
        logger.debug(f"[DEBUG] Process created: PID={process.pid}")

        # Try to get more information about the process
        try:
            import psutil
            proc = psutil.Process(process.pid)
            logger.debug(f"[DEBUG] Process details: name={proc.name()}, exe={proc.exe()}")
            logger.debug(f"[DEBUG] Process parent: {proc.parent().pid if proc.parent() else 'None'}")
            logger.debug(f"[DEBUG] Process children: {[p.pid for p in proc.children()]}")
        except Exception as e:
            logger.debug(f"[DEBUG] Error getting process details: {e}")

        return process

    # Replace the Popen function in subprocess module
    subprocess.Popen = patched_popen

    # Log that we've patched the function
    logging.getLogger(__name__).info("Patched subprocess.Popen with enhanced debugging for Windows")

from selfy_core.global_modules.config import get as config_get
from .base_llm_provider import BaseLLMProvider
from .response import LLMResponse
from .exceptions import (
    LLMError, LLMConfigurationError, LLMConnectionError, LLMAPIError,
    LLMRateLimitError, LLMAuthenticationError, LLMTimeoutError, LLMTokenLimitError
)
from .token_utils import count_tokens
from .logging_helpers import log_llm_call

# Set up logging
logger = logging.getLogger(__name__)

# Try to import the Ollama client
try:
    import ollama
    OLLAMA_CLIENT_AVAILABLE = True
except ImportError:
    logger.warning("'ollama' library not installed. Ollama LLM features disabled.")
    OLLAMA_CLIENT_AVAILABLE = False

# Global variables for Ollama server management
import os
import atexit
import traceback
import threading

# Global variables for Ollama server management
OLLAMA_SERVER_PROCESS = None
OLLAMA_SERVER_STARTED_BY_US = False
OLLAMA_SERVER_CHECKED = False

# Cache for server status and model loading
SERVER_STATUS_CACHE = {
    "status": None,
    "timestamp": 0,
    "ttl": 300  # 5 minutes TTL
}
LOADED_MODELS = set()  # Track which models have been loaded

def check_gpu_availability():
    """Check if GPU is available for Ollama"""
    try:
        # Check for NVIDIA GPUs
        import subprocess
        try:
            # Use our patched subprocess to avoid blank terminal
            output = subprocess.check_output(["nvidia-smi", "-L"], stderr=subprocess.STDOUT, text=True)
            if "GPU" in output:
                logger.info(f"NVIDIA GPU detected: {output.strip()}")
                return True
        except (subprocess.SubprocessError, FileNotFoundError):
            pass

        return False
    except Exception as e:
        logger.warning(f"Error checking GPU availability: {e}")
        return False

def is_ollama_using_gpu():
    """Check if Ollama is using GPU and log detailed information."""
    try:
        # Method 1: Check for NVIDIA GPU processes
        import subprocess
        try:
            output = subprocess.check_output(
                ["nvidia-smi", "--query-compute-apps=pid,process_name,used_memory", "--format=csv"],
                stderr=subprocess.STDOUT,
                text=True,
                creationflags=subprocess.CREATE_NO_WINDOW
            )
            ollama_processes = [line for line in output.split('\n') if 'ollama' in line.lower()]
            if ollama_processes:
                logger.info("NVIDIA-SMI CONFIRMS: Ollama is using GPU:")
                for process in ollama_processes:
                    logger.info(f"  GPU Process: {process}")
                return True
        except Exception as e:
            logger.debug(f"Could not check NVIDIA GPU processes: {e}")

        # Method 2: Check CUDA memory usage
        try:
            import torch
            if torch.cuda.is_available():
                # Get current CUDA memory usage
                allocated = torch.cuda.memory_allocated() / (1024 ** 2)  # MB
                reserved = torch.cuda.memory_reserved() / (1024 ** 2)    # MB
                logger.info(f"CUDA Memory: Allocated={allocated:.2f}MB, Reserved={reserved:.2f}MB")

                # If memory is being used, Ollama might be using GPU
                if allocated > 10:  # Arbitrary threshold
                    logger.info("CUDA MEMORY USAGE CONFIRMS: Ollama likely using GPU")
                    return True
        except Exception as e:
            logger.debug(f"Could not check CUDA memory: {e}")

        # Method 3: Check Ollama API for model info
        try:
            import requests
            response = requests.get("http://localhost:11434/api/tags")
            if response.status_code == 200:
                models = response.json().get("models", [])
                for model in models:
                    if "details" in model and "quantization_level" in model["details"]:
                        quant = model["details"]["quantization_level"]
                        logger.info(f"Model {model['name']} using quantization {quant}")
                        # Q4_K_M and Q5_K_M typically use GPU
                        if "_K_M" in quant:
                            logger.info(f"QUANTIZATION CONFIRMS: Model {model['name']} likely using GPU (K_M quantization)")
                            return True
        except Exception as e:
            logger.debug(f"Could not check Ollama API: {e}")

        # Method 4: Check environment variables
        try:
            if os.environ.get("OLLAMA_USE_GPU") == "1" and os.environ.get("CUDA_VISIBLE_DEVICES") is not None:
                logger.info("ENVIRONMENT VARIABLES SUGGEST: Ollama configured to use GPU")
                logger.info(f"  OLLAMA_USE_GPU={os.environ.get('OLLAMA_USE_GPU')}")
                logger.info(f"  CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')}")
                return True
        except Exception as e:
            logger.debug(f"Could not check environment variables: {e}")

        logger.info("NO CONFIRMATION: Could not confirm Ollama is using GPU")
        return False
    except Exception as e:
        logger.error(f"Error checking GPU usage: {e}")
        return False

def check_ollama_server(max_retries=3, retry_delay=1.0, force_check=False):
    """
    Check if Ollama server is running and responsive with retries and caching

    Args:
        max_retries: Maximum number of retries
        retry_delay: Delay between retries in seconds
        force_check: Force a fresh check ignoring the cache

    Returns:
        bool: True if server is running, False otherwise
    """
    global SERVER_STATUS_CACHE

    # Check cache first unless force_check is True
    now = time.time()
    cache_expired = (now - SERVER_STATUS_CACHE["timestamp"]) > SERVER_STATUS_CACHE["ttl"]

    if not force_check and SERVER_STATUS_CACHE["status"] is not None and not cache_expired:
        logger.debug(f"Using cached Ollama server status: {SERVER_STATUS_CACHE['status']} (cache age: {now - SERVER_STATUS_CACHE['timestamp']:.1f}s)")
        return SERVER_STATUS_CACHE["status"]

    # If we get here, we need to check the server status
    logger.debug(f"Checking Ollama server status (force_check={force_check}, cache_expired={cache_expired})")

    for attempt in range(max_retries):
        try:
            start_time = time.time()

            # Try to connect to the Ollama API
            import requests
            response = requests.get("http://localhost:11434/api/tags", timeout=5)

            if response.status_code == 200:
                duration = time.time() - start_time
                logger.debug(f"Ollama server is running (responded in {duration:.2f}s on attempt {attempt+1}/{max_retries})")

                # Update cache
                SERVER_STATUS_CACHE["status"] = True
                SERVER_STATUS_CACHE["timestamp"] = now

                return True
            else:
                logger.debug(f"Ollama server returned status code {response.status_code} on attempt {attempt+1}/{max_retries}")

                if attempt < max_retries - 1:
                    logger.debug(f"Retrying in {retry_delay} seconds...")
                    time.sleep(retry_delay)
                else:
                    logger.warning(f"Ollama server check failed after {max_retries} attempts")

                    # Update cache
                    SERVER_STATUS_CACHE["status"] = False
                    SERVER_STATUS_CACHE["timestamp"] = now

                    return False

        except Exception as e:
            if attempt < max_retries - 1:
                logger.debug(f"Ollama server check failed on attempt {attempt+1}/{max_retries}: {e}")
                logger.debug(f"Retrying in {retry_delay} seconds...")
                time.sleep(retry_delay)
            else:
                logger.warning(f"Ollama server check failed after {max_retries} attempts: {e}")

                # Update cache
                SERVER_STATUS_CACHE["status"] = False
                SERVER_STATUS_CACHE["timestamp"] = now

                return False

    # Update cache
    SERVER_STATUS_CACHE["status"] = False
    SERVER_STATUS_CACHE["timestamp"] = now

    return False

def capture_ollama_output(process):
    """Capture and log output from the Ollama server process"""
    try:
        for line in iter(process.stdout.readline, ''):
            if not line:
                break
            # Log the Ollama server output
            logger.debug(f"Ollama server: {line.strip()}")
    except Exception as e:
        logger.warning(f"Error capturing Ollama output: {e}")

def ensure_ollama_server(use_gpu=True):
    """
    Ensure Ollama server is running with GPU support if available.
    This function implements a singleton pattern to ensure only one Ollama server is started
    and reused across multiple calls.

    On Windows, it will try to use the Ollama service if available to prevent blank terminal windows.

    Args:
        use_gpu: Whether to enable GPU acceleration if available

    Returns:
        bool: True if server is running or was started, False otherwise
    """
    global OLLAMA_SERVER_PROCESS, OLLAMA_SERVER_STARTED_BY_US, OLLAMA_SERVER_CHECKED, SERVER_STATUS_CACHE

    # If we've already checked and found a running server, return immediately
    if OLLAMA_SERVER_CHECKED and OLLAMA_SERVER_PROCESS is not None:
        logger.debug("Using existing Ollama server process")
        return True

    # First check if server is already running (using cached status if available)
    if check_ollama_server():
        logger.info("Ollama server is already running (external process)")
        OLLAMA_SERVER_CHECKED = True
        OLLAMA_SERVER_STARTED_BY_US = False
        return True

    # On Windows, try to use the Ollama service
    if platform.system() == 'Windows':
        try:
            # Import the service module
            from .ollama_service import (
                is_service_installed, is_service_running,
                start_service, ensure_ollama_service, check_ollama_api
            )

            # Check if the service is installed
            if is_service_installed():
                logger.info("Ollama service is installed, checking if it's running...")

                # Check if the service is running
                if is_service_running():
                    logger.info("Ollama service is already running")

                    # Check if the API is responsive
                    if check_ollama_api():
                        logger.info("Ollama API is responsive")
                        OLLAMA_SERVER_CHECKED = True
                        OLLAMA_SERVER_STARTED_BY_US = False
                        return True
                    else:
                        logger.warning("Ollama service is running but API is not responsive")
                        # Continue with normal startup
                else:
                    logger.info("Ollama service is installed but not running, starting it...")

                    # Start the service
                    if start_service():
                        logger.info("Ollama service started successfully")

                        # Check if the API is responsive
                        if check_ollama_api():
                            logger.info("Ollama API is responsive")
                            OLLAMA_SERVER_CHECKED = True
                            OLLAMA_SERVER_STARTED_BY_US = False
                            return True
                        else:
                            logger.warning("Ollama service started but API is not responsive")
                            # Continue with normal startup
                    else:
                        logger.warning("Failed to start Ollama service")
                        # Continue with normal startup
            else:
                logger.info("Ollama service is not installed, using normal startup")
                # Continue with normal startup
        except ImportError:
            logger.warning("Ollama service module not available, using normal startup")
            # Continue with normal startup
        except Exception as e:
            logger.warning(f"Error checking Ollama service: {e}")
            # Continue with normal startup

    # If we already have a process but it's not responding, clean it up
    if OLLAMA_SERVER_PROCESS is not None:
        logger.warning("Existing Ollama process is not responding, cleaning up")
        try:
            OLLAMA_SERVER_PROCESS.terminate()
            OLLAMA_SERVER_PROCESS.wait(timeout=5)
        except Exception as e:
            logger.warning(f"Error terminating non-responsive Ollama process: {e}")
            try:
                OLLAMA_SERVER_PROCESS.kill()
            except Exception:
                pass
        OLLAMA_SERVER_PROCESS = None
        OLLAMA_SERVER_STARTED_BY_US = False

    # Always try to use GPU regardless of availability check
    # This ensures we always attempt to use GPU acceleration
    logger.info("Attempting to use GPU acceleration for Ollama")
    gpu_available = True

    # For logging purposes only, check if GPU is actually available
    actual_gpu_available = check_gpu_availability()
    if actual_gpu_available:
        logger.info("GPU is available and will be used for Ollama")
    else:
        logger.info("No GPU detected, but will still attempt to use GPU acceleration")

    # If not running, try to start it
    try:
        import subprocess
        import os
        import threading

        # Set environment variables for verbosity and GPU
        env = os.environ.copy()
        env["OLLAMA_LOG_LEVEL"] = "info"

        # Try setting OLLAMA_NO_CONSOLE environment variable
        env["OLLAMA_NO_CONSOLE"] = "1"
        logger.debug("[DEBUG] Setting OLLAMA_NO_CONSOLE=1 in environment")

        # Enable GPU if available
        if gpu_available:
            env["CUDA_VISIBLE_DEVICES"] = "0"  # Use first GPU
            env["OLLAMA_USE_GPU"] = "1"        # Enable GPU for Ollama

        # Build command
        cmd = ["ollama", "serve"]

        # Log detailed information about the server launch
        logger.debug("[DEBUG] ====== OLLAMA SERVER LAUNCH DETAILS ======")
        logger.debug(f"[DEBUG] Command: {' '.join(cmd)}")
        logger.debug("[DEBUG] Environment variables:")
        for key, value in env.items():
            if key.startswith("OLLAMA_") or key.startswith("CUDA_"):
                logger.debug(f"[DEBUG]   {key}={value}")

        # Log Windows-specific information
        if platform.system() == 'Windows':
            logger.debug(f"[DEBUG] CREATE_NO_WINDOW flag: 0x{CREATE_NO_WINDOW:08x}")
            logger.debug(f"[DEBUG] DETACHED_PROCESS flag: 0x{DETACHED_PROCESS:08x}")
            logger.debug(f"[DEBUG] Will use creationflags: 0x{CREATE_NO_WINDOW:08x}")

        logger.debug("[DEBUG] Current working directory: " + os.getcwd())
        logger.debug("[DEBUG] Ollama executable path: " + subprocess.check_output(["where", "ollama"], text=True).strip())
        logger.debug("[DEBUG] =========================================")

        logger.info(f"Starting Ollama server with command: {' '.join(cmd)}")
        logger.info(f"Environment variables: OLLAMA_LOG_LEVEL={env.get('OLLAMA_LOG_LEVEL', 'not set')}, "
                   f"CUDA_VISIBLE_DEVICES={env.get('CUDA_VISIBLE_DEVICES', 'not set')}, "
                   f"OLLAMA_USE_GPU={env.get('OLLAMA_USE_GPU', 'not set')}, "
                   f"OLLAMA_NO_CONSOLE={env.get('OLLAMA_NO_CONSOLE', 'not set')}")

        # Start Ollama server as a background process
        OLLAMA_SERVER_PROCESS = subprocess.Popen(
            cmd,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,  # Line buffered
            universal_newlines=True
        )

        # Mark that we started this process
        OLLAMA_SERVER_STARTED_BY_US = True
        OLLAMA_SERVER_CHECKED = True

        # Start a thread to capture and log the output
        output_thread = threading.Thread(
            target=capture_ollama_output,
            args=(OLLAMA_SERVER_PROCESS,),
            daemon=True
        )
        output_thread.start()

        # Wait for server to start with retries
        logger.info("Waiting for Ollama server to start...")

        # Use more retries and longer delay for initial startup
        if check_ollama_server(max_retries=5, retry_delay=2.0):
            logger.info("Successfully started Ollama server")
            return True
        else:
            logger.warning("Started Ollama process but server is not responding after multiple retries")
            OLLAMA_SERVER_STARTED_BY_US = False
            return False

    except Exception as e:
        logger.error(f"Failed to start Ollama server: {e}")
        logger.debug(f"Error details: {traceback.format_exc()}")
        OLLAMA_SERVER_STARTED_BY_US = False
        return False

def cleanup_ollama_server():
    """Terminate the Ollama server process when the application exits, but only if we started it"""
    global OLLAMA_SERVER_PROCESS, OLLAMA_SERVER_STARTED_BY_US

    # Only terminate the server if we started it ourselves
    if OLLAMA_SERVER_PROCESS is not None and OLLAMA_SERVER_STARTED_BY_US:
        try:
            logger.info("Shutting down Ollama server that we started...")
            OLLAMA_SERVER_PROCESS.terminate()
            OLLAMA_SERVER_PROCESS.wait(timeout=5)
            logger.info("Ollama server shut down successfully")
        except Exception as e:
            logger.warning(f"Error shutting down Ollama server: {e}")
            try:
                OLLAMA_SERVER_PROCESS.kill()
                logger.info("Ollama server killed")
            except Exception as kill_error:
                logger.error(f"Failed to kill Ollama server: {kill_error}")
    elif OLLAMA_SERVER_PROCESS is not None and not OLLAMA_SERVER_STARTED_BY_US:
        logger.info("Not shutting down Ollama server as it was already running")

    # We don't need to stop the Ollama service if it's running
    # The service will continue running in the background

# Register the cleanup function
atexit.register(cleanup_ollama_server)


class OllamaProvider(BaseLLMProvider):
    """
    Ollama LLM provider implementation.

    This class implements the BaseLLMProvider interface for the local Ollama service.
    It provides methods for generating text and responses using the local Ollama models.
    """

    def __init__(self,
                model_name: str = None,
                host: str = "http://localhost:11434",
                max_retries: int = None,
                retry_delay_seconds: float = None,
                timeout_seconds: int = None,
                **kwargs):
        """
        Initialize the Ollama provider.

        Args:
            model_name: The name of the model to use
            host: The Ollama host URL
            max_retries: Maximum number of retries for API calls
            retry_delay_seconds: Delay between retries in seconds
            timeout_seconds: Timeout for API calls in seconds
            **kwargs: Additional provider-specific arguments
        """
        super().__init__(model_name)

        # Store configuration
        self.host = host

        # Get configuration values from config or use defaults
        self.max_retries = max_retries if max_retries is not None else config_get('llm.ollama.max_retries', 1)
        self.retry_delay_seconds = retry_delay_seconds if retry_delay_seconds is not None else config_get('llm.ollama.retry_delay_seconds', 1.0)
        self.timeout_seconds = timeout_seconds if timeout_seconds is not None else config_get('llm.ollama.timeout_seconds', 30)

        # Use default model if not provided
        if not self.model_name:
            self.model_name = config_get('llm.ollama.default_model', 'llama3:8b')

        # Initialize client
        self.client = None
        self.is_available = False

        # Try to initialize the client
        if OLLAMA_CLIENT_AVAILABLE:
            try:
                # Ensure Ollama server is running
                logger.info("Ensuring Ollama server is running during initialization")
                server_running = ensure_ollama_server()
                if not server_running:
                    logger.warning("Failed to ensure Ollama server is running during initialization")
                    self.is_available = False
                    return

                # Set the host
                logger.info(f"Setting Ollama host to {self.host}")
                ollama.host = self.host

                # Test the connection with retry logic
                max_retries = 3
                retry_count = 0
                connection_success = False

                while retry_count < max_retries and not connection_success:
                    try:
                        # Try to get the list of models
                        logger.info(f"About to call ollama.list() (attempt {retry_count + 1}/{max_retries})")
                        start_time = time.time()
                        models = ollama.list()
                        list_time = time.time() - start_time
                        logger.info(f"ollama.list() completed in {list_time:.2f}s")

                        # If we get here, the connection was successful
                        connection_success = True
                        self.is_available = True
                        logger.info(f"Ollama provider initialized with model {self.model_name}")

                        # Check if the model is available
                        available_models = []

                        # Debug the response structure
                        logger.debug(f"Ollama list response: {models}")

                        # Try different ways to extract model names based on the response structure
                        if isinstance(models, dict) and 'models' in models:
                            for model in models.get('models', []):
                                if isinstance(model, dict) and 'name' in model:
                                    available_models.append(model['name'])

                        # If we couldn't extract models from the response, try a direct API call
                        if not available_models:
                            logger.info(f"No models found in response, trying direct API call")
                            import requests
                            import json

                            try:
                                logger.info(f"Making direct HTTP request to {self.host}/api/tags")
                                start_time = time.time()
                                response = requests.get(f"{self.host}/api/tags", timeout=self.timeout_seconds)
                                api_time = time.time() - start_time
                                logger.info(f"Direct API call completed in {api_time:.2f}s with status {response.status_code}")

                                if response.status_code == 200:
                                    data = response.json()
                                    logger.debug(f"Direct API response: {data}")

                                    if 'models' in data:
                                        for model in data.get('models', []):
                                            if isinstance(model, dict) and 'name' in model:
                                                available_models.append(model['name'])
                            except Exception as api_error:
                                logger.warning(f"Failed to get models via direct API call: {api_error}")

                        logger.info(f"Available Ollama models: {available_models}")

                        # Check if our model is in the list
                        if available_models and self.model_name not in available_models:
                            logger.warning(f"Model {self.model_name} not found in available models: {available_models}")
                            logger.warning(f"You may need to pull the model: ollama pull {self.model_name}")
                        elif not available_models:
                            # If we couldn't get any models but Ollama is running, assume the model is available
                            logger.warning("Could not retrieve model list from Ollama, but service is running.")
                            logger.warning("Assuming model is available, will attempt to use it.")

                        # Check if model is already loaded
                        global LOADED_MODELS
                        if self.model_name in LOADED_MODELS:
                            logger.info(f"Model {self.model_name} is already loaded, skipping pre-loading")
                        else:
                            # Pre-load the model to prevent blank window during queries
                            logger.info(f"Pre-loading Ollama model {self.model_name} to prevent blank window during queries...")
                            logger.debug("[DEBUG] ====== MODEL PRE-LOADING DETAILS ======")
                            logger.debug(f"[DEBUG] Model name: {self.model_name}")
                            logger.debug(f"[DEBUG] Ollama host: {ollama.host}")

                            # Log existing Ollama processes before pre-loading
                            try:
                                import psutil
                                ollama_processes = []
                                for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
                                    try:
                                        proc_info = proc.info
                                        cmdline = " ".join(proc_info['cmdline']) if proc_info['cmdline'] else ""
                                        if 'ollama' in cmdline.lower() or 'llama' in cmdline.lower():
                                            ollama_processes.append({
                                                'pid': proc_info['pid'],
                                                'name': proc_info['name'],
                                                'cmdline': cmdline
                                            })
                                    except Exception:
                                        pass

                                logger.debug(f"[DEBUG] Found {len(ollama_processes)} Ollama-related processes before pre-loading:")
                                for proc in ollama_processes:
                                    logger.debug(f"[DEBUG]   PID: {proc['pid']}, Name: {proc['name']}, Cmdline: {proc['cmdline']}")
                            except Exception as e:
                                logger.debug(f"[DEBUG] Error checking processes before pre-loading: {e}")

                            try:
                                # Make a simple query to force model loading
                                logger.debug("[DEBUG] Making simple query to force model loading...")
                                start_time = time.time()

                                self._generate_local_response(
                                    prompt="Hello",
                                    temperature=0.1,
                                    max_tokens=10
                                )

                                duration = time.time() - start_time
                                logger.debug(f"[DEBUG] Pre-loading completed in {duration:.2f} seconds")
                                logger.info(f"Successfully pre-loaded Ollama model {self.model_name}")

                                # Add model to loaded models set
                                LOADED_MODELS.add(self.model_name)

                                # Log Ollama processes after pre-loading
                                try:
                                    import psutil
                                    ollama_processes = []
                                    for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
                                        try:
                                            proc_info = proc.info
                                            cmdline = " ".join(proc_info['cmdline']) if proc_info['cmdline'] else ""
                                            if 'ollama' in cmdline.lower() or 'llama' in cmdline.lower():
                                                ollama_processes.append({
                                                    'pid': proc_info['pid'],
                                                    'name': proc_info['name'],
                                                    'cmdline': cmdline
                                                })
                                        except Exception:
                                            pass

                                    logger.debug(f"[DEBUG] Found {len(ollama_processes)} Ollama-related processes after pre-loading:")
                                    for proc in ollama_processes:
                                        logger.debug(f"[DEBUG]   PID: {proc['pid']}, Name: {proc['name']}, Cmdline: {proc['cmdline']}")
                                except Exception as e:
                                    logger.debug(f"[DEBUG] Error checking processes after pre-loading: {e}")

                            except Exception as e:
                                logger.warning(f"Failed to pre-load Ollama model: {e}")
                                logger.debug(f"[DEBUG] Pre-loading error details: {traceback.format_exc()}")

                            logger.debug("[DEBUG] =======================================")
                    except Exception as e:
                        retry_count += 1
                        logger.warning(f"Failed to connect to Ollama (attempt {retry_count}/{max_retries}): {e}")
                        if retry_count < max_retries:
                            # Wait before retrying
                            time.sleep(2 * retry_count)  # Exponential backoff

                # If we couldn't connect after all retries, mark as unavailable
                if not connection_success:
                    self.is_available = False
                    logger.error("Ollama client marked as unavailable after multiple connection attempts failed")
            except Exception as e:
                logger.error(f"Failed to initialize Ollama client: {e}")
                logger.error(f"Error details: {traceback.format_exc()}")
                self.is_available = False
        else:
            logger.warning("Ollama client not available. Install the 'ollama' package to use Ollama.")

    def _generate_local_response(self,
                               prompt: str,
                               temperature: float = 0.7,
                               max_tokens: Optional[int] = None,
                               stop_sequences: Optional[List[str]] = None,
                               top_p: Optional[float] = None,
                               top_k: Optional[int] = None) -> LLMResponse:
        """
        Generates a response from the local Ollama model using a simple prompt string.

        Args:
            prompt: The input prompt for the LLM
            temperature: The generation temperature (0.0 to 1.0+)
            max_tokens: Maximum number of tokens to generate (not directly supported by Ollama)
            stop_sequences: Optional list of strings that will stop generation when encountered
            top_p: Optional nucleus sampling parameter (0.0 to 1.0)
            top_k: Optional top-k sampling parameter

        Returns:
            An LLMResponse object containing the result and usage statistics

        Raises:
            LLMError: If there's an issue during generation
        """
        start_time = time.time()

        # Check if client is available
        if not self.is_available:
            msg = "Ollama call abort: Client unavailable/uninitialized."
            logger.error(msg)
            raise LLMConnectionError(msg)

        # Check if Ollama is using GPU and log the result
        logger.info("Checking if Ollama is using GPU before generation...")
        is_using_gpu = is_ollama_using_gpu()
        if is_using_gpu:
            logger.info("CONFIRMED: Ollama is using GPU for this generation")

            # Try to get more detailed GPU information
            try:
                import torch
                if torch.cuda.is_available():
                    logger.info(f"CUDA version: {torch.version.cuda}")
                    logger.info(f"CUDA device: {torch.cuda.get_device_name(0)}")
                    allocated = torch.cuda.memory_allocated() / (1024 ** 2)  # MB
                    reserved = torch.cuda.memory_reserved() / (1024 ** 2)    # MB
                    logger.info(f"CUDA Memory: Allocated={allocated:.2f}MB, Reserved={reserved:.2f}MB")
            except Exception as e:
                logger.debug(f"Could not get detailed GPU information: {e}")

            # Try to get nvidia-smi information
            try:
                import subprocess
                output = subprocess.check_output(
                    ["nvidia-smi", "--query-compute-apps=pid,process_name,used_memory", "--format=csv"],
                    stderr=subprocess.STDOUT,
                    text=True,
                    creationflags=subprocess.CREATE_NO_WINDOW
                )
                logger.info(f"NVIDIA-SMI GPU processes:\n{output}")
            except Exception as e:
                logger.debug(f"Could not get nvidia-smi information: {e}")
        else:
            logger.info("NOTE: Ollama appears to be using CPU for this generation")

            # Log possible reasons
            try:
                import os
                logger.info(f"OLLAMA_USE_GPU environment variable: {os.environ.get('OLLAMA_USE_GPU', 'not set')}")
                logger.info(f"CUDA_VISIBLE_DEVICES environment variable: {os.environ.get('CUDA_VISIBLE_DEVICES', 'not set')}")

                # Check if config.json exists
                config_path = os.path.join(os.path.expanduser("~"), ".ollama", "config.json")
                if os.path.exists(config_path):
                    import json
                    with open(config_path, 'r') as f:
                        config = json.load(f)
                        logger.info(f"Ollama config.json: {config}")
                else:
                    logger.info("Ollama config.json not found")
            except Exception as e:
                logger.debug(f"Could not check Ollama configuration: {e}")

        # Prepare options
        options = {
            'temperature': temperature,
            'num_gpu': 1  # Explicitly request GPU usage
        }

        # Add optional parameters if provided
        if max_tokens is not None:
            options['num_predict'] = max_tokens
        if stop_sequences:
            options['stop'] = stop_sequences
        if top_p is not None:
            options['top_p'] = top_p
        if top_k is not None:
            options['top_k'] = top_k

        # Count prompt tokens
        prompt_tokens = count_tokens(prompt, self.model_name)

        # Make the API call with retries
        retries = 0
        wait_time = self.retry_delay_seconds

        while True:
            try:
                # Using ollama.generate for simpler string in/out
                response = ollama.generate(
                    model=self.model_name,
                    prompt=prompt,
                    stream=False,  # Get the full response at once
                    options=options
                )

                # Calculate duration
                duration = time.time() - start_time

                # Access the text response
                if response and hasattr(response, 'response'):
                    response_text = response.response.strip()
                    logger.info(f"Received response from {self.model_name}: {response_text[:100]}...")
                    logger.debug(f"Full local response: {response_text}")

                    # Extract token usage if available
                    completion_tokens = None
                    total_tokens = None

                    if hasattr(response, 'eval_count'):
                        completion_tokens = response.eval_count
                    if hasattr(response, 'prompt_eval_count'):
                        prompt_tokens = response.prompt_eval_count

                    # If not available, estimate
                    if completion_tokens is None:
                        completion_tokens = count_tokens(response_text, self.model_name)

                    total_tokens = prompt_tokens + completion_tokens

                    # Log the call
                    log_llm_call(
                        logger=logger,
                        provider_name="ollama",
                        model_name=self.model_name,
                        prompt_tokens=prompt_tokens,
                        completion_tokens=completion_tokens,
                        duration=duration
                    )

                    # Create response object
                    return LLMResponse(
                        content=response_text,
                        provider_name="ollama",
                        model_name=self.model_name,
                        prompt_tokens=prompt_tokens,
                        completion_tokens=completion_tokens,
                        total_tokens=total_tokens,
                        finish_reason="stop",
                        metadata={
                            "duration": duration
                        }
                    )
                else:
                    # Empty response
                    logger.warning(f"Empty response from {self.model_name}")

                    # Log the call
                    log_llm_call(
                        logger=logger,
                        provider_name="ollama",
                        model_name=self.model_name,
                        prompt_tokens=prompt_tokens,
                        completion_tokens=0,
                        duration=duration,
                        error="Empty response"
                    )

                    # Create response object with empty content
                    return LLMResponse(
                        content="",
                        provider_name="ollama",
                        model_name=self.model_name,
                        prompt_tokens=prompt_tokens,
                        completion_tokens=0,
                        total_tokens=prompt_tokens,
                        finish_reason="error",
                        metadata={
                            "duration": duration,
                            "error": "Empty response"
                        }
                    )

            except Exception as e:
                # Handle retries
                if retries < self.max_retries:
                    logger.warning(f"Ollama error (Attempt {retries+1}/{self.max_retries+1}): {e}")
                    logger.warning(f"Retrying Ollama in {wait_time:.1f}s...")
                    time.sleep(wait_time)
                    retries += 1
                    wait_time *= 1.5
                    continue

                # Log the error
                duration = time.time() - start_time
                log_llm_call(
                    logger=logger,
                    provider_name="ollama",
                    model_name=self.model_name,
                    prompt_tokens=prompt_tokens,
                    completion_tokens=0,
                    duration=duration,
                    error=str(e)
                )

                # Raise appropriate error
                if "connection" in str(e).lower():
                    raise LLMConnectionError(f"Ollama connection error: {e}")
                elif "timeout" in str(e).lower():
                    raise LLMTimeoutError(f"Ollama timeout error: {e}")
                else:
                    raise LLMAPIError(f"Ollama API error: {e}")

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

        # Ollama doesn't directly support max_tokens, so we log a warning
        if max_tokens is not None:
            logger.warning(f"Ollama doesn't fully support max_tokens parameter. Value may be ignored: {max_tokens}")

        # If system prompt is provided, prepend it to the prompt
        if system_prompt:
            full_prompt = f"{system_prompt}\n\n{prompt}"
        else:
            full_prompt = prompt

        # Generate the response
        return self._generate_local_response(
            prompt=full_prompt,
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
        logger.info(f"Generating chat response with {len(messages)} messages")

        # Get format configuration
        user_prefix = config_get('llm.ollama.format.user_prefix', "User: ")
        assistant_prefix = config_get('llm.ollama.format.assistant_prefix', "Assistant: ")
        system_prefix = config_get('llm.ollama.format.system_prefix', "System: ")

        # Extract system prompt from messages if present
        final_system_prompt = system_prompt
        if not final_system_prompt:
            for msg in messages:
                if msg.get('role') == 'system':
                    final_system_prompt = msg.get('content', '')
                    break

        # Format messages as a conversation
        conversation_text = ""
        for msg in messages:
            role = msg.get('role', '')
            content = msg.get('content', '')

            # Skip system messages (handled separately)
            if role == 'system':
                continue

            # Add appropriate prefix based on role
            if role == 'user':
                conversation_text += f"{user_prefix}{content}\n\n"
            elif role == 'assistant':
                conversation_text += f"{assistant_prefix}{content}\n\n"
            else:
                # Unknown role, just add the content
                conversation_text += f"{content}\n\n"

        # Add the final prompt for the assistant to respond
        assistant_response_prefix = config_get('llm.ollama.format.assistant_response_prefix', "Assistant: ")
        conversation_text += assistant_response_prefix

        # Combine system prompt and conversation
        if final_system_prompt:
            full_prompt = f"{final_system_prompt}\n\n{conversation_text}"
        else:
            full_prompt = conversation_text

        # Generate the response
        return self._generate_local_response(
            prompt=full_prompt,
            temperature=temperature,
            max_tokens=max_tokens,
            stop_sequences=stop_sequences,
            top_p=top_p,
            top_k=top_k
        )

