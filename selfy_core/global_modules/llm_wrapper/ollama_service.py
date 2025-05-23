"""
Ollama Windows Service Management Module.

This module provides functions for managing the Ollama service on Windows.
It allows checking, installing, starting, and stopping the Ollama service.
"""

import os
import sys
import subprocess
import logging
import time
import platform
import shutil
import traceback
from typing import Tuple, Optional, Dict, Any

# Set up logging
logger = logging.getLogger(__name__)

# Constants
SERVICE_NAME = "OllamaService2"
NSSM_DOWNLOAD_URL = "https://nssm.cc/release/nssm-2.24.zip"
NSSM_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "nssm")
NSSM_PATH = os.path.join(NSSM_DIR, "nssm.exe")

def is_admin() -> bool:
    """
    Check if the current process has administrator privileges.

    Returns:
        bool: True if running as administrator, False otherwise
    """
    if platform.system() != 'Windows':
        return False

    try:
        import ctypes
        return ctypes.windll.shell32.IsUserAnAdmin() != 0
    except Exception as e:
        logger.warning(f"Failed to check admin status: {e}")
        return False

def get_ollama_path() -> Optional[str]:
    """
    Get the path to the Ollama executable.

    Returns:
        Optional[str]: Path to the Ollama executable, or None if not found
    """
    try:
        # Use appropriate command based on OS
        cmd = "where" if platform.system() == 'Windows' else "which"
        ollama_path = subprocess.check_output([cmd, "ollama"], text=True).strip().split('\n')[0]
        return ollama_path
    except Exception as e:
        logger.warning(f"Failed to find Ollama executable: {e}")
        return None

def is_nssm_available() -> bool:
    """
    Check if NSSM is available.

    Returns:
        bool: True if NSSM is available, False otherwise
    """
    return os.path.exists(NSSM_PATH)

def download_nssm() -> bool:
    """
    Download and extract NSSM.

    Returns:
        bool: True if successful, False otherwise
    """
    import requests
    import zipfile
    import io

    try:
        # Create directory if it doesn't exist
        os.makedirs(NSSM_DIR, exist_ok=True)

        # Download NSSM
        logger.info(f"Downloading NSSM from {NSSM_DOWNLOAD_URL}...")
        response = requests.get(NSSM_DOWNLOAD_URL)
        response.raise_for_status()

        # Extract NSSM
        logger.info("Extracting NSSM...")
        with zipfile.ZipFile(io.BytesIO(response.content)) as zip_file:
            # Find the nssm.exe file in the zip
            for file in zip_file.namelist():
                if file.endswith("nssm.exe") and "win64" in file:
                    # Extract the file
                    with zip_file.open(file) as source, open(NSSM_PATH, "wb") as target:
                        shutil.copyfileobj(source, target)
                    logger.info(f"NSSM extracted to {NSSM_PATH}")
                    return True

        logger.error("Failed to find nssm.exe in the downloaded zip file")
        return False
    except Exception as e:
        logger.error(f"Failed to download and extract NSSM: {e}")
        return False

def ensure_nssm_available() -> bool:
    """
    Ensure NSSM is available, downloading it if necessary.

    Returns:
        bool: True if NSSM is available, False otherwise
    """
    if is_nssm_available():
        return True

    return download_nssm()

def is_service_installed() -> bool:
    """
    Check if the Ollama service is installed.

    Returns:
        bool: True if the service is installed, False otherwise
    """
    try:
        result = subprocess.run(
            ["sc", "query", SERVICE_NAME],
            capture_output=True,
            text=True,
            creationflags=subprocess.CREATE_NO_WINDOW
        )
        return result.returncode == 0
    except Exception as e:
        logger.error(f"Failed to check if service is installed: {e}")
        return False

def is_service_running() -> bool:
    """
    Check if the Ollama service is running.

    Returns:
        bool: True if the service is running, False otherwise
    """
    try:
        result = subprocess.run(
            ["sc", "query", SERVICE_NAME],
            capture_output=True,
            text=True,
            creationflags=subprocess.CREATE_NO_WINDOW
        )
        return result.returncode == 0 and "RUNNING" in result.stdout
    except Exception as e:
        logger.error(f"Failed to check if service is running: {e}")
        return False

def install_service() -> bool:
    """
    Install the Ollama service using NSSM.

    Returns:
        bool: True if successful, False otherwise
    """
    if not is_admin():
        logger.error("Administrator privileges required to install service")
        return False

    if not ensure_nssm_available():
        logger.error("Failed to ensure NSSM is available")
        return False

    ollama_path = get_ollama_path()
    if not ollama_path:
        logger.error("Failed to find Ollama executable")
        return False

    logger.info(f"Found Ollama executable at: {ollama_path}")

    # Check if there are any existing Ollama processes
    try:
        import psutil
        for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
            try:
                if proc.info['name'] and 'ollama' in proc.info['name'].lower():
                    logger.warning(f"Found existing Ollama process: PID={proc.info['pid']}, Name={proc.info['name']}")
                    logger.warning("This might conflict with the service. Consider terminating it first.")
            except Exception:
                pass
    except ImportError:
        logger.warning("psutil not available, skipping process check")

    try:
        # Check if the service already exists
        if is_service_installed():
            logger.info("Ollama service is already installed, removing it first...")
            uninstall_service()

        # Install the service
        logger.info(f"Installing Ollama service using NSSM...")
        logger.info(f"NSSM path: {NSSM_PATH}")
        logger.info(f"Service name: {SERVICE_NAME}")
        logger.info(f"Ollama path: {ollama_path}")

        # Create the service
        install_cmd = [NSSM_PATH, "install", SERVICE_NAME, ollama_path, "serve"]
        logger.info(f"Running install command: {' '.join(install_cmd)}")

        result = subprocess.run(
            install_cmd,
            capture_output=True,
            text=True,
            creationflags=subprocess.CREATE_NO_WINDOW
        )

        logger.info(f"Install command output: {result.stdout}")
        if result.stderr:
            logger.error(f"Install command error: {result.stderr}")

        if result.returncode != 0:
            logger.error(f"Failed to install service with return code {result.returncode}")
            return False

        # Set service description
        logger.info("Setting service description...")
        desc_result = subprocess.run(
            [NSSM_PATH, "set", SERVICE_NAME, "Description", "Ollama API server for local LLM inference"],
            capture_output=True,
            text=True,
            creationflags=subprocess.CREATE_NO_WINDOW
        )

        logger.info(f"Description command output: {desc_result.stdout}")
        if desc_result.stderr:
            logger.warning(f"Description command error: {desc_result.stderr}")

        # Set environment variables
        logger.info("Setting service environment variables...")
        env_result = subprocess.run(
            [NSSM_PATH, "set", SERVICE_NAME, "AppEnvironmentExtra",
             "OLLAMA_NO_CONSOLE=1\nOLLAMA_LOG_LEVEL=info\nOLLAMA_USE_GPU=1\nCUDA_VISIBLE_DEVICES=0"],
            capture_output=True,
            text=True,
            creationflags=subprocess.CREATE_NO_WINDOW
        )

        logger.info(f"Environment command output: {env_result.stdout}")
        if env_result.stderr:
            logger.warning(f"Environment command error: {env_result.stderr}")

        # Set startup type to automatic
        logger.info("Setting service startup type...")
        start_result = subprocess.run(
            [NSSM_PATH, "set", SERVICE_NAME, "Start", "SERVICE_AUTO_START"],
            capture_output=True,
            text=True,
            creationflags=subprocess.CREATE_NO_WINDOW
        )

        logger.info(f"Startup command output: {start_result.stdout}")
        if start_result.stderr:
            logger.warning(f"Startup command error: {start_result.stderr}")

        # Set failure actions
        logger.info("Setting service failure actions...")
        failure_result = subprocess.run(
            [NSSM_PATH, "set", SERVICE_NAME, "AppExit", "Default", "Restart"],
            capture_output=True,
            text=True,
            creationflags=subprocess.CREATE_NO_WINDOW
        )

        logger.info(f"Failure command output: {failure_result.stdout}")
        if failure_result.stderr:
            logger.warning(f"Failure command error: {failure_result.stderr}")

        # Set working directory
        ollama_dir = os.path.dirname(ollama_path)
        logger.info(f"Setting service working directory to: {ollama_dir}")
        dir_result = subprocess.run(
            [NSSM_PATH, "set", SERVICE_NAME, "AppDirectory", ollama_dir],
            capture_output=True,
            text=True,
            creationflags=subprocess.CREATE_NO_WINDOW
        )

        logger.info(f"Directory command output: {dir_result.stdout}")
        if dir_result.stderr:
            logger.warning(f"Directory command error: {dir_result.stderr}")

        logger.info("Ollama service installed successfully")
        return True
    except Exception as e:
        logger.error(f"Failed to install service: {e}")
        logger.error(f"Error details: {traceback.format_exc()}")
        return False

def start_service() -> bool:
    """
    Start the Ollama service.

    Returns:
        bool: True if successful, False otherwise
    """
    if not is_service_installed():
        logger.error("Ollama service is not installed")
        return False

    try:
        # Check if there are any existing Ollama processes
        try:
            import psutil
            for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
                try:
                    if proc.info['name'] and 'ollama' in proc.info['name'].lower():
                        logger.warning(f"Found existing Ollama process: PID={proc.info['pid']}, Name={proc.info['name']}")
                        logger.warning("This might conflict with the service. Consider terminating it first.")
                except Exception:
                    pass
        except ImportError:
            logger.warning("psutil not available, skipping process check")

        # Get service status before starting
        status_before = subprocess.run(
            ["sc", "query", SERVICE_NAME],
            capture_output=True,
            text=True,
            creationflags=subprocess.CREATE_NO_WINDOW
        )
        logger.info(f"Service status before starting: {status_before.stdout}")

        # Start the service
        logger.info("Starting Ollama service...")
        result = subprocess.run(
            ["sc", "start", SERVICE_NAME],
            capture_output=True,
            text=True,
            creationflags=subprocess.CREATE_NO_WINDOW
        )

        logger.info(f"SC start command output: {result.stdout}")
        if result.stderr:
            logger.error(f"SC start command error: {result.stderr}")

        if result.returncode != 0:
            logger.error(f"Failed to start service with return code {result.returncode}")

            # Check Windows service error codes
            if "1053" in result.stdout or "1053" in result.stderr:
                logger.error("Error 1053: The service did not respond to the start or control request in a timely fashion")
                logger.error("This often means the service executable path is incorrect or the service crashed on startup")
            elif "1056" in result.stdout or "1056" in result.stderr:
                logger.error("Error 1056: An instance of the service is already running")
            elif "5" in result.stdout or "5" in result.stderr:
                logger.error("Error 5: Access denied. Make sure you're running with administrator privileges")

            return False

        # Wait for the service to start
        max_retries = 10
        retry_delay = 2  # seconds

        for attempt in range(1, max_retries + 1):
            logger.info(f"Checking if Ollama service is running (attempt {attempt}/{max_retries})...")

            # Get detailed service status
            status = subprocess.run(
                ["sc", "query", SERVICE_NAME],
                capture_output=True,
                text=True,
                creationflags=subprocess.CREATE_NO_WINDOW
            )
            logger.info(f"Service status on attempt {attempt}: {status.stdout}")

            # Check if the service is running
            if is_service_running():
                logger.info("Ollama service started successfully")

                # Check if the API is responsive
                try:
                    import requests
                    response = requests.get("http://localhost:11434/api/tags", timeout=5)
                    if response.status_code == 200:
                        logger.info("Ollama API is responsive")
                    else:
                        logger.warning(f"Ollama API returned status code {response.status_code}")
                except Exception as e:
                    logger.warning(f"Failed to connect to Ollama API: {e}")

                return True

            logger.info(f"Ollama service not running yet, retrying in {retry_delay}s...")
            time.sleep(retry_delay)

        logger.warning("Failed to start Ollama service after maximum retries")

        # Get final service status
        final_status = subprocess.run(
            ["sc", "query", SERVICE_NAME],
            capture_output=True,
            text=True,
            creationflags=subprocess.CREATE_NO_WINDOW
        )
        logger.info(f"Final service status: {final_status.stdout}")

        return False
    except Exception as e:
        logger.error(f"Failed to start service: {e}")
        logger.error(f"Error details: {traceback.format_exc()}")
        return False

def stop_service() -> bool:
    """
    Stop the Ollama service.

    Returns:
        bool: True if successful, False otherwise
    """
    if not is_service_installed():
        logger.error("Ollama service is not installed")
        return False

    try:
        # Stop the service
        logger.info("Stopping Ollama service...")
        result = subprocess.run(
            ["sc", "stop", SERVICE_NAME],
            capture_output=True,
            text=True,
            creationflags=subprocess.CREATE_NO_WINDOW
        )

        if result.returncode != 0:
            logger.error(f"Failed to stop service: {result.stderr}")
            return False

        logger.info("Ollama service stopped successfully")
        return True
    except Exception as e:
        logger.error(f"Failed to stop service: {e}")
        return False

def uninstall_service() -> bool:
    """
    Uninstall the Ollama service.

    Returns:
        bool: True if successful, False otherwise
    """
    if not is_admin():
        logger.error("Administrator privileges required to uninstall service")
        return False

    if not is_service_installed():
        logger.info("Ollama service is not installed")
        return True

    if not ensure_nssm_available():
        logger.error("Failed to ensure NSSM is available")
        return False

    try:
        # Stop the service first
        if is_service_running():
            stop_service()

        # Uninstall the service
        logger.info("Uninstalling Ollama service...")
        result = subprocess.run(
            [NSSM_PATH, "remove", SERVICE_NAME, "confirm"],
            capture_output=True,
            text=True,
            creationflags=subprocess.CREATE_NO_WINDOW
        )

        if result.returncode != 0:
            logger.error(f"Failed to uninstall service: {result.stderr}")
            return False

        logger.info("Ollama service uninstalled successfully")
        return True
    except Exception as e:
        logger.error(f"Failed to uninstall service: {e}")
        return False

def ensure_ollama_service() -> bool:
    """
    Ensure the Ollama service is installed and running.

    Returns:
        bool: True if the service is running, False otherwise
    """
    # Check if the service is already running
    if is_service_running():
        logger.info("Ollama service is already running")
        return True

    # Check if the service is installed but not running
    if is_service_installed():
        logger.info("Ollama service is installed but not running, starting it...")
        return start_service()

    # Service is not installed, check if we have admin privileges
    if not is_admin():
        logger.warning("Administrator privileges required to install Ollama service")
        logger.warning("Please run the install_ollama_service.py script as administrator")
        return False

    # Install and start the service
    logger.info("Ollama service is not installed, installing it...")
    if install_service():
        logger.info("Ollama service installed, starting it...")
        return start_service()

    return False

def check_ollama_api() -> bool:
    """
    Check if the Ollama API is responsive.

    Returns:
        bool: True if the API is responsive, False otherwise
    """
    try:
        import requests
        response = requests.get("http://localhost:11434/api/tags", timeout=5)
        return response.status_code == 200
    except Exception as e:
        logger.debug(f"Failed to connect to Ollama API: {e}")
        return False
