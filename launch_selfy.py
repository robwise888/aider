#!/usr/bin/env python
"""
Launcher script for Selfy that ensures the Ollama server is running before starting the application.

This script:
1. Checks if the Ollama server is running
2. If not, starts it and waits for it to be ready
3. Then launches the main Selfy application
"""

import os
import sys
import time
import subprocess
import logging
import argparse
import platform
import requests
from typing import List  # Optional is not used

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("selfy_launcher")

def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Selfy Launcher")
    parser.add_argument(
        "--config",
        help="Path to configuration file"
    )
    interactive_group = parser.add_mutually_exclusive_group()
    interactive_group.add_argument(
        "--interactive",
        action="store_true",
        help="Run in interactive mode (default on Windows)"
    )
    interactive_group.add_argument(
        "--no-interactive",
        action="store_true",
        help="Don't run in interactive mode"
    )
    parser.add_argument(
        "--skip-ollama-check",
        action="store_true",
        help="Skip checking and starting Ollama server"
    )

    # Add service-related arguments
    service_group = parser.add_argument_group("Ollama Service Options")
    service_usage = service_group.add_mutually_exclusive_group()
    service_usage.add_argument(
        "--use-service",
        action="store_true",
        help="Use Ollama Windows service if available (default on Windows)"
    )
    service_usage.add_argument(
        "--no-use-service",
        action="store_true",
        help="Don't use Ollama Windows service even if available"
    )
    service_group.add_argument(
        "--service-status",
        action="store_true",
        help="Check the status of the Ollama Windows service and exit"
    )
    service_group.add_argument(
        "--service-install",
        action="store_true",
        help="Install Ollama as a Windows service (requires admin privileges)"
    )
    service_group.add_argument(
        "--service-uninstall",
        action="store_true",
        help="Uninstall the Ollama Windows service (requires admin privileges)"
    )
    service_group.add_argument(
        "--service-start",
        action="store_true",
        help="Start the Ollama Windows service"
    )
    service_group.add_argument(
        "--service-stop",
        action="store_true",
        help="Stop the Ollama Windows service"
    )

    return parser.parse_args()

def is_ollama_running() -> bool:
    """
    Check if the Ollama server is running and responsive.

    Returns:
        bool: True if the server is running, False otherwise
    """
    try:
        response = requests.get("http://localhost:11434/api/tags", timeout=5)
        return response.status_code == 200
    except requests.exceptions.RequestException:
        return False

def start_ollama_server() -> bool:
    """
    Start the Ollama server.

    Returns:
        bool: True if the server was started successfully, False otherwise
    """
    try:
        logger.info("Starting Ollama server...")

        # Define constants and variables
        CREATE_NO_WINDOW = 0x08000000
        DETACHED_PROCESS = 0x00000008
        CREATE_NEW_PROCESS_GROUP = 0x00000200
        env_vars = os.environ.copy()

        # Log detailed information about the launch
        logger.info("[DEBUG] ====== LAUNCHER OLLAMA SERVER START DETAILS ======")
        logger.info(f"[DEBUG] OS: {os.name}")
        logger.info(f"[DEBUG] Command: ollama serve")

        if os.name == 'nt':  # Windows
            logger.info(f"[DEBUG] CREATE_NO_WINDOW flag: 0x{CREATE_NO_WINDOW:08x}")
            logger.info(f"[DEBUG] DETACHED_PROCESS flag: 0x{DETACHED_PROCESS:08x}")
            logger.info(f"[DEBUG] CREATE_NEW_PROCESS_GROUP flag: 0x{CREATE_NEW_PROCESS_GROUP:08x}")
            logger.info(f"[DEBUG] Combined flags: 0x{DETACHED_PROCESS | CREATE_NEW_PROCESS_GROUP | CREATE_NO_WINDOW:08x}")

        # Log environment variables
        logger.info("[DEBUG] Environment variables:")
        for key, value in env_vars.items():
            if key.startswith("OLLAMA_") or key.startswith("CUDA_"):
                logger.info(f"[DEBUG]   {key}={value}")

        # Try setting OLLAMA_NO_CONSOLE environment variable
        env_vars["OLLAMA_NO_CONSOLE"] = "1"
        logger.info("[DEBUG] Setting OLLAMA_NO_CONSOLE=1 in environment")

        # Log process information
        logger.info("[DEBUG] Current working directory: " + os.getcwd())
        try:
            # Use appropriate command based on OS
            cmd = "where" if os.name == 'nt' else "which"
            ollama_path = subprocess.check_output([cmd, "ollama"], text=True).strip()
            logger.info("[DEBUG] Ollama executable path: " + ollama_path)
        except Exception as e:
            logger.info(f"[DEBUG] Error getting Ollama path: {e}")

        logger.info("[DEBUG] ==============================================")

        # Start Ollama server as a background process
        logger.info("[DEBUG] Starting Ollama process...")

        # Prepare common arguments
        popen_args = {
            "args": ["ollama", "serve"],
            "stdout": subprocess.PIPE,
            "stderr": subprocess.PIPE,
            "env": env_vars
        }

        # Add OS-specific arguments
        # Since we're on Windows, add Windows-specific flags
        # Note: The IDE warns about unreachable code in the else branch
        # because it knows we're on Windows, but we keep it for completeness
        # and in case this code is ever run on other platforms
        if True:  # Always execute this block on Windows
            # Use CREATE_NEW_PROCESS_GROUP to allow the process to continue running
            # after this script exits
            # Try adding CREATE_NO_WINDOW flag to prevent visible terminal window
            popen_args["creationflags"] = DETACHED_PROCESS | CREATE_NEW_PROCESS_GROUP | CREATE_NO_WINDOW
            logger.info(f"[DEBUG] Using Windows-specific creationflags: 0x{popen_args['creationflags']:08x}")
        # This code is kept for completeness but won't be reached on Windows
        # It would be used if this code were run on Unix/Linux/Mac
        # We're commenting it out to avoid IDE warnings about unreachable code
        # if os.name != 'nt':  # Only on non-Windows platforms
        #     popen_args["start_new_session"] = True
        #     logger.info("[DEBUG] Using Unix-specific start_new_session=True")

        # Log that we're skipping Unix-specific code
        logger.info("[DEBUG] Skipping Unix-specific code (not applicable on Windows)")

        # Create the process
        process = subprocess.Popen(**popen_args)
        logger.info(f"[DEBUG] Popen call completed with args: {popen_args}")

        logger.info(f"[DEBUG] Started Ollama server process with PID: {process.pid}")

        # Try to get more information about the process
        try:
            import psutil
            proc = psutil.Process(process.pid)
            logger.info(f"[DEBUG] Process details: name={proc.name()}, exe={proc.exe()}")
            logger.info(f"[DEBUG] Process parent: {proc.parent().pid if proc.parent() else 'None'}")
            logger.info(f"[DEBUG] Process children: {[p.pid for p in proc.children()]}")
        except Exception as e:
            logger.info(f"[DEBUG] Error getting process details: {e}")

        # Wait for the server to start
        max_retries = 10
        retry_delay = 2  # seconds

        logger.info(f"Waiting for Ollama server to start (max {max_retries} retries with {retry_delay}s delay)...")

        for attempt in range(1, max_retries + 1):
            logger.info(f"Checking if Ollama server is running (attempt {attempt}/{max_retries})...")

            # Wait before checking
            time.sleep(retry_delay)

            # Check if the server is running
            if is_ollama_running():
                logger.info(f"Ollama server started successfully on attempt {attempt}")
                return True

            logger.info(f"Ollama server not responding yet, retrying in {retry_delay}s...")

        logger.warning("Failed to start Ollama server after maximum retries")
        return False

    except Exception as e:
        logger.error(f"Error starting Ollama server: {e}")
        return False

def launch_selfy(args: List[str]) -> int:
    """
    Launch the Selfy application with the given arguments.

    Args:
        args: Command-line arguments to pass to Selfy

    Returns:
        int: The exit code from the Selfy process
    """
    try:
        logger.info(f"Launching Selfy with arguments: {args}")

        # Build the command
        cmd = [sys.executable, os.path.join("selfy_core", "main.py")] + args

        # Launch Selfy
        process = subprocess.Popen(cmd)

        # Wait for Selfy to exit
        return process.wait()
    except Exception as e:
        logger.error(f"Error launching Selfy: {e}")
        return 1

def check_service_status():
    """
    Check the status of the Ollama Windows service.

    Returns:
        bool: True if the service is running, False otherwise
    """
    try:
        # Import the service module
        from selfy_core.global_modules.llm_wrapper.ollama_service import (
            is_service_installed, is_service_running, check_ollama_api
        )

        # Check if the service is installed
        if is_service_installed():
            logger.info("Ollama service is installed")

            # Check if the service is running
            if is_service_running():
                logger.info("Ollama service is running")

                # Check if the API is responsive
                if check_ollama_api():
                    logger.info("Ollama API is responsive")
                else:
                    logger.warning("Ollama API is not responsive")

                return True
            else:
                logger.info("Ollama service is installed but not running")
                return False
        else:
            logger.info("Ollama service is not installed")
            return False
    except ImportError:
        logger.warning("Ollama service module not available")
        return False
    except Exception as e:
        logger.error(f"Error checking Ollama service status: {e}")
        return False

def install_service():
    """
    Install Ollama as a Windows service.

    Returns:
        bool: True if successful, False otherwise
    """
    try:
        # Import the service module
        from selfy_core.global_modules.llm_wrapper.ollama_service import (
            is_admin, install_service as install_ollama_service
        )

        # Check if running as administrator
        if not is_admin():
            logger.error("Administrator privileges required to install Ollama service")
            logger.error("Please run with administrator privileges")
            return False

        # Install the service
        logger.info("Installing Ollama service...")
        if install_ollama_service():
            logger.info("Ollama service installed successfully")
            return True
        else:
            logger.error("Failed to install Ollama service")
            return False
    except ImportError:
        logger.error("Ollama service module not available")
        return False
    except Exception as e:
        logger.error(f"Error installing Ollama service: {e}")
        return False

def uninstall_service():
    """
    Uninstall the Ollama Windows service.

    Returns:
        bool: True if successful, False otherwise
    """
    try:
        # Import the service module
        from selfy_core.global_modules.llm_wrapper.ollama_service import (
            is_admin, uninstall_service as uninstall_ollama_service
        )

        # Check if running as administrator
        if not is_admin():
            logger.error("Administrator privileges required to uninstall Ollama service")
            logger.error("Please run with administrator privileges")
            return False

        # Uninstall the service
        logger.info("Uninstalling Ollama service...")
        if uninstall_ollama_service():
            logger.info("Ollama service uninstalled successfully")
            return True
        else:
            logger.error("Failed to uninstall Ollama service")
            return False
    except ImportError:
        logger.error("Ollama service module not available")
        return False
    except Exception as e:
        logger.error(f"Error uninstalling Ollama service: {e}")
        return False

def start_service():
    """
    Start the Ollama Windows service.

    Returns:
        bool: True if successful, False otherwise
    """
    try:
        # Import the service module
        from selfy_core.global_modules.llm_wrapper.ollama_service import (
            is_service_installed, start_service as start_ollama_service
        )

        # Check if the service is installed
        if not is_service_installed():
            logger.error("Ollama service is not installed")
            return False

        # Start the service
        logger.info("Starting Ollama service...")
        if start_ollama_service():
            logger.info("Ollama service started successfully")
            return True
        else:
            logger.error("Failed to start Ollama service")
            return False
    except ImportError:
        logger.error("Ollama service module not available")
        return False
    except Exception as e:
        logger.error(f"Error starting Ollama service: {e}")
        return False

def stop_service():
    """
    Stop the Ollama Windows service.

    Returns:
        bool: True if successful, False otherwise
    """
    try:
        # Import the service module
        from selfy_core.global_modules.llm_wrapper.ollama_service import (
            is_service_installed, stop_service as stop_ollama_service
        )

        # Check if the service is installed
        if not is_service_installed():
            logger.error("Ollama service is not installed")
            return False

        # Stop the service
        logger.info("Stopping Ollama service...")
        if stop_ollama_service():
            logger.info("Ollama service stopped successfully")
            return True
        else:
            logger.error("Failed to stop Ollama service")
            return False
    except ImportError:
        logger.error("Ollama service module not available")
        return False
    except Exception as e:
        logger.error(f"Error stopping Ollama service: {e}")
        return False

def ensure_ollama_running(use_service=False):
    """
    Ensure Ollama is running, using either the service or the normal process.

    Args:
        use_service: Whether to try using the Ollama service first

    Returns:
        bool: True if Ollama is running, False otherwise
    """
    # Check if Ollama is already running
    if is_ollama_running():
        logger.info("Ollama server is already running")
        return True

    # If using service is enabled, try that first
    if use_service:
        try:
            # Import the service module
            from selfy_core.global_modules.llm_wrapper.ollama_service import (
                is_service_installed, is_service_running,
                start_service as start_ollama_service, check_ollama_api
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
                        return True
                    else:
                        logger.warning("Ollama service is running but API is not responsive")
                        # Fall back to normal startup
                else:
                    logger.info("Ollama service is installed but not running, starting it...")

                    # Start the service
                    if start_ollama_service():
                        logger.info("Ollama service started successfully")

                        # Check if the API is responsive
                        if check_ollama_api():
                            logger.info("Ollama API is responsive")
                            return True
                        else:
                            logger.warning("Ollama service started but API is not responsive")
                            # Fall back to normal startup
                    else:
                        logger.warning("Failed to start Ollama service")
                        # Fall back to normal startup
            else:
                logger.info("Ollama service is not installed")
                # Fall back to normal startup
        except ImportError:
            logger.warning("Ollama service module not available")
            # Fall back to normal startup
        except Exception as e:
            logger.warning(f"Error checking Ollama service: {e}")
            # Fall back to normal startup

    # Fall back to normal startup
    logger.info("Starting Ollama server normally...")
    if start_ollama_server():
        logger.info("Ollama server started successfully")
        return True
    else:
        logger.error("Failed to start Ollama server")
        return False

def main():
    """Main entry point for the launcher."""
    try:
        # Parse command-line arguments
        args = parse_arguments()

        # Enable use-service and interactive by default on Windows
        if platform.system() == 'Windows':
            # Only set these defaults if they weren't explicitly set by the user
            if not args.use_service and not args.no_use_service:
                logger.info("Enabling Ollama service usage by default")
                args.use_service = True
            elif args.no_use_service:
                logger.info("Disabling Ollama service usage as requested")
                args.use_service = False

            if not args.interactive and not args.no_interactive:
                logger.info("Enabling interactive mode by default")
                args.interactive = True
            elif args.no_interactive:
                logger.info("Disabling interactive mode as requested")
                args.interactive = False
        else:
            # On non-Windows platforms, disable service features
            logger.warning("Ollama service features are only available on Windows")
            args.use_service = False
            args.service_status = False
            args.service_install = False
            args.service_uninstall = False
            args.service_start = False
            args.service_stop = False

        # Handle service-related commands
        if args.service_status:
            check_service_status()
            return 0

        if args.service_install:
            if install_service():
                return 0
            else:
                return 1

        if args.service_uninstall:
            if uninstall_service():
                return 0
            else:
                return 1

        if args.service_start:
            if start_service():
                return 0
            else:
                return 1

        if args.service_stop:
            if stop_service():
                return 0
            else:
                return 1

        # Check if we should skip Ollama check
        if not args.skip_ollama_check:
            # Ensure Ollama is running
            if not ensure_ollama_running(use_service=args.use_service):
                logger.error("Failed to ensure Ollama is running. Continuing anyway...")

        # Build arguments for Selfy
        selfy_args = []
        if args.config:
            selfy_args.extend(["--config", args.config])
        if args.interactive:
            selfy_args.append("--interactive")

        # Launch Selfy
        return launch_selfy(selfy_args)
    except Exception as e:
        logger.error(f"Unhandled exception: {e}", exc_info=True)
        return 1

if __name__ == "__main__":
    sys.exit(main())
