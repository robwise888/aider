# Blank Terminal Window Fix

This document explains the solution implemented to fix the issue with blank terminal windows appearing during Selfy operation.

## Problem

During Selfy launch, a blank terminal window would appear when the Ollama model was being loaded. This happened because:

1. The Ollama server creates a separate process for each model it loads
2. On Windows, these model processes would appear as blank terminal windows
3. The application had no direct control over how the Ollama server creates these processes

## Solution

The solution implemented is to run Ollama as a Windows service. This approach has several benefits:

1. **No Blank Terminal Windows**: Services run in the background without any visible UI elements
2. **Improved Reliability**: The service automatically restarts if it crashes
3. **Automatic Startup**: The service starts automatically when Windows boots
4. **Resource Management**: The service runs with consistent resource allocation

## Implementation

The implementation consists of the following components:

1. **Ollama Service Management Module** (`selfy_core/global_modules/llm_wrapper/ollama_service.py`):
   - Functions for installing, starting, stopping, and checking the service
   - Uses NSSM (Non-Sucking Service Manager) to create and manage the service

2. **Service Installation Script** (`install_ollama_service.py`):
   - Standalone script for service installation and management
   - Requires administrator privileges for installation/uninstallation

3. **Enhanced Launcher** (`launch_selfy.py`):
   - Now enables service usage and interactive mode by default on Windows
   - Provides options to disable these defaults if needed
   - Checks for and uses the service if available

## Usage

### Default Behavior

On Windows, Selfy now uses the Ollama service by default. Simply run:

```
python launch_selfy.py
```

This will:
1. Use the Ollama service if it's installed and running
2. Install and start the service if it's not already installed (requires admin privileges)
3. Run in interactive mode

### Command-Line Options

The launcher now supports the following options:

- `--use-service`: Use the Ollama service (default on Windows)
- `--no-use-service`: Don't use the Ollama service
- `--interactive`: Run in interactive mode (default on Windows)
- `--no-interactive`: Don't run in interactive mode
- `--service-status`: Check the status of the Ollama service
- `--service-install`: Install the Ollama service (requires admin privileges)
- `--service-uninstall`: Uninstall the Ollama service (requires admin privileges)
- `--service-start`: Start the Ollama service
- `--service-stop`: Stop the Ollama service

### Service Management

To manage the Ollama service directly:

```
# Check service status
python install_ollama_service.py --status

# Install service (requires admin privileges)
python install_ollama_service.py --install

# Uninstall service (requires admin privileges)
python install_ollama_service.py --uninstall

# Start service
python install_ollama_service.py --start

# Stop service
python install_ollama_service.py --stop
```

## Troubleshooting

### Service Installation Issues

If service installation fails:

1. Make sure you're running with administrator privileges
2. Check that Ollama is installed and in your PATH
3. Check the logs for specific error messages
4. Ensure no other Ollama processes are running

### Service Startup Issues

If the service starts but the API is not responsive:

1. Check the Windows Event Viewer for service-related errors
2. Try stopping and restarting the service
3. Check if another instance of Ollama is already running

### Service Removal Issues

If the service is marked for deletion but not fully removed:

1. Restart your computer to complete the removal process
2. After restarting, try installing the service again

## Technical Details

The service is configured with the following settings:

- **Service Name**: OllamaService
- **Executable**: The path to the Ollama executable
- **Arguments**: serve
- **Environment Variables**:
  - OLLAMA_NO_CONSOLE=1
  - OLLAMA_LOG_LEVEL=info
  - OLLAMA_USE_GPU=1
  - CUDA_VISIBLE_DEVICES=0

The service is installed using NSSM, which is automatically downloaded if not already available.
