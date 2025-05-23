# Ollama Windows Service

This document explains how to use Ollama as a Windows service to prevent blank terminal windows from appearing during Selfy operation.

## Overview

Running Ollama as a Windows service provides several benefits:

1. **No Blank Terminal Windows**: The service runs in the background without any visible UI elements.
2. **Improved Reliability**: The service automatically restarts if it crashes.
3. **Automatic Startup**: The service can be configured to start automatically when Windows boots.
4. **Resource Management**: The service runs with consistent resource allocation.

## Requirements

- Windows 10 or later
- Administrator privileges (for service installation/uninstallation)
- Ollama installed and working

## Installation

There are two ways to install the Ollama service:

### Method 1: Using the Launcher Script

Run the launcher script with the `--service-install` flag:

```
python launch_selfy.py --service-install
```

This requires administrator privileges. If you're not running as administrator, the script will prompt you to run it with administrator privileges.

### Method 2: Using the Installation Script

Run the installation script:

```
python install_ollama_service.py --install
```

This also requires administrator privileges.

## Usage

Once the service is installed, you can use Selfy with the `--use-service` flag to make it use the Ollama service:

```
python launch_selfy.py --use-service
```

This will:
1. Check if the Ollama service is installed and running
2. Start the service if it's not running
3. Use the service for Ollama operations

## Service Management

The launcher script provides several options for managing the Ollama service:

- `--service-status`: Check the status of the Ollama service
- `--service-install`: Install Ollama as a Windows service
- `--service-uninstall`: Uninstall the Ollama Windows service
- `--service-start`: Start the Ollama service
- `--service-stop`: Stop the Ollama service

Examples:

```
# Check service status
python launch_selfy.py --service-status

# Start the service
python launch_selfy.py --service-start

# Stop the service
python launch_selfy.py --service-stop
```

## Troubleshooting

### Service Installation Fails

If service installation fails, check the following:

1. Make sure you're running with administrator privileges
2. Check that Ollama is installed and in your PATH
3. Check the logs for specific error messages

### Service Starts but API is Not Responsive

If the service starts but the API is not responsive:

1. Check the Windows Event Viewer for service-related errors
2. Try stopping and restarting the service
3. Check if another instance of Ollama is already running

### Uninstalling the Service

If you need to uninstall the service:

```
python launch_selfy.py --service-uninstall
```

This requires administrator privileges.

## How It Works

The Ollama service implementation uses NSSM (Non-Sucking Service Manager) to create and manage the Windows service. NSSM is a lightweight tool that allows running any executable as a Windows service.

The service is configured with the following settings:

- **Service Name**: OllamaService
- **Executable**: The path to the Ollama executable
- **Arguments**: serve
- **Environment Variables**:
  - OLLAMA_NO_CONSOLE=1
  - OLLAMA_LOG_LEVEL=info
  - OLLAMA_USE_GPU=1
  - CUDA_VISIBLE_DEVICES=0

## Technical Details

The service implementation consists of the following components:

1. **ollama_service.py**: Core module for service management
2. **install_ollama_service.py**: Standalone script for service installation
3. **launch_selfy.py**: Enhanced launcher with service support
4. **ollama_provider.py**: Modified provider that can use the service

The implementation automatically downloads NSSM if it's not already available.
