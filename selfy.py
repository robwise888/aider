#!/usr/bin/env python
"""
Unified launcher script for Selfy Production.

This script provides a single entry point for running the production version of Selfy in various modes:
- normal: Standard production mode
- debug: Debug mode with verbose logging
- fast: Fast startup mode that skips optional components

Usage:
    python selfy.py [options]

Examples:
    python selfy.py                          # Run in normal mode
    python selfy.py --mode debug             # Run in debug mode
    python selfy.py --mode fast              # Run in fast startup mode
    python selfy.py --skip-rag --skip-ollama # Skip RAG and Ollama initialization
    python selfy.py --log-level DEBUG        # Set log level to DEBUG
"""

import os
import sys
import subprocess
import argparse
import logging
from dotenv import load_dotenv

# Add the production directory to the Python path
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, script_dir)

# Create logs directory if it doesn't exist
os.makedirs(os.path.join(script_dir, 'logs'), exist_ok=True)

# Ensure we're running in the correct Python environment
def ensure_venv():
    """Ensure we're running in the virtual environment."""
    # Check if we're already in a virtual environment
    in_venv = hasattr(sys, 'real_prefix') or (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix)
    
    if in_venv:
        return sys.executable
    
    # Get the directory of this script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(script_dir)
    
    # Look for the virtual environment
    venv_dirs = ['selfy_env', 'venv', '.venv', 'env']
    for venv_dir in venv_dirs:
        venv_python = os.path.join(parent_dir, venv_dir, 'Scripts', 'python.exe')
        if os.path.exists(venv_python):
            print(f"Not running in virtual environment. Restarting with: {venv_python}")
            
            # Re-run the script with the correct Python interpreter
            args = [venv_python] + sys.argv
            os.execv(venv_python, args)
    
    # If we get here, no virtual environment was found
    print("WARNING: No virtual environment found. Running with system Python.")
    print("This may cause dependency issues. Consider creating a virtual environment.")
    return sys.executable

# Check dependencies
def check_dependencies():
    """Check if required dependencies are installed."""
    required_env_vars = ['GROQ_API_KEY']
    missing_vars = [var for var in required_env_vars if not os.getenv(var)]

    if missing_vars:
        print(f"Error: Missing required environment variables: {', '.join(missing_vars)}")
        print("Please set these variables in your .env file or environment.")
        return False

    # Check for required directories
    required_dirs = ['core', 'utils', 'config', 'data']
    missing_dirs = [dir for dir in required_dirs if not os.path.isdir(os.path.join(script_dir, dir))]

    if missing_dirs:
        print(f"Error: Missing required directories: {', '.join(missing_dirs)}")
        return False

    return True

# Set up logging
def setup_logging(log_level='INFO'):
    """Set up logging configuration."""
    numeric_level = getattr(logging, log_level.upper(), logging.INFO)
    
    # Configure logging
    logging.basicConfig(
        level=numeric_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(os.path.join(script_dir, 'logs', 'selfy.log')),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    # Set debug level for all loggers if in debug mode
    if log_level.upper() == 'DEBUG':
        for name in logging.root.manager.loggerDict:
            logging.getLogger(name).setLevel(logging.DEBUG)
    
    return logging.getLogger("Selfy.Launcher")

# Clear log file
def clear_log_file(log_file):
    """Clear the specified log file."""
    with open(os.path.join(script_dir, 'logs', log_file), 'w') as f:
        f.write("")

# Seed the RAG database
def seed_rag_database(logger):
    """Seed the RAG database with relevant documentation."""
    print("\nSeeding RAG database...")

    try:
        # Check if the seed_rag_knowledge.py script exists
        seed_script = os.path.join(script_dir, 'seed_rag_knowledge.py')
        if os.path.exists(seed_script):
            # Run the script
            result = subprocess.run([sys.executable, seed_script], capture_output=True, text=True)

            if result.returncode == 0:
                print("RAG database seeded successfully!")
                return True
            else:
                logger.error(f"Failed to seed RAG database: {result.stderr}")
                print(f"Failed to seed RAG database: {result.stderr}")
                return False
        else:
            logger.warning(f"RAG database seeding script not found: {seed_script}")
            print(f"RAG database seeding script not found: {seed_script}")
            return False
    except Exception as e:
        logger.error(f"Error seeding RAG database: {e}")
        print(f"Error seeding RAG database: {e}")
        return False

# Run Selfy in normal mode
def run_normal_mode(args, logger):
    """Run Selfy in normal production mode."""
    logger.info("Starting Selfy in normal mode")
    
    # Import required modules
    try:
        from production.core.autonomous_agent import AutonomousAgent
        from production.core.groq_wrapper import GroqLLMWrapper
        from production.core.ollama_wrapper import OllamaWrapper
        from production.core.rag_manager import RAGManager
        from production.utils.sandbox_manager import SandboxManager
        from production.utils.component_manager import ComponentManager
        import production.utils.config as config
        
        # Verify that AgentState can be imported correctly
        from production.core.agent_state import AgentState
        
        logger.info("Successfully imported core modules")
    except ImportError as e:
        logger.error(f"Failed to import core modules: {e}")
        print(f"Error: Failed to import core modules: {e}")
        return False
    
    # Initialize startup timer
    import time
    startup_start_time = time.time()
    
    # Initialize components
    try:
        # Get API keys
        groq_api_key = os.getenv('GROQ_API_KEY')
        
        # Create component manager
        component_manager = ComponentManager()
        
        # Initialize Groq wrapper
        logger.info("Initializing Groq wrapper")
        groq_wrapper = GroqLLMWrapper(api_key=groq_api_key, model_name=config.GROQ_DEFAULT_MODEL)
        
        # Initialize Ollama wrapper (if available and not skipped)
        ollama_wrapper = None
        if not args.skip_ollama:
            logger.info("Initializing Ollama wrapper")
            try:
                ollama_wrapper = OllamaWrapper(model_name=config.OLLAMA_DEFAULT_MODEL)
                # Register the Ollama wrapper with the component manager
                component_manager.register_initializer('ollama_wrapper', lambda: ollama_wrapper)
                logger.info("Registered Ollama wrapper with component manager")
            except Exception as e:
                logger.warning(f"Failed to initialize Ollama wrapper: {e}")
        else:
            logger.info("Skipping Ollama initialization (--skip-ollama)")
        
        # Initialize RAG manager (if not skipped)
        rag_manager = None
        if not args.skip_rag:
            logger.info("Initializing RAG manager")
            try:
                rag_manager = RAGManager()
                # Register the RAG manager with the component manager
                component_manager.register_initializer('rag_manager', lambda: rag_manager)
                logger.info("Registered RAG manager with component manager")
            except Exception as e:
                logger.warning(f"Failed to initialize RAG manager: {e}")
        else:
            logger.info("Skipping RAG initialization (--skip-rag)")
        
        # Initialize sandbox manager (if not skipped)
        sandbox_manager = None
        if not args.skip_sandbox:
            logger.info("Initializing sandbox manager")
            try:
                sandbox_manager = SandboxManager()
                # Register the sandbox manager with the component manager
                component_manager.register_initializer('sandbox_manager', lambda: sandbox_manager)
                logger.info("Registered sandbox manager with component manager")
            except Exception as e:
                logger.warning(f"Failed to initialize sandbox manager: {e}")
        else:
            logger.info("Skipping sandbox initialization (--skip-sandbox)")
        
        # Initialize the agent
        logger.info("Initializing autonomous agent")
        agent = AutonomousAgent(
            main_llm_wrapper=groq_wrapper,
            local_llm_wrapper=ollama_wrapper,
            rag_manager=rag_manager,
            sandbox_manager=sandbox_manager,
            agent_name=config.AGENT_NAME,
            enable_web_interaction=not args.no_web,
            enable_user_modeling=not args.no_user_model,
            component_manager=component_manager
        )
        
        # Calculate startup time
        startup_time = time.time() - startup_start_time
        logger.info(f"Selfy initialized in {startup_time:.2f} seconds")
        
        # Run the agent
        logger.info("Starting Selfy agent")
        agent.run()
        
        return True
    except Exception as e:
        logger.error(f"Failed to initialize Selfy: {e}", exc_info=True)
        print(f"Error: Failed to initialize Selfy: {e}")
        return False

# Run Selfy in debug mode
def run_debug_mode(args, logger):
    """Run Selfy in debug mode with verbose logging."""
    logger.info("Starting Selfy in debug mode")
    
    # Clear the debug log file
    clear_log_file('selfy_debug.log')
    
    # Create a timestamp for the log file
    import datetime
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_log = os.path.join(script_dir, "logs", f"selfy_output_{timestamp}.log")
    
    logger.info(f"Debug output will be saved to: {output_log}")
    
    # Run normal mode with debug logging
    return run_normal_mode(args, logger)

# Run Selfy in fast mode
def run_fast_mode(args, logger):
    """Run Selfy in fast startup mode, skipping optional components."""
    logger.info("Starting Selfy in fast startup mode")
    
    # Force skip flags for fast startup
    args.skip_rag = True
    args.skip_ollama = True
    args.skip_sandbox = True
    
    # Run normal mode with components skipped
    return run_normal_mode(args, logger)

def main():
    """Main entry point for the unified launcher."""
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Unified launcher for Selfy Production")
    
    # Mode selection
    parser.add_argument("--mode", choices=["normal", "debug", "fast"], 
                        default="normal", help="Launch mode (default: normal)")
    
    # Logging options
    parser.add_argument("--log-level", choices=["DEBUG", "INFO", "WARNING", "ERROR"], 
                        default="INFO", help="Log level (default: INFO)")
    
    # Component options
    parser.add_argument("--skip-rag", action="store_true", help="Skip RAG initialization")
    parser.add_argument("--skip-ollama", action="store_true", help="Skip Ollama initialization")
    parser.add_argument("--skip-sandbox", action="store_true", help="Skip Sandbox initialization")
    
    # Feature options
    parser.add_argument("--no-web", action="store_true", help="Disable web interaction")
    parser.add_argument("--no-user-model", action="store_true", help="Disable user modeling")
    
    args = parser.parse_args()
    
    # Ensure we're running in the virtual environment
    python_executable = ensure_venv()
    
    # Load environment variables
    load_dotenv()
    
    # Set up logging
    log_level = "DEBUG" if args.mode == "debug" else args.log_level
    logger = setup_logging(log_level)
    logger.info(f"Starting Selfy unified launcher in {args.mode} mode")
    
    # Check dependencies
    if not check_dependencies():
        input("Press Enter to exit...")
        sys.exit(1)
    
    # Seed the RAG database (unless skipped)
    if not args.skip_rag:
        seed_rag_database(logger)
    
    # Launch in the requested mode
    try:
        if args.mode == "normal":
            success = run_normal_mode(args, logger)
        elif args.mode == "debug":
            success = run_debug_mode(args, logger)
        elif args.mode == "fast":
            success = run_fast_mode(args, logger)
        else:
            logger.error(f"Unknown mode: {args.mode}")
            print(f"Error: Unknown mode: {args.mode}")
            success = False
        
        if not success:
            input("Press Enter to exit...")
            sys.exit(1)
    except KeyboardInterrupt:
        print("\nSelfy terminated by user.")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Unhandled exception: {e}", exc_info=True)
        print(f"\nUnhandled exception: {e}")
        input("Press Enter to exit...")
        sys.exit(1)

if __name__ == "__main__":
    main()
