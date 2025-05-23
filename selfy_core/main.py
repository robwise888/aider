"""
Main entry point for the Selfy agent.

This module provides the main entry point for the Selfy agent, including
initialization of the user chat pipeline and processing of user requests.
"""

import os
import sys
import uuid
import argparse
import logging
import signal
import atexit
from typing import Dict, Any, Optional

# Add the parent directory to the path
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

# Import from global modules
try:
    from selfy_core.global_modules.config import setup as setup_config, get as config_get, load_config_file
    from selfy_core.global_modules.logging import setup_logging, get_logger
except ImportError as e:
    # Fall back to standard logging if the logging module is not available
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    logger.error(f"Failed to import from global modules: {e}")
    sys.exit(1)

# Set up logging and configuration
setup_config()
setup_logging()
logger = get_logger(__name__)

def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Selfy Agent")
    parser.add_argument(
        "--config",
        help="Path to configuration file"
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Run in interactive mode"
    )
    return parser.parse_args()


def initialize(config_path: Optional[str] = None) -> bool:
    """
    Initialize the Selfy agent.

    Args:
        config_path: Path to the configuration file (optional)

    Returns:
        True if initialization was successful, False otherwise
    """
    try:
        # Start Ollama server at the very beginning of initialization
        logger.info("Starting Ollama server at application launch...")
        try:
            # Import the ensure_ollama_server function and OLLAMA_SERVER_CHECKED flag
            from selfy_core.global_modules.llm_wrapper.ollama_provider import ensure_ollama_server, OLLAMA_SERVER_CHECKED

            # Reset the checked flag to force a fresh check during initialization
            # This ensures we don't rely on a stale flag value
            logger.debug(f"OLLAMA_SERVER_CHECKED before reset: {OLLAMA_SERVER_CHECKED}")
            OLLAMA_SERVER_CHECKED = False
            logger.debug(f"OLLAMA_SERVER_CHECKED after reset: {OLLAMA_SERVER_CHECKED}")

            # Start the Ollama server with GPU acceleration
            server_running = ensure_ollama_server(use_gpu=True)

            if server_running:
                logger.info(f"Ollama server started successfully at application launch (OLLAMA_SERVER_CHECKED={OLLAMA_SERVER_CHECKED})")
            else:
                logger.warning("Failed to start Ollama server at application launch, will retry when needed")
        except Exception as ollama_error:
            logger.warning(f"Error starting Ollama server at application launch: {ollama_error}")
            logger.info("Will attempt to start Ollama server when needed")

        # Set up configuration
        if not setup_config():
            logger.error("Failed to set up configuration")
            return False

        # Load configuration file if provided
        if config_path:
            if os.path.exists(config_path):
                load_config_file(config_path)
                logger.info(f"Loaded configuration from {config_path}")
            else:
                logger.error(f"Configuration file not found: {config_path}")
                return False
        else:
            # Use default configuration
            default_config_path = os.path.join(os.path.dirname(__file__), "config", "default_config.json")
            if os.path.exists(default_config_path):
                load_config_file(default_config_path)
                logger.info(f"Loaded default configuration from {default_config_path}")
            else:
                logger.warning(f"Default configuration file not found: {default_config_path}")
                # Continue with hardcoded defaults

        # Verify critical configuration values
        logger.info("Verifying critical configuration values...")
        confidence_threshold = config_get('context_engine.request_analyzer.confidence_threshold')
        skip_final_analysis_threshold = config_get('context_engine.request_analyzer.skip_final_analysis_threshold')
        memory_score_threshold = config_get('memory.evaluator.min_score_threshold')
        consolidate_on_shutdown = config_get('memory.consolidation.consolidate_on_shutdown', True)

        logger.info(f"Context Engine: confidence_threshold={confidence_threshold}, "
                   f"skip_final_analysis_threshold={skip_final_analysis_threshold}")
        logger.info(f"Memory: min_score_threshold={memory_score_threshold}, "
                   f"consolidate_on_shutdown={consolidate_on_shutdown}")

        # Set up pipeline
        from selfy_core.user_pipeline.pipeline import setup_pipeline
        if not setup_pipeline():
            logger.error("Failed to set up pipeline")
            return False

        logger.info("Selfy agent initialized successfully")
        return True
    except Exception as e:
        logger.error(f"Failed to initialize Selfy agent: {e}", exc_info=True)
        return False


def process_request(user_input: str, user_id: str = None, session_id: str = None,
                 output_format: str = "text") -> Dict[str, Any]:
    """
    Process a user request.

    Args:
        user_input: The raw user input
        user_id: The user ID (optional, will be generated if not provided)
        session_id: The session ID (optional, will be generated if not provided)
        output_format: The desired output format (text, markdown, json)

    Returns:
        Dictionary containing the response and metadata
    """
    # Generate user_id and session_id if not provided
    if not user_id:
        user_id = str(uuid.uuid4())
    if not session_id:
        session_id = str(uuid.uuid4())

    # Get pipeline
    from selfy_core.user_pipeline.pipeline import get_pipeline
    pipeline = get_pipeline()
    if not pipeline:
        logger.error("Pipeline not initialized")
        return {
            "status": "error",
            "error": "Pipeline not initialized",
            "response": "I'm sorry, but I'm not fully initialized yet. Please try again later."
        }

    # Process request
    return pipeline.process_request(
        user_input=user_input,
        user_id=user_id,
        session_id=session_id,
        output_format=output_format
    )


def run_interactive_mode():
    """Run the Selfy agent in interactive mode."""
    logger.info("Starting interactive mode...")

    # Create a simple console interface
    print("\nSelfy Agent")
    print("Type 'exit' or 'quit' to exit")
    print("Type 'help' for help")
    print("----------------------------\n")

    user_id = str(uuid.uuid4())
    session_id = str(uuid.uuid4())

    while True:
        try:
            # Get user input
            user_input = input("\nYou: ")

            # Check for exit commands
            if user_input.lower() in ["exit", "quit"]:
                print("\nGoodbye!")
                break

            # Check for help command
            if user_input.lower() == "help":
                print("\nCommands:")
                print("  exit, quit - Exit the chat")
                print("  help - Show this help message")
                continue

            # Process the request
            print("\nSelfy: ", end='', flush=True)
            process_result = process_request(
                user_input=user_input,
                user_id=user_id,
                session_id=session_id
            )

            # Print the response
            print(process_result.get('response', 'No response generated.'))

            # Print a newline after streaming is complete
            print()

            # Log processing time
            if 'processing_time' in process_result:
                logger.info(f"Request processed in {process_result['processing_time']:.2f} seconds")

        except KeyboardInterrupt:
            print("\nKeyboard interrupt detected. Shutting down...")
            logger.info("Keyboard interrupt detected. Shutting down...")
            break
        except Exception as e:
            logger.error(f"Error processing request: {e}", exc_info=True)
            print("\nSelfy: I'm sorry, an error occurred while processing your request.")


# Global variable to track if we're in interactive mode
interactive_mode = False

def signal_handler(sig, frame):
    """Handle termination signals."""
    if sig == signal.SIGINT:
        print("\nKeyboard interrupt detected. Shutting down...")
    else:
        print(f"\nSignal {sig} received. Shutting down...")

    logger.info(f"Signal {sig} received. Shutting down...")

    # Run memory consolidation before shutdown
    shutdown()

    # For interactive mode, we'll handle the "press enter to close" in the main function
    # to avoid re-entering readline
    sys.exit(0)

def shutdown():
    """Perform cleanup tasks before shutting down."""
    logger.info("Performing shutdown tasks...")

    # Run memory consolidation if enabled
    consolidate_on_shutdown = config_get('memory.consolidation.consolidate_on_shutdown', True)
    if consolidate_on_shutdown:
        try:
            # Import memory consolidation service and working memory
            from selfy_core.global_modules.memory.memory_consolidation import get_memory_consolidation_service
            from selfy_core.global_modules.memory.core import get_memory_system
            from selfy_core.global_modules.memory.working_memory import get_working_memory

            # Get the consolidation service and memory system
            consolidation_service = get_memory_consolidation_service()
            memory_system = get_memory_system()

            if consolidation_service and memory_system:
                logger.info("Running final memory consolidation before shutdown")

                # Get all active sessions
                active_sessions = []
                try:
                    # Try to get all sessions from working memory
                    # This is a simplified approach - in a real implementation,
                    # we would have a proper way to get all active sessions
                    working_memory = get_working_memory()  # Use the standalone function instead of a method
                    if working_memory:
                        # Extract unique session IDs from working memory
                        for item in working_memory.get_all_items():
                            if 'session_id' in item.metadata:
                                session_id = item.metadata['session_id']
                                if session_id not in active_sessions:
                                    active_sessions.append(session_id)
                except Exception as e:
                    logger.error(f"Error getting active sessions: {e}")

                # Consolidate each active session
                total_items_stored = 0
                for session_id in active_sessions:
                    logger.info(f"Consolidating session {session_id}")
                    try:
                        # Get conversation turns for this session
                        conversation_turns = memory_system.get_recent_conversation(session_id, 100)

                        if conversation_turns:
                            # Evaluate conversation for long-term storage
                            valuable_items = consolidation_service.memory_evaluator.evaluate_conversation(conversation_turns)

                            if valuable_items:
                                # Store valuable items in long-term memory
                                for item in valuable_items:
                                    memory_system.add_memory(item, use_intelligent_filtering=False)
                                    total_items_stored += 1

                                logger.info(f"Stored {len(valuable_items)} valuable items from session {session_id}")
                            else:
                                logger.info(f"No valuable items found in session {session_id}")
                        else:
                            logger.info(f"No conversation turns found for session {session_id}")
                    except Exception as e:
                        logger.error(f"Error consolidating session {session_id}: {e}")

                logger.info(f"Memory consolidation complete. Stored {total_items_stored} items in long-term memory.")
            else:
                logger.warning("Memory consolidation service or memory system not available")
        except Exception as e:
            logger.error(f"Error during memory consolidation: {e}")
    else:
        logger.info("Memory consolidation on shutdown is disabled")

    logger.info("Shutdown tasks completed")

def setup_signal_handlers():
    """Set up signal handlers for graceful termination."""
    signal.signal(signal.SIGINT, signal_handler)
    if hasattr(signal, 'SIGTERM'):
        signal.signal(signal.SIGTERM, signal_handler)

def main():
    """Main entry point for the Selfy agent."""
    global interactive_mode

    try:
        # Set up signal handlers
        setup_signal_handlers()

        # Parse command-line arguments
        args = parse_arguments()

        # Set interactive mode flag
        interactive_mode = args.interactive

        # Initialize the agent
        if not initialize(args.config):
            logger.error("Failed to initialize Selfy agent")
            sys.exit(1)

        # Run in interactive mode if requested
        if interactive_mode:
            run_interactive_mode()
        else:
            logger.info("Selfy agent initialized. Use --interactive to start chat mode.")

        # Perform shutdown tasks
        shutdown()
    except Exception as e:
        logger.error(f"Unhandled exception: {e}", exc_info=True)
        sys.exit(1)
    finally:
        # Keep terminal open after completion or error
        if interactive_mode:
            print("\nPress Enter to close this window...")
            input()

if __name__ == "__main__":
    main()
