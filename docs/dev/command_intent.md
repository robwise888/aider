# Command Intent Detection System

The command intent detection system allows Aider to automatically detect when a user message implies a command should be executed, even if the user doesn't explicitly use the `/command` syntax.

## Architecture

The system consists of the following components:

1. **CommandIntentDetector**: Analyzes user messages to detect command intents
2. **Commands Integration**: Processes detected intents and executes commands
3. **RAG Integration**: Provides context for command detection
4. **Feedback Loop**: Learns from user interactions to improve detection

## Command Intent Detector

The `CommandIntentDetector` class is responsible for analyzing user messages and determining if they imply a command should be executed. It uses the LLM to perform this analysis.

### Key Methods

- `detect_command_intent(user_message)`: Analyzes a user message to detect command intents
- `_create_intent_detection_prompt(user_message)`: Creates a prompt for the LLM
- `_query_llm(prompt)`: Queries the LLM with the prompt
- `_parse_intent_response(response)`: Parses the LLM response
- `log_detection_result(intent, was_executed, user_feedback)`: Logs detection results

## Commands Integration

The `Commands` class is extended to handle command intents:

- `initialize_intent_detector(model, config)`: Initializes the detector
- `process_message(message)`: Processes a message, potentially executing commands
- `cmd_auto_detect(arg)`: Command to control detection settings
- `cmd_command_stats(arg)`: Command to view statistics and update learning

## RAG Integration

The system integrates with the RAG system to provide context for command detection:

- `_get_rag_context_for_commands(query)`: Gets relevant context from the RAG system
- `retrieve_command_context(query, max_results)`: Retrieves context for command detection
- `_initialize_command_examples()`: Initializes command examples in the vector database

## Feedback Loop

The `CommandLearningSystem` class implements the feedback loop:

- `analyze_logs()`: Analyzes command detection logs
- `generate_examples()`: Generates command examples from successful detections
- `update_command_examples(repo_map)`: Updates command examples in the vector database

## Configuration

The system can be configured with the following parameters:

- `auto_detect_commands`: Enable/disable automatic detection
- `command_detection_threshold`: Confidence threshold for detection
- `command_confirmation_threshold`: Threshold above which no confirmation is required
- `command_detection_log`: Log file for detection results

## Initialization Flow

1. The `Coder` class initializes the command detection parameters during its initialization
2. The `run_one` method initializes the `CommandIntentDetector` if it doesn't exist
3. The `process_message` method is called to check if the message implies a command
4. If a command is detected with high confidence, it is executed

## Detection Process

1. The user sends a message
2. The `process_message` method is called
3. If the message starts with `/` or `!`, it is handled as a normal command
4. Otherwise, the `detect_command_intent` method is called
5. The LLM analyzes the message and returns a command, arguments, and confidence
6. If the confidence is above the threshold, the command is executed
7. If the confidence is below the confirmation threshold, the user is asked for confirmation
8. The result is logged for future improvement

## Log Format

The command detection log contains JSON entries with the following fields:

```json
{
  "timestamp": "2023-01-01T00:00:00.000000",
  "message": "How do I use this tool?",
  "detected_command": "help",
  "confidence": 0.95,
  "was_executed": true,
  "user_feedback": null,
  "reasoning": "The user is asking for help"
}
```

## Example Initialization

```python
# Initialize command intent detector
config = {
    "auto_detect_commands": True,
    "command_detection_threshold": 0.85,
    "command_confirmation_threshold": 0.95,
    "command_detection_log": ".aider.command.log",
    "verbose": False
}
commands.initialize_intent_detector(model, config)
```

## Example Detection

```python
# Process a message
command_executed = commands.process_message("How do I use this tool?")
if command_executed:
    return
```

## Integration with EnhancedRepoMap

The `EnhancedRepoMap` class initializes command examples in the vector database:

```python
def _initialize_command_examples(self):
    """Initialize command examples for the RAG system"""
    if not self.vector_db_manager:
        return
    
    # Create a file with command examples
    from aider.command_learning import CommandLearningSystem
    
    # Create learning system
    learning = CommandLearningSystem()
    
    # Get default examples
    command_examples = learning.get_default_examples()
    
    # Add to vector database
    command_file_path = ".aider.commands.md"
    self.vector_db_manager.add_file(command_file_path, command_examples)
```

## Testing

The system includes unit tests for the `CommandIntentDetector` class and integration tests for the command detection process. These tests verify that:

1. The command registry is built correctly
2. Commands are detected with the correct confidence
3. Commands are executed when confidence is high
4. User confirmation is requested when confidence is low
5. Detection results are logged correctly
