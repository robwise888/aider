# Automatic Command Detection

Aider can automatically detect when your message implies a command should be executed, even if you don't explicitly use the `/command` syntax.

## How It Works

When you send a message, Aider analyzes it to determine if you're implicitly requesting a command to be executed. For example, if you say "Show me what files we've changed", Aider might detect that you want to run the `/diff` command.

If Aider is highly confident about the detected command, it will execute it automatically. If it's less confident, it will ask for your confirmation first.

## Configuration

You can control the command detection behavior with the following settings:

- `--auto-detect-commands`: Enable or disable automatic command detection (default: enabled)
- `--command-detection-threshold`: Confidence threshold for detection (default: 0.85)
- `--command-confirmation-threshold`: Threshold above which no confirmation is required (default: 0.95)
- `--command-detection-log`: Log file for detection results (default: .aider.command.log)

## Commands

### /auto-detect

You can control the command detection feature during a session using the `/auto-detect` command:

- `/auto-detect status`: Show the current status
- `/auto-detect on`: Enable automatic command detection
- `/auto-detect off`: Disable automatic command detection

### /command-stats

View statistics about command detection and update the learning system:

- `/command-stats`: Show detection statistics
- `/command-stats update`: Update command examples in the vector database

## Feedback Loop

Aider learns from your interactions to improve command detection over time. When you confirm or reject a detected command, this feedback is recorded and used to improve future detections.

## Integration with RAG

The command detection system is integrated with Aider's Retrieval-Augmented Generation (RAG) system. This means that Aider can use context from previous interactions to better understand when to trigger commands.

For example, if you've previously used the `/diff` command after asking "What changes have we made?", Aider will learn this pattern and suggest the `/diff` command when you ask similar questions in the future.

## Examples

Here are some examples of messages that might trigger automatic command detection:

- "Show me the help documentation" → `/help`
- "What files are in this project?" → `/ls`
- "Let's add this file to our conversation" → `/add`
- "Show me what changes we've made" → `/diff`
- "Save our changes to git" → `/commit`
- "Let's search for all occurrences of this function" → `/search`

## Technical Details

The command detection system uses the following components:

1. **CommandIntentDetector**: Analyzes user messages to detect command intents
2. **CommandLearningSystem**: Learns from user interactions to improve detection
3. **RAG Integration**: Provides context for command detection

The system uses a lightweight LLM model to analyze messages and determine if they imply a command. The analysis includes:

- Confidence score for the detected command
- Arguments for the command
- Reasoning for why the command was detected

This information is logged and used to improve future detections.
