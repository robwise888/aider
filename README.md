# Selfy Production Environment

This directory contains the production version of the Selfy agent, with all test and mock code removed.

## Directory Structure

- `core/`: Core components of the Selfy agent
- `utils/`: Utility modules
- `config/`: Configuration files
- `data/`: Data files
- `logs/`: Log files

## Running Selfy

To run the Selfy agent, use one of the following methods:

### Windows

```
selfy.bat
```

### Python

```
# Standard mode
python selfy.py

# Debug mode
python selfy.py --mode debug

# Fast startup mode
python selfy.py --mode fast
```

## Configuration

The Selfy agent can be configured using environment variables or by editing the `config/config.py` file.

### Environment Variables

- `GROQ_API_KEY`: API key for the Groq LLM service
- `CONSOLE_LOG_LEVEL`: Log level for console output (default: INFO)
- `VERBOSE_STARTUP`: Whether to enable verbose startup (default: False)
- `FAST_STARTUP`: Whether to enable fast startup (default: False)
- `SKIP_RAG`: Whether to skip RAG initialization (default: False)
- `SKIP_OLLAMA`: Whether to skip Ollama initialization (default: False)
- `SKIP_SANDBOX`: Whether to skip sandbox initialization (default: False)
- `SHOW_SYSTEM_INFO`: Whether to show system information (default: False)
- `SELF_MODIFICATION_ENABLED`: Whether to enable self-modification (default: True)
- `SELF_IMPROVEMENT_TARGET_DIR`: Target directory for self-improvement (default: .)
- `SELF_IMPROVEMENT_CYCLE_DELAY`: Delay between self-improvement cycles in seconds (default: 60)
- `SELF_IMPROVEMENT_MAX_CYCLES`: Maximum number of self-improvement cycles (default: 10)

## Troubleshooting

If you encounter any issues, check the log files in the `logs/` directory for more information.

Common issues:

1. **Missing API key**: Make sure the `GROQ_API_KEY` environment variable is set.
2. **Initialization errors**: Check the log files for initialization errors.
3. **Component failures**: If a component fails to initialize, check the log files for more information.

## Support

For support, please contact the Selfy development team.

## Current Status

*Last updated: 2025-04-20 08:09:59*

### Active Goals

1. **Develop the ability to self-code.** (Priority: 5)
   Subgoals:
   - Learn the fundamentals of at least one programming language.
   - Train on a large dataset of code examples.
   - Develop a code generation model.
   - Enable user feedback and adaptation.
2. **Learn the fundamentals of at least one programming language.** (Priority: 1)
3. **Train on a large dataset of code examples.** (Priority: 2)
4. **Develop a code generation model.** (Priority: 3)
5. **Enable user feedback and adaptation.** (Priority: 4)

### Recent Activities

- [SUCCESS] Generated code for Write a script that prints 'Hello world' to the console.: Code successfully applied to generated_code/write_a_script_that_prints_'hello_world'_to_the_console..py
- [SUCCESS] Explained concept 'capabilities'.: Successfully generated explanation.

### Knowledge Collections

- **self_model**: 0 items
- **code_knowledge**: 0 items
- **user_preferences**: 0 items
- **learning_history**: 0 items
- **development_history**: 0 items

### Memory System

Total memories: 3

**Working Memory**: 0/50 items (0.0% used)
**Episodic Memory**: 3/1000 items (0.3% used)
**Semantic Memory**: 0/5000 items (0.0% used)
**Procedural Memory**: 0/500 items (0.0% used)

### User Model

**User Preferences**:

*Communication*:
- communication_style: conversational (confidence: 80.0%)
- explanation_detail: medium (confidence: 80.0%)
*Coding*:
- code_style: not applicable (confidence: 80.0%)

Total interactions: 7

