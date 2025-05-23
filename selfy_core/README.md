# Selfy Core

This is the core implementation of the Selfy agent, focusing on the user chat pipeline.

## Architecture

The Selfy agent is built with a modular architecture, consisting of:

### Global Modules

- **Memory System**: Provides working memory and long-term storage for the agent
- **LLM Wrapper**: Provides a unified interface for interacting with different LLM providers
- **Capability Manifest**: Serves as a registry for capabilities that the agent can use

### User Chat Pipeline

- **Input Handling**: Processes user input, retrieves conversation history, and prepares input for the context engine
- **Context Engine**: Analyzes requests, matches them to capabilities, and generates responses
- **Output Handling**: Validates, formats, and delivers responses to users
- **Identity System**: Ensures consistent agent identity and filters inputs/outputs

## Directory Structure

- `global_modules/`: Shared modules used by all components
  - `config/`: Configuration system
  - `logging/`: Logging system
  - `memory/`: Memory system (working memory and long-term storage)
  - `llm_wrapper/`: Unified interface for local and cloud LLMs
  - `capability_manifest/`: Registry of agent capabilities
- `user_pipeline/`: Components for user interaction
  - `input_handling/`: Input processing and validation
  - `context_engine/`: Request analysis and context building
  - `identity/`: Identity management and filtering
  - `output_handling/`: Output processing and delivery
  - `pipeline.py`: Main pipeline orchestration
- `config/`: Configuration files
  - `default_config.json`: Default configuration

## Getting Started

### Prerequisites

- Python 3.8 or higher
- Required packages (see requirements.txt)

### Configuration

The agent can be configured using a JSON configuration file. A default configuration is provided in `selfy_core/config/default_config.json`.

Key configuration areas include:

- **LLM Providers**: Settings for Groq (cloud) and Ollama (local) LLM providers
- **Memory System**: Configuration for ChromaDB and working memory
- **Logging**: Comprehensive logging with rotation and structured formats
- **Identity**: Agent identity profile and filtering settings
- **Pipeline**: Settings for input and output handling

### Environment Variables

The following environment variables can be set:

- `GROQ_API_KEY`: API key for the Groq LLM provider

### Running the Agent

To run the agent in interactive mode:

```bash
python -m selfy_core.main --interactive
```

To use a custom configuration file:

```bash
python -m selfy_core.main --config path/to/config.json --interactive
```

## Data Storage

- **Memory Database**: Located in `./memory_db`
- **Capability Manifest**: Located in `./data/capability_manifest.json`
- **Log Files**: Located in the `logs/` directory

## Pipeline Flow

1. User input is processed by the Input Handler
2. The Context Engine analyzes the request and generates a response
3. The Output Handler validates and formats the response
4. The response is returned to the user

## Components

### Memory System

- **Working Memory**: In-memory cache for recently accessed items
- **Embedding Service**: Generates vector embeddings for semantic search
- **LTS Vector Store**: Long-term storage for memory items with vector search

### LLM Wrapper

- **Base Provider**: Abstract interface for LLM providers
- **Groq Provider**: Implementation for Groq cloud LLM service
- **Ollama Provider**: Implementation for local Ollama LLM service

### Capability Manifest

- **Registry**: Stores and manages capabilities
- **Execution**: Executes capabilities with parameter validation
- **Access**: Functions for accessing capabilities

Note: The capability discovery functionality is intended for the self-development pipeline and is not implemented in the production environment. The capability manifest in production contains capabilities that were migrated from the development environment.

### Input Handling

- **Processor**: Validates input and retrieves conversation history
- **Memory Integration**: Stores input in memory

### Context Engine

- **Request Analyzer**: Analyzes requests and matches them to capabilities
- **Context Builder**: Builds appropriate contexts for LLM interactions
- **Execution Planner**: Plans and executes capability calls

### Output Handling

- **Processor**: Orchestrates output processing
- **Validator**: Validates output for quality and safety
- **Formatter**: Formats output in different formats

### Identity System

- **Manager**: Manages the agent's identity profile
- **Filter**: Filters inputs and outputs for identity consistency
