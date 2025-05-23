# Configuration Guide

## Introduction

This document provides guidance on configuring the Context Engine V2. It describes the configuration options, their default values, and how to customize them.

## Configuration File

The Context Engine V2 is configured using a JSON configuration file. The default location for this file is `config/context_engine.json`. Here's an example configuration file:

```json
{
    "context_engine": {
        "enable_adaptive_planning": true,
        "max_execution_attempts": 3,
        "enable_memory_integration": true,
        "enable_identity_filtering": true
    },
    "llm": {
        "local": {
            "enabled": true,
            "model": "llama3:8b",
            "api_base": "http://localhost:11434",
            "options": {}
        },
        "cloud": {
            "enabled": true,
            "model": "llama3-70b-8192",
            "api_key": "",
            "api_base": "https://api.groq.com/openai/v1"
        },
        "confidence_threshold": 0.7,
        "instruction_refinement_threshold": 0.8,
        "max_retries": 3,
        "backoff_time": 2
    },
    "analyzer": {
        "use_two_stage_analysis": true,
        "min_confidence_threshold": 0.7,
        "use_local_preprocessing": true,
        "use_cloud_detailed_analysis": true,
        "max_analysis_attempts": 3
    },
    "capability": {
        "manifest_path": "data/capability_manifest.json",
        "min_confidence_threshold": 0.7,
        "max_matches": 5,
        "use_semantic_matching": true,
        "use_embeddings": true,
        "embedding_model": "text-embedding-ada-002"
    },
    "execution": {
        "enable_adaptive_planning": true,
        "max_execution_attempts": 3,
        "enable_chain_of_thought": true,
        "error_recovery_methods": ["alternative_tool", "llm", "simplified_parameters"],
        "timeout": 60
    },
    "error": {
        "max_retries": 3,
        "initial_backoff": 1,
        "backoff_factor": 2,
        "use_llm_analysis": true,
        "log_level": "ERROR",
        "include_details_in_response": false
    },
    "memory": {
        "enable_memory_integration": true,
        "store_execution_logs": true,
        "store_learned_facts": true,
        "max_conversation_history_length": 10,
        "max_learned_facts": 10,
        "relevance_threshold": 0.7
    },
    "input": {
        "max_input_length": 1000,
        "enable_input_validation": true,
        "enable_input_sanitization": true,
        "max_conversation_history_length": 10,
        "include_user_preferences": true,
        "include_learned_facts": true
    },
    "output": {
        "enable_identity_filtering": true,
        "identity_violations": ["as an ai", "as an assistant", "i'm an ai", "i'm an assistant"],
        "keywords_to_flag": ["i think", "i feel", "i believe"],
        "default_response_format": "standard",
        "store_responses_in_memory": true,
        "max_response_length": 2000
    },
    "logging": {
        "level": "INFO",
        "file": "logs/context_engine.log",
        "max_size": 10485760,
        "backup_count": 5,
        "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    }
}
```

## Configuration Options

### Context Engine

- `enable_adaptive_planning`: Whether to enable adaptive planning (default: true)
- `max_execution_attempts`: Maximum number of execution attempts (default: 3)
- `enable_memory_integration`: Whether to enable memory integration (default: true)
- `enable_identity_filtering`: Whether to enable identity filtering (default: true)

### LLM

- `local.enabled`: Whether to enable the local provider (default: true)
- `local.model`: The model to use for the local provider (default: "llama3:8b")
- `local.api_base`: The base URL for the local provider API (default: "http://localhost:11434")
- `local.options`: Additional options for the local provider (default: {})
- `cloud.enabled`: Whether to enable the cloud provider (default: true)
- `cloud.model`: The model to use for the cloud provider (default: "llama3-70b-8192")
- `cloud.api_key`: The API key for the cloud provider (default: "")
- `cloud.api_base`: The base URL for the cloud provider API (default: "https://api.groq.com/openai/v1")
- `confidence_threshold`: Default confidence threshold for general tasks (default: 0.7)
- `instruction_refinement_threshold`: Confidence threshold for instruction refinement (default: 0.8)
- `max_retries`: Maximum number of retries for rate limit errors (default: 3)
- `backoff_time`: Initial backoff time in seconds (default: 2)

### Analyzer

- `use_two_stage_analysis`: Whether to use two-stage analysis (default: true)
- `min_confidence_threshold`: Minimum confidence threshold for analysis (default: 0.7)
- `use_local_preprocessing`: Whether to use local preprocessing (default: true)
- `use_cloud_detailed_analysis`: Whether to use cloud for detailed analysis (default: true)
- `max_analysis_attempts`: Maximum number of analysis attempts (default: 3)

### Capability

- `manifest_path`: Path to the capability manifest file (default: "data/capability_manifest.json")
- `min_confidence_threshold`: Minimum confidence threshold for capability matches (default: 0.7)
- `max_matches`: Maximum number of capabilities to match (default: 5)
- `use_semantic_matching`: Whether to use semantic matching (default: true)
- `use_embeddings`: Whether to use embeddings for semantic matching (default: true)
- `embedding_model`: The model to use for embeddings (default: "text-embedding-ada-002")

### Execution

- `enable_adaptive_planning`: Whether to enable adaptive planning (default: true)
- `max_execution_attempts`: Maximum number of execution attempts (default: 3)
- `enable_chain_of_thought`: Whether to enable chain-of-thought reasoning (default: true)
- `error_recovery_methods`: List of error recovery methods to use (default: ["alternative_tool", "llm", "simplified_parameters"])
- `timeout`: Timeout for plan execution in seconds (default: 60)

### Error

- `max_retries`: Maximum number of retries for retryable errors (default: 3)
- `initial_backoff`: Initial backoff time in seconds (default: 1)
- `backoff_factor`: Backoff factor for exponential backoff (default: 2)
- `use_llm_analysis`: Whether to use LLM-based error analysis (default: true)
- `log_level`: Log level for errors (default: "ERROR")
- `include_details_in_response`: Whether to include error details in the response (default: false)

### Memory

- `enable_memory_integration`: Whether to enable memory integration (default: true)
- `store_execution_logs`: Whether to store execution logs in memory (default: true)
- `store_learned_facts`: Whether to store learned facts in memory (default: true)
- `max_conversation_history_length`: Maximum number of conversation turns to retrieve (default: 10)
- `max_learned_facts`: Maximum number of learned facts to retrieve (default: 10)
- `relevance_threshold`: Minimum relevance score for retrieved memory items (default: 0.7)

### Input

- `max_input_length`: Maximum length of user input (default: 1000)
- `enable_input_validation`: Whether to enable input validation (default: true)
- `enable_input_sanitization`: Whether to enable input sanitization (default: true)
- `max_conversation_history_length`: Maximum number of messages in conversation history (default: 10)
- `include_user_preferences`: Whether to include user preferences in input metadata (default: true)
- `include_learned_facts`: Whether to include learned facts in input metadata (default: true)

### Output

- `enable_identity_filtering`: Whether to enable identity filtering (default: true)
- `identity_violations`: List of identity violations to check for (default: ["as an ai", "as an assistant", "i'm an ai", "i'm an assistant"])
- `keywords_to_flag`: List of keywords to flag in output (default: ["i think", "i feel", "i believe"])
- `default_response_format`: Default response format (default: "standard")
- `store_responses_in_memory`: Whether to store responses in memory (default: true)
- `max_response_length`: Maximum length of response (default: 2000)

### Logging

- `level`: Log level (default: "INFO")
- `file`: Log file path (default: "logs/context_engine.log")
- `max_size`: Maximum log file size in bytes (default: 10485760)
- `backup_count`: Number of backup log files to keep (default: 5)
- `format`: Log format (default: "%(asctime)s - %(name)s - %(levelname)s - %(message)s")

## Environment Variables

The Context Engine V2 can also be configured using environment variables. Environment variables take precedence over the configuration file. The environment variables use the following naming convention:

```
CONTEXT_ENGINE_<SECTION>_<OPTION>
```

For example, to set the `local.model` option in the `llm` section, you would use the environment variable:

```
CONTEXT_ENGINE_LLM_LOCAL_MODEL=llama3:8b
```

## Configuration Loading

The Context Engine V2 loads configuration in the following order:

1. Default values
2. Configuration file
3. Environment variables

This allows you to override specific configuration options without modifying the configuration file.

## Configuration Validation

The Context Engine V2 validates the configuration when it is loaded. If any configuration options are invalid, the Context Engine will log an error and use the default value for that option.

## Configuration Examples

### Local-Only Configuration

```json
{
    "llm": {
        "local": {
            "enabled": true,
            "model": "llama3:8b",
            "api_base": "http://localhost:11434",
            "options": {}
        },
        "cloud": {
            "enabled": false
        }
    }
}
```

### Cloud-Only Configuration

```json
{
    "llm": {
        "local": {
            "enabled": false
        },
        "cloud": {
            "enabled": true,
            "model": "llama3-70b-8192",
            "api_key": "your-api-key",
            "api_base": "https://api.groq.com/openai/v1"
        }
    }
}
```

### High-Performance Configuration

```json
{
    "llm": {
        "local": {
            "enabled": true,
            "model": "llama3:8b",
            "api_base": "http://localhost:11434",
            "options": {}
        },
        "cloud": {
            "enabled": true,
            "model": "llama3-70b-8192",
            "api_key": "your-api-key",
            "api_base": "https://api.groq.com/openai/v1"
        },
        "confidence_threshold": 0.5
    },
    "analyzer": {
        "use_two_stage_analysis": true,
        "min_confidence_threshold": 0.5
    },
    "execution": {
        "enable_adaptive_planning": true,
        "max_execution_attempts": 5,
        "enable_chain_of_thought": true
    }
}
```

### Memory-Intensive Configuration

```json
{
    "memory": {
        "enable_memory_integration": true,
        "store_execution_logs": true,
        "store_learned_facts": true,
        "max_conversation_history_length": 20,
        "max_learned_facts": 20,
        "relevance_threshold": 0.5
    },
    "input": {
        "include_user_preferences": true,
        "include_learned_facts": true
    },
    "output": {
        "store_responses_in_memory": true
    }
}
```

### Error-Tolerant Configuration

```json
{
    "error": {
        "max_retries": 5,
        "initial_backoff": 0.5,
        "backoff_factor": 1.5,
        "use_llm_analysis": true,
        "log_level": "WARNING",
        "include_details_in_response": true
    },
    "execution": {
        "enable_adaptive_planning": true,
        "max_execution_attempts": 5,
        "error_recovery_methods": ["alternative_tool", "llm", "simplified_parameters", "retry"],
        "timeout": 120
    }
}
```

## Configuration Best Practices

1. **Start with Defaults**: Start with the default configuration and customize only what you need.
2. **Use Environment Variables for Secrets**: Use environment variables for sensitive information like API keys.
3. **Validate Configuration**: Validate the configuration before deploying to production.
4. **Monitor Configuration**: Monitor the configuration to ensure it is working as expected.
5. **Document Configuration**: Document any custom configuration options you use.
6. **Version Configuration**: Version your configuration files to track changes.
7. **Test Configuration Changes**: Test configuration changes in a non-production environment before deploying to production.
8. **Use Different Configurations for Different Environments**: Use different configurations for development, testing, and production environments.
9. **Backup Configuration**: Backup your configuration files regularly.
10. **Review Configuration Regularly**: Review your configuration regularly to ensure it is still appropriate.
