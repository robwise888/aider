# Context Engine Explanation

The Context Engine is a core component of the Selfy agent architecture, responsible for processing user requests and generating appropriate responses. Here's a detailed explanation of how it works:

## Overview

The Context Engine sits between the Input Handling and Output Handling components in the user pipeline. Its primary responsibilities are:

1. Analyzing user requests to understand their intent
2. Building appropriate contexts for LLM interactions
3. Planning and executing the steps needed to fulfill the request
4. Generating responses based on the execution results

## Components

The Context Engine consists of three main components:

1. **RequestAnalyzer**: Analyzes user requests to determine their type and parameters
2. **ContextBuilder**: Builds appropriate contexts for LLM interactions
3. **ExecutionPlanner**: Plans and executes the steps needed to fulfill the request

### RequestAnalyzer

The RequestAnalyzer is responsible for analyzing user requests to determine:
- The type of request (e.g., general query, code generation, action)
- The confidence level of the analysis
- The parameters needed to fulfill the request
- The capabilities required to fulfill the request

It uses LLMs to perform this analysis, with a fallback mechanism that uses a more powerful LLM if the initial analysis fails or has low confidence.

### ContextBuilder

The ContextBuilder is responsible for building appropriate contexts for LLM interactions. It:
- Builds different types of contexts based on the request type
- Includes relevant capabilities in the context
- Includes user preferences in the context
- Applies identity filtering to the context

The ContextBuilder supports various context types, including:
- Default context
- Capability query context
- Action context
- Code generation context
- Error recovery context
- Alternative plan context

### ExecutionPlanner

The ExecutionPlanner is responsible for planning and executing the steps needed to fulfill the request. It:
- Creates an execution plan based on the request analysis
- Executes the plan step by step
- Handles errors and retries if necessary
- Generates a final response based on the execution results

## Processing Flow

1. The user request is received from the Input Handling component
2. The RequestAnalyzer analyzes the request to determine its type and parameters
3. The ContextBuilder builds an appropriate context for LLM interaction
4. The ExecutionPlanner creates and executes a plan to fulfill the request
5. The final response is generated and passed to the Output Handling component

## Error Handling

The Context Engine includes robust error handling mechanisms:
- If the initial request analysis fails, it falls back to a more powerful LLM
- If a step in the execution plan fails, it can retry or create an alternative plan
- If all else fails, it generates an error response explaining what went wrong

## Integration with Other Components

The Context Engine integrates with several other components of the Selfy agent:
- It uses the LLM Wrapper to interact with language models
- It uses the Capability Manifest to determine available capabilities
- It uses the Identity System to filter prompts and responses
- It uses the Memory System to retrieve relevant information

## Code Structure

The Context Engine code is organized as follows:
- `core.py`: Contains the main ContextEngine class
- `request_analyzer.py`: Contains the RequestAnalyzer class
- `context_builder.py`: Contains the ContextBuilder class
- `execution_planner.py`: Contains the ExecutionPlanner class
- `utils/`: Contains utility functions for the Context Engine

## Example Flow

1. User sends a request: "Can you help me write a Python function to sort a list?"
2. RequestAnalyzer determines this is a code generation request with parameters: language=Python, task=sort a list
3. ContextBuilder builds a code generation context with these parameters
4. ExecutionPlanner creates a plan with steps: generate code, explain code
5. The plan is executed, generating Python code and an explanation
6. The final response is returned to the user

This is a high-level overview of how the Context Engine works in the Selfy agent architecture. The actual implementation includes many more details and edge cases to handle various types of requests and scenarios.
