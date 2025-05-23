"""
LLM utility functions for the Context Engine.

This module provides utility functions for working with LLMs, including:
- Robust JSON extraction from LLM responses
- Balanced bracket parsing for extracting JSON objects and arrays
- Advanced JSON sanitization for handling malformed JSON
- Conversation history formatting
- LLM call logging

The JSON extraction functions are designed to handle various edge cases:
- Text before or after JSON blocks
- Markdown code blocks
- Custom tags
- Malformed JSON with common errors
- Balanced bracket parsing for nested structures
"""

import logging
import json
import re
import time
from typing import Dict, Any, Optional, List

logger = logging.getLogger(__name__)

def extract_first_json_block(text: str) -> Dict[str, Any]:
    """
    Extracts the first valid JSON object from a string,
    even if the string contains text before or after the JSON.

    This function uses a balanced bracket approach to find valid JSON objects.

    Args:
        text: The text containing a JSON object

    Returns:
        The extracted JSON as a dictionary

    Raises:
        ValueError: If no valid JSON object is found
    """
    # Look for the first `{` and try to find the balanced closing `}`
    stack = []
    start = None
    in_string = False
    escape_next = False

    for i, c in enumerate(text):
        # Handle string literals properly
        if c == '\\' and not escape_next:
            escape_next = True
            continue

        if c == '"' and not escape_next:
            in_string = not in_string

        escape_next = False

        # Only process brackets when not in a string
        if not in_string:
            if c == '{':
                if not stack:
                    start = i
                stack.append(c)
            elif c == '}':
                if stack and stack[-1] == '{':
                    stack.pop()
                    if not stack and start is not None:
                        try:
                            json_block = text[start:i+1]
                            return json.loads(json_block)
                        except json.JSONDecodeError:
                            # Continue searching if this block isn't valid
                            continue

    # If we get here, we didn't find a valid JSON object
    raise ValueError("No valid JSON object found in the input.")

def extract_json_from_response(response: str) -> Optional[Dict[str, Any]]:
    """
    Extract a JSON object from an LLM response.

    Args:
        response: The LLM response to extract JSON from

    Returns:
        The extracted JSON object, or None if extraction failed
    """
    if not response:
        logger.warning("Empty response provided to extract_json_from_response")
        return None

    logger.debug(f"Extracting JSON from response: {response[:100]}...")

    # Method 0: Try to find JSON between ```json and ``` delimiters (our new preferred format)
    json_pattern = r"```json\s*([\s\S]*?)```"
    matches = re.findall(json_pattern, response)

    if matches:
        for match in matches:
            try:
                json_obj = json.loads(match.strip())
                logger.info("Successfully extracted JSON using ```json delimiter method")
                return json_obj
            except json.JSONDecodeError as e:
                logger.debug(f"Failed to parse JSON from ```json delimiter: {e}")
                # Try to sanitize and parse
                sanitized_json = _sanitize_json(match.strip())
                if sanitized_json:
                    try:
                        json_obj = json.loads(sanitized_json)
                        logger.info("Successfully extracted JSON using ```json delimiter with sanitization")
                        return json_obj
                    except json.JSONDecodeError:
                        logger.debug("Failed to parse sanitized JSON from ```json delimiter")
                        continue

    # Method 0.5: Try to find JSON between plain ``` and ``` delimiters (without json specifier)
    # This is needed for Groq responses which often use ``` without the json specifier
    plain_backtick_pattern = r"```\s*([\s\S]*?)```"
    matches = re.findall(plain_backtick_pattern, response)

    if matches:
        for match in matches:
            try:
                # Skip if it's clearly not JSON (e.g., starts with a language specifier)
                if match.strip().startswith("python") or match.strip().startswith("javascript"):
                    continue

                json_obj = json.loads(match.strip())
                logger.info("Successfully extracted JSON using plain ``` delimiter method")
                return json_obj
            except json.JSONDecodeError as e:
                logger.debug(f"Failed to parse JSON from plain ``` delimiter: {e}")
                # Try to sanitize and parse
                sanitized_json = _sanitize_json(match.strip())
                if sanitized_json:
                    try:
                        json_obj = json.loads(sanitized_json)
                        logger.info("Successfully extracted JSON using plain ``` delimiter with sanitization")
                        return json_obj
                    except json.JSONDecodeError:
                        logger.debug("Failed to parse sanitized JSON from plain ``` delimiter")
                        continue

    # Try to extract using the balanced JSON extraction method
    try:
        json_obj = extract_first_json_block(response)
        if json_obj and isinstance(json_obj, dict):
            logger.info("Successfully extracted JSON using balanced extraction method")
            return json_obj
    except Exception as e:
        logger.debug(f"Balanced JSON extraction failed: {e}")

    # Method 1: Try to find JSON between triple backticks (without json specifier)
    json_pattern = r"```(?!json)\s*([\s\S]*?)```"
    matches = re.findall(json_pattern, response)

    if matches:
        for match in matches:
            try:
                json_obj = json.loads(match.strip())
                logger.info("Successfully extracted JSON using generic ``` delimiter method")
                return json_obj
            except json.JSONDecodeError as e:
                logger.debug(f"Failed to parse JSON from code block: {e}")
                # Try to sanitize and parse
                sanitized_json = _sanitize_json(match.strip())
                if sanitized_json:
                    try:
                        json_obj = json.loads(sanitized_json)
                        logger.info("Successfully extracted JSON using generic ``` delimiter with sanitization")
                        return json_obj
                    except json.JSONDecodeError:
                        logger.debug("Failed to parse sanitized JSON from code block")
                        continue

    # Method 2: Try to find JSON between <json> tags
    json_pattern = r"<json>([\s\S]*?)</json>"
    matches = re.findall(json_pattern, response)

    if matches:
        for match in matches:
            try:
                return json.loads(match.strip())
            except json.JSONDecodeError:
                # Try to sanitize and parse
                sanitized_json = _sanitize_json(match.strip())
                if sanitized_json:
                    try:
                        return json.loads(sanitized_json)
                    except json.JSONDecodeError:
                        continue

    # Method 3: Try to find JSON between response_json tags
    json_pattern = r"<response_json>([\s\S]*?)</response_json>"
    matches = re.findall(json_pattern, response)

    if matches:
        for match in matches:
            try:
                return json.loads(match.strip())
            except json.JSONDecodeError:
                # Try to sanitize and parse
                sanitized_json = _sanitize_json(match.strip())
                if sanitized_json:
                    try:
                        return json.loads(sanitized_json)
                    except json.JSONDecodeError:
                        continue

    # Method 4: Try to parse the entire response as JSON
    try:
        return json.loads(response.strip())
    except json.JSONDecodeError as e:
        logger.debug(f"Failed to parse entire response as JSON: {e}")
        # Try to sanitize and parse
        sanitized_json = _sanitize_json(response.strip())
        if sanitized_json:
            try:
                return json.loads(sanitized_json)
            except json.JSONDecodeError:
                logger.debug("Failed to parse sanitized entire response")

    # Method 5: Try to find any JSON-like structure in the response
    json_pattern = r"\{[\s\S]*?\}"
    matches = re.findall(json_pattern, response)

    if matches:
        for match in matches:
            try:
                return json.loads(match.strip())
            except json.JSONDecodeError:
                # Try to sanitize and parse
                sanitized_json = _sanitize_json(match.strip())
                if sanitized_json:
                    try:
                        return json.loads(sanitized_json)
                    except json.JSONDecodeError:
                        continue

    # Method 6: Last resort - try to fix common JSON errors and parse again
    try:
        # Replace single quotes with double quotes (common error)
        fixed_content = response.replace("'", '"')
        # Try to parse the fixed content
        return json.loads(fixed_content.strip())
    except json.JSONDecodeError as e:
        logger.warning(f"Failed to extract JSON from response after multiple attempts: {e}")
        logger.debug(f"Response content (preview): {response[:200]}...")

        # Return a default structure instead of None
        return {
            "request_type": "unknown",
            "confidence": 0.0,
            "request_description": "Failed to parse response",
            "parameters": {},
            "error": True,
            "error_message": f"JSON extraction failed: {e}"
        }

def _sanitize_json(json_str: str) -> Optional[str]:
    """
    Attempt to sanitize and repair malformed JSON.

    Args:
        json_str: The JSON string to sanitize

    Returns:
        Sanitized JSON string or None if sanitization failed
    """
    try:
        logger.debug(f"Attempting to sanitize JSON: {json_str[:100]}...")

        # Remove any leading/trailing whitespace
        json_str = json_str.strip()

        # Check if the JSON is wrapped in backticks or other markers
        if json_str.startswith('```') and '```' in json_str[3:]:
            json_str = json_str[3:].split('```')[0].strip()

        # Remove any BOM characters
        if json_str.startswith('\ufeff'):
            json_str = json_str[1:]
            logger.debug("Removed BOM character from JSON")

        # Remove any comments (lines starting with // or /* */)
        json_str = re.sub(r'^\s*//.*$', '', json_str, flags=re.MULTILINE)
        json_str = re.sub(r'/\*[\s\S]*?\*/', '', json_str)

        # Replace single quotes with double quotes for keys and string values
        json_str = re.sub(r"'([^']+)'\s*:", r'"\1":', json_str)
        json_str = re.sub(r":\s*'([^']+)'([,}])", r':"\1"\2', json_str)

        # Fix common issues with trailing commas
        json_str = re.sub(r',\s*}', '}', json_str)
        json_str = re.sub(r',\s*]', ']', json_str)

        # Fix unquoted property names
        json_str = re.sub(r'([{,])\s*(\w+)\s*:', r'\1"\2":', json_str)

        # Fix missing quotes around string values
        json_str = re.sub(r':\s*(\w+)([,}])', r':"\1"\2', json_str)

        # Fix boolean and null values (ensure they're not quoted)
        json_str = re.sub(r':\s*"(true|false|null)"([,}])', r':\1\2', json_str)

        # Fix newlines in string values
        json_str = re.sub(r'"\s*\n\s*([^"]*)"', r'"\1"', json_str)

        # Fix extra colons in keys
        json_str = re.sub(r'"([^"]+):([^"]+)":', r'"\1\2":', json_str)

        # Fix missing commas between properties
        json_str = re.sub(r'}\s*{', '},{', json_str)
        json_str = re.sub(r']\s*{', '],{', json_str)
        json_str = re.sub(r'}\s*\[', '},\[', json_str)
        json_str = re.sub(r']\s*\[', '],\[', json_str)

        # Fix missing commas between object properties
        json_str = re.sub(r'"([^"]+)"\s*:\s*("[^"]*"|[\d.]+|true|false|null)\s*"', r'"\1": \2,"', json_str)

        # Log the sanitized JSON for debugging
        logger.debug(f"Sanitized JSON (preview): {json_str[:100]}...")

        # Validate the sanitized JSON
        try:
            json.loads(json_str)
            logger.info("Successfully sanitized JSON")
            return json_str
        except json.JSONDecodeError as e:
            logger.warning(f"Initial sanitization failed: {e}")
            error_msg = str(e)

            # Apply error-specific fixes based on the error message
            if "Expecting value" in error_msg and "line 1 column 1" in error_msg:
                # Text before JSON - try to find the first '{'
                start_idx = json_str.find('{')
                if start_idx >= 0:
                    json_str = json_str[start_idx:]
                    logger.debug("Fixed: Removed text before first '{'")

                    # Try parsing again after removing text
                    try:
                        json.loads(json_str)
                        logger.info("Successfully sanitized JSON after removing prefix text")
                        return json_str
                    except json.JSONDecodeError:
                        # Continue with other fixes if this didn't work
                        pass

            elif "Expecting ',' delimiter" in error_msg:
                # Try to insert comma at the error location
                match = re.search(r"line (\d+) column (\d+)", error_msg)
                if match:
                    line, col = int(match.group(1)), int(match.group(2))
                    lines = json_str.split('\n')
                    if 0 < line <= len(lines):
                        # Insert comma at the specified position
                        line_content = lines[line-1]
                        if col <= len(line_content):
                            lines[line-1] = line_content[:col] + ',' + line_content[col:]
                            json_str = '\n'.join(lines)
                            logger.debug(f"Fixed: Inserted missing comma at line {line}, column {col}")

                            # Try parsing again after inserting comma
                            try:
                                json.loads(json_str)
                                logger.info("Successfully sanitized JSON after inserting comma")
                                return json_str
                            except json.JSONDecodeError:
                                # Continue with other fixes if this didn't work
                                pass

            elif "Expecting property name enclosed in double quotes" in error_msg:
                # Try to fix unquoted property names more aggressively
                json_str = re.sub(r'([{,])\s*([a-zA-Z0-9_]+)\s*:', r'\1"\2":', json_str)
                logger.debug("Fixed: Added quotes around property names")

                # Try parsing again after fixing property names
                try:
                    json.loads(json_str)
                    logger.info("Successfully sanitized JSON after fixing property names")
                    return json_str
                except json.JSONDecodeError:
                    # Continue with other fixes if this didn't work
                    pass

            elif "Unterminated string" in error_msg:
                # Try to fix unterminated strings
                # This is a complex fix that would require tracking the exact position
                # For now, we'll use a simpler approach to add missing quotes
                json_str = re.sub(r':\s*"([^"]*?)(?=\s*[,}])', r': "\1"', json_str)
                logger.debug("Fixed: Added closing quotes to unterminated strings")

                # Try parsing again after fixing strings
                try:
                    json.loads(json_str)
                    logger.info("Successfully sanitized JSON after fixing unterminated strings")
                    return json_str
                except json.JSONDecodeError:
                    # Continue with other fixes if this didn't work
                    pass

            # Try more aggressive fixes if error-specific fixes didn't work
            # Replace unquoted boolean values
            json_str = re.sub(r':\s*true\s*([,}])', r': true\1', json_str)
            json_str = re.sub(r':\s*false\s*([,}])', r': false\1', json_str)

            # Replace unquoted null values
            json_str = re.sub(r':\s*null\s*([,}])', r': null\1', json_str)

            # Try to fix unclosed quotes
            json_str = re.sub(r':\s*"([^",}]*?)(?=\s*[,}])', r': "\1"', json_str)

            # Try to fix unclosed objects/arrays
            open_braces = json_str.count('{')
            close_braces = json_str.count('}')
            if open_braces > close_braces:
                json_str += '}' * (open_braces - close_braces)
                logger.debug(f"Fixed: Added {open_braces - close_braces} closing braces")

            open_brackets = json_str.count('[')
            close_brackets = json_str.count(']')
            if open_brackets > close_brackets:
                json_str += ']' * (open_brackets - close_brackets)
                logger.debug(f"Fixed: Added {open_brackets - close_brackets} closing brackets")

            # Try to parse again
            try:
                json.loads(json_str)
                logger.info("Advanced JSON sanitization successful")
                return json_str
            except json.JSONDecodeError:
                # Last resort: try to extract just the valid part
                try:
                    # Find the longest valid JSON substring
                    for i in range(len(json_str), 0, -1):
                        try:
                            test_str = json_str[:i]
                            # Make sure it ends with } or ]
                            if test_str.rstrip().endswith('}') or test_str.rstrip().endswith(']'):
                                json.loads(test_str)
                                logger.info("Partial JSON extraction successful")
                                return test_str
                        except:
                            continue
                except:
                    pass

                logger.error("Failed to sanitize JSON after multiple attempts")
                return None

    except Exception as e:
        logger.error(f"Error sanitizing JSON: {e}")
        return None

def extract_first_list_block(text: str) -> List[Any]:
    """
    Extracts the first valid JSON list from a string,
    even if the string contains text before or after the list.

    This function uses a balanced bracket approach to find valid JSON lists.

    Args:
        text: The text containing a JSON list

    Returns:
        The extracted JSON as a list

    Raises:
        ValueError: If no valid JSON list is found
    """
    # Look for the first `[` and try to find the balanced closing `]`
    stack = []
    start = None
    in_string = False
    escape_next = False

    for i, c in enumerate(text):
        # Handle string literals properly
        if c == '\\' and not escape_next:
            escape_next = True
            continue

        if c == '"' and not escape_next:
            in_string = not in_string

        escape_next = False

        # Only process brackets when not in a string
        if not in_string:
            if c == '[':
                if not stack:
                    start = i
                stack.append(c)
            elif c == ']':
                if stack and stack[-1] == '[':
                    stack.pop()
                    if not stack and start is not None:
                        try:
                            json_block = text[start:i+1]
                            result = json.loads(json_block)
                            if isinstance(result, list):
                                return result
                        except json.JSONDecodeError:
                            # Continue searching if this block isn't valid
                            continue

    # If we get here, we didn't find a valid JSON list
    raise ValueError("No valid JSON list found in the input.")

def extract_list_from_response(response: str) -> Optional[List[Any]]:
    """
    Extract a list from an LLM response.

    Args:
        response: The LLM response to extract a list from

    Returns:
        The extracted list, or None if extraction failed
    """
    if not response:
        logger.warning("Empty response provided to extract_list_from_response")
        return None

    logger.debug(f"Extracting list from response: {response[:100]}...")

    # Method 0: Try to find list between ```json and ``` delimiters (our new preferred format)
    list_pattern = r"```json\s*([\s\S]*?)```"
    matches = re.findall(list_pattern, response)

    if matches:
        for match in matches:
            try:
                result = json.loads(match.strip())
                if isinstance(result, list):
                    logger.info("Successfully extracted list using ```json delimiter method")
                    return result
            except json.JSONDecodeError as e:
                logger.debug(f"Failed to parse list from ```json delimiter: {e}")
                # Try to sanitize and parse
                sanitized_json = _sanitize_json(match.strip())
                if sanitized_json:
                    try:
                        result = json.loads(sanitized_json)
                        if isinstance(result, list):
                            logger.info("Successfully extracted list using ```json delimiter with sanitization")
                            return result
                    except json.JSONDecodeError:
                        logger.debug("Failed to parse sanitized list from ```json delimiter")
                        continue

    # Try to extract using the balanced list extraction method
    try:
        json_list = extract_first_list_block(response)
        if json_list and isinstance(json_list, list):
            logger.info("Successfully extracted list using balanced extraction method")
            return json_list
    except Exception as e:
        logger.debug(f"Balanced list extraction failed: {e}")

    # Try to find a list between triple backticks (without json specifier)
    list_pattern = r"```(?!json)\s*([\s\S]*?)```"
    matches = re.findall(list_pattern, response)

    if matches:
        for match in matches:
            try:
                result = json.loads(match.strip())
                if isinstance(result, list):
                    return result
            except json.JSONDecodeError:
                # Try to sanitize and parse
                sanitized_json = _sanitize_json(match.strip())
                if sanitized_json:
                    try:
                        result = json.loads(sanitized_json)
                        if isinstance(result, list):
                            return result
                    except json.JSONDecodeError:
                        continue

    # Try to parse the entire response as a list
    try:
        result = json.loads(response.strip())
        if isinstance(result, list):
            return result
    except json.JSONDecodeError:
        # Try to sanitize and parse
        sanitized_json = _sanitize_json(response.strip())
        if sanitized_json:
            try:
                result = json.loads(sanitized_json)
                if isinstance(result, list):
                    return result
            except json.JSONDecodeError:
                pass

    # Try to find any list-like structure in the response
    list_pattern = r"\[[\s\S]*?\]"
    matches = re.findall(list_pattern, response)

    if matches:
        for match in matches:
            try:
                result = json.loads(match.strip())
                if isinstance(result, list):
                    return result
            except json.JSONDecodeError:
                # Try to sanitize and parse
                sanitized_json = _sanitize_json(match.strip())
                if sanitized_json:
                    try:
                        result = json.loads(sanitized_json)
                        if isinstance(result, list):
                            return result
                    except json.JSONDecodeError:
                        continue

    # Last resort - try to fix common JSON errors and parse again
    try:
        # Replace single quotes with double quotes (common error)
        fixed_content = response.replace("'", '"')
        # Try to parse the fixed content
        result = json.loads(fixed_content.strip())
        if isinstance(result, list):
            return result
    except json.JSONDecodeError as e:
        logger.warning(f"Failed to extract list from response after multiple attempts: {e}")
        return None

def format_conversation_history(conversation_history: List[Dict[str, Any]]) -> str:
    """
    Format conversation history for inclusion in a context.

    Args:
        conversation_history: The conversation history to format

    Returns:
        A formatted string representation of the conversation history
    """
    if not conversation_history:
        return "No conversation history available."

    formatted_history = []
    for i, turn in enumerate(conversation_history):
        content = turn.get("content", "")
        role = turn.get("role", "unknown")
        formatted_turn = f"{role.capitalize()}: {content}"
        formatted_history.append(formatted_turn)

    return "\n\n".join(formatted_history)

def log_llm_call(logger, provider_name: str, model_name: str, prompt_tokens: int,
              completion_tokens: int, duration: float, error: str = None, streaming: bool = False) -> None:
    """
    Log an LLM call.

    Args:
        logger: The logger to use
        provider_name: The name of the provider (e.g., 'groq', 'ollama')
        model_name: The name of the model used
        prompt_tokens: The number of tokens in the prompt
        completion_tokens: The number of tokens in the completion
        duration: The duration of the call in seconds
        error: Optional error message if the call failed
        streaming: Whether the call was a streaming call
    """
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    stream_indicator = " (streaming)" if streaming else ""

    if error:
        logger.info(f"[{timestamp}] LLM call to {provider_name}/{model_name}{stream_indicator}: "
                   f"prompt_tokens={prompt_tokens}, completion_tokens={completion_tokens}, "
                   f"duration={duration:.2f}s, ERROR: {error}")
    else:
        logger.info(f"[{timestamp}] LLM call to {provider_name}/{model_name}{stream_indicator}: "
                   f"prompt_tokens={prompt_tokens}, completion_tokens={completion_tokens}, "
                   f"duration={duration:.2f}s")
