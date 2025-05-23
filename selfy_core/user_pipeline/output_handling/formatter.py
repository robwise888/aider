"""
Output Formatter for the Output Handling module.

This module provides the OutputFormatter class, which is responsible for
formatting output in different formats.
"""

import json
import time
from typing import Dict, List, Any, Optional

from selfy_core.global_modules.config import get as config_get
from selfy_core.global_modules.logging import get_logger

# Set up logger
logger = get_logger(__name__)


class OutputFormatter:
    """
    Formats output in different formats.
    
    The OutputFormatter is responsible for:
    1. Formatting output in different formats (text, markdown, json)
    2. Applying basic formatting rules
    """
    
    def __init__(self):
        """
        Initialize the output formatter.
        
        Loads configuration values needed for formatting.
        """
        # Load configuration
        self.default_format = config_get('pipeline.output_handling.formatting.default_format', 'text')
        
        logger.info(f"OutputFormatter initialized with default_format={self.default_format}")
    
    def format(self, text: str, target_format: str) -> str:
        """
        Format output for the specified format.
        
        Args:
            text: The output to format
            target_format: The format to use ('text', 'markdown', 'json')
            
        Returns:
            The formatted output
        """
        # Format based on the specified format
        if target_format.lower() == 'markdown':
            return self.format_as_markdown(text)
        elif target_format.lower() == 'json':
            return self.format_as_json(text)
        else:  # Default to text
            return self.format_as_text(text)
    
    def format_as_text(self, text: str) -> str:
        """
        Format output as plain text.
        
        Args:
            text: The output to format
            
        Returns:
            The formatted output
        """
        # For plain text, just return the output as is
        return text
    
    def format_as_markdown(self, text: str) -> str:
        """
        Format output as Markdown.
        
        Args:
            text: The output to format
            
        Returns:
            The formatted output
        """
        # For Markdown, ensure code blocks are properly formatted
        # This is a simple example; in a real implementation, more sophisticated
        # Markdown formatting might be applied
        
        # Check if there are unclosed code blocks
        if text.count('```') % 2 != 0:
            # Add a closing code block
            text += "\n```"
        
        return text
    
    def format_as_json(self, text: str) -> str:
        """
        Format output as JSON.
        
        Args:
            text: The output to format
            
        Returns:
            The formatted output as JSON
        """
        # Create a JSON object with the output
        json_obj = {
            "response": text
        }
        
        # Convert to JSON string
        return json.dumps(json_obj, indent=2)
