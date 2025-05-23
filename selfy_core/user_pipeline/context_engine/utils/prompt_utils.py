"""
Prompt Utilities for the Context Engine.

This module provides utilities for validating and optimizing prompts
before they are sent to LLMs.
"""

import re
import logging
import difflib
from typing import Dict, List, Any, Tuple, Optional

# Set up logging
logger = logging.getLogger(__name__)

def validate_and_deduplicate_prompt(prompt_text: str) -> Tuple[str, Dict[str, Any]]:
    """
    Validate prompt text and remove duplications.
    
    This function detects and removes various types of duplicated content in prompts:
    1. Exact duplicated paragraphs
    2. Duplicated system instructions
    3. Duplicated user messages
    
    Args:
        prompt_text: The prompt text to validate and deduplicate
        
    Returns:
        Tuple of (cleaned_prompt_text, stats) where stats is a dictionary with:
            - original_length: Length of original prompt
            - cleaned_length: Length of cleaned prompt
            - duplications_found: Number of duplications found
            - characters_removed: Number of characters removed
    """
    if not prompt_text:
        return prompt_text, {
            "original_length": 0,
            "cleaned_length": 0,
            "duplications_found": 0,
            "characters_removed": 0
        }
    
    original_length = len(prompt_text)
    duplications_found = 0
    
    # First check for exact duplicated system prompts
    cleaned_text, system_duplications = detect_system_prompt_duplication(prompt_text)
    duplications_found += system_duplications
    
    # Then check for duplicated paragraphs
    cleaned_text, paragraph_duplications = detect_paragraph_duplication(cleaned_text)
    duplications_found += paragraph_duplications
    
    # Finally check for duplicated user messages
    cleaned_text, user_duplications = detect_user_message_duplication(cleaned_text)
    duplications_found += user_duplications
    
    cleaned_length = len(cleaned_text)
    characters_removed = original_length - cleaned_length
    
    stats = {
        "original_length": original_length,
        "cleaned_length": cleaned_length,
        "duplications_found": duplications_found,
        "characters_removed": characters_removed
    }
    
    if duplications_found > 0:
        logger.info(f"Prompt validation removed {duplications_found} duplications ({characters_removed} characters)")
        logger.debug(f"Prompt validation stats: {stats}")
    
    return cleaned_text, stats

def detect_system_prompt_duplication(prompt_text: str) -> Tuple[str, int]:
    """
    Detect and remove duplicated system prompts.
    
    This function specifically looks for duplicated system prompts that follow patterns like:
    "You are X...", "Your role is...", etc.
    
    Args:
        prompt_text: The prompt text to check
        
    Returns:
        Tuple of (cleaned_text, duplications_found)
    """
    # Common patterns for system prompts
    system_patterns = [
        r"You are [^.]+\.",
        r"Your (role|job|task) is [^.]+\.",
        r"You (should|must|need to) [^.]+\.",
        r"Your core values are[^.]+\.",
        r"Your tone is [^.]+\.",
        r"Your strengths include[^.]+\."
    ]
    
    duplications_found = 0
    cleaned_text = prompt_text
    
    # Check each pattern
    for pattern in system_patterns:
        matches = re.findall(pattern, prompt_text, re.IGNORECASE)
        if len(matches) > 1:
            # Find all occurrences with context
            occurrences = []
            for match in re.finditer(pattern, prompt_text, re.IGNORECASE):
                # Get some context around the match (up to 100 chars)
                start = max(0, match.start() - 50)
                end = min(len(prompt_text), match.end() + 50)
                context = prompt_text[start:end]
                occurrences.append((match.start(), match.end(), context))
            
            # If we have multiple occurrences, keep only the first one
            if len(occurrences) > 1:
                # Sort by position
                occurrences.sort(key=lambda x: x[0])
                
                # Keep track of what we've removed to avoid index issues
                offset = 0
                
                # Remove all but the first occurrence
                for i in range(1, len(occurrences)):
                    start, end, _ = occurrences[i]
                    # Adjust for previous removals
                    start -= offset
                    end -= offset
                    
                    # Get the text to remove
                    text_to_remove = cleaned_text[start:end]
                    
                    # Remove the text
                    cleaned_text = cleaned_text[:start] + cleaned_text[end:]
                    
                    # Update offset
                    offset += (end - start)
                    
                    # Log what we removed
                    logger.debug(f"Removed duplicated system prompt: '{text_to_remove}'")
                    duplications_found += 1
    
    # Special case for exact system prompt duplication
    if "You are Selfy" in prompt_text:
        # Count occurrences
        count = prompt_text.count("You are Selfy")
        if count > 1:
            # Replace with a single occurrence
            cleaned_text = re.sub(r"(You are Selfy.*?(?:\n\n|\Z))(You are Selfy.*?(?:\n\n|\Z))+", 
                                r"\1", cleaned_text, flags=re.DOTALL)
            duplications_found += (count - 1)
            logger.debug(f"Removed {count-1} duplicated 'You are Selfy' system prompts")
    
    return cleaned_text, duplications_found

def detect_paragraph_duplication(prompt_text: str) -> Tuple[str, int]:
    """
    Detect and remove duplicated paragraphs.
    
    This function looks for exact duplicated paragraphs in the prompt.
    
    Args:
        prompt_text: The prompt text to check
        
    Returns:
        Tuple of (cleaned_text, duplications_found)
    """
    # Split into paragraphs
    paragraphs = re.split(r'\n\s*\n', prompt_text)
    
    if len(paragraphs) <= 1:
        return prompt_text, 0
    
    # Check for duplicates
    unique_paragraphs = []
    duplications_found = 0
    
    for paragraph in paragraphs:
        # Skip empty paragraphs
        if not paragraph.strip():
            continue
            
        # Check if this paragraph is already in unique_paragraphs
        is_duplicate = False
        for existing in unique_paragraphs:
            # Use similarity ratio to account for minor differences
            similarity = difflib.SequenceMatcher(None, paragraph, existing).ratio()
            if similarity > 0.9:  # 90% similarity threshold
                is_duplicate = True
                duplications_found += 1
                logger.debug(f"Removed duplicated paragraph: '{paragraph[:50]}...'")
                break
                
        if not is_duplicate:
            unique_paragraphs.append(paragraph)
    
    # Reconstruct the text
    cleaned_text = '\n\n'.join(unique_paragraphs)
    
    return cleaned_text, duplications_found

def detect_user_message_duplication(prompt_text: str) -> Tuple[str, int]:
    """
    Detect and remove duplicated user messages.
    
    This function looks for duplicated user messages in the prompt.
    
    Args:
        prompt_text: The prompt text to check
        
    Returns:
        Tuple of (cleaned_text, duplications_found)
    """
    # Look for patterns like "User: ...", "User request: ...", etc.
    user_patterns = [
        r"User: [^\n]+",
        r"User request: [^\n]+",
        r"User query: [^\n]+",
        r"User input: [^\n]+"
    ]
    
    duplications_found = 0
    cleaned_text = prompt_text
    
    # Check each pattern
    for pattern in user_patterns:
        matches = re.findall(pattern, prompt_text, re.IGNORECASE)
        if len(matches) > 1:
            # Find all occurrences with context
            occurrences = []
            for match in re.finditer(pattern, prompt_text, re.IGNORECASE):
                # Get some context around the match (up to 100 chars)
                start = max(0, match.start() - 50)
                end = min(len(prompt_text), match.end() + 50)
                context = prompt_text[start:end]
                occurrences.append((match.start(), match.end(), context))
            
            # If we have multiple occurrences, keep only the first one
            if len(occurrences) > 1:
                # Sort by position
                occurrences.sort(key=lambda x: x[0])
                
                # Keep track of what we've removed to avoid index issues
                offset = 0
                
                # Remove all but the first occurrence
                for i in range(1, len(occurrences)):
                    start, end, _ = occurrences[i]
                    # Adjust for previous removals
                    start -= offset
                    end -= offset
                    
                    # Get the text to remove
                    text_to_remove = cleaned_text[start:end]
                    
                    # Remove the text
                    cleaned_text = cleaned_text[:start] + cleaned_text[end:]
                    
                    # Update offset
                    offset += (end - start)
                    
                    # Log what we removed
                    logger.debug(f"Removed duplicated user message: '{text_to_remove}'")
                    duplications_found += 1
    
    return cleaned_text, duplications_found
