"""
Semantic matching for capabilities.

This module provides functions for semantically matching capabilities to queries.
"""

import logging
from typing import List, Dict, Any, Optional, Tuple, Union

from selfy_core.global_modules.config import get as config_get
from selfy_core.global_modules.capability_manifest.exceptions import CapabilityError

# Set up logger
logger = logging.getLogger(__name__)


def semantic_match_capabilities(query_text: str, capabilities: List[Dict[str, Any]], 
                              top_k: int = 5, llm_wrapper=None) -> List[Tuple[str, float]]:
    """
    Match capabilities to a query using semantic matching.

    Args:
        query_text: The query to match
        capabilities: The capabilities to match against
        top_k: Maximum number of capabilities to return
        llm_wrapper: Optional LLM wrapper to use for matching

    Returns:
        List of (capability_id, score) tuples
    """
    try:
        # If LLM wrapper is provided, use LLM-based matching
        if llm_wrapper:
            return _llm_match_capabilities(query_text, capabilities, top_k, llm_wrapper)
        
        # Otherwise, fall back to keyword matching
        return _keyword_match_capabilities(query_text, capabilities, top_k)
    except Exception as e:
        logger.error(f"Error in semantic matching: {e}", exc_info=True)
        # Fall back to keyword matching
        return _keyword_match_capabilities(query_text, capabilities, top_k)


def _llm_match_capabilities(query_text: str, capabilities: List[Dict[str, Any]], 
                          top_k: int, llm_wrapper) -> List[Tuple[str, float]]:
    """
    Match capabilities to a query using an LLM.

    Args:
        query_text: The query to match
        capabilities: The capabilities to match against
        top_k: Maximum number of capabilities to return
        llm_wrapper: LLM wrapper to use for matching

    Returns:
        List of (capability_id, score) tuples
    """
    try:
        # Create a prompt for the LLM
        prompt = _create_capability_matching_prompt(query_text, capabilities)
        
        # Generate a response
        response = llm_wrapper.generate_text(prompt, temperature=0.2)
        
        # Parse the response
        matches = _parse_llm_capability_matches(response, capabilities)
        
        # Limit to top_k
        return matches[:top_k]
    except Exception as e:
        logger.error(f"Error in LLM-based capability matching: {e}", exc_info=True)
        # Fall back to keyword matching
        return _keyword_match_capabilities(query_text, capabilities, top_k)


def _create_capability_matching_prompt(query_text: str, capabilities: List[Dict[str, Any]]) -> str:
    """
    Create a prompt for matching capabilities using an LLM.

    Args:
        query_text: The query to match
        capabilities: The capabilities to match against

    Returns:
        The prompt
    """
    # Create a list of capability descriptions
    capability_descriptions = []
    for capability in capabilities:
        name = capability.get("name", "")
        description = capability.get("description", "")
        category = capability.get("category", "")
        
        capability_descriptions.append(f"- {name}: {description} (Category: {category})")
    
    # Create the prompt
    prompt = f"""
    You are a capability matching system. Your task is to match the user's query to the most relevant capabilities.
    
    User Query: {query_text}
    
    Available Capabilities:
    {chr(10).join(capability_descriptions)}
    
    For each capability, assign a relevance score from 0.0 to 1.0, where 1.0 means the capability is perfectly relevant to the query.
    
    Return your response in the following format:
    capability_name1: score1
    capability_name2: score2
    ...
    
    Only include capabilities with a score of 0.5 or higher.
    """
    
    return prompt


def _parse_llm_capability_matches(response: Any, capabilities: List[Dict[str, Any]]) -> List[Tuple[str, float]]:
    """
    Parse the LLM response to extract capability matches.

    Args:
        response: The LLM response
        capabilities: The capabilities that were matched against

    Returns:
        List of (capability_id, score) tuples
    """
    try:
        # Get the response text
        if hasattr(response, 'content'):
            response_text = response.content
        else:
            response_text = str(response)
        
        # Parse the response
        matches = []
        for line in response_text.strip().split('\n'):
            line = line.strip()
            if not line or ':' not in line:
                continue
            
            # Split the line into capability name and score
            parts = line.split(':', 1)
            if len(parts) != 2:
                continue
            
            capability_name = parts[0].strip()
            try:
                score = float(parts[1].strip())
            except ValueError:
                continue
            
            # Add to matches
            matches.append((capability_name, score))
        
        # Sort by score in descending order
        matches.sort(key=lambda x: x[1], reverse=True)
        
        return matches
    except Exception as e:
        logger.error(f"Error parsing LLM capability matches: {e}", exc_info=True)
        return []


def _keyword_match_capabilities(query_text: str, capabilities: List[Dict[str, Any]], 
                              top_k: int) -> List[Tuple[str, float]]:
    """
    Match capabilities to a query using keywords.

    Args:
        query_text: The query to match
        capabilities: The capabilities to match against
        top_k: Maximum number of capabilities to return

    Returns:
        List of (capability_id, score) tuples
    """
    # Convert query to lowercase for case-insensitive matching
    query_lower = query_text.lower()
    query_words = query_lower.split()
    
    # Calculate scores for each capability
    matches = []
    for capability in capabilities:
        name = capability.get("name", "").lower()
        description = capability.get("description", "").lower()
        category = capability.get("category", "").lower()
        
        # Calculate score based on word matches
        score = 0.0
        
        # Check for exact matches in name
        if query_lower in name:
            score += 0.8
        
        # Check for word matches in name
        for word in query_words:
            if word in name:
                score += 0.5
        
        # Check for word matches in description
        for word in query_words:
            if word in description:
                score += 0.3
        
        # Check for word matches in category
        for word in query_words:
            if word in category:
                score += 0.2
        
        # Normalize score to [0, 1]
        score = min(1.0, score)
        
        # Add to matches if score is above threshold
        if score >= 0.1:
            matches.append((capability.get("name", ""), score))
    
    # Sort by score in descending order
    matches.sort(key=lambda x: x[1], reverse=True)
    
    # Limit to top_k
    return matches[:top_k]
