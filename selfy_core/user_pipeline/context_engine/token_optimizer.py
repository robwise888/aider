"""
Token Optimizer for the Selfy agent.

This module provides the TokenOptimizer class, which is responsible for
optimizing contexts to reduce token usage while preserving meaning.
"""

import logging
import re
import json
from typing import Dict, List, Any, Optional

logger = logging.getLogger(__name__)

class TokenOptimizer:
    """
    Optimizes contexts to reduce token usage while preserving meaning.
    
    The TokenOptimizer is responsible for:
    1. Estimating token counts for different content types
    2. Summarizing conversation history
    3. Compressing context while preserving meaning
    4. Optimizing context for different LLM providers
    """
    
    def __init__(self, llm_wrapper=None, max_tokens=4000):
        """
        Initialize the token optimizer.
        
        Args:
            llm_wrapper: The LLM wrapper to use for optimization
            max_tokens: The maximum number of tokens to target
        """
        logger.info("Initializing TokenOptimizer")
        
        self.llm_wrapper = llm_wrapper
        self.max_tokens = max_tokens
        
        # Token estimation multipliers for different content types
        self.token_multipliers = {
            "general": 0.25,  # General text (1 token ≈ 4 characters)
            "code": 0.2,      # Code (1 token ≈ 5 characters)
            "json": 0.3,      # JSON (1 token ≈ 3.3 characters)
            "markdown": 0.22  # Markdown (1 token ≈ 4.5 characters)
        }
        
        logger.info("TokenOptimizer initialized successfully")
    
    def optimize(self, context: str, preferred_llm: str = "cloud") -> str:
        """
        Optimize a context to reduce token usage while preserving meaning.
        
        Args:
            context: The context to optimize
            preferred_llm: The preferred LLM to use for optimization
            
        Returns:
            The optimized context
        """
        # Only optimize if the context is large enough to warrant optimization
        estimated_tokens = self.estimate_tokens(context)
        if estimated_tokens <= self.max_tokens:
            logger.info(f"Context already within token limit ({estimated_tokens} <= {self.max_tokens}), skipping optimization")
            return context
            
        logger.info(f"Optimizing context: estimated {estimated_tokens} tokens (target: {self.max_tokens})")
        
        # Use LLM to optimize the context
        if self.llm_wrapper:
            return self._optimize_with_llm(context, preferred_llm)
        else:
            # Fallback to basic optimization if no LLM wrapper is available
            return self._basic_optimize(context)
    
    def estimate_tokens(self, text: str) -> int:
        """
        Estimate the number of tokens in a text.
        
        Args:
            text: The text to estimate tokens for
            
        Returns:
            The estimated number of tokens
        """
        if not text:
            return 0
            
        # Detect content type
        if re.search(r'```[a-z]*\n', text):
            # Contains code blocks
            multiplier = self.token_multipliers["code"]
        elif re.search(r'[{}\[\]":]', text) and len(re.findall(r'[{}\[\]":]', text)) > len(text) * 0.05:
            # Likely JSON
            multiplier = self.token_multipliers["json"]
        elif re.search(r'[#*_`]', text) and len(re.findall(r'[#*_`]', text)) > len(text) * 0.02:
            # Likely Markdown
            multiplier = self.token_multipliers["markdown"]
        else:
            # General text
            multiplier = self.token_multipliers["general"]
            
        # Estimate tokens based on character count and content type
        return int(len(text) * multiplier)
    
    def _optimize_with_llm(self, context: str, preferred_llm: str = "cloud") -> str:
        """
        Use LLM to optimize the context.
        
        Args:
            context: The context to optimize
            preferred_llm: The preferred LLM to use for optimization
            
        Returns:
            The optimized context
        """
        logger.info(f"Optimizing context with LLM (preferred: {preferred_llm})")
        
        # Create a prompt for context optimization
        prompt = f"""
        You are an expert at optimizing contexts to reduce token usage while preserving meaning.
        Your task is to optimize the following context to reduce its token count while preserving
        all essential information and meaning.
        
        Original context:
        {context}
        
        Please optimize this context by:
        1. Removing redundant or unnecessary information
        2. Summarizing verbose sections
        3. Using more concise language
        4. Preserving all critical information and capabilities
        5. Maintaining the overall structure and flow
        
        Target token count: {self.max_tokens} tokens
        
        Return ONLY the optimized context, without any explanations or additional text.
        """
        
        try:
            # Use the appropriate LLM based on preference
            if preferred_llm == "cloud" and hasattr(self.llm_wrapper, 'generate_chat_completion'):
                # Use cloud LLM for optimization
                logger.info("Using cloud LLM for context optimization")
                messages = [
                    {"role": "system", "content": "You are an expert at optimizing contexts to reduce token usage."},
                    {"role": "user", "content": prompt}
                ]
                response = self.llm_wrapper.generate_chat_completion(messages, temperature=0.3)
                optimized_context = response.content
            else:
                # Use local LLM for optimization
                logger.info("Using local LLM for context optimization")
                if hasattr(self.llm_wrapper, 'generate_text'):
                    response = self.llm_wrapper.generate_text(prompt, temperature=0.3)
                    optimized_context = response.content
                else:
                    # Fallback to cloud if local is not available
                    logger.warning("Local LLM not available, falling back to cloud LLM")
                    messages = [
                        {"role": "system", "content": "You are an expert at optimizing contexts to reduce token usage."},
                        {"role": "user", "content": prompt}
                    ]
                    response = self.llm_wrapper.generate_chat_completion(messages, temperature=0.3)
                    optimized_context = response.content
            
            # Check if optimization was successful
            if optimized_context and len(optimized_context.strip()) > 0:
                original_tokens = self.estimate_tokens(context)
                optimized_tokens = self.estimate_tokens(optimized_context)
                logger.info(f"Context optimization successful: {original_tokens} tokens -> {optimized_tokens} tokens")
                return optimized_context
            else:
                logger.warning("Context optimization returned empty result, using original context")
                return context
        except Exception as e:
            logger.error(f"Error optimizing context with LLM: {e}")
            return context  # Return original context on error
    
    def _basic_optimize(self, context: str) -> str:
        """
        Basic optimization without using an LLM.
        
        Args:
            context: The context to optimize
            
        Returns:
            The optimized context
        """
        logger.info("Using basic optimization (no LLM available)")
        
        # Split context into sections
        sections = re.split(r'\n\s*\n', context)
        
        # Keep important sections
        important_sections = []
        for section in sections:
            # Keep sections with important keywords
            if re.search(r'(capability|function|parameter|argument|required|user request|instruction)', section, re.IGNORECASE):
                important_sections.append(section)
            # Keep short sections
            elif len(section) < 200:
                important_sections.append(section)
            # Truncate long sections
            else:
                truncated = section[:200] + "..."
                important_sections.append(truncated)
        
        # Reassemble the context
        optimized_context = "\n\n".join(important_sections)
        
        original_tokens = self.estimate_tokens(context)
        optimized_tokens = self.estimate_tokens(optimized_context)
        logger.info(f"Basic optimization: {original_tokens} tokens -> {optimized_tokens} tokens")
        
        return optimized_context
