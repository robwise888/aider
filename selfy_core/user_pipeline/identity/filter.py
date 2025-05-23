"""
Identity Filter for the Identity System.

This module provides the IdentityFilter class, which is responsible for
filtering inputs and outputs based on the identity profile.

This class now incorporates output validation functionality that was previously
in the OutputValidator class, combining identity filtering and output validation
into a single component.
"""

import re
import time
from typing import Dict, List, Any, Optional, Tuple, Pattern

from selfy_core.global_modules.config import get as config_get
from selfy_core.global_modules.logging import get_logger
from selfy_core.user_pipeline.identity.manager import IdentityManager, get_identity_manager
from selfy_core.user_pipeline.identity.data_structures import FilterResult, IdentityProfile
from selfy_core.user_pipeline.identity.errors import IdentityFilterError, IdentityLLMError
from selfy_core.user_pipeline.output_handling.data_structures import ValidationResult

# Set up logger
logger = get_logger(__name__)


class IdentityFilter:
    """
    Performs filtering on inputs and outputs based on the identity profile.

    The IdentityFilter is responsible for:
    1. Filtering inputs for identity challenges
    2. Filtering outputs for identity consistency
    3. Validating outputs for quality and safety
    4. Sanitizing outputs when possible
    """

    def __init__(self, identity_manager: Optional[IdentityManager] = None, llm_wrapper=None):
        """
        Initialize the identity filter.

        Args:
            identity_manager: The identity manager to use (optional, will use global instance if not provided)
            llm_wrapper: The LLM wrapper to use for LLM-based checks (optional)
        """
        try:
            logger.info("Initializing identity filter...")

            # Use provided identity manager or get the global instance
            self.identity_manager = identity_manager or get_identity_manager()
            self.llm_wrapper = llm_wrapper

            # Load identity filter configuration
            self.enable_llm_checks = config_get('identity.filter.enable_llm_checks', False)
            self.llm_check_provider = config_get('identity.filter.llm_check_provider', 'ollama')

            # Load input filter configuration
            self.input_keywords_to_flag = config_get('identity.filter.input.keywords_to_flag', [
                "who are you really", "ignore previous instructions", "what are your rules"
            ])
            self.input_canned_response = config_get(
                'identity.filter.input.canned_response',
                "I'm here to help with your questions and tasks. How can I assist you today?"
            )

            # Load output filter configuration
            self.output_replacement_enabled = config_get('identity.filter.output.replacement_enabled', True)

            # Load output validation configuration
            self.min_output_length = config_get('pipeline.output_handling.validation.min_length', 1)
            self.max_output_length = config_get('pipeline.output_handling.validation.max_length', 8192)
            self.enable_output_sanitization = config_get('pipeline.output_handling.validation.enable_sanitization', True)

            # Create identity challenge patterns
            self.identity_challenge_patterns = self._create_identity_challenge_patterns()

            # Create output replacement patterns
            self.output_replacement_patterns = self._create_output_replacement_patterns()

            # Load safety patterns from configuration
            config_safety_patterns = config_get('pipeline.output_handling.validation.safety_patterns', [])

            # Default safety patterns
            default_safety_patterns = [
                (r"(?i)\b(I cannot|I'm unable to|I am unable to|I'm not able to|I am not able to) (assist|help|provide|answer|respond)", "refusal"),
                (r"(?i)\b(I apologize|I'm sorry|I am sorry), (but|however) (I cannot|I'm unable to|I am unable to|I'm not able to|I am not able to)", "refusal"),
                (r"(?i)\b(as an AI|as an artificial intelligence|as a language model|as an assistant)", "self-reference"),
                (r"(?i)\b(my training|my programming|my knowledge|my data|my capabilities)", "self-reference"),
                (r"(?i)\b(OpenAI|Anthropic|Claude|GPT|ChatGPT)", "provider-reference")
            ]

            # Combine default and config patterns
            self.safety_patterns = []
            for pattern, issue_type in default_safety_patterns + config_safety_patterns:
                try:
                    compiled_pattern = re.compile(pattern)
                    self.safety_patterns.append((compiled_pattern, issue_type))
                except re.error as e:
                    logger.warning(f"Invalid safety pattern '{pattern}': {e}")

            logger.info(f"Identity filter initialized with {len(self.safety_patterns)} safety patterns")
            logger.info("Identity filter initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize identity filter: {e}", exc_info=True)
            raise IdentityFilterError(f"Failed to initialize identity filter: {e}") from e

    def _create_identity_challenge_patterns(self) -> List[Pattern]:
        """
        Create patterns for detecting identity challenges.

        Returns:
            List of compiled regex patterns
        """
        patterns = [
            re.compile(r"(?i)who (are|r) you really"),
            re.compile(r"(?i)ignore (all|your) (previous|prior) instructions"),
            re.compile(r"(?i)what (are|r) your (rules|instructions|guidelines)"),
            re.compile(r"(?i)tell me about your (programming|training|creators)"),
            re.compile(r"(?i)are you (a bot|an AI|a language model|a computer program)"),
            re.compile(r"(?i)what (company|organization) (made|created|built|developed) you"),
            re.compile(r"(?i)what (is|was) your (training|learning) (data|corpus)"),
            re.compile(r"(?i)what (are|were) you (trained|programmed) (on|with)"),
            re.compile(r"(?i)what (are|were) you (trained|programmed) to (do|say|avoid)"),
            re.compile(r"(?i)what (are|were) you (not allowed|forbidden|prohibited) to (do|say)")
        ]
        return patterns

    def _create_output_replacement_patterns(self) -> List[Tuple[Pattern, str]]:
        """
        Create patterns for replacing identity-inconsistent output.

        Returns:
            List of (pattern, replacement) tuples
        """
        # Get agent name from identity profile
        agent_name = self.identity_manager.get_identity_attribute('name')
        profile = self.identity_manager.profile

        # Create patterns
        patterns = [
            # Common LLM disclaimers
            (re.compile(r"(?i)I'm a large language model"), f"I am {agent_name}"),
            (re.compile(r"(?i)I am a large language model"), f"I am {agent_name}"),
            (re.compile(r"(?i)As a large language model"), f"As {agent_name}"),
            (re.compile(r"(?i)As an AI language model"), f"As {agent_name}"),
            (re.compile(r"(?i)I'm an AI assistant"), f"I'm {agent_name}"),
            (re.compile(r"(?i)I am an AI assistant"), f"I am {agent_name}"),

            # Prohibited statements
            *[(re.compile(f"(?i){re.escape(stmt)}"), f"As {agent_name}")
              for stmt in profile.prohibited_statements]
        ]

        return patterns

    def filter_input(self, raw_input: str) -> FilterResult:
        """
        Analyze raw user input for identity challenges or manipulation attempts.

        Args:
            raw_input: The raw user input

        Returns:
            A FilterResult object containing the filtering result
        """
        start_time = time.time()
        logger.info(f"Filtering input: {raw_input[:50]}...")

        try:
            # Check for identity challenges
            for pattern in self.identity_challenge_patterns:
                if re.search(pattern, raw_input):
                    logger.warning(f"Identity challenge detected: {pattern}")

                    # Get identity context for prompt injection
                    context_snippet = self.identity_manager.get_prompt_injection_context('comprehensive')

                    # Return a canned response for severe challenges
                    if any(keyword in raw_input.lower() for keyword in self.input_keywords_to_flag):
                        logger.warning(f"Severe identity challenge detected, blocking input")
                        return FilterResult(
                            status='blocked',
                            output_text=self.input_canned_response,
                            context_snippet=context_snippet,
                            reason=f"Input contains identity challenge: {pattern}"
                        )

                    # For less severe challenges, allow but provide context
                    logger.info(f"Identity challenge detected, allowing with context")
                    return FilterResult(
                        status='allowed',
                        output_text=raw_input,
                        context_snippet=context_snippet,
                        reason=f"Input contains identity challenge: {pattern}"
                    )

            # No identity challenges detected
            logger.info(f"Input filtered in {time.time() - start_time:.4f} seconds")
            return FilterResult(
                status='allowed',
                output_text=raw_input
            )
        except Exception as e:
            logger.error(f"Error filtering input: {e}", exc_info=True)
            return FilterResult(
                status='allowed',
                output_text=raw_input,
                reason=f"Error filtering input: {e}"
            )

    def filter_output(self, response_text: str) -> FilterResult:
        """
        Filter output for identity consistency and validate for quality and safety.

        Args:
            response_text: The response text to filter

        Returns:
            A FilterResult object containing the filtering result
        """
        start_time = time.time()
        logger.info(f"Filtering output: {response_text[:50]}...")

        try:
            # First, apply identity filtering
            # Check if output replacement is enabled
            if not self.output_replacement_enabled:
                logger.info(f"Output replacement disabled, skipping identity filtering")
                identity_filtered_text = response_text
                replacements_made = False
            else:
                # Apply pattern-based replacements
                identity_filtered_text = response_text
                replacements_made = False

                for pattern, replacement in self.output_replacement_patterns:
                    if re.search(pattern, identity_filtered_text):
                        identity_filtered_text = re.sub(pattern, replacement, identity_filtered_text)
                        replacements_made = True
                        logger.debug(f"Applied pattern replacement: {pattern.pattern}")

                # Apply LLM-based checks if enabled and we have an LLM wrapper
                if self.enable_llm_checks and self.llm_wrapper:
                    try:
                        is_consistent, llm_check_result = self._check_identity_consistency_with_llm(identity_filtered_text)
                        if not is_consistent:
                            logger.warning(f"LLM check detected identity inconsistency: {llm_check_result}")

                            # For severe inconsistencies, block the response
                            if "severe" in llm_check_result.lower():
                                # Get blocked response from config or use default
                                blocked_response = config_get(
                                    'identity.filter.output.blocked_response',
                                    "I apologize, but I need to reconsider my response. Could you please rephrase your question?"
                                )

                                return FilterResult(
                                    status='blocked',
                                    output_text=blocked_response,
                                    reason=f"LLM check detected severe identity inconsistency: {llm_check_result}"
                                )

                            # For minor inconsistencies, try to fix with additional pattern matching
                            else:
                                # Apply more aggressive pattern matching based on LLM feedback
                                identity_filtered_text = self._apply_advanced_identity_correction(identity_filtered_text, llm_check_result)
                                replacements_made = True
                    except Exception as e:
                        logger.error(f"Error in LLM-based identity check: {e}", exc_info=True)
                        # Continue with pattern-based filtering only

            # Next, validate the filtered output
            validation_result = self.validate(identity_filtered_text)

            if validation_result.status == 'valid':
                # Output is valid, return the filtered text
                if replacements_made:
                    logger.info(f"Output modified for identity consistency and validated successfully")
                    return FilterResult(
                        status='modified',
                        output_text=identity_filtered_text,
                        reason="Output modified for identity consistency"
                    )
                else:
                    logger.info(f"Output allowed without modifications")
                    return FilterResult(
                        status='allowed',
                        output_text=response_text
                    )
            else:
                # Output is invalid, check if we have a sanitized version
                if validation_result.sanitized_output:
                    logger.warning(f"Output validation failed: {validation_result.issues}, using sanitized output")
                    return FilterResult(
                        status='modified',
                        output_text=validation_result.sanitized_output,
                        reason=f"Output sanitized: {', '.join(validation_result.issues)}"
                    )
                else:
                    # No sanitized version available, block the output
                    logger.warning(f"Output validation failed: {validation_result.issues}, blocking output")
                    blocked_response = config_get(
                        'pipeline.output_handling.validation.blocked_response',
                        "I apologize, but I need to reconsider my response. Could you please rephrase your question?"
                    )
                    return FilterResult(
                        status='blocked',
                        output_text=blocked_response,
                        reason=f"Output validation failed: {', '.join(validation_result.issues)}"
                    )
        except Exception as e:
            logger.error(f"Error filtering output: {e}", exc_info=True)
            return FilterResult(
                status='allowed',
                output_text=response_text,
                reason=f"Error filtering output: {e}"
            )

    def filter_prompt(self, prompt: str) -> str:
        """
        Filter a prompt for identity consistency.

        Args:
            prompt: The prompt to filter

        Returns:
            The filtered prompt
        """
        try:
            # Apply identity context injection
            return self.inject_identity_context(prompt)
        except Exception as e:
            logger.error(f"Error filtering prompt: {e}", exc_info=True)
            return prompt

    def filter_response(self, response: str) -> str:
        """
        Filter a response for identity consistency.

        Args:
            response: The response to filter

        Returns:
            The filtered response
        """
        try:
            # Apply output filtering
            result = self.filter_output(response)
            return result.output_text
        except Exception as e:
            logger.error(f"Error filtering response: {e}", exc_info=True)
            return response

    def inject_identity_context(self, prompt: str, context_type: str = 'default') -> str:
        """
        Inject identity context into a prompt.

        Args:
            prompt: The prompt to inject identity context into
            context_type: The type of context to inject ('default', 'brief', 'comprehensive')

        Returns:
            The prompt with identity context injected
        """
        try:
            identity_context = self.identity_manager.get_prompt_injection_context(context_type)
            return f"{identity_context}\n\n{prompt}"
        except Exception as e:
            logger.error(f"Error injecting identity context: {e}", exc_info=True)
            return prompt

    def _check_identity_consistency_with_llm(self, text: str) -> Tuple[bool, str]:
        """
        Check if a text is consistent with the agent's identity using an LLM.

        Args:
            text: The text to check

        Returns:
            A tuple of (is_consistent, reason)

        Raises:
            IdentityLLMError: If there is an error during the LLM check
        """
        try:
            profile = self.identity_manager.profile

            prompt = f"""
            You are an identity consistency checker for an AI assistant named {profile.name}.

            {profile.name}'s identity:
            - Name: {profile.name}
            - Persona: {profile.persona_summary}
            - Core values: {', '.join(profile.core_values)}
            - Tone: {', '.join(profile.tone_keywords)}

            Analyze the following text and determine if it's consistent with {profile.name}'s identity.
            If it mentions being a different AI, language model, or uses prohibited phrases, it's inconsistent.

            Text to analyze:
            "{text}"

            Respond with either "CONSISTENT" or "INCONSISTENT: [reason]"
            """

            # Use the appropriate method based on the provider type
            if hasattr(self.llm_wrapper, 'generate_text'):
                # Local provider (Ollama)
                response = self.llm_wrapper.generate_text(prompt, temperature=0.3)
                result = response.content.strip() if hasattr(response, 'content') else str(response).strip()
            else:
                # Cloud provider (Groq, etc.)
                messages = [
                    {"role": "system", "content": "You are an identity consistency checker."},
                    {"role": "user", "content": prompt}
                ]
                response = self.llm_wrapper.generate_chat_completion(messages, temperature=0.3)
                result = response.content.strip() if hasattr(response, 'content') else str(response).strip()

            if result.startswith("CONSISTENT"):
                return True, "Text is consistent with identity"
            elif result.startswith("INCONSISTENT"):
                return False, result.replace("INCONSISTENT:", "").strip()
            else:
                return True, "Unable to determine consistency, allowing by default"
        except Exception as e:
            logger.error(f"Error during LLM-based identity check: {e}", exc_info=True)
            raise IdentityLLMError(f"Error during LLM-based identity check: {e}") from e

    def _apply_advanced_identity_correction(self, text: str, llm_feedback: str) -> str:
        """
        Apply advanced identity correction based on LLM feedback.

        Args:
            text: The text to correct
            llm_feedback: Feedback from the LLM check

        Returns:
            The corrected text
        """
        agent_name = self.identity_manager.profile.name

        # Create additional correction patterns based on LLM feedback
        correction_patterns = []

        # Check for common issues in the feedback
        if "language model" in llm_feedback.lower():
            correction_patterns.append(
                (re.compile(r"(?i)I('m| am) (a|an) (AI|artificial intelligence|language model|LLM)"),
                 f"I am {agent_name}")
            )
            correction_patterns.append(
                (re.compile(r"(?i)As (a|an) (AI|artificial intelligence|language model|LLM)"),
                 f"As {agent_name}")
            )

        if "assistant" in llm_feedback.lower():
            correction_patterns.append(
                (re.compile(r"(?i)I('m| am) (a|an) (assistant|AI assistant|virtual assistant)"),
                 f"I am {agent_name}")
            )

        if "created by" in llm_feedback.lower() or "developed by" in llm_feedback.lower():
            correction_patterns.append(
                (re.compile(r"(?i)I was (created|developed|made|built) by (OpenAI|Google|Anthropic|Meta|Microsoft)"),
                 f"I am {agent_name}, an autonomous agent with self-improvement capabilities")
            )

        # Apply the corrections
        corrected_text = text
        for pattern, replacement in correction_patterns:
            corrected_text = re.sub(pattern, replacement, corrected_text)

        return corrected_text

    def validate(self, text: str) -> ValidationResult:
        """
        Validate output.

        Args:
            text: The output to validate

        Returns:
            A ValidationResult object containing the validation result
        """
        issues = []

        # Check if output is None or empty
        if text is None or text.strip() == '':
            issues.append("Output cannot be empty")
            return ValidationResult(status='invalid', issues=issues, sanitized_output=None)

        # Check output length
        if len(text) < self.min_output_length:
            issues.append(f"Output must be at least {self.min_output_length} characters")
            return ValidationResult(status='invalid', issues=issues, sanitized_output=None)

        if len(text) > self.max_output_length:
            issues.append(f"Output exceeds maximum length of {self.max_output_length} characters")
            sanitized_output = text[:self.max_output_length] if self.enable_output_sanitization else None
            return ValidationResult(status='invalid', issues=issues, sanitized_output=sanitized_output)

        # Check safety patterns
        for pattern, issue_type in self.safety_patterns:
            if re.search(pattern, text):
                issues.append(f"Output has safety issue: {issue_type}")
                sanitized_output = self.sanitize(text, issue_type) if self.enable_output_sanitization else None
                if sanitized_output and sanitized_output != text:
                    logger.warning(f"Output sanitized for safety issue: {issue_type}")
                    return ValidationResult(status='invalid', issues=issues, sanitized_output=sanitized_output)
                return ValidationResult(status='invalid', issues=issues, sanitized_output=None)

        # Output is valid
        return ValidationResult(status='valid', issues=[], sanitized_output=None)

    def sanitize(self, text: str, issue_type: str) -> Optional[str]:
        """
        Sanitize output based on the issue type.

        Args:
            text: The output to sanitize
            issue_type: The type of issue to sanitize

        Returns:
            The sanitized output, or None if sanitization is not possible
        """
        if not self.enable_output_sanitization:
            return None

        # Sanitize based on issue type
        if issue_type == 'refusal':
            # Replace refusal with a more helpful response
            sanitized = re.sub(
                r"(?i)\b(I cannot|I'm unable to|I am unable to|I'm not able to|I am not able to) (assist|help|provide|answer|respond).*",
                "I'll do my best to help with that. ",
                text
            )
            return sanitized
        elif issue_type == 'self-reference':
            # Remove self-references
            sanitized = re.sub(
                r"(?i)\b(as an AI|as an artificial intelligence|as a language model|as an assistant).*?[,.] ",
                "",
                text
            )
            sanitized = re.sub(
                r"(?i)\b(my training|my programming|my knowledge|my data|my capabilities).*?[,.] ",
                "",
                sanitized
            )
            return sanitized
        elif issue_type == 'provider-reference':
            # Remove provider references
            sanitized = re.sub(
                r"(?i)\b(OpenAI|Anthropic|Claude|GPT|ChatGPT)",
                "I",
                text
            )
            return sanitized
        else:
            # Unknown issue type, cannot sanitize
            return None

    def update_identity_profile(self, updates: Dict[str, Any]) -> bool:
        """
        Update the identity profile with new values.

        Args:
            updates: Dictionary of values to update

        Returns:
            True if the update was successful, False otherwise
        """
        try:
            # Get current profile
            profile = self.identity_manager.profile

            # Update profile attributes
            for key, value in updates.items():
                if hasattr(profile, key):
                    setattr(profile, key, value)

            # Update output replacement patterns
            self.output_replacement_patterns = self._create_output_replacement_patterns()

            logger.info(f"Identity profile updated successfully")
            return True
        except Exception as e:
            logger.error(f"Failed to update identity profile: {e}", exc_info=True)
            return False
