"""
Execution Planner for the Context Engine.

This module provides the ExecutionPlanner class, which is responsible for
planning and executing capability calls to fulfill user requests.
"""

import logging
import time
import json
import uuid
from typing import Dict, List, Any, Optional, Tuple, Union

from selfy_core.global_modules.config import get as config_get
from selfy_core.user_pipeline.context_engine.data_structures import ExecutionPlan, ExecutionStep
from selfy_core.user_pipeline.context_engine.utils.llm_utils import extract_json_from_response, log_llm_call
from selfy_core.user_pipeline.context_engine.token_optimizer import TokenOptimizer

# Custom JSON encoder for handling non-serializable objects
class CustomJSONEncoder(json.JSONEncoder):
    """Custom JSON encoder for handling non-serializable objects."""
    def default(self, obj):
        try:
            return super().default(obj)
        except TypeError:
            return str(obj)

def safe_json_dumps(obj, indent=2):
    """
    Safely convert an object to a JSON string, handling non-serializable objects.

    Args:
        obj: The object to convert
        indent: The indentation level

    Returns:
        A JSON string representation of the object
    """
    return json.dumps(obj, indent=indent, cls=CustomJSONEncoder)

logger = logging.getLogger(__name__)

class ExecutionPlanner:
    """
    Plans and executes capability calls to fulfill user requests.

    The ExecutionPlanner is responsible for:
    1. Creating execution plans based on request analysis
    2. Executing plans by calling capabilities
    3. Handling errors and retries
    4. Adaptive planning and execution
    5. Plan repair and alternative plan generation
    6. Generating final responses
    """

    def __init__(self, llm_wrapper=None, capability_manifest=None, identity_filter=None, memory_system=None):
        """
        Initialize the execution planner.

        Args:
            llm_wrapper: The LLM wrapper to use for planning
            capability_manifest: The capability manifest to use for capability execution
            identity_filter: The identity filter to use for filtering prompts and responses
            memory_system: The memory system for storing execution history
        """
        logger.info("Initializing ExecutionPlanner")

        self.llm_wrapper = llm_wrapper
        self.capability_manifest = capability_manifest
        self.identity_filter = identity_filter
        self.memory_system = memory_system

        # Configuration
        self.max_execution_steps = config_get('context_engine.execution_planner.max_execution_steps', 5)
        self.max_retries = config_get('context_engine.execution_planner.max_retries', 2)
        self.retry_delay = config_get('context_engine.execution_planner.retry_delay', 1.0)

        # Adaptive planning configuration
        self.max_plan_attempts = config_get('context_engine.execution_planner.max_plan_attempts', 3)
        self.plan_retry_delay = config_get('context_engine.execution_planner.plan_retry_delay', 2.0)
        self.enable_adaptive_planning = config_get('context_engine.execution_planner.enable_adaptive_planning', True)
        self.enable_plan_repair = config_get('context_engine.execution_planner.enable_plan_repair', True)
        self.confidence_threshold = config_get('context_engine.execution_planner.confidence_threshold', 0.7)

        # Working memory for the current execution
        self.working_memory = {
            "current_chunk_sizes": {},  # Track chunk sizes for different operations
            "plans_tried": set(),       # Track plans that have been tried
            "approaches_tried": set()   # Track approaches that have been tried
        }

        # Initialize token optimizer
        self.token_optimizer = TokenOptimizer(
            llm_wrapper=llm_wrapper,
            max_tokens=config_get('context_engine.execution_planner.max_tokens', 4000)
        )

        logger.info(f"ExecutionPlanner initialized with adaptive planning: {self.enable_adaptive_planning}")
        logger.info(f"Memory system: {self.memory_system is not None}")
        logger.info("ExecutionPlanner initialized successfully")

    def create_execution_plan(self, request: str, analysis: Dict[str, Any],
                            context: Optional[str] = None) -> ExecutionPlan:
        """
        Create an execution plan based on request analysis.

        Args:
            request: The user's request
            analysis: The analysis of the request
            context: The context for the request

        Returns:
            An execution plan
        """
        logger.info(f"Creating execution plan for request: {request[:50]}...")

        # Initialize execution plan
        execution_plan = ExecutionPlan()

        # Get request type
        request_type = analysis.get("request_type", "unknown")

        # Check if this is a high-confidence general query with a potential answer
        if (request_type == "general_query" and
            "potential_answer" in analysis and
            analysis.get("confidence", 0.0) >= 0.9):
            # For high-confidence general queries with potential answers, create an empty plan
            # This will cause the _generate_final_response method to use the potential_answer directly
            logger.info("Creating empty plan for high-confidence general query with potential answer")
            # Create an empty plan with no steps
            execution_plan = ExecutionPlan()
            execution_plan.final_response_template = "{{potential_answer}}"
            logger.info("Created empty execution plan (will use potential answer directly)")
            return execution_plan

        # Handle different request types
        if request_type == "general_query":
            # For general queries, use the LLM directly
            execution_plan = self._create_general_query_plan(request, analysis, context)
        elif request_type == "capability_query":
            # For capability queries, use the capability manifest
            execution_plan = self._create_capability_query_plan(request, analysis, context)
        elif request_type == "code_generation":
            # For code generation, use the LLM directly
            execution_plan = self._create_code_generation_plan(request, analysis, context)
        elif request_type == "action":
            # For actions, use the matched capabilities
            execution_plan = self._create_action_plan(request, analysis, context)
        else:
            # For unknown request types, use the LLM directly
            execution_plan = self._create_general_query_plan(request, analysis, context)

        logger.info(f"Created execution plan with {len(execution_plan.steps)} steps")
        return execution_plan

    def execute_plan(self, plan: ExecutionPlan, request: str, analysis: Dict[str, Any],
                   context: Optional[str] = None) -> Dict[str, Any]:
        """
        Execute an execution plan.

        Args:
            plan: The execution plan to execute
            request: The user's request
            analysis: The analysis of the request
            context: The context for the request

        Returns:
            A dictionary containing the execution results
        """
        # Log detailed flow information
        logger.debug(f"CONTEXT ENGINE FLOW - EXECUTION PLANNER INPUT: Request: {request}")
        logger.debug(f"CONTEXT ENGINE FLOW - EXECUTION PLANNER ANALYSIS: {safe_json_dumps(analysis)}")
        logger.debug(f"CONTEXT ENGINE FLOW - EXECUTION PLANNER PLAN: {safe_json_dumps(plan.__dict__)}")

        # If adaptive planning is enabled, use the adaptive execution method
        if self.enable_adaptive_planning:
            logger.info("Using adaptive planning for execution")
            return self.execute_plan_with_persistence(
                plan, request, analysis, context
            )

        # Otherwise, use the standard execution method
        start_time = time.time()
        logger.info(f"Executing plan with {len(plan.steps)} steps")

        # Initialize results
        results = {
            "request": request,
            "success": False,
            "response": "",
            "steps_executed": 0,
            "steps_succeeded": 0,
            "steps_failed": 0,
            "execution_time": 0.0,
            "capabilities_used": [],
            "execution_steps": []
        }

        # Execute each step in the plan
        for i, step in enumerate(plan.steps):
            logger.info(f"Executing step {i+1}/{len(plan.steps)}: {step.description}")

            # Execute the step
            step_result = self._execute_step(step, context, analysis)

            # Record step result
            step_info = {
                "step_id": step.step_id,
                "description": step.description,
                "tool": step.tool,
                "parameters": step.parameters,
                "success": step_result["success"],
                "result": step_result["result"],
                "error": step_result.get("error", "")
            }
            results["execution_steps"].append(step_info)

            # Update step result
            step.result = step_result["result"]

            # Update counters
            results["steps_executed"] += 1
            if step_result["success"]:
                results["steps_succeeded"] += 1
                if step.tool not in results["capabilities_used"]:
                    results["capabilities_used"].append(step.tool)
            else:
                results["steps_failed"] += 1
                logger.warning(f"Step {i+1} failed: {step_result.get('error', 'Unknown error')}")

                # Try fallback steps if available
                if plan.fallback_steps:
                    logger.info(f"Trying fallback steps")
                    for fallback_step in plan.fallback_steps:
                        fallback_result = self._execute_step(fallback_step, context, analysis)
                        if fallback_result["success"]:
                            logger.info(f"Fallback step succeeded")
                            step.result = fallback_result["result"]
                            results["steps_succeeded"] += 1
                            if fallback_step.tool not in results["capabilities_used"]:
                                results["capabilities_used"].append(fallback_step.tool)
                            break
                        else:
                            logger.warning(f"Fallback step failed: {fallback_result.get('error', 'Unknown error')}")

                # If all steps and fallbacks failed, try to generate a response with the LLM
                if results["steps_failed"] > 0 and results["steps_succeeded"] == 0:
                    logger.info(f"All steps failed, generating response with LLM")
                    response = self._generate_error_response(request, analysis, context, results)
                    results["response"] = response
                    results["success"] = False
                    results["execution_time"] = time.time() - start_time
                    return results

        # Generate final response
        response = self._generate_final_response(plan, results, request, analysis, context)
        results["response"] = response
        results["success"] = True
        results["execution_time"] = time.time() - start_time

        logger.info(f"Plan execution completed in {results['execution_time']:.2f}s")

        # Log the final results for detailed flow tracking
        logger.debug(f"CONTEXT ENGINE FLOW - EXECUTION PLANNER OUTPUT: {safe_json_dumps(results)}")
        logger.debug(f"CONTEXT ENGINE FLOW - FINAL RESPONSE: {response}")

        return results

    def execute_plan_with_persistence(self, plan: ExecutionPlan, request: str, analysis: Dict[str, Any],
                                    context: Optional[str] = None) -> Dict[str, Any]:
        """
        Execute a plan with persistence, retrying and adapting as needed.

        This method implements adaptive execution, which includes:
        1. Retrying failed steps with exponential backoff
        2. Attempting to repair the plan if steps fail
        3. Generating alternative plans if the original plan fails
        4. Tracking execution history for better error recovery

        Args:
            plan: The execution plan to execute
            request: The user's request
            analysis: The analysis of the request
            context: The context for the request

        Returns:
            A dictionary containing the execution results
        """
        start_time = time.time()
        logger.info(f"Executing plan with persistence: {len(plan.steps)} steps")

        # Initialize results
        results = {
            "request": request,
            "success": False,
            "response": "",
            "steps_executed": 0,
            "steps_succeeded": 0,
            "steps_failed": 0,
            "execution_time": 0.0,
            "capabilities_used": [],
            "execution_steps": [],
            "plan_attempts": 0,
            "plan_repairs": 0,
            "alternative_plans_tried": 0
        }

        # Track the current plan attempt
        plan_attempt = 0
        max_attempts = self.max_plan_attempts

        # Keep trying until we succeed or run out of attempts
        while plan_attempt < max_attempts:
            plan_attempt += 1
            results["plan_attempts"] = plan_attempt

            logger.info(f"Plan attempt {plan_attempt}/{max_attempts}")

            # Reset step counters for this attempt
            steps_executed = 0
            steps_succeeded = 0
            steps_failed = 0

            # Execute each step in the plan
            for i, step in enumerate(plan.steps):
                logger.info(f"Executing step {i+1}/{len(plan.steps)}: {step.description}")

                # Try to execute the step with retries
                retry_count = 0
                step_succeeded = False
                step_result = None

                while retry_count <= self.max_retries and not step_succeeded:
                    if retry_count > 0:
                        # Calculate exponential backoff delay
                        delay = self.retry_delay * (2 ** (retry_count - 1))
                        logger.info(f"Retrying step {i+1} after {delay:.2f}s delay (attempt {retry_count+1}/{self.max_retries+1})")
                        time.sleep(delay)

                    # Execute the step
                    step_result = self._execute_step(step, context, analysis)
                    step_succeeded = step_result["success"]

                    if not step_succeeded:
                        retry_count += 1
                        logger.warning(f"Step {i+1} failed: {step_result.get('error', 'Unknown error')}")

                # Record step result
                step_info = {
                    "step_id": step.step_id,
                    "description": step.description,
                    "tool": step.tool,
                    "parameters": step.parameters,
                    "success": step_succeeded,
                    "result": step_result["result"] if step_result else None,
                    "error": step_result.get("error", "") if step_result else "Unknown error",
                    "retries": retry_count
                }
                results["execution_steps"].append(step_info)

                # Update step result
                if step_result:
                    step.result = step_result["result"]

                # Update counters
                steps_executed += 1
                if step_succeeded:
                    steps_succeeded += 1
                    if step.tool not in results["capabilities_used"]:
                        results["capabilities_used"].append(step.tool)
                else:
                    steps_failed += 1

                    # Try fallback steps if available
                    fallback_succeeded = False
                    if plan.fallback_steps:
                        logger.info(f"Trying fallback steps for step {i+1}")
                        for fallback_step in plan.fallback_steps:
                            # Check if this fallback applies to the current step
                            trigger_match = False
                            for trigger in getattr(fallback_step, 'triggers', [f"error_in_step_{i+1}"]):
                                if trigger == f"error_in_step_{i+1}":
                                    trigger_match = True
                                    break

                            if not trigger_match:
                                continue

                            fallback_result = self._execute_step(fallback_step, context, analysis)
                            if fallback_result["success"]:
                                logger.info(f"Fallback step succeeded")
                                step.result = fallback_result["result"]
                                steps_succeeded += 1
                                fallback_succeeded = True
                                if fallback_step.tool not in results["capabilities_used"]:
                                    results["capabilities_used"].append(fallback_step.tool)
                                break
                            else:
                                logger.warning(f"Fallback step failed: {fallback_result.get('error', 'Unknown error')}")

                    # If step failed and no fallback succeeded, try to repair the plan
                    if not fallback_succeeded and self.enable_plan_repair:
                        logger.info(f"Attempting to repair plan after step {i+1} failed")

                        # Try to repair the plan
                        repaired_plan = self._repair_plan(plan, i, step_result, request, analysis, context)
                        if repaired_plan and repaired_plan != plan:
                            logger.info(f"Plan repaired successfully, restarting execution")
                            results["plan_repairs"] += 1
                            plan = repaired_plan
                            break  # Restart execution with the repaired plan
                        else:
                            logger.warning(f"Failed to repair plan")

                    # If we couldn't repair the plan, continue to the next step
                    # The final response will be generated based on partial results

            # Update overall counters
            results["steps_executed"] += steps_executed
            results["steps_succeeded"] += steps_succeeded
            results["steps_failed"] += steps_failed

            # Check if the plan succeeded overall
            if steps_failed == 0 or steps_succeeded > 0:
                # Generate final response
                response = self._generate_final_response(plan, results, request, analysis, context)
                results["response"] = response
                results["success"] = True
                results["execution_time"] = time.time() - start_time

                logger.info(f"Plan execution completed successfully in {results['execution_time']:.2f}s")
                return results

            # If we get here, the plan failed completely
            if plan_attempt < max_attempts:
                # Try an alternative plan
                logger.info(f"Plan attempt {plan_attempt} failed completely, trying alternative plan")
                alternative_plan = self._generate_alternative_plan(request, analysis, context, results)
                if alternative_plan:
                    logger.info(f"Generated alternative plan with {len(alternative_plan.steps)} steps")
                    plan = alternative_plan
                    results["alternative_plans_tried"] += 1
                else:
                    logger.warning(f"Failed to generate alternative plan")
                    break  # Give up if we can't generate an alternative plan
            else:
                logger.warning(f"Reached maximum plan attempts ({max_attempts})")
                break

            # Add a delay before trying the next plan
            if plan_attempt < max_attempts:
                delay = self.plan_retry_delay * (2 ** (plan_attempt - 1))
                logger.info(f"Waiting {delay:.2f}s before trying next plan")
                time.sleep(delay)

        # If we get here, all plan attempts failed
        logger.error(f"All plan attempts failed after {results['execution_time']:.2f}s")
        response = self._generate_error_response(request, analysis, context, results)
        results["response"] = response
        results["success"] = False
        results["execution_time"] = time.time() - start_time

        return results

    def _create_general_query_plan(self, request: str, analysis: Dict[str, Any],
                                 context: Optional[str] = None) -> ExecutionPlan:
        """
        Create an execution plan for a general query.

        Args:
            request: The user's request
            analysis: The analysis of the request
            context: The context for the request

        Returns:
            An execution plan
        """
        # Create a plan with a single step to generate a response with the LLM
        plan = ExecutionPlan()

        step = ExecutionStep(
            step_id=str(uuid.uuid4()),
            description="Generate a response to the user's query",
            tool="llm_generate",
            parameters={
                "prompt": context or request,
                "temperature": 0.7,
                "max_tokens": 1000
            },
            expected_output="A natural language response to the user's query"
        )

        plan.steps.append(step)
        plan.final_response_template = "{{result}}"

        return plan

    def _create_capability_query_plan(self, request: str, analysis: Dict[str, Any],
                                    context: Optional[str] = None) -> ExecutionPlan:
        """
        Create an execution plan for a capability query.

        Args:
            request: The user's request
            analysis: The analysis of the request
            context: The context for the request

        Returns:
            An execution plan
        """
        # Create a plan with a single step to generate a response about capabilities
        plan = ExecutionPlan()

        step = ExecutionStep(
            step_id=str(uuid.uuid4()),
            description="Generate a response about available capabilities",
            tool="llm_generate",
            parameters={
                "prompt": context or request,
                "temperature": 0.7,
                "max_tokens": 1000
            },
            expected_output="A natural language response about available capabilities"
        )

        plan.steps.append(step)
        plan.final_response_template = "{{result}}"

        return plan

    def _create_code_generation_plan(self, request: str, analysis: Dict[str, Any],
                                   context: Optional[str] = None) -> ExecutionPlan:
        """
        Create an execution plan for a code generation request.

        Args:
            request: The user's request
            analysis: The analysis of the request
            context: The context for the request

        Returns:
            An execution plan
        """
        # Create a plan with a single step to generate code with the LLM
        plan = ExecutionPlan()

        step = ExecutionStep(
            step_id=str(uuid.uuid4()),
            description="Generate code based on the user's request",
            tool="llm_generate",
            parameters={
                "prompt": context or request,
                "temperature": 0.2,
                "max_tokens": 2000
            },
            expected_output="Code that fulfills the user's request"
        )

        plan.steps.append(step)
        plan.final_response_template = "{{result}}"

        return plan

    def _create_action_plan(self, request: str, analysis: Dict[str, Any],
                          context: Optional[str] = None) -> ExecutionPlan:
        """
        Create an execution plan for an action request.

        Args:
            request: The user's request
            analysis: The analysis of the request
            context: The context for the request

        Returns:
            An execution plan
        """
        # Create a plan with steps to execute the matched capabilities
        plan = ExecutionPlan()

        # Get matched capabilities
        matched_capabilities = analysis.get("matched_capabilities", [])

        if not matched_capabilities:
            # If no capabilities matched, use the LLM directly
            return self._create_general_query_plan(request, analysis, context)

        # Create steps for each matched capability
        for i, capability in enumerate(matched_capabilities[:self.max_execution_steps]):
            capability_name = capability.get("name", f"capability_{i}")
            capability_description = capability.get("description", "No description available")

            # Get parameters for the capability
            parameters = {}
            for param_name, param_info in capability.get("parameters", {}).items():
                if param_name in analysis.get("parameters", {}):
                    parameters[param_name] = analysis["parameters"][param_name]

            # Create a step for the capability
            step = ExecutionStep(
                step_id=str(uuid.uuid4()),
                description=f"Execute capability: {capability_name}",
                tool=capability_name,
                parameters=parameters,
                expected_output=f"Result of executing {capability_name}"
            )

            plan.steps.append(step)

        # Add a final step to generate a response based on the results
        final_step = ExecutionStep(
            step_id=str(uuid.uuid4()),
            description="Generate a response based on the results",
            tool="llm_generate",
            parameters={
                "prompt": "{{context}}\n\nResults:\n{{results}}",
                "temperature": 0.7,
                "max_tokens": 1000
            },
            expected_output="A natural language response based on the results"
        )

        plan.steps.append(final_step)
        plan.final_response_template = "{{result}}"

        return plan

    def _execute_step(self, step: ExecutionStep, context: Optional[str] = None,
                    analysis: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Execute a single step in an execution plan.

        Args:
            step: The step to execute
            context: The context for the request
            analysis: The analysis of the request (optional)

        Returns:
            A dictionary containing the execution results
        """
        logger.info(f"Executing step: {step.description}")

        # Initialize result
        result = {
            "success": False,
            "result": None,
            "error": ""
        }

        try:
            # Handle different tools
            if step.tool == "llm_generate":
                # Generate text with the LLM
                prompt = step.parameters.get("prompt", "")
                temperature = step.parameters.get("temperature", 0.7)
                max_tokens = step.parameters.get("max_tokens", 1000)

                # Apply identity filter if available
                if self.identity_filter:
                    prompt = self.identity_filter.filter_prompt(prompt)

                # Get the cloud and local provider names from configuration
                from selfy_core.global_modules.config import get as config_get
                cloud_provider = config_get('llm.cloud_provider', 'groq')
                local_provider = config_get('llm.local_provider', 'ollama')

                # Get the preferred LLM from the analysis if available
                preferred_llm = analysis.get("preferred_llm", local_provider) if analysis else local_provider

                logger.info(f"Preferred LLM for text generation: {preferred_llm}")

                # Initialize response_text to avoid variable scope issues
                response_text = ""

                # Optimize the prompt if using cloud LLM
                if preferred_llm != local_provider:
                    logger.info(f"Using token optimization for cloud LLM (preferred: {preferred_llm})")
                    prompt = self.token_optimizer.optimize(prompt, preferred_llm)

                # Use non-streaming mode
                try:
                    if preferred_llm == local_provider:
                        # Local provider
                        logger.info(f"Using local LLM ({local_provider}) for text generation")
                        # Get a new instance of the local provider
                        from selfy_core.global_modules.llm_wrapper import get_llm_provider
                        local_llm = get_llm_provider(local_provider)
                        response = local_llm.generate_text(prompt, temperature=temperature, max_tokens=max_tokens)
                        response_text = response.content
                    else:
                        # Cloud provider
                        logger.info(f"Using cloud LLM ({preferred_llm}) for text generation")
                        # Get a new instance of the cloud provider
                        from selfy_core.global_modules.llm_wrapper import get_llm_provider
                        cloud_llm = get_llm_provider(cloud_provider)
                        messages = [
                            {"role": "system", "content": "You are a helpful assistant."},
                            {"role": "user", "content": prompt}
                        ]
                        response = cloud_llm.generate_chat_completion(messages, temperature=temperature, max_tokens=max_tokens)
                        response_text = response.content
                except Exception as e:
                    logger.error(f"Error generating text with LLM: {e}")
                    response_text = f"I'm sorry, I encountered an error while processing your request: {str(e)}"

                # Apply identity filter if available
                if self.identity_filter:
                    response_text = self.identity_filter.filter_response(response_text)

                result["result"] = response_text
                result["success"] = True
            elif self.capability_manifest:
                # Execute a capability
                capability = self.capability_manifest.get_capability(step.tool)

                if capability:
                    # Execute the capability
                    capability_result = self.capability_manifest.execute_capability(step.tool, step.parameters)

                    result["result"] = capability_result
                    result["success"] = True
                else:
                    result["error"] = f"Capability not found: {step.tool}"
            else:
                result["error"] = f"Unknown tool: {step.tool}"
        except Exception as e:
            logger.error(f"Error executing step: {e}")
            result["error"] = str(e)

        logger.info(f"Step execution {'succeeded' if result['success'] else 'failed'}")
        return result

    def _generate_final_response(self, plan: ExecutionPlan, results: Dict[str, Any],
                               request: str, analysis: Dict[str, Any],
                               context: Optional[str] = None) -> str:
        """
        Generate a final response based on the execution results.

        Args:
            plan: The execution plan
            results: The execution results
            request: The user's request
            analysis: The analysis of the request
            context: The context for the request

        Returns:
            The final response
        """
        logger.info("Generating final response")

        # Log the full analysis for debugging
        logger.debug(f"Analysis in _generate_final_response: {json.dumps(analysis, default=str)}")

        # Check for potential_answer in analysis
        if "potential_answer" in analysis:
            logger.info(f"Analysis contains potential_answer: {analysis['potential_answer'][:100]}...")
            logger.info(f"Analysis confidence: {analysis.get('confidence', 0.0):.2f}")
        else:
            logger.info("Analysis does not contain a potential_answer")

        # Check for knowledge domains in analysis
        if "required_knowledge_domains" in analysis and analysis["required_knowledge_domains"]:
            knowledge_domains = analysis["required_knowledge_domains"]
            logger.info(f"Analysis contains knowledge domains: {knowledge_domains}")

            # Add knowledge domains to the context if they're not already there
            if context and "Required knowledge domains:" not in context:
                knowledge_domains_str = ", ".join(knowledge_domains)
                knowledge_domains_section = f"""
                Required knowledge domains:
                {knowledge_domains_str}
                """
                context += knowledge_domains_section
                logger.info(f"Added knowledge domains to context: {knowledge_domains}")

        # Log the context to see if it contains the potential answer
        if context:
            logger.debug(f"Context in _generate_final_response (first 200 chars): {context[:200]}...")
            if "potential answer" in context.lower():
                logger.info("Context contains reference to potential answer")
            else:
                logger.info("Context does not contain reference to potential answer")

        # If there's a potential answer in the analysis with high confidence, use it directly
        if "potential_answer" in analysis and analysis.get("confidence", 0.0) >= 0.9:
            logger.info(f"Using potential answer from analysis (confidence: {analysis.get('confidence', 0.0):.2f})")
            logger.info(f"Returning potential answer: {analysis['potential_answer']}")
            return analysis["potential_answer"]

        # If there's only one step and it's an LLM generation, use its result directly
        if len(plan.steps) == 1 and plan.steps[0].tool == "llm_generate":
            logger.info("Using result from single LLM generation step")
            result = plan.steps[0].result or "I'm sorry, I couldn't generate a response."
            logger.info(f"Result from LLM generation step: {result[:100]}...")
            return result

        # If there's a final step that's an LLM generation, use its result
        if plan.steps[-1].tool == "llm_generate":
            logger.info("Using result from final LLM generation step")
            result = plan.steps[-1].result or "I'm sorry, I couldn't generate a response."
            logger.info(f"Result from final LLM generation step: {result[:100]}...")
            return result

        # Otherwise, use the template to generate a response
        try:
            # Replace placeholders in the template
            response = plan.final_response_template

            # Replace {{potential_answer}} with the potential answer from analysis
            if "{{potential_answer}}" in response and "potential_answer" in analysis:
                response = response.replace("{{potential_answer}}", str(analysis["potential_answer"]))
                logger.info("Used potential_answer in final response template")
                return response

            # Replace {{result}} with the result of the last step
            if "{{result}}" in response and plan.steps and plan.steps[-1].result:
                response = response.replace("{{result}}", str(plan.steps[-1].result))

            # Replace {{results}} with a summary of all results
            if "{{results}}" in response:
                results_summary = ""
                for i, step in enumerate(plan.steps):
                    if step.result:
                        results_summary += f"Step {i+1}: {step.description}\n"
                        results_summary += f"Result: {step.result}\n\n"
                response = response.replace("{{results}}", results_summary)

            # Replace {{context}} with the context
            if "{{context}}" in response and context:
                response = response.replace("{{context}}", context)

            # Replace {{request}} with the request
            if "{{request}}" in response:
                response = response.replace("{{request}}", request)

            return response
        except Exception as e:
            logger.error(f"Error generating final response: {e}")
            return "I'm sorry, I couldn't generate a response."

    def _repair_plan(self, plan: ExecutionPlan, failed_step_index: int,
                    step_result: Dict[str, Any], request: str,
                    analysis: Dict[str, Any], context: Optional[str] = None) -> Optional[ExecutionPlan]:
        """
        Attempt to repair a plan after a step failure.

        Args:
            plan: The execution plan to repair
            failed_step_index: The index of the failed step
            step_result: The result of the failed step
            request: The user's request
            analysis: The analysis of the request
            context: The context for the request

        Returns:
            A repaired execution plan, or None if repair failed
        """
        logger.info(f"Attempting to repair plan after step {failed_step_index+1} failed")

        if not self.llm_wrapper:
            logger.warning("No LLM wrapper available for plan repair")
            return None

        try:
            # Create a repair prompt
            repair_prompt = self._create_repair_prompt(plan, failed_step_index, step_result, request, analysis, context)

            # Apply identity filter if available
            if self.identity_filter:
                repair_prompt = self.identity_filter.filter_prompt(repair_prompt)

            # Get provider names from config
            from selfy_core.global_modules.config import get as config_get
            cloud_provider = config_get('llm.cloud_provider', 'groq')

            # Always use cloud provider for plan repair
            from selfy_core.global_modules.llm_wrapper import get_llm_provider
            cloud_llm = get_llm_provider(cloud_provider)

            messages = [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": repair_prompt}
            ]
            response = cloud_llm.generate_chat_completion(messages, temperature=0.7, max_tokens=2000)
            response_text = response.content

            # Extract JSON from response
            try:
                import json
                from selfy_core.user_pipeline.context_engine.utils.llm_utils import extract_json_from_response

                json_str = extract_json_from_response(response_text)
                if not json_str:
                    logger.warning("Failed to extract JSON from repair response")
                    return None

                repair_data = json.loads(json_str)

                # Create a new plan from the repair data
                repaired_plan = ExecutionPlan()

                # Add steps from the repair data
                for step_data in repair_data.get("steps", []):
                    step = ExecutionStep(
                        step_id=step_data.get("step_id", str(uuid.uuid4())),
                        description=step_data.get("description", "Repaired step"),
                        tool=step_data.get("tool", "direct_response"),
                        parameters=step_data.get("parameters", {}),
                        expected_output=step_data.get("expected_output", "Step output")
                    )
                    repaired_plan.steps.append(step)

                # Add fallback steps from the repair data
                for fallback_data in repair_data.get("fallback_steps", []):
                    fallback_step = ExecutionStep(
                        step_id=fallback_data.get("step_id", f"f{uuid.uuid4()}"),
                        description=fallback_data.get("description", "Fallback step"),
                        tool=fallback_data.get("tool", "direct_response"),
                        parameters=fallback_data.get("parameters", {}),
                        expected_output=fallback_data.get("expected_output", "Fallback output")
                    )
                    # Add triggers if available
                    if "triggers" in fallback_data:
                        setattr(fallback_step, "triggers", fallback_data["triggers"])
                    repaired_plan.fallback_steps.append(fallback_step)

                # Set final response template
                if "final_response_template" in repair_data:
                    repaired_plan.final_response_template = repair_data["final_response_template"]

                return repaired_plan
            except Exception as e:
                logger.error(f"Error parsing repair data: {e}")
                return None

        except Exception as e:
            logger.error(f"Error repairing plan: {e}")
            return None

    def _create_repair_prompt(self, plan: ExecutionPlan, failed_step_index: int,
                            step_result: Dict[str, Any], request: str,
                            analysis: Dict[str, Any], context: Optional[str] = None) -> str:
        """
        Create a prompt for repairing a plan.

        Args:
            plan: The execution plan to repair
            failed_step_index: The index of the failed step
            step_result: The result of the failed step
            request: The user's request
            analysis: The analysis of the request
            context: The context for the request

        Returns:
            A prompt for repairing the plan
        """
        # Get the failed step
        failed_step = plan.steps[failed_step_index] if failed_step_index < len(plan.steps) else None

        # Format the plan as JSON
        plan_json = {
            "steps": [
                {
                    "step_id": step.step_id,
                    "description": step.description,
                    "tool": step.tool,
                    "parameters": step.parameters,
                    "expected_output": step.expected_output
                }
                for step in plan.steps
            ],
            "fallback_steps": [
                {
                    "step_id": step.step_id,
                    "description": step.description,
                    "tool": step.tool,
                    "parameters": step.parameters,
                    "expected_output": step.expected_output,
                    "triggers": getattr(step, "triggers", [])
                }
                for step in plan.fallback_steps
            ],
            "final_response_template": plan.final_response_template
        }

        # Format the plan as a string
        import json
        plan_str = json.dumps(plan_json, indent=2)

        # Create the repair prompt
        prompt = f"""
        I need to repair an execution plan that failed. The user asked:

        "{request}"

        The original plan was:

        {plan_str}

        Step {failed_step_index + 1} failed with the following error:

        {step_result.get('error', 'Unknown error')}

        Please create a repaired plan that addresses this issue. You can:

        1. Modify the failed step to fix the error
        2. Replace the failed step with alternative steps
        3. Add additional steps before or after the failed step
        4. Add fallback steps to handle similar errors in the future

        Return the repaired plan in JSON format with the following structure:

        {{
            "steps": [
                {{
                    "step_id": 1,
                    "description": "Description of the step",
                    "tool": "The tool to use",
                    "parameters": {{
                        "param1": "value1",
                        "param2": "value2",
                        ...
                    }},
                    "expected_output": "Description of the expected output"
                }},
                ...
            ],
            "fallback_steps": [
                {{
                    "step_id": "f1",
                    "description": "Description of the fallback step",
                    "tool": "The tool to use",
                    "parameters": {{
                        "param1": "value1",
                        "param2": "value2",
                        ...
                    }},
                    "expected_output": "Description of the expected output",
                    "triggers": ["error_in_step_1", "no_results_from_step_2"]
                }},
                ...
            ],
            "final_response_template": "Template for the final response to the user"
        }}
        """

        return prompt

    def _generate_alternative_plan(self, request: str, analysis: Dict[str, Any],
                                 context: Optional[str] = None,
                                 results: Optional[Dict[str, Any]] = None) -> Optional[ExecutionPlan]:
        """
        Generate an alternative execution plan.

        Args:
            request: The user's request
            analysis: The analysis of the request
            context: The context for the request
            results: The execution results from previous attempts

        Returns:
            An alternative execution plan, or None if generation failed
        """
        logger.info("Generating alternative execution plan")

        if not self.llm_wrapper:
            logger.warning("No LLM wrapper available for alternative plan generation")
            return None

        try:
            # Create an alternative plan prompt
            alternative_prompt = self._create_alternative_plan_prompt(request, analysis, context, results)

            # Apply identity filter if available
            if self.identity_filter:
                alternative_prompt = self.identity_filter.filter_prompt(alternative_prompt)

            # Get provider names from config
            from selfy_core.global_modules.config import get as config_get
            cloud_provider = config_get('llm.cloud_provider', 'groq')

            # Always use cloud provider for alternative plan generation
            from selfy_core.global_modules.llm_wrapper import get_llm_provider
            cloud_llm = get_llm_provider(cloud_provider)

            messages = [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": alternative_prompt}
            ]
            response = cloud_llm.generate_chat_completion(messages, temperature=0.7, max_tokens=2000)
            response_text = response.content

            # Extract JSON from response
            try:
                import json
                from selfy_core.user_pipeline.context_engine.utils.llm_utils import extract_json_from_response

                json_str = extract_json_from_response(response_text)
                if not json_str:
                    logger.warning("Failed to extract JSON from alternative plan response")
                    return None

                plan_data = json.loads(json_str)

                # Create a new plan from the plan data
                alternative_plan = ExecutionPlan()

                # Add steps from the plan data
                for step_data in plan_data.get("steps", []):
                    step = ExecutionStep(
                        step_id=step_data.get("step_id", str(uuid.uuid4())),
                        description=step_data.get("description", "Alternative step"),
                        tool=step_data.get("tool", "direct_response"),
                        parameters=step_data.get("parameters", {}),
                        expected_output=step_data.get("expected_output", "Step output")
                    )
                    alternative_plan.steps.append(step)

                # Add fallback steps from the plan data
                for fallback_data in plan_data.get("fallback_steps", []):
                    fallback_step = ExecutionStep(
                        step_id=fallback_data.get("step_id", f"f{uuid.uuid4()}"),
                        description=fallback_data.get("description", "Fallback step"),
                        tool=fallback_data.get("tool", "direct_response"),
                        parameters=fallback_data.get("parameters", {}),
                        expected_output=fallback_data.get("expected_output", "Fallback output")
                    )
                    # Add triggers if available
                    if "triggers" in fallback_data:
                        setattr(fallback_step, "triggers", fallback_data["triggers"])
                    alternative_plan.fallback_steps.append(fallback_step)

                # Set final response template
                if "final_response_template" in plan_data:
                    alternative_plan.final_response_template = plan_data["final_response_template"]

                return alternative_plan
            except Exception as e:
                logger.error(f"Error parsing alternative plan data: {e}")
                return None

        except Exception as e:
            logger.error(f"Error generating alternative plan: {e}")
            return None

    def _create_alternative_plan_prompt(self, request: str, analysis: Dict[str, Any],
                                      context: Optional[str] = None,
                                      results: Optional[Dict[str, Any]] = None) -> str:
        """
        Create a prompt for generating an alternative plan.

        Args:
            request: The user's request
            analysis: The analysis of the request
            context: The context for the request
            results: The execution results from previous attempts

        Returns:
            A prompt for generating an alternative plan
        """
        # Format the failed steps
        failed_steps_str = ""
        if results:
            for step in results.get("execution_steps", []):
                if not step.get("success", False):
                    failed_steps_str += f"\nStep: {step.get('description', 'Unknown step')}"
                    failed_steps_str += f"\nTool: {step.get('tool', 'Unknown tool')}"
                    failed_steps_str += f"\nParameters: {step.get('parameters', {})}"
                    failed_steps_str += f"\nError: {step.get('error', 'Unknown error')}\n"

        # Create the alternative plan prompt
        prompt = f"""
        I need to generate an alternative execution plan. The user asked:

        "{request}"

        Previous execution attempts failed with the following issues:

        {failed_steps_str if failed_steps_str else "No specific issues recorded."}

        Please create a completely different plan to fulfill the user's request. The plan should:

        1. Use different tools or approaches than the previous attempts
        2. Include robust error handling with fallback steps
        3. Be simpler and more direct if possible
        4. Focus on the core intent of the user's request

        Return the alternative plan in JSON format with the following structure:

        {{
            "steps": [
                {{
                    "step_id": 1,
                    "description": "Description of the step",
                    "tool": "The tool to use",
                    "parameters": {{
                        "param1": "value1",
                        "param2": "value2",
                        ...
                    }},
                    "expected_output": "Description of the expected output"
                }},
                ...
            ],
            "fallback_steps": [
                {{
                    "step_id": "f1",
                    "description": "Description of the fallback step",
                    "tool": "The tool to use",
                    "parameters": {{
                        "param1": "value1",
                        "param2": "value2",
                        ...
                    }},
                    "expected_output": "Description of the expected output",
                    "triggers": ["error_in_step_1", "no_results_from_step_2"]
                }},
                ...
            ],
            "final_response_template": "Template for the final response to the user"
        }}
        """

        return prompt

    def _generate_error_response(self, request: str, analysis: Dict[str, Any],
                               context: Optional[str] = None,
                               results: Optional[Dict[str, Any]] = None) -> str:
        """
        Generate an error response when execution fails.

        Args:
            request: The user's request
            analysis: The analysis of the request
            context: The context for the request
            results: The execution results

        Returns:
            The error response
        """
        logger.info("Generating error response")

        # Create a prompt for the error response
        prompt = f"""
        The user asked: "{request}"

        I tried to fulfill this request, but encountered an error. Please generate a helpful response that:
        1. Acknowledges the error
        2. Explains what went wrong (if possible)
        3. Suggests alternative approaches or workarounds
        4. Maintains a helpful and apologetic tone

        Error details:
        """

        if results:
            for step in results.get("execution_steps", []):
                if not step.get("success", False):
                    prompt += f"\nStep: {step.get('description', 'Unknown step')}"
                    prompt += f"\nError: {step.get('error', 'Unknown error')}"

        # Apply identity filter if available
        if self.identity_filter:
            prompt = self.identity_filter.filter_prompt(prompt)

        # Generate error response
        try:
            # Get provider names from config
            from selfy_core.global_modules.config import get as config_get
            cloud_provider = config_get('llm.cloud_provider', 'groq')
            local_provider = config_get('llm.local_provider', 'ollama')

            # Always use cloud provider for error responses
            from selfy_core.global_modules.llm_wrapper import get_llm_provider
            cloud_llm = get_llm_provider(cloud_provider)

            messages = [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt}
            ]
            response = cloud_llm.generate_chat_completion(messages, temperature=0.7, max_tokens=1000)
            response_text = response.content

            # Apply identity filter if available
            if self.identity_filter:
                response_text = self.identity_filter.filter_response(response_text)

            return response_text
        except Exception as e:
            logger.error(f"Error generating error response: {e}")
            return "I'm sorry, I encountered an error while processing your request. Please try again later."
