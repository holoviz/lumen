"""
Lumen AI 2.0 - Base Agent Framework

Hierarchical AI agent system:
- Roadmap: High-level plan with milestones
- Milestone: Execute steps until complete
- Actions: Individual tasks that do the work

Goal → Roadmap → Milestone → Actions
                    ↑         ↓
                Checkpoint ← Outputs

This should only house generic agent functionality;
other domain-specific agents should extend this base class.
"""

import inspect
import json
import time

from collections.abc import Callable
from functools import wraps
from textwrap import dedent
from typing import Any

import param

from jinja2 import Template, TemplateError, UndefinedError
from pydantic import BaseModel, Field, create_model

from ...ai.llm import Llm
from .logger import AgentLogger
from .models import (
    Action, ActionDecision, ActionInfo, ActionParameter, ActionResult,
    ActionsInfo, Checkpoint, Checkpoints, GoalResult, PartialEscapeModel,
    Roadmap,
)

MAX_ITERATIONS = 3
MAX_ERRORS = 3


def action(func: Callable = None, *, locked: bool = False) -> Callable:
    """Decorator to mark methods as callable actions with automatic error handling

    Args:
        func: The function to decorate
        locked: Whether this action must be executed exclusively (cannot be parallelized)
    """

    def decorator(f: Callable) -> Callable:
        @wraps(f)
        async def wrapper(self, *args, **kwargs):
            return await f(self, *args, **kwargs)

        wrapper._is_action = True
        wrapper._locked = locked
        return wrapper

    # Handle both @action and @action(locked=True) syntax
    if func is None:
        return decorator
    else:
        return decorator(func)


class BaseAgent(param.Parameterized):
    """Base class for all Lumen AI agents

    Purpose:
    Provides a framework for creating AI agents that can perform specific tasks
    using LLM capabilities. Uses a three-tier hierarchical architecture:

    1. High-level roadmap generation (once per goal)
    2. Milestone execution with decisive completion
    3. Atomic @action methods for individual tasks
    """

    # ===== Configuration =====

    name = param.String(doc="Name of the agent")
    llm = param.ClassSelector(class_=Llm, doc="LLM instance for this agent")
    log_level = param.String(default="DEBUG", doc="Logging level (DEBUG, INFO, WARNING, ERROR)")

    # ===== Initialization =====

    def __init__(self, **params):
        super().__init__(**params)
        if not self.name:
            self.name = self.__class__.__name__.lower()

        # Initialize logger
        self.logger = AgentLogger(f"agent.{self.name}", self.log_level)

        self._discover_actions()
        self.logger.info(f"Initialized {self.__class__.__name__} with {len(self._actions)} actions")

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name='{self.name}')"

    # ===== Internal Methods =====

    def _render_prompt(self, method_name: str, **template_vars) -> str:
        """Render the prompt template for a method with variables

        Args:
            method_name: Name of the method (action or regular method)
            **template_vars: Variables to pass to the Jinja2 template

        Returns:
            Rendered prompt string
        """
        # First try to get as an action
        action_info = self._actions_info.get(method_name)
        if action_info:
            docstring = action_info.docstring
        # Try to get as a regular method
        elif hasattr(self, method_name):
            method = getattr(self, method_name)
            docstring = method.__doc__ or ""
        else:
            raise ValueError(f"Method '{method_name}' not found")

        if not docstring.strip():
            raise ValueError(f"Method '{method_name}' has no prompt template in docstring")

        # Handle template inheritance with {{ super() }}
        if "{{ super() }}" in docstring:
            # Find the parent class method
            for base_class in self.__class__.__mro__[1:]:  # Skip self class
                if hasattr(base_class, method_name):
                    base_method = getattr(base_class, method_name)
                    if hasattr(base_method, "__doc__") and base_method.__doc__:
                        parent_docstring = base_method.__doc__
                        # Replace {{ super() }} with the parent docstring
                        docstring = docstring.replace("{{ super() }}", parent_docstring)
                        break
            else:
                # If no parent method found, just remove {{ super() }}
                docstring = docstring.replace("{{ super() }}", "")

        docstring_lines = [line for line in docstring.splitlines() if line.strip()]
        dedented_docstring = docstring_lines[0] + "\n" + dedent("\n".join(docstring_lines[1:]))

        # Render using Jinja2 with error handling
        try:
            template = Template(dedented_docstring)
            rendered_template = template.render(**template_vars)
        except UndefinedError as e:
            # Extract the undefined variable name from the error
            undefined_var = str(e).split("'")[1] if "'" in str(e) else "unknown"
            raise ValueError(
                f"Template for method '{method_name}' references undefined variable '{undefined_var}'. Available variables: {list(template_vars.keys())}"
            ) from e
        except TemplateError as e:
            raise ValueError(f"Template error in method '{method_name}': {e}. Check the Jinja2 syntax in the docstring.") from e
        except Exception as e:
            raise ValueError(f"Unexpected error rendering template for method '{method_name}': {e}") from e

        # Log the rendered template
        self.logger.log_template_render(method_name, len(rendered_template))
        self.logger.log_prompt_content(method_name, rendered_template)

        return rendered_template

    def _build_action_info(self, action_name: str, method: Callable) -> ActionInfo:
        """Build ActionInfo for a specific action method"""
        sig = inspect.signature(method)

        # Get parameter info
        params = {}
        for param_name, param_value in sig.parameters.items():
            if param_name not in ("self",):  # Skip 'self' parameter
                params[param_name] = ActionParameter(
                    name=param_name,
                    annotation=param_value.annotation,
                    default=param_value.default if param_value.default != inspect.Parameter.empty else None,
                )

        # Create human-readable signature string using ActionParameter's __str__
        sig_params = []
        for param_name, param_info in params.items():
            sig_params.append(str(param_info))

        signature_str = ", ".join(sig_params)

        return ActionInfo(
            name=action_name,
            callable=method,
            signature=sig,
            signature_str=signature_str,
            docstring=method.__doc__ or "",
            parameters=params,
        )

    def _discover_actions(self):
        """Discover all methods decorated with @action

        This method scans the class for methods decorated with @action
        and registers them in the internal _actions dictionary.
        Also builds ActionInfo cache for efficiency.

        This is called automatically during initialization, but can be
        called manually if actions are added dynamically.
        """
        self._actions = {}
        actions_dict = {}

        for name in dir(self):
            method = getattr(self, name)
            if callable(method) and hasattr(method, "_is_action"):
                self._actions[name] = method
                # Build and cache ActionInfo immediately
                actions_dict[name] = self._build_action_info(name, method)

        # Create ActionsInfo wrapper for utility methods
        self._actions_info = ActionsInfo(actions=actions_dict)

    def _create_action_model(self, action_name: str, action_info: ActionInfo) -> type:
        """Create a dynamic Pydantic model for a specific action"""
        signature = action_info.signature
        fields = {}

        # Add reasoning field first
        fields["reasoning"] = (str, Field(description=f"Think through why {action_name} is needed and what it should accomplish"))

        # Add action name as a literal field
        fields["action"] = (str, Field(default=action_name, description=f"Action name: {action_name}"))

        # Convert signature parameters to Pydantic fields
        for param_name, param_value in signature.parameters.items():
            if param_name in ("self", "context"):  # Skip these parameters
                continue

            # Get the parameter type
            param_type = param_value.annotation if param_value.annotation != inspect.Parameter.empty else Any

            # Create field with default if available
            if param_value.default != inspect.Parameter.empty:
                if param_value.default is None:
                    fields[param_name] = (param_type | None, Field(default=None, description=f"Parameter: {param_name}"))
                else:
                    fields[param_name] = (param_type, Field(default=param_value.default, description=f"Parameter: {param_name}"))
            else:
                fields[param_name] = (param_type, Field(description=f"Parameter: {param_name}"))

        # Create the dynamic model
        model_name = f"{action_name.title()}Step"
        dynamic_model = create_model(model_name, __base__=PartialEscapeModel, **fields)

        return dynamic_model

    async def _execute_single_action_safe(self, action: Action) -> Any:
        """Execute a single action with error handling"""
        if action.name not in self._actions:
            error_msg = f"Unknown action: {action.name}"
            self.logger.log_action_error(action.name, "Unknown action")
            if hasattr(self, "raise_on_error") and self.raise_on_error:
                raise ValueError(error_msg)
            return error_msg

        method = self._actions[action.name]
        try:
            # Always use keyword arguments to avoid positional argument issues
            return await method(**action.parameters)
        except Exception as e:
            error_msg = f"Error executing {action.name}: {e!s}"
            self.logger.log_action_error(action.name, str(e))
            if hasattr(self, "raise_on_error") and self.raise_on_error:
                raise RuntimeError(error_msg) from e
            return error_msg

    def _serialize_output(self, output: str | dict | BaseModel) -> str:
        if isinstance(output, str):
            return output.strip()

        if isinstance(output, BaseModel):
            return output.model_dump_json(indent=2, exclude_none=True, exclude_unset=True)
        elif isinstance(output, dict):
            # Filter out None and empty string values
            filtered_dict = self._filter_empty_values(output)
            return json.dumps(filtered_dict, indent=2, ensure_ascii=False)
        else:
            raise ValueError(f"Unsupported output type: {type(output)}. Expected str, dict, or BaseModel.")

    def _filter_empty_values(self, obj: Any) -> Any:
        """Recursively filter out None and empty string values from dictionaries and lists"""
        if isinstance(obj, dict):
            filtered = {}
            for key, value in obj.items():
                if value is not None and value != "":
                    filtered_value = self._filter_empty_values(value)
                    # Only include if the filtered value is not empty
                    if filtered_value is not None and filtered_value != "" and filtered_value != {} and filtered_value != []:
                        filtered[key] = filtered_value
            return filtered
        elif isinstance(obj, list):
            filtered = []
            for item in obj:
                filtered_item = self._filter_empty_values(item)
                # Only include non-empty items
                if filtered_item is not None and filtered_item not in ("", {}, []):
                    filtered.append(filtered_item)
            return filtered
        else:
            return obj

    async def _call_llm(self, method_name: str, messages: list[dict], response_model: type, **template_vars) -> Any:
        """Internal method to call LLM with automatic prompt rendering and logging

        Args:
            method_name: Name of the calling method (used for prompt template and logging)
            messages: Messages to send to LLM
            response_model: Pydantic model for response
            **template_vars: Variables to pass to the prompt template

        Returns:
            Parsed response from LLM
        """
        system_prompt = self._render_prompt(method_name, **template_vars)

        # Log input messages
        for i, message in enumerate(messages):
            role = message.get("role", "unknown")
            content = message.get("content", "")
            # Truncate very long messages for readability
            if len(content) > 500:
                display_content = content[:500] + f"... [+{len(content) - 500} chars]"
            else:
                display_content = content
            # vary emoji based on role
            if role == "user":
                emoji = "👤"
            elif role == "assistant":
                emoji = "🤖"
            else:
                emoji = "💬"
            self.logger.debug(f"{emoji} {i + 1} ({role}): {display_content}")

        self.logger.log_llm_call(method_name, len(messages))
        response = await self.llm.invoke(messages=messages, system=system_prompt, response_model=response_model)

        # Log response
        if hasattr(response, "model_dump_json"):
            response_json = response.model_dump_json(indent=2, exclude_none=True)
            if len(response_json) > 1000:
                display_response = response_json[:1000] + f"... [+{len(response_json) - 1000} chars]"
            else:
                display_response = response_json
            self.logger.debug(f"📤 Response content: {display_response}")
        else:
            response_str = str(response)
            if len(response_str) > 500:
                display_response = response_str[:500] + f"... [+{len(response_str) - 500} chars]"
            else:
                display_response = response_str
            self.logger.debug(f"📤 Response content: {display_response}")

        return response

    # ===== Action Discovery & Management =====

    @property
    def actions(self) -> dict[str, Callable]:
        """Get all available actions"""
        return self._actions

    # ===== Two-Loop System Core Methods =====

    async def achieve_goal(self, goal: str) -> GoalResult:
        """Main entry point for achieving a goal using the two-loop system

        Args:
            goal: The high-level goal to achieve

        Returns:
            GoalResult containing the final output and execution summary
        """
        start_time = time.time()
        self.logger.log_goal_start(goal)

        checkpoints = Checkpoints()

        # Create roadmap once at the beginning
        roadmap = await self.create_roadmap(goal, checkpoints)
        roadmap.goal = goal

        # Log roadmap creation
        milestone_goals = [m.goal for m in roadmap.milestones]
        self.logger.info(f"📋 Roadmap created with {len(roadmap.milestones)} milestones")
        for i, goal_text in enumerate(milestone_goals, 1):
            self.logger.info(f"  {i}. {goal_text}")

        # Execute each milestone in sequence
        for milestone in roadmap.milestones:
            if milestone.skipped:
                self.logger.info(f"⏭️ Skipping milestone: {milestone.goal}")
                continue

            # Execute the milestone
            self.logger.log_milestone_start(milestone.goal)
            checkpoint = await self.execute_milestone(milestone.goal, goal, checkpoints)
            checkpoints.append(checkpoint)

            # Mark milestone as completed
            milestone.completed = True

            # Log checkpoint
            self.logger.log_checkpoint_created(milestone.goal, checkpoint.overall_goal_completed)

            # Check if we can complete the goal early
            if checkpoint.overall_goal_completed:
                self.logger.info("🎯 Goal can be completed early based on checkpoint")
                break

        duration = time.time() - start_time
        self.logger.log_goal_complete(goal, len(checkpoints), duration)

        # Get all artifacts from checkpoints
        all_artifacts = checkpoints.get_all_artifacts()

        # Collect all errors from checkpoints
        all_errors = []
        for checkpoint in checkpoints:
            all_errors.extend(checkpoint.errors)

        # Determine success and summary
        if checkpoints and checkpoints[-1].overall_goal_result:
            # Use the checkpoint's result directly
            success = True
            summary = checkpoints[-1].overall_goal_result
            self.logger.info("🎉 Goal completed with final checkpoint result")
        else:
            # Create basic result from checkpoints
            success = bool(checkpoints) and not any(checkpoint.has_errors for checkpoint in checkpoints)
            summary_parts = [f"Completed {len(checkpoints)} milestones:"]
            for i, checkpoint in enumerate(checkpoints, 1):
                summary_parts.append(f"{i}. {checkpoint.milestone_goal}: {checkpoint.summary}")
            summary = "\n".join(summary_parts)
            self.logger.info(f"🎉 Goal completed with {len(checkpoints)} checkpoints")

        # Create final result
        result = GoalResult(
            goal=goal,
            success=success,
            summary=summary,
            artifacts=all_artifacts,
            errors=all_errors,
            checkpoints=checkpoints.checkpoints,  # Extract the list for backwards compatibility
            total_milestones=len(checkpoints),
            iterations=1,  # Static roadmap means single iteration
            duration=duration,
        )

        self.logger.info(f"📊 Final summary: {result.summary}")
        return result

    def _extract_final_data(self, checkpoints: Checkpoints) -> dict[str, Any]:
        """Extract final data from checkpoints - to be overridden by domain-specific agents"""
        return {}

    async def create_roadmap(self, goal: str, checkpoints: Checkpoints) -> Roadmap:
        """Create the strategic roadmap based on goal - called only once at the start

        You are a strategic planning AI that creates high-level roadmaps for achieving goals.

        Create a complete roadmap of 2-4 substantial milestones that will achieve the goal.
        Each milestone should be a meaningful step that accomplishes significant progress.

        IMPORTANT GUIDELINES:
        - Combine related operations into single milestones (e.g., "Generate and execute query to find top 5 customers")
        - Consider prerequisites like schema discovery for complex database queries
        - Create substantial milestones that can stand alone, not tiny incremental steps
        - Plan the complete path to success upfront

        Create a complete roadmap that will achieve the goal efficiently.
        """
        # Build messages for planning phase - only checkpoints, not execution details
        messages = []

        # Add the current goal
        messages.append({"role": "user", "content": f"Goal: {goal}"})

        roadmap = await self._call_llm("create_roadmap", messages, Roadmap)

        return roadmap

    async def execute_milestone(self, milestone_goal: str, overall_goal: str, previous_checkpoints: Checkpoints) -> Checkpoint:
        """Execute a single milestone using single-action iterations

        Args:
            milestone_goal: The specific milestone to achieve
            overall_goal: The overall goal for context
            previous_checkpoints: Checkpoints collection with completed checkpoints and artifacts

        Returns:
            Checkpoint summary for the planning phase
        """
        # Start with milestone goal and context from previous checkpoints
        messages = [{"role": "user", "content": f"Milestone: {milestone_goal}"}]

        # Add previous checkpoint summaries and artifacts as context
        if previous_checkpoints:
            context_msg = previous_checkpoints.build_context_summary()
            messages.insert(0, {"role": "assistant", "content": context_msg})
            self.logger.debug(f"💡 Added context from {len(previous_checkpoints)} previous checkpoints")

        outputs = {}
        step_summaries = []  # Accumulate step summaries

        iteration = 0
        while iteration < MAX_ITERATIONS:
            iteration += 1
            self.logger.log_milestone_step(iteration, milestone_goal, 0)

            # Step 1: Execute single action
            result = await self.execute_milestone_step(messages)
            outputs[result.action_name] = result.output

            # Step 2: Accumulate step summary (don't add to messages yet)
            step_summary = f"Step {iteration}: Executed {result.action_name}"
            if result.output:
                step_summary += f"\nOutput: {self._serialize_output(result.output)}"
            step_summaries.append(step_summary)

            # Step 3: Evaluate completion and create checkpoint
            try:
                checkpoint = await self.evaluate_and_checkpoint(milestone_goal, overall_goal, messages, outputs)
                # If we get here, milestone is complete (no missing_info exception)

                # Add consolidated message for all steps
                if step_summaries:
                    consolidated_message = "\n\n".join(step_summaries)
                    messages.append({"role": "assistant", "content": consolidated_message})

                self.logger.log_complete(milestone_goal, iteration)
                return checkpoint
            except Exception as e:
                if "Agent needs more information" in str(e):
                    # Extract the missing info message
                    missing_info = str(e).replace("Agent needs more information: ", "")
                    # Log incomplete evaluation
                    self.logger.log_checkpoint_evaluation_result(milestone_goal, False, missing_info)
                    self.logger.debug(f"Milestone not complete after step {iteration}: {e}")

                    # Add consolidated message for steps so far (for context in next iteration)
                    if step_summaries:
                        consolidated_message = "\n\n".join(step_summaries)
                        messages.append({"role": "assistant", "content": consolidated_message})
                        step_summaries = []  # Reset for next batch

                    continue
                else:
                    # Re-raise unexpected errors
                    raise

        # If we reach max iterations, create checkpoint anyway
        self.logger.warning(f"Milestone '{milestone_goal}' hit max iterations ({MAX_ITERATIONS}), creating checkpoint anyway")

        # Add final consolidated message
        if step_summaries:
            consolidated_message = "\n\n".join(step_summaries)
            messages.append({"role": "assistant", "content": consolidated_message})

        try:
            return await self.evaluate_and_checkpoint(milestone_goal, overall_goal, messages, outputs)
        except RuntimeError:
            # Even the fallback failed, create a minimal checkpoint
            checkpoint = Checkpoint(
                summary=f"Attempted {MAX_ITERATIONS} iterations but could not complete milestone. Partial progress made.",
                overall_goal_completed=False,
            )
            checkpoint.milestone_goal = milestone_goal
            return checkpoint

    async def execute_milestone_step(self, messages: list[dict]) -> ActionResult:
        """Execute a single milestone step.

        You are executing a milestone step. Based on the milestone goal and conversation history,
        determine what single action to take next.

        IMPORTANT: Review the conversation history to understand what has already been accomplished.
        Build upon previous work rather than duplicating it. If artifacts or outputs from previous
        milestones can be used directly, prefer actions that utilize them over actions that recreate them.

        When choosing action parameters, consider the outputs from previous actions in this milestone.
        For example, if a previous action discovered table names, use those exact names in subsequent actions.

        Available actions:
        {{ actions_list }}

        Choose the most appropriate action to make progress toward the milestone goal.
        """
        # Step 1: Decide which action to take
        decision = await self._call_llm("execute_milestone_step", messages, ActionDecision, actions_list=self._actions_info.build_action_list())

        # Step 2: Create dynamic model for the chosen action
        if decision.action_name not in self._actions_info:
            raise ValueError(f"Unknown action: {decision.action_name}")

        action_info = self._actions_info[decision.action_name]
        ActionModel = self._create_action_model(decision.action_name, action_info)

        # Step 3: Get structured action with parameters
        action_prompt = f"Execute {decision.action_name} based on this reasoning: {decision.reasoning}"
        action_messages = [{"role": "user", "content": action_prompt}]

        structured_action = await self._call_llm(decision.action_name, action_messages, ActionModel, action_info=action_info, reasoning=decision.reasoning)

        # Step 4: Execute the action
        method = self._actions[decision.action_name]
        # Extract parameters (skip reasoning, action fields) - preserve Pydantic model instances
        params = {}
        for field_name, field_value in structured_action:
            if field_name not in ("reasoning", "action", "missing_info"):
                params[field_name] = field_value

        output = await method(**params)

        # Step 5: Return action result
        return ActionResult(action_name=decision.action_name, output=output)

    async def evaluate_and_checkpoint(self, milestone_goal: str, overall_goal: str, messages: list[dict], outputs: dict[str, Any]) -> Checkpoint:
        """Evaluate milestone completion and create checkpoint with smart artifact extraction

        Based on the milestone goal, conversation history, and action outputs:

        1. Evaluate if the milestone is complete
        2. Extract ONLY goal-relevant information from outputs as artifacts
        3. Create a summary for strategic planning
        4. If the overall goal is now complete, provide comprehensive final result summary
        5. Track any errors that occurred during execution

        Overall goal: {{ overall_goal }}

        {% if output_summaries %}
        Action outputs from this milestone:
        {{ output_summaries }}
        {% endif %}

        CRITICAL for milestone completion evaluation:
        - Break down the milestone goal into its component parts and verify each part is complete
        - Look for conjunctions like "and", "then", "to" that indicate multiple requirements
        - If the milestone uses compound phrases, ALL parts must be satisfied
        - Be precise about what has been accomplished vs. what was requested
        - Set milestone_completed to True ONLY if the milestone goal has been fully achieved
        - If any required component is missing or incomplete, set milestone_completed to False

        IMPORTANT for artifacts field:
        - Store ONLY information needed to achieve the overall goal
        - Use descriptive keys like "customer_table", "aggregation_query", etc.
        - Don't store raw data - extract the essential information relevant to the goal
        - Key numeric results or row counts
        - Any error messages or issues encountered
        - Example: Artifact(key="orders_table_columns", value="order_id, customer_id, total_amount", description="Available columns in orders table")

        IMPORTANT for error handling:
        - If any actions failed or produced errors, list them in the errors field
        - Set milestone_completed to False if critical errors prevent milestone completion
        - Include specific error messages to help with debugging and retry logic

        IMPORTANT for overall_goal_result:
        - If this milestone produces the final answer to the overall goal, provide it here
        - Example: "The top 5 customers by total order amount are: 1. Customer A ($500), 2. Customer B ($450)..."
        - Only fill this if the overall goal is actually completed
        """
        # Log evaluation start
        self.logger.log_checkpoint_evaluation_start(milestone_goal, bool(outputs))

        # Build output summaries using str() on each output
        output_summaries = []
        for name, output in outputs.items():
            output_summaries.append(f"{name}: {output}")
        output_summary_text = "\n".join(output_summaries) if output_summaries else ""

        checkpoint = await self._call_llm(
            "evaluate_and_checkpoint",
            messages,  # Don't add milestone again - it's already in messages
            Checkpoint,
            overall_goal=overall_goal,
            output_summaries=output_summary_text,
        )

        # Set milestone_goal after LLM generation (excluded field)
        checkpoint.milestone_goal = milestone_goal

        # Log checkpoint details
        self.logger.info(f"Checkpoint created: {checkpoint.summary}")
        self.logger.log_checkpoint_content(checkpoint.summary)

        # Log evaluation result - checkpoint is complete if no missing_info exception was raised
        self.logger.log_checkpoint_evaluation_result(milestone_goal, checkpoint.milestone_completed, None)

        # Log any errors
        if checkpoint.has_errors:
            for error in checkpoint.errors:
                self.logger.warning(f"Checkpoint error: {error}")

        # Validate that milestone was actually completed if no errors
        if not checkpoint.has_errors and not checkpoint.milestone_completed:
            self.logger.warning(f"Milestone marked as incomplete despite no errors: {milestone_goal}")

        return checkpoint
