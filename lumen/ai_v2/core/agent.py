"""
Lumen AI 2.0 - Base Agent Framework

Hierarchical AI agent system:
- Roadmap: High-level plan with milestones
- Milestone: Execute steps until complete
- Actions: Individual tasks that do the work

Goal → Roadmap → Milestone → Actions
                    ↑         ↓
                Checkpoint ← Outputs
"""

import asyncio
import inspect
import json
import time

from collections.abc import Callable
from functools import wraps
from textwrap import dedent
from typing import Any

import param

from jinja2 import Template
from pydantic import BaseModel, Field, create_model

from ...ai.llm import Llm
from .logger import AgentLogger
from .models import (
    Action, ActionInfo, ActionParameter, Checkpoint, MilestoneStep,
    PartialEscapeModel, Roadmap,
)

MAX_ITERATIONS = 2
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

        # Render using Jinja2
        template = Template(dedented_docstring)
        rendered_template = template.render(**template_vars)

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

        # Create human-readable signature string with required/optional indicators
        sig_params = []
        for param_name, param_value in sig.parameters.items():
            if param_name not in ("self", "context"):  # Skip 'self' and 'context'
                param_str = param_name
                if param_value.annotation != inspect.Parameter.empty:
                    param_str += f": {getattr(param_value.annotation, '__name__', str(param_value.annotation))}"
                if param_value.default != inspect.Parameter.empty:
                    param_str += f" = {param_value.default}"
                else:
                    param_str += " (required)"
                sig_params.append(param_str)

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
        self._actions_info = {}  # Cache ActionInfo objects

        for name in dir(self):
            method = getattr(self, name)
            if callable(method) and hasattr(method, "_is_action"):
                self._actions[name] = method
                # Build and cache ActionInfo immediately
                self._actions_info[name] = self._build_action_info(name, method)

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

        # Add completion fields
        fields["completes_milestone"] = (str, Field(default="", description="Exact milestone text this action completes (if any)"))
        fields["is_final"] = (bool, Field(default=False, description="True if this action should complete the current milestone"))

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
        self.logger.debug(f"📨 Input messages for {method_name}: {len(messages)} messages")
        for i, message in enumerate(messages):
            role = message.get("role", "unknown")
            content = message.get("content", "")
            # Truncate very long messages for readability
            if len(content) > 500:
                display_content = content[:500] + f"... [+{len(content) - 500} chars]"
            else:
                display_content = content
            self.logger.debug(f"📨 Message {i + 1} ({role}): {display_content}")

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

    async def achieve_goal(self, goal: str) -> dict[str, Any]:
        """Main entry point for achieving a goal using the two-loop system

        Args:
            goal: The high-level goal to achieve

        Returns:
            Dictionary containing the final output and execution summary
        """
        start_time = time.time()
        self.logger.log_goal_start(goal)

        checkpoints = []
        iterations = 0

        while True:
            iterations += 1
            self.logger.debug(f"🔄 Roadmap iteration {iterations}")

            # Outer loop: Strategic planning
            roadmap = await self.create_roadmap(goal, checkpoints)

            # Find the next milestone to execute
            next_milestone = None
            for milestone in roadmap.milestones:
                if not milestone.completed and not milestone.skipped:
                    next_milestone = milestone
                    break

            if next_milestone is None:
                # All milestones completed or skipped
                self.logger.info("✅ All milestones completed or skipped")
                break

            # Inner loop: Tactical execution
            self.logger.log_milestone_start(next_milestone.goal)
            checkpoint = await self.execute_milestone(next_milestone.goal, goal)
            checkpoints.append(checkpoint)

            # Mark milestone as completed
            next_milestone.completed = True

            # Log checkpoint
            self.logger.log_checkpoint_created(checkpoint.milestone_goal, checkpoint.enables_goal_completion)

            # Check if we can complete the goal early
            if checkpoint.enables_goal_completion:
                self.logger.info("🎯 Goal can be completed early based on checkpoint")
                break

        duration = time.time() - start_time
        self.logger.log_goal_complete(goal, len(checkpoints), duration)

        return {
            "goal": goal,
            "checkpoints": checkpoints,
            "total_milestones": len(checkpoints),
            "iterations": iterations,
            "duration": duration,
            "final_data": {},
        }

    async def create_roadmap(self, goal: str, checkpoints: list[Checkpoint]) -> Roadmap:
        """Create or update the strategic roadmap based on goal and previous checkpoints

        You are a strategic planning AI that creates high-level roadmaps for achieving goals.

        Based on the goal and any previous checkpoint summaries, create a roadmap of 2-4 milestones.
        Each milestone should be a clear, actionable step toward the goal.

        If previous checkpoints show that certain milestones are no longer needed, mark them as skipped.
        Focus on what still needs to be done to achieve the goal.

        Available actions for milestone execution:
        {% for action_name, action_info in actions.items() %}
        - {{ action_name }}: {{ action_info.docstring.split('.')[0] if action_info.docstring else 'No description' }}
        {%- endfor %}

        Previous checkpoints:
        {% for checkpoint in checkpoints %}
        ✅ {{ checkpoint.milestone_goal }}: {{ checkpoint.summary }}
        {% endfor %}

        Create a roadmap that builds on these completed milestones.
        """
        # Build messages for planning phase - only checkpoints, not execution details
        messages = []

        # Add checkpoint summaries as assistant messages
        for checkpoint in checkpoints:
            messages.append({"role": "assistant", "content": f"✅ Completed: {checkpoint.summary}"})

        # Add the current goal
        messages.append({"role": "user", "content": f"Goal: {goal}"})

        roadmap = await self._call_llm("create_roadmap", messages, Roadmap, checkpoints=checkpoints, actions=self._actions_info)

        # Log roadmap creation
        if not checkpoints:
            milestone_goals = [m.goal for m in roadmap.milestones]
            self.logger.info(f"📋 Roadmap created with {len(roadmap.milestones)} milestones")
            for i, goal in enumerate(milestone_goals, 1):
                self.logger.info(f"  {i}. {goal}")
        else:
            completed = len(checkpoints)
            skipped = sum(1 for m in roadmap.milestones if m.skipped)
            remaining = len(roadmap.milestones) - skipped
            self.logger.info(f"📋 Roadmap updated: {completed} completed, {skipped} skipped, {remaining} remaining")
            for i, milestone in enumerate(roadmap.milestones, 1):
                status = "✅ Completed" if milestone.completed else "⏭️ Skipped" if milestone.skipped else "📋 Pending"
                self.logger.info(f"  {i}. {status}: {milestone.goal}")

        return roadmap

    async def evaluate_and_checkpoint(self, milestone_goal: str, overall_goal: str, messages: list[dict], outputs: dict[str, Any]) -> Checkpoint:
        """Evaluate milestone completion and create checkpoint

        Based on the milestone goal, conversation history, and action outputs:

        If milestone is COMPLETE:
        - Leave missing_info empty
        - Provide comprehensive summary for strategic planning

        If milestone is INCOMPLETE:
        - Set missing_info describing what's still needed
        - Provide summary of progress so far

        Overall goal: {{ overall_goal }}
        Raw outputs: {{ raw_results }}
        """
        # Include raw outputs for semantic understanding
        raw_results = []
        if outputs:
            for action_name, output in outputs.items():
                raw_results.append(f"{action_name}: {self._serialize_output(output)}")

        raw_results_text = "\n\n".join(raw_results) if raw_results else "No outputs"

        # Add outputs to the evaluation context
        eval_messages = messages + [{"role": "user", "content": f"Milestone: {milestone_goal}"}]

        checkpoint = await self._call_llm("evaluate_and_checkpoint", eval_messages, Checkpoint, overall_goal=overall_goal, raw_results=raw_results_text)

        # Log checkpoint details
        self.logger.info(f"Checkpoint created: {checkpoint.summary}")
        return checkpoint

    async def execute_milestone(self, milestone_goal: str, overall_goal: str) -> Checkpoint:
        """Execute a single milestone using tactical iterations

        Args:
            milestone_goal: The specific milestone to achieve
            overall_goal: The overall goal for context

        Returns:
            Checkpoint summary for the planning phase
        """
        # Isolated conversation for this milestone
        messages = [{"role": "user", "content": f"Milestone: {milestone_goal}"}]

        iteration = 0
        while iteration < MAX_ITERATIONS:
            iteration += 1
            self.logger.log_milestone_step(iteration, milestone_goal, 0)

            # Step 1: Decide what actions to take and execute them
            step = await self.execute_milestone_step(messages)

            # Step 2: Get the outputs from the executed actions
            outputs = step.outputs or {}

            # Step 3: Evaluate completion and create checkpoint if complete
            try:
                checkpoint = await self.evaluate_and_checkpoint(milestone_goal, overall_goal, messages, outputs)
                # If we get here, milestone is complete (no missing_info exception)
                self.logger.log_complete(milestone_goal, iteration)
                return checkpoint
            except Exception as e:
                if "Agent needs more information" in str(e):
                    # missing_info was set, continue to next iteration
                    self.logger.debug(f"Milestone not complete after step {iteration}: {e}")

                    # Add step to conversation history
                    step_summary = f"Step {iteration}: {step.reasoning}"
                    if step.actions:
                        action_names = [action.name for action in step.actions]
                        step_summary += f"\nActions: {action_names}"
                    if outputs:
                        result_summaries = []
                        for action, output in outputs.items():
                            result_str = str(output)
                            result_summaries.append(f"{action}: {result_str}")
                            if "Error" in result_str:
                                continue
                        step_summary += f"\nOutputs: {result_summaries}"
                    step_summary += f"\nStatus: {str(e).replace('Agent needs more information: ', '')}"

                    messages.append({"role": "assistant", "content": step_summary})
                    continue

        # If we reach max iterations, create checkpoint anyway
        self.logger.warning(f"Milestone '{milestone_goal}' hit max iterations ({MAX_ITERATIONS}), creating checkpoint anyway")
        try:
            return await self.evaluate_and_checkpoint(milestone_goal, overall_goal, messages, {})
        except RuntimeError:
            # Even the fallback failed, create a minimal checkpoint
            checkpoint = Checkpoint(
                milestone_goal=milestone_goal,
                summary=f"Attempted {MAX_ITERATIONS} iterations but could not complete milestone. Partial progress made.",
                enables_goal_completion=False,
            )
            return checkpoint

    async def execute_milestone_step(self, messages: list[dict]) -> MilestoneStep:
        """Execute a single milestone step with batched actions

        You are executing a milestone step. Based on the milestone goal and conversation history,
        determine what actions to take next.

        You can execute multiple actions in a single step for efficiency.
        Set complete to True when the milestone is fully achieved.

        Available actions:
        {%- for action_name, action_info in actions.items() %}
        - {{ action_name }}({{ action_info.signature_str }}): {{ action_info.docstring.split('.')[0] if action_info.docstring else 'No description' }}
        {%- endfor %}

        Focus on making progress toward the milestone goal.
        """
        step = await self._call_llm("execute_milestone_step", messages, MilestoneStep, actions=self._actions_info)

        # Execute the actions and store outputs
        if step.actions:
            step.outputs = await self.execute_actions(step.actions)

            # Log outputs
            self.logger.debug(f"Action outputs: {len(step.outputs)} outputs")
            for action_name, output in step.outputs.items():
                if isinstance(output, str) and "Error" in output:
                    self.logger.error(f"Output {action_name}: {output}")
                else:
                    self.logger.debug(f"Output {action_name}: {type(output).__name__}")

        return step

    async def execute_actions(self, actions: list[Action]) -> dict[str, Any]:
        """Execute a list of actions with order-preserving parallel execution

        Returns:
            Dictionary mapping action names to their outputs
        """
        if not actions:
            return {}

        start_time = time.time()
        self.logger.log_actions_start(actions)

        outputs = []
        current_parallel_group = []

        # Process actions in order, grouping parallelizable ones
        for action in actions:
            if not getattr(self._actions.get(action.name), "_locked", False):
                current_parallel_group.append(action)
            else:
                # Execute any pending parallel group first
                if current_parallel_group:
                    if len(current_parallel_group) == 1:
                        # Single action - execute directly
                        output = await self._execute_single_action_safe(current_parallel_group[0])
                        outputs.append(output)
                    else:
                        # Multiple actions - execute in parallel
                        tasks = [asyncio.create_task(self._execute_single_action_safe(action)) for action in current_parallel_group]
                        parallel_results = await asyncio.gather(*tasks, return_exceptions=True)
                        processed_results = [str(output) if isinstance(output, Exception) else output for output in parallel_results]
                        outputs.extend(processed_results)
                    current_parallel_group = []

                # Execute locked action alone
                output = await self._execute_single_action_safe(action)
                outputs.append(output)

        # Execute any remaining parallel group
        if current_parallel_group:
            if len(current_parallel_group) == 1:
                # Single action - execute directly
                output = await self._execute_single_action_safe(current_parallel_group[0])
                outputs.append(output)
            else:
                # Multiple actions - execute in parallel
                tasks = [asyncio.create_task(self._execute_single_action_safe(action)) for action in current_parallel_group]
                parallel_results = await asyncio.gather(*tasks, return_exceptions=True)
                processed_results = [str(output) if isinstance(output, Exception) else output for output in parallel_results]
                outputs.extend(processed_results)

        duration = time.time() - start_time
        self.logger.log_actions_complete(actions, duration)

        # Map action names to outputs
        action_results = {}
        for i, action in enumerate(actions):
            if i < len(outputs):
                action_results[action.name] = outputs[i]
            else:
                action_results[action.name] = f"Error: No output for {action.name}"

        return action_results
