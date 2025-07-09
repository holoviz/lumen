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
from typing import Any, Literal

import param

from instructor.exceptions import InstructorRetryException
from jinja2 import Template, TemplateError, UndefinedError
from pydantic import BaseModel, Field, create_model
from tenacity import (
    retry, retry_if_exception_type, stop_after_attempt, wait_exponential,
)

from ...ai.llm import Llm
from ...ai.models import PartialBaseModel
from .logger import AgentLogger
from .models import (
    Action, ActionInfo, ActionParameter, ActionResult, ActionsInfo, Checkpoint,
    Checkpoints, GoalResult, MissingDeliverablesError, Roadmap,
)

MAX_ITERATIONS = 3
MAX_ERRORS = 3


def action(func: Callable | None = None, *, locked: bool = False) -> Callable:
    """Decorator to mark methods as callable actions with automatic error handling

    Args:
        func: The function to decorate
        locked: Whether this action must be executed exclusively
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
        for param_info in params.values():
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

        fields = {
            "reasoning": (str, Field(description="Think through what needs to be done next and why this specific action is the right choice")),
            "action_name": (Literal[tuple(self._actions.keys())], Field(description="Name of the action to execute")),
        }
        self._action_decision_model = create_model("ActionDecision", __base__=PartialBaseModel, **fields)

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
        dynamic_model = create_model(model_name, __base__=PartialBaseModel, **fields)

        return dynamic_model

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
                    if filtered_value is not None and filtered_value not in ("", {}, []):
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
        self.logger.log_prompt_content(method_name, system_prompt)

        self.logger.log_llm_call(method_name, len(messages))
        for i, message in enumerate(messages):
            role = message.get("role", "unknown")
            content = message.get("content", "")
            if role == "user":
                emoji = "👤"
            elif role == "assistant":
                emoji = "🤖"
            else:
                emoji = "💬"
            self.logger.debug(f"{emoji} {i + 1} ({role}): {content}")
        response = await self.llm.invoke(messages=messages, system=system_prompt, response_model=response_model)

        # Log response
        if hasattr(response, "model_dump_json"):
            response_json = response.model_dump_json(indent=2, exclude_none=True)
            # self.logger.debug(f"📤 Response content: {response_json}")
        else:
            response_str = str(response)
            self.logger.debug(f"📤 Response content: {response_str}")

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

        # Create roadmap once at the beginning
        roadmap = await self.create_roadmap(goal)
        roadmap.goal = goal

        # Log roadmap creation
        milestone_goals = [m.goal for m in roadmap.milestones]
        self.logger.info(f"📋 Roadmap created with {len(roadmap.milestones)} milestones")
        for i, goal_text in enumerate(milestone_goals, 1):
            self.logger.info(f"  {i}. {goal_text}")

        # Execute each milestone in sequence
        checkpoints = Checkpoints()
        for milestone in roadmap.milestones:
            if milestone.skipped:
                self.logger.info(f"⏭️ Skipping milestone: {milestone.goal}")
                continue

            # Execute the milestone
            checkpoint = await self.execute_milestone(milestone.goal, goal, checkpoints)
            checkpoints.append(checkpoint)

            # Mark milestone as completed
            milestone.completed = True

            # Check if we can complete the goal early
            if checkpoint.overall_goal_completed:
                self.logger.info("🎯 Goal can be completed early based on checkpoint")
                break

        duration = time.time() - start_time
        self.logger.log_goal_complete(goal, len(checkpoints), duration)

        # Determine success and summary
        if checkpoints:
            success = True
            summary = checkpoints[-1].overall_goal_result
            self.logger.info("🎉 Goal completed with final checkpoint result")

        # Create final result
        result = GoalResult(
            goal=goal,
            success=success,
            summary=summary,
            checkpoints=checkpoints,
            duration=duration,
        )

        self.logger.info(f"📊 Final summary: {result.summary}")
        return result

    async def create_roadmap(self, goal: str) -> Roadmap:
        """Create strategic roadmap for achieving a high-level goal

        Usually 2-4 atomic milestones that build logically toward the goal.
        Consider prerequisites, dependencies, and plan complete path upfront.
        """
        messages = [{"role": "user", "content": f"Goal: {goal}"}]
        roadmap = await self._call_llm("create_roadmap", messages, Roadmap)
        return roadmap

    async def execute_milestone(self, milestone_goal: str, overall_goal: str, checkpoints: Checkpoints | None = None) -> Checkpoint:
        """Execute a single milestone using single-action iterations

        Args:
            milestone_goal: The specific milestone to achieve
            overall_goal: The overall goal for context
            checkpoints: Checkpoints collection with completed checkpoints and artifacts

        Returns:
            Checkpoint summary for the planning phase
        """
        # Build conversation with proper user/assistant alternation
        messages = []
        if checkpoints:
            messages.extend(checkpoints.serialize())
        messages.append({"role": "user", "content": milestone_goal})

        iteration = 0
        while iteration < MAX_ITERATIONS:
            iteration += 1
            self.logger.log_milestone_step(iteration, milestone_goal, 0)

            # Step 1: Execute single action
            result = await self.execute_milestone_step(messages)

            # Step 2: Add assistant message with action result
            step_message = f"Executed {result.action_name!r}, outputting:\n{result}"
            messages.append({"role": "assistant", "content": step_message})

            # Step 3: Evaluate completion and create checkpoint
            try:
                checkpoint = await self.evaluate_checkpoint(milestone_goal, messages)
                # If we get here, milestone is complete (no missing_parts exception)
                self.logger.log_complete(milestone_goal, iteration)
                return checkpoint
            except (MissingDeliverablesError, InstructorRetryException) as e:
                messages.append({"role": "user", "content": f"Checkpoint incomplete: {e!s}"})

        # If we reach max iterations, create checkpoint anyway
        self.logger.warning(f"Milestone '{milestone_goal}' hit max iterations ({MAX_ITERATIONS}), creating checkpoint anyway")

        try:
            return await self.evaluate_checkpoint(milestone_goal, messages)
        except RuntimeError:
            # Even the fallback failed, create a minimal checkpoint
            checkpoint = Checkpoint(
                summary=f"Attempted {MAX_ITERATIONS} iterations but could not complete milestone. Partial progress made.",
                overall_goal_completed=False,
            )
            checkpoint.milestone_goal = milestone_goal
            return checkpoint

    @retry(
        stop=stop_after_attempt(MAX_ERRORS),
        retry=retry_if_exception_type(ValueError),
    )
    async def execute_milestone_step(self, messages: list[dict]) -> ActionResult:
        """Execute a single milestone step.

        You are executing a milestone step. Based on the milestone goal and conversation history,
        determine what single action to take next.

        Available actions:
        {{ actions_info }}

        Choose the most appropriate action to make progress toward the milestone goal.
        """
        # Step 1: Decide which action to take
        decision = await self._call_llm("execute_milestone_step", messages, self._action_decision_model, actions_info=self._actions_info)

        action_info = self._actions_info[decision.action_name]
        ActionModel = self._create_action_model(decision.action_name, action_info)

        # Step 2: Get structured action with parameters
        action_prompt = f"Execute {decision.action_name} based on this reasoning: {decision.reasoning}"

        action_messages = messages.copy()
        if action_messages[-1]["role"] == "user":
            # Combine the last message with the action prompt
            action_messages[-1]["content"] += f"\n\n{action_prompt}"
        else:
            action_messages.append({"role": "user", "content": action_prompt})

        structured_action = await self._call_llm(decision.action_name, action_messages, ActionModel, action_info=action_info, reasoning=decision.reasoning)

        # Step 3: Execute the action
        method = self._actions[decision.action_name]
        # Extract parameters (skip reasoning, action fields) - preserve Pydantic model instances
        params = {}
        for field_name, field_value in structured_action:
            if field_name not in ("reasoning", "action", "missing_parts"):
                params[field_name] = field_value

        output = await method(**params)

        # Step 5: Return action result
        return ActionResult(action_name=decision.action_name, output=output)

    async def evaluate_checkpoint(self, milestone_goal: str, messages: list[dict]) -> Checkpoint:
        """Evaluate milestone completion and create checkpoint with smart artifact extraction

        Based on the milestone goal (user request) and conversation history:
        1. Evaluate if the milestone is complete
        2. Extract goal-relevant artifacts for future milestones
        3. Create a summary for strategic planning
        4. Track any errors that occurred
        """
        checkpoint = await self._call_llm(
            "evaluate_checkpoint",
            messages,
            Checkpoint,
        )

        # Set milestone_goal after LLM generation (excluded field)
        checkpoint.milestone_goal = milestone_goal

        # Log checkpoint details with yellow highlighting
        self.logger.log_checkpoint_created(checkpoint.summary, len(checkpoint.artifacts))
        return checkpoint
