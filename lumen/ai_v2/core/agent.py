"""
Lumen AI 2.0 - Base Agent Framework
"""

import inspect
import time

from collections.abc import Callable
from functools import wraps
from textwrap import dedent
from typing import Any, Literal

import param

from jinja2 import Template
from pydantic import Field, create_model

from ...ai.llm import Llm
from .events import print_execution_event
from .models import (
    ActionError, ActionInfo, ActionParameter, ExecutionContext, ExecutionEvent,
    PartialEscapeModel, Roadmap, SpecFragment,
)

MAX_ITERATIONS = 8
MAX_ERRORS = 3


def action(func: Callable) -> Callable:
    """Decorator to mark methods as callable actions with automatic error handling"""

    @wraps(func)
    async def wrapper(self, *args, **kwargs):
        try:
            return await func(self, *args, **kwargs)
        except Exception as e:
            # Print full traceback for debugging
            import traceback

            print(f"\n🚨 Exception in action {func.__name__}:")  # noqa: T201
            traceback.print_exc()

            # Auto-generate error type
            class_name = self.__class__.__name__.lower().replace("agent", "")
            func_name = func.__name__
            error_type = f"{class_name}_{func_name}_error"

            # Create error fragment
            fragment = SpecFragment.create(
                "error",
                {
                    "error_type": error_type,
                    "message": str(e),
                    "action": func.__name__,
                    "agent": self.name if hasattr(self, "name") else self.__class__.__name__,
                    "args": list(args),
                    "kwargs": {k: str(v) for k, v in kwargs.items()},  # Convert to strings for serialization
                    "traceback": traceback.format_exc(),  # Include full traceback in fragment
                },
            )
            return fragment

    wrapper._is_action = True
    return wrapper


class BaseAgent(param.Parameterized):
    """Base class for all Lumen AI agents

    Purpose:
    Provides a framework for creating AI agents that can perform specific tasks
    using LLM capabilities. Uses a three-tier hierarchical architecture:

    1. High-level roadmap generation (once per goal)
    2. Milestone execution with decisive completion
    3. Atomic @action methods for individual tasks

    Core Capabilities:
    - Action discovery and registration via @action decorator
    - Dynamic Pydantic model generation for perfect type safety
    - Two-step planning: action selection + parameter collection
    - Prompt template rendering with Jinja2
    - Error handling with resilience and recovery
    - Hierarchical planning with roadmap → milestone → micro-actions
    - Production-ready execution with user control and observability
    - Memory management with fragment ring buffer
    - Decisive completion to prevent endless loops

    Public Planning Methods:
    - plan(): Main orchestration method using hierarchical architecture
    - create_roadmap(): Generate strategic roadmap with milestones
    - execute_milestone(): Execute single milestone with decisive completion
    - select_action(): Choose the best action for a milestone
    - execute_action(): Execute chosen action with dynamically typed parameters
    - get_actions_info(): Extract action signatures and documentation

    Public Extension Points:
    - get_completion_message(): Custom completion message

    Example Usage:
    ```python
    agent = SQLAgent(llm=llm, source=source)

        # Full automated planning with event handling
            def event_handler(event: ExecutionEvent):
                print(f"{event.action}: {event.milestone}")

            result = await agent.plan("Find customer purchases", event_handler=event_handler)

            # Manual two-step control
            selected_action = await agent.select_action("Find schemas", context)
            micro_step = await agent.execute_action(selected_action, "Find schemas", context)

            # Manual milestone control
            context = ExecutionContext(goal="Find customers", roadmap=roadmap, ...)
            success = await agent.execute_milestone("Find database schemas", context)
            ```
    """

    # ===== Configuration =====

    name = param.String(doc="Name of the agent")
    llm = param.ClassSelector(class_=Llm, doc="LLM instance for this agent")
    max_tasks = param.Integer(default=20, bounds=(1, 100), doc="Maximum number of tasks in iterative planning")
    roadmap_update_frequency = param.Integer(default=2, bounds=(1, 10), doc="How often to review and update roadmap (every N steps)")

    # ===== Initialization =====

    def __init__(self, **params):
        super().__init__(**params)
        if not self.name:
            self.name = self.__class__.__name__.lower()
        self._discover_actions()

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
        action_info = self.get_action_info(method_name)
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
        print(f"\033[90m{rendered_template}\033[0m")  # noqa: T201
        return rendered_template

    def _discover_actions(self):
        """Discover all methods decorated with @action

        This method scans the class for methods decorated with @action
        and registers them in the internal _actions dictionary.

        This is called automatically during initialization, but can be
        called manually if actions are added dynamically.
        """
        self._actions = {}
        for name in dir(self):
            method = getattr(self, name)
            if callable(method) and hasattr(method, "_is_action"):
                self._actions[name] = method

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

    def _get_milestone(self, context: ExecutionContext) -> str | None:
        """Get the first uncompleted milestone"""
        for milestone in context.roadmap.milestones:
            if "- [ ]" in milestone:
                return milestone.replace("- [ ]", "").strip()
            elif "- [x]" not in milestone:
                # Handle milestones without checkbox format
                milestone_clean = milestone.strip()
                if milestone_clean and milestone_clean not in context.completed_milestones:
                    return milestone_clean
        return None

    def _mark_complete(self, milestone: str, context: ExecutionContext):
        """Mark milestone as complete using checkbox syntax"""
        milestone_clean = milestone.replace("- [ ]", "").strip()

        # Add to completed milestones list
        if milestone_clean not in context.completed_milestones:
            context.completed_milestones.append(milestone_clean)

        # Update roadmap milestones to use checkbox format
        for i, roadmap_milestone in enumerate(context.roadmap.milestones):
            if milestone_clean in roadmap_milestone:
                # Convert to completed checkbox format
                if "- [ ]" in roadmap_milestone:
                    context.roadmap.milestones[i] = roadmap_milestone.replace("- [ ]", "- [x]")
                elif "- [x]" not in roadmap_milestone:
                    # Plain text milestone - convert to checkbox format
                    context.roadmap.milestones[i] = f"- [x] {roadmap_milestone.strip()}"
                break

    def _is_roadmap_complete(self, context: ExecutionContext) -> bool:
        """Check if all milestones are completed"""
        if not context.roadmap.milestones:
            return True

        # Check if all milestones are completed (handle both checkbox and plain formats)
        for milestone in context.roadmap.milestones:
            if "- [ ]" in milestone:
                return False  # Found uncompleted checkbox
            elif "- [x]" in milestone:
                continue  # This one is completed
            else:
                # Plain text milestone - check if it's in completed list
                milestone_clean = milestone.strip()
                if milestone_clean and milestone_clean not in context.completed_milestones:
                    return False

        return True

    # ===== Action Discovery & Management =====

    @property
    def actions(self) -> dict[str, Callable]:
        """Get all available actions"""
        return self._actions

    def get_action_info(self, action_name: str) -> ActionInfo | None:
        """Get detailed information about a specific action"""
        if action_name not in self._actions:
            return None

        method = self._actions[action_name]
        sig = inspect.signature(method)

        # Get parameter info
        params = {}
        for param_name, param_value in sig.parameters.items():
            if param_name not in ("self",):  # Skip 'self' parameter
                params[param_name] = ActionParameter(
                    name=param_name,
                    annotation=param_value.annotation,
                    default=param_value.default if param_value.default != inspect.Parameter.empty else None,
                    kind=param_value.kind.name,
                )

        # Create human-readable signature string
        sig_params = []
        for param_name, param_value in sig.parameters.items():
            if param_name not in ("self", "context"):  # Skip 'self' and 'context'
                param_str = param_name
                if param_value.annotation != inspect.Parameter.empty:
                    param_str += f": {getattr(param_value.annotation, '__name__', str(param_value.annotation))}"
                if param_value.default != inspect.Parameter.empty:
                    param_str += f" = {param_value.default}"
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

    def get_actions_info(self) -> dict[str, ActionInfo]:
        """Get detailed information about all available actions"""
        detailed_actions = {}

        for action_name in self._actions:
            action_info = self.get_action_info(action_name)
            if action_info:
                detailed_actions[action_name] = action_info

        return detailed_actions

    # ===== Main Planning Methods =====

    async def plan(
        self, goal: str, context: dict[str, Any] | None = None, event_handler: Callable[[ExecutionEvent], None] | None = None, user_control: bool = True
    ) -> ExecutionContext:
        """Execute hierarchical roadmap → milestone → micro-actions plan

        This is the main planning method that orchestrates goal achievement through:
        1. Creating a strategic roadmap with milestones (once)
        2. Executing milestones sequentially with decisive completion
        3. Using micro-actions (@action methods) as atomic building blocks

        Args:
            goal: The goal to achieve
            context: Optional initial context
            event_handler: Optional event handler for observability
            user_control: Whether to enable user stop/pause controls

        Returns:
            ExecutionContext with final results and progress tracking
        """
        # Initialize context with just the required fields
        execution_context = ExecutionContext(
            goal=goal,
            memory=context or {},
        )

        event_handler = event_handler or self.default_event_handler

        # Phase 1: Generate strategic roadmap (once)
        try:
            roadmap = await self.create_roadmap(goal, context)
            execution_context.roadmap = roadmap

            event_handler(ExecutionEvent(milestone="roadmap_creation", iteration=0, action="roadmap_complete", completed=True))

        except Exception as e:
            event_handler(ExecutionEvent(milestone="roadmap_creation", iteration=0, action="roadmap_failed", error=str(e)))
            return execution_context

        # Phase 2: Execute milestones with decisive completion
        while not self._is_roadmap_complete(execution_context):
            next_milestone = self._get_milestone(execution_context)  # TODO: maybe pop instead?
            if not next_milestone:
                break

            # Check for user stop request
            if execution_context.should_stop:
                break

            try:
                success = await self.execute_milestone(next_milestone, execution_context, event_handler)
                if not success:
                    # Milestone failed - continue with next or stop based on error handling
                    pass

            except Exception as e:
                event_handler(ExecutionEvent(milestone=next_milestone, iteration=execution_context.iteration, action="milestone_failed", error=str(e)))
                # Continue with next milestone or escalate based on error severity

        # Phase 3: Final completion
        event_handler(ExecutionEvent(milestone="plan_complete", iteration=0, action="all_milestones_complete", completed=True))

        return execution_context

    async def create_roadmap(self, goal: str, context: dict[str, Any] | None = None) -> Roadmap:
        """Create strategic roadmap with markdown checklist

        Create roadmap for: {{ goal }}

        {% if context %}
        Context: {{ context }}
        {% endif %}

        Generate 2-4 milestones in markdown checklist format:
        - [ ] First major milestone
        - [ ] Second major milestone
        - [ ] Final milestone

        Each milestone should be:
        - A clear, completable objective
        - Action-oriented: "Discover X", "Find Y", "Generate Z"
        - Building toward the goal logically

        Think through your approach first, then list milestones.
        """
        system_prompt = self._render_prompt("create_roadmap", goal=goal, context=context or {})
        messages = [{"role": "user", "content": f"Create roadmap for: {goal}"}]
        return await self.llm.invoke(messages, system=system_prompt, response_model=Roadmap)

    async def execute_milestone(self, milestone: str, context: ExecutionContext, event_handler: Callable[[ExecutionEvent], None] | None = None) -> bool:
        """Execute milestone with decisive completion and error resilience"""
        event_handler = event_handler or self.default_event_handler
        context.current_milestone = milestone
        context.iteration = 0
        context.error_count = 0

        event_handler(ExecutionEvent(milestone=milestone, iteration=0, action="milestone_start", result_type="start"))

        for iteration in range(MAX_ITERATIONS):
            context.iteration = iteration + 1

            # Check for user control signals
            if context.should_stop:
                return await self.handle_early_stop(milestone, context, event_handler)

            if context.pause_requested:
                # Handle pause request (placeholder for future implementation)
                context.pause_requested = False  # Reset after handling

            start_time = time.perf_counter()

            try:
                # Select the best action for this milestone
                selected_action = await self.select_action(milestone, context)

                # Execute the selected action (consolidated method)
                result = await self.execute_action(selected_action, milestone, context)

                # Track execution time
                duration_ms = (time.perf_counter() - start_time) * 1000

                # Store result with memory management
                context.add_fragment(f"step_{iteration + 1}", result)

                # Log event
                event_handler(
                    ExecutionEvent(
                        milestone=milestone, iteration=iteration + 1, action=selected_action, duration_ms=duration_ms, result_type=result.type, completed=False
                    )
                )

                # Check for completion
                self._mark_complete(milestone, context)
                event_handler(ExecutionEvent(milestone=milestone, iteration=iteration + 1, action="milestone_complete", completed=True))
                return True

            except ActionError as e:
                context.error_count += 1
                await self.handle_action_error(e, context, event_handler)

                # Too many errors - force completion or escalate
                if context.error_count >= MAX_ERRORS:
                    return await self.handle_excessive_errors(milestone, context, event_handler)

            except Exception as e:
                # Unexpected error - log and continue or escalate
                event_handler(ExecutionEvent(milestone=milestone, iteration=iteration + 1, action="system_error", error=str(e)))

                # Critical system error - escalate to user
                if await self.is_critical_error(e):
                    return await self.escalate_to_user(milestone, e, context)

        # Hit max iterations - force completion
        return await self.force_completion(milestone, context, event_handler)

    # ===== Action Selection & Execution =====

    async def select_action(self, milestone: str, context: ExecutionContext) -> str:
        """Select best action for milestone

        Current milestone: {{ milestone }}
        Iteration: {{ iteration }}/8

        Recent results:
        {% for key, fragment in recent_fragments.items() %}
        - {{ key }}: {{ fragment.type }} - {{ fragment.content | truncate(80) }}
        {% endfor %}

        {% if iteration >= 6 %}
        🚨 APPROACHING MAX ITERATIONS - Focus on completion!
        Choose actions that will complete this milestone quickly.
        {% elif iteration >= 3 %}
        ⚠️ Consider completion - do you have sufficient results to move forward?
        {% endif %}

        Available actions:
        {%- for action_name, action_info in available_actions.items() %}
        - {{ action_name }}: {{ action_info.docstring.split('\n')[0] }}
        {% endfor %}

        Choose the BEST action for this milestone. Think about what specific task is needed.
        """
        # Get available actions
        available_actions = self.get_actions_info()
        action_names = list(available_actions.keys())

        # Create action selector model
        action_selector_fields = {
            "reasoning": (str, Field(description="Think through what action is needed for this milestone")),
            "action": (Literal[tuple(action_names)], Field(description=f"Choose from: {', '.join(action_names)}")),
        }

        ActionSelector = create_model("ActionSelector", __base__=PartialEscapeModel, **action_selector_fields)

        # Render prompt for action selection
        recent_fragments = {k: context.fragments[k] for k in list(context.fragments.keys())[-3:]}

        system_prompt = self._render_prompt(
            "select_action",
            milestone=milestone,
            iteration=context.iteration,
            recent_fragments=recent_fragments,
            available_actions=available_actions,
        )

        messages = [{"role": "user", "content": f"Choose the best action for milestone: {milestone}"}]

        # Get action choice
        action_choice = await self.llm.invoke(messages, system=system_prompt, response_model=ActionSelector)

        return action_choice.action

    async def execute_action(self, action_name: str, milestone: str, context: ExecutionContext) -> SpecFragment:
        """Execute the selected action by collecting proper parameters and executing it

        Execute action: {{ action_name }}
        Milestone: {{ milestone }}

        Action signature: {{ action_name }}({{ action_info.signature_str }})
        Description: {{ action_info.docstring[:200] }}...

        Fill in the required parameters to accomplish the milestone.
        """
        # Get action info for parameter collection
        available_actions = self.get_actions_info()
        action_info = available_actions[action_name]

        # Create dynamic model for the selected action
        DynamicActionModel = self._create_action_model(action_name, action_info)

        # Render the prompt template
        system_prompt = self._render_prompt("execute_action", action_name=action_name, milestone=milestone, action_info=action_info)

        messages = [{"role": "user", "content": system_prompt}]

        # Get the fully specified action with parameters
        action_inputs = await self.llm.invoke(messages, system=system_prompt, response_model=DynamicActionModel)

        # Get the action method
        action_method = self.actions[action_name]
        if not action_method:
            raise ActionError(f"Action '{action_name}' not found", is_critical=False)

        kwargs = {}
        signature = action_info.signature
        for param_name in signature.parameters:
            if param_name == "self":
                continue
            if hasattr(action_inputs, param_name):
                value = getattr(action_inputs, param_name)
                if value is not None:  # Only include non-None values
                    kwargs[param_name] = value

        # Execute the action directly - @action decorator handles all error cases
        result = await action_method(**kwargs)
        return result if isinstance(result, SpecFragment) else SpecFragment(type="unknown", content={})

    # ===== Error Handling & User Control =====

    async def handle_action_error(self, error: ActionError, context: ExecutionContext, event_handler: Callable[[ExecutionEvent], None]):
        """Handle action errors with recovery strategies"""

        # Log the error
        event_handler(ExecutionEvent(milestone=context.current_milestone, iteration=context.iteration, action="error_recovery", error=str(error)))

        # Store error fragment for learning
        error_fragment = SpecFragment.create(
            "error", {"message": str(error), "recoverable": error.recoverable, "critical": error.is_critical, "iteration": context.iteration}
        )
        context.add_fragment(f"error_{context.iteration}", error_fragment)

    async def handle_early_stop(self, milestone: str, context: ExecutionContext, event_handler: Callable[[ExecutionEvent], None]) -> bool:
        """Handle user-requested early termination"""
        event_handler(ExecutionEvent(milestone=milestone, iteration=context.iteration, action="user_stop", completed=False))

        # Complete with partial results
        self._mark_complete(milestone, context)
        return True

    async def handle_excessive_errors(self, milestone: str, context: ExecutionContext, event_handler: Callable[[ExecutionEvent], None]) -> bool:
        """Handle too many errors by forcing completion"""
        self._mark_complete(milestone, context)
        return True

    async def is_critical_error(self, error: Exception) -> bool:
        """Determine if an error is critical and requires escalation"""
        return False  # Default: no errors are critical

    async def escalate_to_user(self, milestone: str, error: Exception, context: ExecutionContext) -> bool:
        """Escalate critical errors to user"""
        # In a real implementation, this would use UserInteraction
        return False

    async def force_completion(self, milestone: str, context: ExecutionContext, event_handler: Callable[[ExecutionEvent], None]) -> bool:
        """Force completion when hitting max iterations"""
        self._mark_complete(milestone, context)
        event_handler(ExecutionEvent(milestone=milestone, iteration=context.iteration, action="forced_completion", completed=True))
        return True

    # ===== Extension Points for Subclasses =====

    def get_completion_message(self, goal: str, context: ExecutionContext) -> str:
        """Get a custom completion message for the goal"""
        return "Goal completed successfully!"

    # ===== Utilities =====

    def default_event_handler(self, event: ExecutionEvent):
        """Default event handler that prints progress"""
        print_execution_event(event)
