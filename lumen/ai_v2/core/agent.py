"""
Lumen AI 2.0 - Base Agent Framework
"""

import inspect
import json

from collections.abc import Callable
from functools import wraps
from textwrap import dedent
from typing import Any

import param

from jinja2 import Template
from pydantic import BaseModel, Field, create_model

from ...ai.llm import Llm
from .models import ActionInfo, ActionParameter, PartialEscapeModel

MAX_ITERATIONS = 8
MAX_ERRORS = 3


def action(func: Callable) -> Callable:
    """Decorator to mark methods as callable actions with automatic error handling"""

    @wraps(func)
    async def wrapper(self, *args, **kwargs):
        return await func(self, *args, **kwargs)

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
    """

    # ===== Configuration =====

    name = param.String(doc="Name of the agent")
    llm = param.ClassSelector(class_=Llm, doc="LLM instance for this agent")

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

    def _serialize_output(self, output: str | dict | BaseModel) -> str:
        if isinstance(output, str):
            return output.strip()

        if isinstance(output, BaseModel):
            return output.model_dump_json(indent=2, exclude_none=True)
        elif isinstance(output, dict):
            return json.dumps(output, indent=2, ensure_ascii=False)
        else:
            raise ValueError(f"Unsupported output type: {type(output)}. Expected str, dict, or BaseModel.")

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
