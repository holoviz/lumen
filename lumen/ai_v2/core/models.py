"""
Lumen AI 2.0 - Core Response Models
Production-ready hierarchical agent models for decisive execution

Note: reasoning, chain_of_thought always comes before
the bools, actions, etc to give the LLM time to reflect
and build up context!
"""

import time

from collections.abc import Callable
from inspect import Signature
from typing import Any

from instructor.dsl.partial import PartialLiteralMixin
from pydantic import BaseModel, Field, field_validator

# ===== Base Models =====

class PartialBaseModel(BaseModel, PartialLiteralMixin):
    """Base model that supports streaming through async generators"""

class PartialEscapeModel(PartialBaseModel):
    """Base model with escape hatch - set missing_info to halt execution when you need more information"""

    missing_info: str = Field(
        default="", description="If you need more information to proceed, describe what's missing here. This will halt execution with the message."
    )

    @field_validator("missing_info")
    def validate_missing_info(cls, v: str) -> str:
        if v.strip():
            raise RuntimeError(f"Agent needs more information: {v}")
        return v

# ===== NEW: Production-Ready Hierarchical Models =====

class Roadmap(PartialEscapeModel):
    """Strategic roadmap with reasoning and milestones"""

    reasoning: str = Field(default="", description="Think through the goal step-by-step to plan your approach")
    milestones: list[str] = Field(default_factory=list, description="High-level goals to address user's query in markdown checklist format")


class ActionParameter(PartialBaseModel):
    """Information about an action parameter"""

    name: str = Field(description="Parameter name")
    annotation: Any = Field(description="Type annotation")
    default: Any = Field(default=None, description="Default value if any")
    kind: str = Field(description="Parameter kind (e.g. 'POSITIONAL_OR_KEYWORD')")

    class Config:
        arbitrary_types_allowed = True


class ActionInfo(PartialBaseModel):
    """Structured information about an action method"""

    name: str = Field(description="Action method name")
    callable: Callable = Field(description="The actual callable method")
    signature: Signature = Field(description="Method signature")
    signature_str: str = Field(description="Human-readable signature string")
    docstring: str = Field(default="", description="Method docstring")
    parameters: dict[str, ActionParameter] = Field(default_factory=dict, description="Parameter information")

    class Config:
        arbitrary_types_allowed = True


class ExecutionEvent(PartialBaseModel):
    """Structured event for observability"""

    timestamp: float = Field(default_factory=time.time)
    milestone: str
    iteration: int
    action: str
    duration_ms: float = Field(default=0)
    result_type: str = Field(default="")
    completed: bool = Field(default=False)
    error: str | None = Field(default=None)


class UserInteraction(PartialEscapeModel):
    """When agent needs user input to proceed"""

    question: str = Field(description="Question to ask the user")
    options: list[str] = Field(description="Available response options")
    context: str = Field(description="Context about why this question is being asked")
    default_choice: str = Field(default="", description="Default option if user doesn't respond")


class ActionError(Exception):
    """Structured error for action failures"""

    def __init__(self, message: str, is_critical: bool = False, recoverable: bool = True):
        super().__init__(message)
        self.is_critical = is_critical
        self.recoverable = recoverable

# ===== Core Data Structures =====

class SpecFragment(PartialBaseModel):
    """Base class for all Lumen specification fragments"""

    type: str = Field(default="", description="Type of spec fragment")
    content: dict[str, Any] = Field(default_factory=dict, description="The actual spec content")

    def __str__(self) -> str:
        """Default string representation"""
        return f"{self.type}: {str(self.content)[:100]}..."

    @classmethod
    def create(cls, fragment_type: str, content: dict[str, Any]) -> "SpecFragment":
        """Factory method to create appropriate fragment subclass"""
        # Import SQL fragments dynamically to avoid circular imports
        from ..sql.models import (
            QueryExecutionFragment, SchemaDiscoveryFragment,
            SQLQueriesFragment,
        )

        fragment_classes = {
            "sql_queries": SQLQueriesFragment,
            "query_execution": QueryExecutionFragment,
            "schema_discovery": SchemaDiscoveryFragment,
            "error": ErrorFragment,
        }

        fragment_class = fragment_classes.get(fragment_type, SpecFragment)
        return fragment_class(type=fragment_type, content=content)


class ErrorFragment(SpecFragment):
    """Fragment for error information"""

    def __str__(self) -> str:
        error_type = self.content.get("error_type", "unknown")
        message = self.content.get("message", "No message")
        return f"Error ({error_type}): {message}"


class ExecutionResult(PartialBaseModel):
    """Result from executing an agent step"""

    success: bool = Field(description="Whether execution was successful")
    output: SpecFragment = Field(default_factory=SpecFragment, description="Generated spec fragment")
    error: str = Field(default="", description="Error message if failed")


class ExecutionContext(PartialBaseModel):
    """Production execution context with user control and memory management"""

    goal: str = Field(description="User's original request")
    roadmap: Roadmap = Field(default_factory=Roadmap, description="Strategic plan with milestones")
    current_milestone: str = Field(default="", description="Which milestone is currently being worked on")
    fragments: dict[str, SpecFragment] = Field(default_factory=dict, description="Recent execution results (last 10)")
    fragment_summaries: dict[str, str] = Field(default_factory=dict, description="Summaries of older results")
    completed_milestones: list[str] = Field(default_factory=list, description="Milestones that have been finished")
    iteration: int = Field(default=0, description="Current iteration number within the milestone")

    # User Control
    should_stop: bool = Field(default=False, description="User-requested early termination")
    pause_requested: bool = Field(default=False, description="User requested pause")

    # Production Features
    start_time: float = Field(default_factory=time.time, description="Execution start timestamp")
    error_count: int = Field(default=0, description="Number of errors encountered")

    class Config:
        arbitrary_types_allowed = True

    def add_fragment(self, key: str, fragment: SpecFragment):
        """Add fragment with memory management"""
        # Keep only last 10 fragments in detail for memory efficiency
        if len(self.fragments) >= 10:
            oldest_key = min(self.fragments.keys())
            self.fragment_summaries[oldest_key] = self._summarize_fragment(self.fragments[oldest_key])
            del self.fragments[oldest_key]
        self.fragments[key] = fragment

    def _summarize_fragment(self, fragment: SpecFragment) -> str:
        """Create summary of fragment for memory efficiency"""
        return str(fragment)
