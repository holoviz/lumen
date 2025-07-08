"""
Lumen AI 2.0 - Core Response Models

Note: reasoning always comes before
the bools, actions, etc to give the LLM time to reflect
and build up context!
"""

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


# ===== Action System =====


class ActionParameter(PartialBaseModel):
    """Information about an action parameter"""

    name: str
    annotation: Any
    default: Any = Field(default=None, description="Default value for parameter. If None, parameter is required")

    @property
    def required(self) -> bool:
        """Whether this parameter is required (has no default value)"""
        return self.default is None

    class Config:
        arbitrary_types_allowed = True


class ActionInfo(PartialBaseModel):
    """Structured information about an action method"""

    name: str
    callable: Callable
    signature: Signature
    signature_str: str = Field(description="Human-readable signature string")
    docstring: str = ""
    parameters: dict[str, ActionParameter] = Field(default_factory=dict)

    class Config:
        arbitrary_types_allowed = True


class Action(PartialBaseModel):
    """Represents a single action to be executed"""

    name: str
    parameters: dict[str, Any] = Field(default_factory=dict, description="Parameters for the action")


# ===== Milestone Execution =====


class MilestoneStep(PartialBaseModel):
    """Represents a single step in milestone execution"""

    reasoning: str
    actions: list[Action]
    outputs: dict[str, Any] = Field(default_factory=dict)

    class Config:
        arbitrary_types_allowed = True


# ===== Strategic Planning =====


class Milestone(PartialBaseModel):
    """Represents a single milestone in the roadmap"""

    goal: str
    skipped: bool = Field(default=False)
    completed: bool = Field(default=False)


class Roadmap(PartialBaseModel):
    """Represents the strategic roadmap for achieving a goal"""

    reasoning: str
    milestones: list[Milestone]


class Checkpoint(PartialEscapeModel):
    """Combined evaluation and checkpoint creation"""

    summary: str = Field(description="If milestone is complete: comprehensive summary for strategic planning. If incomplete: leave empty.")

    enables_goal_completion: bool = Field(default=False, description="Whether this checkpoint enables completing the overall goal")
