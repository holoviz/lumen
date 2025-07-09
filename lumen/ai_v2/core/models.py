"""
Lumen AI 2.0 - Core Response Models

Note: reasoning always comes before
the bools, actions, etc to give the LLM time to reflect
and build up context! Also LLM generated models cannot accept dict[str, Any];
create a PartialBaseModel subclass instead.
"""

from collections.abc import Callable
from inspect import Signature
from typing import Any

from instructor.dsl.partial import PartialLiteralMixin
from pydantic import BaseModel, Field, field_validator
from pydantic.json_schema import SkipJsonSchema

# ===== Base Models =====


class PartialBaseModel(BaseModel, PartialLiteralMixin):
    """Base model that supports streaming through async generators"""


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

    def __str__(self) -> str:
        """String representation of the parameter"""
        type_str = getattr(self.annotation, "__name__", str(self.annotation)) if self.annotation else "Any"
        if self.required:
            return f"{self.name}: {type_str}"
        else:
            return f"{self.name}: {type_str} = {self.default}"

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

    def __str__(self) -> str:
        """String representation of the action for prompts"""
        description = self.docstring.split(".")[0] if self.docstring else "No description"
        return f"{self.name}({self.signature_str}): {description}"


class Action(PartialBaseModel):
    """Represents a single action to be executed"""

    name: str
    parameters: dict[str, Any] = Field(default_factory=dict, description="Parameters for the action")


# ===== Milestone Execution =====


class ActionResult(PartialBaseModel):
    """Result of executing a single action"""

    action_name: str
    output: Any = Field(description="The output returned by the action")

    class Config:
        arbitrary_types_allowed = True

    def __str__(self) -> str:
        """String representation of the action result"""
        if not self.output:
            return "No results returned"

        # Use the model's __str__ method if it exists and is not the default object one
        if hasattr(self.output, "__str__") and self.output.__class__.__str__ is not object.__str__:
            summary = str(self.output)
            # Truncate very long summaries for readability
            if len(summary) > 1000:
                summary = summary[:997] + "..."
            return summary

        # Fallback to basic info
        return f"Completed {self.action_name} successfully"


# ===== Strategic Planning =====


class Milestone(PartialBaseModel):
    """Represents a single milestone in the roadmap"""

    goal: str
    skipped: bool = Field(default=False)

    # Set programmatically after execution
    completed: SkipJsonSchema[bool] = False

    def __str__(self) -> str:
        """String representation of the milestone"""
        status = "✅ Completed" if self.completed else "⏭️ Skipped" if self.skipped else "📋 Pending"
        return f"{status}: {self.goal}"


class Roadmap(PartialBaseModel):
    """Represents the strategic roadmap for achieving a goal"""

    reasoning: str
    milestones: list[Milestone]

    # Metadata set after generation
    iteration: SkipJsonSchema[int] = 0
    completed: SkipJsonSchema[int] = 0
    goal: SkipJsonSchema[str] = ""


class Artifact(PartialBaseModel):
    """Represents a reusable artifact from milestone execution"""

    description: str = Field(default="", description="Optional description of what this artifact represents")
    key: str = Field(description="Unique identifier for this artifact (e.g., 'customer_table', 'aggregation_query')")
    value: str = Field(description="The artifact content as a string")

    def __str__(self) -> str:
        """String representation of the artifact"""
        if self.description:
            return f"{self.key}: {self.value} ({self.description})"
        else:
            return f"{self.key}: {self.value}"


class MissingDeliverablesError(Exception):
    """Raised when a checkpoint has missing deliverables that need to be addressed before proceeding"""

    def __init__(self, missing_parts: list[str]):
        super().__init__(f"Checkpoint has these missing deliverables:\n{', '.join(missing_parts)}")
        self.missing_parts = missing_parts


class Checkpoint(PartialBaseModel):
    """Combined evaluation and checkpoint creation"""

    summary: str = Field(
        description=(
            "Comprehensive summary of what was accomplished and what outputs/artifacts are now available "
            "for future milestones to build upon. Focus on concrete deliverables and usable results."
        )
    )
    artifacts: list[Artifact] = Field(default_factory=list, description="Goal-relevant artifacts extracted from outputs for future milestones")

    overall_goal_result: str = Field(
        default="", description="If the overall goal is now complete, provide a comprehensive summary of the final result based on the overall goal. Leave empty if more work is needed."
    )

    missing_deliverables: list[str] = Field(
        default_factory=list,
        description="""
        List specific parts of the milestone goal that remain incomplete or unfinished.
        If any part of the milestone goal was not fully accomplished, document what's missing.
        This halts execution until the missing work is completed.
        """,
    )

    # Not included in LLM generation - set programmatically
    milestone_goal: SkipJsonSchema[str] = ""

    @property
    def overall_goal_completed(self) -> bool:
        """Whether the overall goal has been completed"""
        return bool(self.overall_goal_result.strip())

    @field_validator("missing_deliverables", mode="before")
    @classmethod
    def validate_missing_deliverables(cls, v: list[str]) -> list[str]:
        if not v or v[0] is None:
            return v
        raise MissingDeliverablesError(v)


class Checkpoints(PartialBaseModel):
    """Collection of checkpoints with context building utilities"""

    checkpoints: list[Checkpoint] = Field(default_factory=list)

    def __len__(self) -> int:
        """Return number of checkpoints"""
        return len(self.checkpoints)

    def __iter__(self):
        """Allow iteration over checkpoints"""
        return iter(self.checkpoints)

    def __getitem__(self, index):
        """Allow indexing into checkpoints"""
        return self.checkpoints[index]

    def append(self, checkpoint: Checkpoint) -> None:
        """Add a checkpoint to the collection"""
        self.checkpoints.append(checkpoint)

    def serialize(self) -> list[dict[str, str]]:
        """Build a list of messages"""
        if not self.checkpoints:
            return []

        messages = []
        for checkpoint in self.checkpoints:
            user_message = {"role": "user", "content": checkpoint.milestone_goal}
            assistant_content = [checkpoint.summary]
            if checkpoint.artifacts:
                assistant_content.append("Available artifacts:")
                for artifact in checkpoint.artifacts:
                    assistant_content.append(f" - {artifact}")
            assistant_content = "\n".join(assistant_content)
            assistant_message = {"role": "assistant", "content": assistant_content}
            messages.extend([user_message, assistant_message])
        return messages


class ActionsInfo(PartialBaseModel):
    """Collection of action information with utilities for prompt generation"""

    actions: dict[str, ActionInfo] = Field(default_factory=dict)

    def __len__(self) -> int:
        """Return number of actions"""
        return len(self.actions)

    def __iter__(self):
        """Allow iteration over action items"""
        return iter(self.actions.items())

    def __getitem__(self, key: str) -> ActionInfo:
        """Allow indexing into actions"""
        return self.actions[key]

    def __contains__(self, key: str) -> bool:
        """Check if action exists"""
        return key in self.actions

    def items(self):
        """Return action items"""
        return self.actions.items()

    def get(self, key: str, default=None):
        """Get action with default"""
        return self.actions.get(key, default)

    def __str__(self) -> str:
        """
        String representation of all actions for prompt generation.
        """
        if not self.actions:
            return "No actions available"

        parts = []
        for action_name, action_info in self.actions.items():
            parts.append(f"`{action_name}`: {action_info}")
        return "\n- ".join(parts)


class GoalResult(PartialBaseModel):
    """Final result of achieving a goal"""

    goal: str = Field(description="The original goal that was achieved")
    success: bool = Field(description="Whether the goal was successfully achieved")
    summary: str = Field(description="Comprehensive summary of what was accomplished")

    # Execution metadata
    checkpoints: Checkpoints
    duration: float = 0.0

    class Config:
        arbitrary_types_allowed = True
