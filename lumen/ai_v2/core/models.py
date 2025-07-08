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


class PartialEscapeModel(PartialBaseModel):
    """Base model with escape hatch - set missing_parts to halt execution when you need more information"""

    missing_parts: list[str] = Field(
        default_factory=list,
        description="""
        If you need more information to proceed or the results are incomplete, describe what's missing here.
        This will halt execution with the message. Be specific about about what additional work is needed.
        """
    )

    @field_validator("missing_parts")
    def validate_missing_parts(cls, v: list[str]) -> list[str]:
        if not v:
            return v
        raise RuntimeError(f"Agent needs more information: {', '.join(v)}")


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


class ActionDecision(PartialBaseModel):
    """Decides which single action to take next in milestone execution"""

    reasoning: str = Field(description="Think through what needs to be done next and why this specific action is the right choice")
    action_name: str = Field(description="Name of the action to execute")


class ActionResult(PartialBaseModel):
    """Result of executing a single action"""

    action_name: str
    output: Any = Field(description="The output returned by the action")

    class Config:
        arbitrary_types_allowed = True


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


class Checkpoint(PartialEscapeModel):
    """Combined evaluation and checkpoint creation"""

    summary: str = Field(
        description=(
            "Comprehensive summary of what was accomplished and what outputs/artifacts are now available "
            "for future milestones to build upon. Focus on concrete deliverables and usable results."
        )
    )
    artifacts: list[Artifact] = Field(default_factory=list, description="Goal-relevant artifacts extracted from outputs for future milestones")
    overall_goal_result: str = Field(
        default="", description="If the overall goal is now complete, provide a comprehensive summary of the final result. Leave empty if more work is needed."
    )
    milestone_completed: bool = Field(
        default=True, description="Whether this milestone was actually completed successfully. Set to False if there were critical errors."
    )
    errors: list[str] = Field(
        default_factory=list, description="List of any errors encountered during this milestone that should be addressed."
    )

    # Not included in LLM generation - set programmatically
    milestone_goal: SkipJsonSchema[str] = ""

    @property
    def overall_goal_completed(self) -> bool:
        """Whether the overall goal has been completed"""
        return bool(self.overall_goal_result.strip())

    @property
    def has_errors(self) -> bool:
        """Whether this checkpoint has any errors"""
        return bool(self.errors)


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

    def build_context_summary(self) -> str:
        """Build a context summary string for prompts"""
        if not self.checkpoints:
            return ""

        context_parts = ["Previous completed work:"]
        for i, checkpoint in enumerate(self.checkpoints, 1):
            context_parts.append(f"{i}. {checkpoint.summary}")
            # Include artifacts if available
            if checkpoint.artifacts:
                context_parts.append("   Available artifacts:")
                for artifact in checkpoint.artifacts:
                    context_parts.append(f"   - {artifact}")
        return "\n".join(context_parts)

    def build_checkpoint_list(self) -> str:
        """Build a simple checkpoint list for roadmap planning"""
        if not self.checkpoints:
            return ""

        parts = []
        for checkpoint in self.checkpoints:
            parts.append(f"✅ {checkpoint.milestone_goal}: {checkpoint.summary}")
        return "\n".join(parts)

    def get_all_artifacts(self) -> list[Artifact]:
        """Get all artifacts from all checkpoints"""
        all_artifacts = []
        for checkpoint in self.checkpoints:
            all_artifacts.extend(checkpoint.artifacts)
        return all_artifacts


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

    def build_action_list(self, prefix: str = "-") -> str:
        """Build a formatted action list for prompts"""
        if not self.actions:
            return "No actions available"

        parts = []
        for action_name, action_info in self.actions.items():
            parts.append(f"{prefix} {action_info}")
        return "\n".join(parts)


class GoalResult(PartialBaseModel):
    """Final result of achieving a goal"""

    goal: str = Field(description="The original goal that was achieved")
    success: bool = Field(description="Whether the goal was successfully achieved")
    summary: str = Field(description="Comprehensive summary of what was accomplished")
    final_data: dict[str, Any] = Field(default_factory=dict, description="Key final outputs and results from achieving the goal")
    artifacts: list[Artifact] = Field(default_factory=list, description="All artifacts created during goal achievement")
    errors: list[str] = Field(default_factory=list, description="List of any errors encountered during goal execution")

    # Execution metadata
    checkpoints: list[Checkpoint] = Field(default_factory=list)
    total_milestones: int = 0
    iterations: int = 0
    duration: float = 0.0

    class Config:
        arbitrary_types_allowed = True

    @property
    def has_errors(self) -> bool:
        """Whether this goal result has any errors"""
        return bool(self.errors) or any(checkpoint.has_errors for checkpoint in self.checkpoints)
