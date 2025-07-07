"""
Lumen AI 2.0 - Core Response Models

Note: reasoning, chain_of_thought always comes before
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

# ===== Hierarchical Models =====

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
