from typing import Annotated, Literal

from instructor.dsl.partial import PartialLiteralMixin
from pydantic import BaseModel, Field, model_validator

from .config import MissingContextError


class PartialBaseModel(BaseModel, PartialLiteralMixin):
    ...


class EscapeBaseModel(PartialBaseModel):

    insufficient_context_reason: str = Field(
        description="If lacking sufficient context, explain why; else use ''. Do not base off the user query; only from the data context provided.",
        examples=[
            "A timeseries is requested but SQL only provides customer and order data; please include a time dimension",
            "The previous result is one aggregated value; try a different aggregation or more dimensions",
            ""
        ]
    )

    insufficient_context: bool = Field(
        description="True if lacking context, else False. If True, leave other fields empty.",
    )

    def model_post_init(self, __context):
        """
        After model initialization, check if insufficient_context. If it is,
        raise a MissingContextError with the provided explanation to stop further processing.
        """
        if self.insufficient_context:
            raise MissingContextError(self.insufficient_context_reason)


class ErrorDescription(BaseModel):
    """
    Represents a user-facing error explanation.

    Style rules:
    - Do not use first-person pronouns (“I”, “we”).
    - All outputs must be written in neutral, system-level language.

    Guidance:
    - If you think it's an internal error, indicate that the user should rerun.
    - Otherwise, ask the user to either replan OR clarify their questions.
    - You are given the error type, message, the user query, and the plan the planning agent came up with.
    """

    explanation: str = Field(
        description="A brief description of the error suitable for a non-technical user.",
        max_length=300
    )
class YesNo(BaseModel):

    yes: bool = Field(description="True if yes, otherwise False.")


class ThinkingYesNo(BaseModel):

    chain_of_thought: str = Field(
        description="Briefly explain your reasoning as to why you will be answering yes or no.")

    yes: bool = Field(description="True if yes, otherwise False.")


class InsertLine(BaseModel):
    op: Literal["insert"] = "insert"
    line_no: int = Field(ge=1, description=(
        "Insert BEFORE this 1-based line number. "
        "Use line_no == len(lines) to append at the end."
    ))
    line: str = Field(min_length=1, description="Content for the new line (must be non-empty).")

class ReplaceLine(BaseModel):
    op: Literal["replace"] = "replace"
    line_no: int = Field(ge=1, description="The 1-based line number to replace.")
    line: str = Field(description="The new content for the line (empty string is allowed).")

class DeleteLine(BaseModel):
    op: Literal["delete"] = "delete"
    line_no: int = Field(ge=1, description="The 1-based line number to delete.")

LineEdit = Annotated[
    InsertLine | ReplaceLine | DeleteLine,
    Field(discriminator="op")
]

class RetrySpec(BaseModel):
    """Represents a revision of text with its content and changes."""

    chain_of_thought: str = Field(description="In a sentence or two, explain the plan to revise the text based on the feedback provided.")
    edits: list[LineEdit] = Field(description="A list of line edits based on the chain_of_thought.")

    @model_validator(mode="after")
    def validate_indices_nonconflicting(self) -> "RetrySpec":
        # Disallow more than one delete/replace on the same original index.
        seen_replace_or_delete: set[int] = set()
        for e in self.edits:
            if e.op in ("replace", "delete"):
                if e.line_no in seen_replace_or_delete:
                    raise ValueError(
                        f"Multiple {e.op}/delete edits targeting the same line_no={e.line_no} "
                        "are ambiguous. Combine them or split into separate patches."
                    )
                seen_replace_or_delete.add(e.line_no)
        return self
