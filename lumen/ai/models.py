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
        description="In 1-2 sentences, explain your reasoning for yes or no.",
        examples=[
            "The query asks for top 10 hosts which requires extending the existing aggregation from LIMIT 1 to LIMIT 10.",
            "The data contains location and count columns needed for a bar chart visualization."
        ])

    yes: bool = Field(description="True if yes, otherwise False.")


class SearchReplace(PartialBaseModel):
    """A single search/replace edit operation."""

    old_str: str = Field(
        description=(
            "COPY-PASTE the exact text from the document. Must appear EXACTLY ONCE. "
            "Include 3-5 lines of context to ensure uniqueness. "
            "Preserve exact whitespace and indentation—do not reconstruct from memory."
        ),
    )
    new_str: str = Field(
        description=(
            "The replacement text. Use empty string to delete. "
            "Preserve the same indentation style as old_str."
        ),
    )


class SearchReplaceSpec(PartialBaseModel):
    """Revision using search/replace patterns (no line numbers needed)."""

    chain_of_thought: str = Field(
        description="In 2-3 sentences: (1) what changes to make, (2) identify the EXACT lines to copy from the document for old_str (quote them), (3) note indentation level.",
        examples=[
            "Adding tooltips to the first layer. The document has '- encoding:\n    x:' at 2-space indent inside layer. Will copy that block and insert tooltip before x.",
            "Formatting text labels. The document shows '    text:\n      field: Life Expectancy\n      type: quantitative' - will copy exactly and add format: ',.1f'."
        ]
    )
    edits: list[SearchReplace] = Field(
        description=(
            "List of search/replace operations. Each old_str must be unique in the document. "
        )
    )

    @model_validator(mode='after')
    def validate_unique_old_str(self):
        """Validate that old_str values are unique within edits."""
        old_strs = [edit.old_str for edit in self.edits]
        duplicates = [s for s in old_strs if old_strs.count(s) > 1]
        if duplicates:
            # Get unique duplicates
            unique_duplicates = list(set(duplicates))
            raise ValueError(f"Duplicate old_str: {unique_duplicates}")
        return self
