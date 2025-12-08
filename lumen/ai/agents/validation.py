from typing import Any, NotRequired

import param

from panel_material_ui import Button
from pydantic import Field

from ..config import PROMPTS_DIR
from ..context import ContextModel, TContext
from ..llm import Message
from ..models import PartialBaseModel
from .base import Agent


class QueryCompletionValidation(PartialBaseModel):
    """Validation of whether the executed plan answered the user's query"""

    chain_of_thought: str = Field(
        description="Restate intent and results succinctly; then explain your reasoning as to why you will be answering yes or no.")

    missing_elements: list[str] = Field(
        default_factory=list,
        description="List of specific elements from the user's query that weren't addressed"
    )
    suggestions: list[str] = Field(
        default_factory=list,
        description="Suggestions for additional steps that could complete the query if not fully answered"
    )
    correct: bool = Field(description="True if query correctly solves user request, otherwise False.")


class ValidationInputs(ContextModel):

    data: NotRequired[Any]

    sql: NotRequired[str]

    view: NotRequired[Any]


class ValidationOutputs(ContextModel):

    validation_result: str


class ValidationAgent(Agent):
    """
    ValidationAgent focuses solely on validating whether the executed plan
    fully answered the user's original query. It identifies missing elements
    and suggests next steps when validation fails.
    """

    conditions = param.List(
        default=[
            "Use to validate whether executed plans fully answered user queries",
            "Use to identify missing elements from the original user request",
            "NOT for data analysis, pattern identification, or technical programming questions",
        ]
    )

    purpose = param.String(
        default="""
        Validates whether executed plans fully answered the user's original query.
        Identifies missing elements, assesses completeness, and suggests next steps
        when validation fails. Acts as a quality gate for plan execution."""
    )

    prompts = param.Dict(
        default={
            "main": {"template": PROMPTS_DIR / "ValidationAgent" / "main.jinja2", "response_model": QueryCompletionValidation, "tools": []},
        }
    )

    input_schema = ValidationInputs

    output_schema = ValidationOutputs

    async def respond(
        self,
        messages: list[Message],
        context: TContext,
        step_title: str | None = None,
    ) -> tuple[list[Any], ValidationOutputs]:
        interface = self.interface
        def on_click(event):
            if messages:
                user_messages = [msg for msg in reversed(messages) if msg.get("role") == "user"]
                original_query = user_messages[0].get("content", "").split("-- For context...")[0]
            suggestions_list = '\n- '.join(result.suggestions)
            interface.send(f"Follow these suggestions to fulfill the original intent {original_query}\n\n{suggestions_list}")

        executed_steps = None
        if "plan" in context:
            executed_steps = [
                f"{step[0].__class__.__name__}: {step.instruction}" for step in context["plan"]
            ]

        system_prompt = await self._render_prompt("main", messages, context, executed_steps=executed_steps)
        model_spec = self.prompts["main"].get("llm_spec", self.llm_spec_key)

        result = await self.llm.invoke(
            messages=messages,
            system=system_prompt,
            model_spec=model_spec,
            response_model=QueryCompletionValidation,
        )
        response_parts = []
        if result.correct:
            return [result], {"validation_result": result}

        response_parts.append(f"**Query Validation: ✗ Incomplete** - {result.chain_of_thought}")
        if result.missing_elements:
            response_parts.append(f"**Missing Elements:** {', '.join(result.missing_elements)}")
        if result.suggestions:
            response_parts.append("**Suggested Next Steps:**")
            for i, suggestion in enumerate(result.suggestions, 1):
                response_parts.append(f"{i}. {suggestion}")

        button = Button(name="Rerun", on_click=on_click)
        footer_objects = [button]
        formatted_response = "\n\n".join(response_parts)
        interface.stream(formatted_response, user=self.user, max_width=self._max_width, footer_objects=footer_objects)
        return [result], {"validation_result": result}
