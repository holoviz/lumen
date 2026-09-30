import json

from typing import Any, NotRequired

import param

from panel_material_ui import Button
from pydantic import Field

from ..config import PROMPTS_DIR
from ..context import ContextModel, TContext, input_dependency_keys
from ..decisions import Noul
from ..llm import Message
from ..models import BaseModel
from ..utils import content_to_text, log_debug, truncate_string
from .base import Agent

# Keeps the decision state small; the start of each output is enough to judge completeness.
VALIDATION_STATE_MAX_CHARS = 2000


class QueryCompletionValidation(BaseModel):
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

    chat: NotRequired[str]

    data: NotRequired[Any]

    listing: NotRequired[str]

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

    user = param.String(default="Validation")

    input_schema = ValidationInputs

    output_schema = ValidationOutputs

    async def _gather_prompt_context(self, prompt_name, messages, context, **kwargs):
        """
        The coordinator's context persists across plans, so keys like
        'chat' from a *previous* plan can leak into this plan's
        validation.  Rather than removing stale keys (which could lose
        useful context), we label them so the prompt can distinguish
        current-plan outputs from previous-plan leftovers.
        """
        ctx = await super()._gather_prompt_context(prompt_name, messages, context, **kwargs)

        plan = context.get("plan")
        previous_keys = set()
        if plan is not None:
            required_keys = set()
            for task in plan:
                if task.actor is not self:
                    required_keys |= input_dependency_keys(task.input_schema)
            produced = {k for task in plan for k in task.out_context}
            for key in ("chat", "sql", "view", "listing"):
                if key in context and key not in produced:
                    if key in required_keys:
                        continue
                    previous_keys.add(key)
        ctx["previous_keys"] = previous_keys
        return ctx

    async def _decide_complete(self, messages: list[Message], context: TContext) -> QueryCompletionValidation | None:
        """
        Skip the validation call when the decision model is confident the
        result is complete. An incomplete verdict still needs the LLM for the
        missing elements and suggestions, so only a confident "yes" is used.
        """
        if self.decision_model is None:
            return None
        content = next((m.get("content") for m in reversed(messages) if m.get("role") == "user"), None)
        if not isinstance(content, str) or not content:
            return None
        state = {"user_request": content}
        for key in ("sql", "data", "chat", "listing", "view"):
            value = context.get(key)
            if key == "view" and value:
                # A chart request is answered by the spec, not by the SQL or data.
                value = json.dumps(value, default=str)
            if isinstance(value, str) and value:
                state[key] = truncate_string(value, VALIDATION_STATE_MAX_CHARS)
        if len(state) == 1:
            return None
        outcome = (await self._decide("validation.complete", state, {"validation.complete": Noul(
            instructions="Does the executed result fully answer every part of the user's request?",
            criteria={
                "true": "Every requested value, breakdown, filter and output is present in the result.",
                "false": "Some requested part is missing, wrong in kind, or only partly answered.",
            },
        )}))["validation.complete"]
        if not (outcome.accepted and outcome.value is True):
            return None
        return QueryCompletionValidation(
            chain_of_thought="The decision model judged the result complete.", correct=True,
        )

    async def respond(
        self,
        messages: list[Message],
        context: TContext,
        step_title: str | None = None,
    ) -> tuple[list[Any], ValidationOutputs]:
        interface = self.interface
        def on_click(event):
            user_messages = [msg for msg in reversed(messages) if msg.get("role") == "user"]
            if not user_messages:
                return
            text_content = content_to_text(user_messages[0].get("content", ""))
            suggestions_list = '\n- '.join(result.suggestions)
            interface.send(f"Follow these suggestions to fulfill the original intent:\n\n> {text_content}\n\n{suggestions_list}")

        if (result := await self._decide_complete(messages, context)) is not None:
            return [result], {"validation_result": result}
        try:
            result = await self._invoke_prompt("main", messages, context)
        except Exception as e:
            # Validation is a quality gate, not part of the answer: the plan's
            # actual outputs were already produced/displayed. If the structured
            # validation call itself fails (e.g. the LLM won't return parseable
            # structured output after retries), degrade to "assume complete" so
            # a gate failure can never abort an otherwise-successful plan.
            log_debug(
                f"ValidationAgent could not run; treating plan as complete: "
                f"{type(e).__name__}: {e}"
            )
            result = QueryCompletionValidation(
                chain_of_thought=(
                    "Validation could not be completed due to an LLM error; "
                    "treating the executed plan as complete."
                ),
                correct=True,
            )
            return [result], {"validation_result": result}

        if result.correct:
            return [result], {"validation_result": result}

        response_parts = [f"**Query Validation: ✗ Incomplete** - {result.chain_of_thought}"]
        if result.missing_elements:
            response_parts.append(f"**Missing Elements:** {', '.join(result.missing_elements)}")
        if result.suggestions:
            response_parts.append("**Suggested Next Steps:**")
            for i, suggestion in enumerate(result.suggestions, 1):
                response_parts.append(f"{i}. {suggestion}")

        button = Button(icon="autorenew", label="Rerun", on_click=on_click)
        footer_objects = [button]
        formatted_response = "\n\n".join(response_parts)
        if interface is not None:
            interface.stream(formatted_response, user=self.user, max_width=self._max_width, footer_objects=footer_objects)
        return [result], {"validation_result": result}
