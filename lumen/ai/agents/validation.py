from typing import Any, NotRequired

import param

from panel_material_ui import Button
from pydantic import Field, computed_field

from ..config import PROMPTS_DIR
from ..context import ContextModel, TContext, input_dependency_keys
from ..llm import Message
from ..models import BaseModel
from ..utils import content_to_text, log_debug
from .base import Agent


class RequirementCheck(BaseModel):
    """One explicit element of the request, checked against the outputs."""

    requirement: str = Field(description="One thing the user asked for, in their words.")
    evidence: str = Field(
        description=(
            "The exact spec key/value, SQL clause or response sentence that addresses it, "
            "and what it does; 'not found' if nothing does."
        )
    )
    satisfied: bool


class QueryCompletionValidation(BaseModel):
    """
    Validation of whether the executed plan answered the user's query.

    The model checks each explicit requirement against quoted evidence and the
    verdict is derived from those checks in code. Asked for one holistic
    yes/no, the model (with reasoning off) judged by the request's wording and
    the chart's title instead of the spec: it passed charts that contradicted
    the request and failed correct ones, while answering the same questions
    correctly when asked about each property directly.
    """

    checks: list[RequirementCheck] = Field(
        default_factory=list,
        description="One check per explicit requirement. Empty when the request names none.",
    )

    chain_of_thought: str = Field(default="", description="One sentence summing up the checks.")

    suggestions: list[str] = Field(
        default_factory=list,
        description="Only for unsatisfied checks: steps that would satisfy them."
    )

    @computed_field
    @property
    def missing_elements(self) -> list[str]:
        return [f"{c.requirement}: {c.evidence}" for c in self.checks if not c.satisfied]

    @computed_field
    @property
    def correct(self) -> bool:
        return all(c.satisfied for c in self.checks)


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
            for key in ("chat", "sql", "data", "view", "listing"):
                if key in context and key not in produced:
                    if key in required_keys:
                        continue
                    previous_keys.add(key)
        ctx["previous_keys"] = previous_keys
        if context.get("view"):
            ctx["view_spec"] = self._compact_view(context["view"])
        return ctx

    @staticmethod
    def _compact_view(view: Any) -> Any:
        """
        Keep the parts of a view spec that decide whether it answers the
        request (marks, encodings, transforms, titles). The pipeline repeats
        the SQL already shown, and the theme config (a 20-colour palette,
        number formats) is boilerplate: both cost tokens and invite the model
        to second-guess choices the user never asked about.
        """
        if not isinstance(view, dict):
            return view
        view = {k: v for k, v in view.items() if k not in ("sizing_mode", "min_height")}
        pipeline = view.get("pipeline")
        if isinstance(pipeline, dict):
            view["pipeline"] = {"table": pipeline.get("table")}
        spec = view.get("spec")
        if isinstance(spec, dict):
            view["spec"] = {k: v for k, v in spec.items() if k not in ("config", "$schema")}
        return view

    @staticmethod
    def _without_roadmap(messages: list[Message]) -> list[Message]:
        """
        Drop the plan roadmap the coordinator appends to the history. It
        describes each step as done ("Create a horizontal bar chart"), and the
        model takes that as evidence instead of reading the spec, so it passed
        charts whose spec contradicted the request.
        """
        return [
            msg for msg in messages
            if not (msg.get("role") == "system" and str(msg.get("content", "")).lstrip().startswith("<roadmap>"))
        ]

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

        try:
            result = await self._invoke_prompt("main", self._without_roadmap(messages), context)
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
            )
            return [result], {"validation_result": result}

        if result.correct:
            return [result], {"validation_result": result}

        response_parts = [f"**Query Validation: ✗ Incomplete** - {result.chain_of_thought}"]
        response_parts.append("**Missing Elements:**\n" + "\n".join(f"- {m}" for m in result.missing_elements))
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
