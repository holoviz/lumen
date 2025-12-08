from typing import Any, NotRequired

import param

from ...pipeline import Pipeline
from ...sources.base import Source
from ..config import PROMPTS_DIR
from ..context import ContextModel, TContext
from ..llm import Message
from .chat import ChatAgent


class AnalystInputs(ContextModel):

    data: NotRequired[Any]

    source: Source

    pipeline: Pipeline

    sql: NotRequired[str]


class AnalystAgent(ChatAgent):
    conditions = param.List(
        default=[
            "Use for interpreting and analyzing results from executed queries",
            "NOT for initial data queries, data exploration, or technical programming questions",
        ]
    )

    purpose = param.String(
        default="""
        Responsible for analyzing results and providing clear, concise, and actionable insights.
        Focuses on breaking down complex data findings into understandable points for
        high-level decision-making. Emphasizes detailed interpretation, trends, and
        relationships within the data, while avoiding general overviews or
        superficial descriptions.""")

    prompts = param.Dict(
        default={
            "main": {"template": PROMPTS_DIR / "AnalystAgent" / "main.jinja2", "tools": []},
        }
    )

    input_schema = AnalystInputs

    async def respond(
        self,
        messages: list[Message],
        context: TContext,
        step_title: str | None = None,
    ) -> tuple[list[Any], TContext]:
        messages, out_context = await super().respond(messages, context, step_title=step_title)
        if len(context.get("data", [])) == 0 and context.get("sql"):
            context["sql"] = f"{context['sql']}\n-- No data was returned from the query."
        return messages, out_context
