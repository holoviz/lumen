
from typing import Any, NotRequired

import param

from ...pipeline import Pipeline
from ...sources.base import Source
from ..config import PROMPTS_DIR
from ..context import ContextModel, TContext
from ..llm import Message
from ..schemas import get_metaset
from .base import Agent


class ChatAgent(Agent):
    """
    ChatAgent provides general information about available data
    and other topics  to the user.
    """

    conditions = param.List(
        default=[
            "Use for high-level data information or general conversation",
            "Use for technical questions about programming, functions, methods, libraries, APIs, software tools, or 'how to' code usage",
            "NOT for data-specific questions that require querying data",
        ]
    )

    purpose = param.String(
        default="""
        Engages in conversations about high-level data topics, programming questions,
        technical documentation, and general conversation. Handles questions about
        specific functions, methods, libraries, and provides coding guidance and
        technical explanations.""")

    prompts = param.Dict(
        default={
            "main": {
                "template": PROMPTS_DIR / "ChatAgent" / "main.jinja2",
            },
        }
    )

    async def respond(
        self,
        messages: list[Message],
        context: TContext,
        step_title: str | None = None,
    ) -> tuple[list[Any], dict[str, Any]]:
        prompt_context = {"tool_context": await self._use_tools("main", messages, context)}
        if "metaset" not in context and "source" in context and "table" in context:
            context["metaset"] = await get_metaset(
                [context["source"]], [context["table"]]
            )
        system_prompt = await self._render_prompt("main", messages, context, **prompt_context)
        return [await self._stream(messages, system_prompt)], {}


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
