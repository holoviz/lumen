from typing import Any

import param

from ..config import PROMPTS_DIR
from ..context import TContext
from ..llm import Message
from ..schemas import get_metaset
from .base import Agent


class ChatAgent(Agent):
    """
    ChatAgent provides general information about available data
    and other topics to the user. When data is available, it acts
    as an analyst providing insights and interpretations.
    """

    conditions = param.List(
        default=[
            "Use for general conversation that doesn't require fetching or querying data",
            "Use for technical questions about programming, functions, methods, libraries, or APIs",
            "Use when interpreting existing results - NOT for generating new data or queries",
            "NOT when user asks to 'show', 'get', 'fetch', 'query', 'find', or 'display' data",
            "NOT for initial data retrieval - only for explaining data that's already been obtained",
        ]
    )

    purpose = param.String(
        default="""
        Provides conversational assistance and interprets existing results.
        Handles general questions, technical documentation, and programming help.
        When data has been retrieved, explains findings in accessible terms.""")

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

        # Handle empty data case for analyst mode
        if len(context.get("data", [])) == 0 and context.get("sql"):
            context["sql"] = f"{context['sql']}\n-- No data was returned from the query."

        system_prompt = await self._render_prompt("main", messages, context, **prompt_context)
        return [await self._stream(messages, system_prompt)], {}
