from typing import Any, NotRequired

import param

from ..config import PROMPTS_DIR
from ..context import ContextModel, TContext
from ..llm import Message
from ..schemas import get_metaset
from ..utils import truncate_string
from .base import Agent


def format_document_sources(sources: list[dict[str, Any]]) -> str:
    """
    Render retrieved document chunks as a Markdown sources block.

    Each file appears once, best match first, followed by the start of the chunk
    it was found in. The excerpt is short because the answer stays the focus.
    """
    best: dict[str, dict[str, Any]] = {}
    for source in sources:
        filename = source.get("filename")
        if not filename:
            continue
        current = best.get(filename)
        if current is None or source.get("similarity", 0.0) > current.get("similarity", 0.0):
            best[filename] = source
    if not best:
        return ""
    ordered = sorted(best.values(), key=lambda source: source.get("similarity", 0.0), reverse=True)
    lines = [
        f"- `{source['filename']}` (similarity {source.get('similarity', 0.0):.2f}): "
        f"{truncate_string(source.get('text', ''), 300)}"
        for source in ordered
    ]
    return "\n\n---\n\n**Sources**\n\n" + "\n".join(lines)


class ChatOutputs(ContextModel):

    chat: NotRequired[str]


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
            "Use when user asks to 'explain', 'interpret', 'analyze', 'summarize', or 'comment on' existing data in context",
            "NOT when user asks to 'query', 'filter', 'calculate', 'aggregate', or 'transform' data",
            "NOT for creating new data transformations - only for explaining data that already exists",
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

    output_schema = ChatOutputs

    async def respond(
        self,
        messages: list[Message],
        context: TContext,
        step_title: str | None = None,
    ) -> tuple[list[Any], dict[str, Any]]:
        # Citations belong to this turn; a previous answer's sources must not
        # reappear when the new turn does not search the documents.
        context.pop("document_sources", None)
        prompt_context = {"tool_context": await self._use_tools("main", messages, context)}
        if "metaset" not in context and "source" in context and "table" in context:
            context["metaset"] = await get_metaset(
                [context["source"]], [context["table"]]
            )

        # Handle empty data case for analyst mode
        if len(context.get("data", [])) == 0 and context.get("sql"):
            context["sql"] = f"{context['sql']}\n-- No data was returned from the query."

        result = await self._stream(messages, context, **prompt_context)
        answer = result.object
        sources = context.pop("document_sources", None)
        if sources:
            result.object = f"{answer}{format_document_sources(sources)}"
        return [result], {"chat": answer}
