"""Keeping the conversation short: data lookups the agent has acted on are not sent again."""
from __future__ import annotations

import dataclasses

from pydantic_ai.messages import ModelMessage, ModelRequest, ToolReturnPart

# Tools whose results only inform a query, and the tools that place one.
LOOKUP_TOOLS = frozenset({"list_tables", "describe_table", "distinct_values", "run_sql", "check_data"})
PLACING_TOOLS = frozenset({"add_sql", "add_chart"})

SUPERSEDED = "[Shown before a node was placed; call the tool again if you need it.]"


def _placed(part) -> bool:
    return (
        isinstance(part, ToolReturnPart) and part.tool_name in PLACING_TOOLS
        and isinstance(part.content, str) and part.content.startswith("Placed")
    )


def stub_superseded_lookups(messages: list[ModelMessage]) -> list[ModelMessage]:
    """
    Replace the results of lookup tools that came before the last node the
    agent placed with a short stub. The agent has acted on them, and they
    would otherwise be sent with every later request.
    """
    last = max((i for i, m in enumerate(messages) if isinstance(m, ModelRequest) and any(map(_placed, m.parts))), default=None)
    if last is None:
        return messages
    trimmed = list(messages)
    for index, message in enumerate(messages[:last]):
        if not isinstance(message, ModelRequest):
            continue
        parts = [
            dataclasses.replace(part, content=SUPERSEDED)
            if isinstance(part, ToolReturnPart) and part.tool_name in LOOKUP_TOOLS and part.content != SUPERSEDED
            else part
            for part in message.parts
        ]
        if parts != list(message.parts):
            trimmed[index] = dataclasses.replace(message, parts=parts)
    return trimmed
