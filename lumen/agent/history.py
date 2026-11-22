"""
Keeping the conversation short: data lookups the agent acted on in earlier
turns are not sent again.

Shortening rewrites earlier messages, which ends the provider's cached prefix
there, so it only happens when a new user turn starts: within a run every
request extends the previous one and stays cacheable. A model trace records
the rewrite as a rewind followed by the shortened history, so after a rewind
the trace pairs later responses with the stubbed results; what was actually
sent before is in the records before the rewind (``Cassette.from_trace``
replays from those).
"""
from __future__ import annotations

import dataclasses

from pydantic_ai.messages import (
    ModelMessage, ModelRequest, ToolReturnPart, UserPromptPart,
)

# Tools whose results only inform a query, and the tools that place or change one.
LOOKUP_TOOLS = frozenset({"list_tables", "describe_table", "distinct_values", "run_sql", "check_data"})
PLACING_TOOLS = frozenset({"add_sql", "add_chart", "update_chart"})

SUPERSEDED = "[Shown before a node was placed; call the tool again if you need it.]"


def _placed(part) -> bool:
    return (
        isinstance(part, ToolReturnPart) and part.tool_name in PLACING_TOOLS
        and isinstance(part.content, str) and part.content.startswith(("Placed", "Changed"))
    )


def stub_superseded_lookups(messages: list[ModelMessage]) -> list[ModelMessage]:
    """
    At the first request of a user turn, replace the results of lookup tools
    that came before the last node the agent placed in earlier turns with a
    short stub. The agent has acted on them, and they would otherwise be
    sent with every later request.
    """
    if not messages or not isinstance(messages[-1], ModelRequest) or not any(
        isinstance(part, UserPromptPart) for part in messages[-1].parts
    ):
        return messages
    last = max((i for i, m in enumerate(messages[:-1]) if isinstance(m, ModelRequest) and any(map(_placed, m.parts))), default=None)
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
