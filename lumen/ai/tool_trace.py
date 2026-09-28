"""Scoped collection of LLM model and tool calls."""

from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import Any

from .usage import Usage


@dataclass
class ToolCall:
    name: str
    arguments: dict[str, Any]
    result: str


@dataclass
class ModelCall:
    model: str
    messages: list[Any]
    response: Any
    duration: float
    error: str | None = None
    usage: list[Usage] = field(default_factory=list)


TraceEvent = ToolCall | ModelCall


_scopes: ContextVar[tuple[tuple[object, list[TraceEvent]], ...]] = ContextVar(
    "lumen_llm_tool_call_scopes", default=(),
)


@contextmanager
def capture_trace(llm: object):
    calls: list[TraceEvent] = []
    token = _scopes.set((*_scopes.get(), (llm, calls)))
    try:
        yield calls
    finally:
        _scopes.reset(token)


def record_trace(llm: object, call: TraceEvent):
    for owner, calls in _scopes.get():
        if owner is llm:
            calls.append(call)


def is_tracing(llm: object) -> bool:
    return any(owner is llm for owner, _ in _scopes.get())
