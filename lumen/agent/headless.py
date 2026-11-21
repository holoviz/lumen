"""Run the agent on a dashboard with nobody looking at it: one source node, a prompt per turn."""
from __future__ import annotations

import asyncio
import os
import re

from collections.abc import Callable
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import Any

from panel_agents import events as ev
from panel_agents.adapters.pydantic_ai import AgentRunner, ModelTraceRecorder
from panel_flowdash import FlowDash
from panel_flowdash.agent import (
    DashboardOps, FlowDashDeps, approval_policy, dashboard_to_yaml,
)
from pydantic_ai.messages import (
    ModelMessage, ModelResponse, TextPart, ToolCallPart,
)
from pydantic_ai.models.function import DeltaToolCall, FunctionModel
from pydantic_ai.models.wrapper import WrapperModel

from ..flowdash import (
    COMPONENTS, SOURCE_ID, SQL_ID, bind_source, unbind_source,
)
from ..sources.base import Source
from . import _graph
from .assistant import create_agent

_GENERATED_NAME = re.compile(r"^[A-Za-z]*Source\d+$")

# Components only the host places or configures; the agent's attempts need approval.
SENSITIVE_COMPONENTS = (SOURCE_ID,)

DENIED = (
    "Refused: source nodes are placed and configured by the app, not by the assistant. "
    "Use the source nodes already on the dashboard."
)


def source_node_name(source: Source) -> str:
    """The source's name as a node name, or ``data`` for a generated one like ``DuckDBSource00012``."""
    name = re.sub(r"\W+", "_", source.name).strip("_")
    if not name or _GENERATED_NAME.match(source.name) or not re.match(r"[A-Za-z_]", name):
        return "data"
    return name.replace("__", "_")


def streamable(model: Any) -> Any:
    """
    A ``FunctionModel`` that also streams: its function's response, part by
    part. ``AgentRunner`` streams every request, which a ``FunctionModel``
    without a ``stream_function`` refuses.
    """
    if not isinstance(model, FunctionModel) or model.stream_function is not None or model.function is None:
        return model
    function = model.function

    async def stream(messages, info):
        response = function(messages, info)
        if hasattr(response, "__await__"):
            response = await response
        for index, part in enumerate(response.parts):
            if isinstance(part, TextPart):
                yield part.content
            elif isinstance(part, ToolCallPart):
                yield {index: DeltaToolCall(name=part.tool_name, json_args=part.args_as_json_str(),
                                            tool_call_id=part.tool_call_id)}

    return FunctionModel(function, stream_function=stream, model_name=model.model_name)


class Meter(WrapperModel):
    """Keeps every response of the wrapped model, including those of runs that failed."""

    def __init__(self, wrapped: Any):
        super().__init__(wrapped)
        self.responses: list[ModelResponse] = []

    async def request(self, *args, **kwargs) -> ModelResponse:
        response = await super().request(*args, **kwargs)
        self.responses.append(response)
        return response

    @asynccontextmanager
    async def request_stream(self, *args, **kwargs):
        async with super().request_stream(*args, **kwargs) as stream:
            try:
                yield stream
            finally:
                self.responses.append(stream.get())


@dataclass
class TurnResult:
    """What one turn did: its final text, the dashboard after it and the nodes it added or changed."""

    prompt: str
    output: str | None
    graph: str
    nodes: list[str]
    changed: list[str] = field(default_factory=list)
    """Nodes that existed before the turn and whose config it changed."""
    messages: list[ModelMessage] = field(default_factory=list)
    responses: list[ModelResponse] = field(default_factory=list)
    """Every model response of the turn, also those of a failed run, which ``messages`` leaves out."""
    tool_results: dict[str, str] = field(default_factory=dict)
    """What each tool call of the turn returned, by tool call id, before the history was shortened."""
    error: str | None = None


class AgentSession:
    """
    A FlowDash editor with one source node bound to ``source``, and the agent
    working on it across the turns of one conversation.

    Parameters
    ----------
    source
        The Lumen source the source node is bound to.
    model
        The pydantic-ai model of every run, e.g. a ``Cassette`` to play a recording back.
    agent
        The agent; defaults to ``create_agent(model, instructions=instructions)``.
    instructions
        Extra instructions for the default agent.
    model_settings
        Model settings for the default agent, by default ``model_settings_for``
        the model of each request.
    trace
        A path the model trace of the conversation is appended to.
    approve
        Decides the tool calls FlowDash's approval policy holds back, e.g.
        placing or configuring a source node, from the tool name and
        arguments. Nobody is there to approve them headless, so by default
        they are refused.
    """

    def __init__(self, source: Source, model: Any, *, agent: Any = None, instructions: str = "",
                 model_settings: dict[str, Any] | None = None,
                 trace: str | os.PathLike | None = None,
                 approve: Callable[[str, dict[str, Any]], bool] | None = None):
        self.source = source
        self.model = Meter(streamable(model))
        self.editor = FlowDash(components=COMPONENTS)
        self.ops = DashboardOps(self.editor)
        self.source_node = source_node_name(source)
        bind_source(self.editor, source, self.source_node)
        self.ops.add_node(SOURCE_ID, {"ref": self.source_node}, name=self.source_node)
        if agent is None:
            agent = create_agent(self.model, instructions=instructions or None, model_settings=model_settings)
        self.agent = agent
        self.deps = FlowDashDeps(self.ops)
        policy = approval_policy(sensitive_components=SENSITIVE_COMPONENTS, ops=self.ops)
        self.runner = AgentRunner(self.agent, deps=self.deps, run_kwargs={"model": self.model}, approval_policy=policy)
        self.approve = approve
        self._results: dict[str, str] = {}
        self._unsubscribe = self.runner.state.subscribe(self._on_event)
        self.recorder = ModelTraceRecorder(self.runner, trace) if trace is not None else None

    def _on_event(self, event: ev.Event):
        if isinstance(event, ev.ToolCallResult):
            self._results[event.tool_call_id] = event.content if isinstance(event.content, str) else str(event.content)
        if isinstance(event, ev.ApprovalRequested):
            asyncio.get_running_loop().call_soon(self._decide, event)

    def _decide(self, event: ev.ApprovalRequested):
        approved = bool(self.approve and self.approve(event.tool_name, dict(event.args)))
        self.runner.state.resolve_approval(event.tool_call_id, approved, reason=None if approved else DENIED)

    def add_sql(self, name: str, sql: str) -> str:
        """Place a SQL node on the source node without the agent, e.g. a query a case starts from."""
        self.ops.add_node(SQL_ID, {"sql": sql, "table": name}, name=name)
        self.ops.connect(f"{self.source_node}.source", f"{name}.source")
        return name

    def _configs(self) -> dict[str, dict[str, Any]]:
        return {name: _graph.config_of(self.ops, name) for name in _graph.nodes(self.ops)}

    def graph(self) -> str:
        return dashboard_to_yaml(self.editor)

    async def ask(self, prompt: str) -> TurnResult:
        """Run the agent on ``prompt``, continuing the conversation, and report what the turn did."""
        before = self._configs()
        self._results = {}
        history = len(self.runner.messages)
        metered = len(self.model.responses)
        self.deps.start_request()
        try:
            output = await self.runner.run(prompt)
        finally:
            self.deps.end_request()
        error = self.runner.error
        after = self._configs()
        added = [name for name in after if name not in before]
        changed = [name for name in after if name in before and after[name] != before[name]]
        return TurnResult(
            prompt=prompt,
            output=output if isinstance(output, str) else None,
            graph=self.graph(),
            nodes=added,
            changed=changed,
            messages=list(self.runner.messages[history:]),
            responses=self.model.responses[metered:],
            tool_results=dict(self._results),
            error=f"{type(error).__name__}: {error}" if error is not None else None,
        )

    def close(self):
        if self.recorder is not None:
            self.recorder.close()
        self._unsubscribe()
        self.runner.close()
        unbind_source(self.editor, self.source_node)
        self.editor.dispose()
