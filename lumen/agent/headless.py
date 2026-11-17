"""Run the agent on a dashboard with nobody looking at it: one source node, a prompt per turn."""
from __future__ import annotations

import os
import re

from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import Any

from panel_agents.adapters.pydantic_ai import AgentRunner, ModelTraceRecorder
from panel_flowdash import FlowDash
from panel_flowdash.agent import DashboardOps, FlowDashDeps, dashboard_to_yaml
from pydantic_ai.messages import (
    ModelMessage, ModelResponse, TextPart, ToolCallPart,
)
from pydantic_ai.models.function import DeltaToolCall, FunctionModel
from pydantic_ai.models.wrapper import WrapperModel

from ..flowdash import (
    COMPONENTS, SOURCE_ID, SQL_ID, bind_source, source_config, unbind_source,
)
from ..sources.base import Source
from . import _graph
from .assistant import create_agent

_GENERATED_NAME = re.compile(r"^[A-Za-z]*Source\d+$")


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
            yield stream
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
    trace
        A path the model trace of the conversation is appended to.
    """

    def __init__(self, source: Source, model: Any, *, agent: Any = None, instructions: str = "",
                 trace: str | os.PathLike | None = None):
        self.source = source
        self.model = Meter(streamable(model))
        self.editor = FlowDash(components=COMPONENTS)
        self.ops = DashboardOps(self.editor)
        self.source_node = source_node_name(source)
        self._ref = bind_source(source, self.source_node)
        self.ops.add_node(SOURCE_ID, source_config(source, self._ref), name=self.source_node)
        self.agent = agent if agent is not None else create_agent(self.model, instructions=instructions or None)
        self.deps = FlowDashDeps(self.ops)
        self.runner = AgentRunner(self.agent, deps=self.deps, run_kwargs={"model": self.model})
        self.recorder = ModelTraceRecorder(self.runner, trace) if trace is not None else None

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
            error=f"{type(error).__name__}: {error}" if error is not None else None,
        )

    def close(self):
        if self.recorder is not None:
            self.recorder.close()
        self.runner.close()
        unbind_source(self._ref)
        self.editor.dispose()
