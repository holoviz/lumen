"""The agent as a subject of Lumen's eval harness, scored on what it leaves on the dashboard."""
from __future__ import annotations

import asyncio
import functools
import hashlib
import json
import os
import re

from collections.abc import Callable, Iterable
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, ClassVar

from pydantic_ai.messages import (
    ModelMessagesTypeAdapter, ModelRequest, ModelResponse, RetryPromptPart,
    TextPart, ToolCallPart, ToolReturnPart,
)
from pydantic_ai.models.function import AgentInfo, FunctionModel

from ..ai.evals.harness import (
    MAX_ANSWER_LENGTH, MAX_ROWS, MAX_TOOL_RESULT_LENGTH, OPENROUTER_ENDPOINT,
    PRICING_PER_MILLION, Inputs, Output, Turn, TurnArtifacts, Usage,
    register_subject,
)
from ..ai.tool_trace import ToolCall
from ..ai.usage import _rates
from ..flowdash import METRIC_ID, SQL_ID, VIEW_ID
from ..flowdash.metrics import table_sql
from ..pipeline import Pipeline
from ..sources.duckdb import DuckDBSource
from ..transforms.sql import MetricAggregate
from ..util import as_pandas
from . import _graph
from .assistant import MODEL_SETTINGS, OTHER_MODEL_SETTINGS, model_settings_for
from .headless import AgentSession, TurnResult

# Name of the SQL node a case's pre-seeded query is placed as.
SEEDED_QUERY = "result"

TRACES_ENV = "LUMEN_AGENT_TRACES"


def build_model(model: str, *, provider: str = "openai", api_key: str | None = None, api: str | None = None) -> Any:
    """
    A pydantic-ai model for the harness's model options: OpenAI, or
    OpenRouter's OpenAI-compatible endpoint. On OpenRouter's chat
    completions it is an ``OpenRouterModel``, as an ``openrouter:`` model
    picked in the app is, so it gets the same settings: the cache breakpoints
    an Anthropic model needs, and reasoning at its lowest effort on a model
    that cannot turn it off.
    """
    from pydantic_ai.models.openai import OpenAIChatModel, OpenAIResponsesModel
    from pydantic_ai.providers.openai import OpenAIProvider

    if provider == "openrouter" and (api or "chat_completions") == "chat_completions":
        from pydantic_ai.models.openrouter import OpenRouterModel
        from pydantic_ai.providers.openrouter import OpenRouterProvider

        return OpenRouterModel(model, provider=OpenRouterProvider(api_key=api_key))
    if provider == "openrouter":
        provider_ = OpenAIProvider(base_url=OPENROUTER_ENDPOINT, api_key=api_key)
        api = api or "chat_completions"
    elif provider == "openai":
        provider_ = OpenAIProvider(api_key=api_key)
        api = api or "responses"
    else:
        raise ValueError(f"Unknown provider {provider!r}")
    cls = OpenAIResponsesModel if api == "responses" else OpenAIChatModel
    return cls(model, provider=provider_)


def usage(responses: list[ModelResponse]) -> Usage | None:
    """Tokens and cost of ``responses``, priced like the Lumen subject's calls."""
    if not responses:
        return None
    inputs = sum(r.usage.input_tokens for r in responses)
    outputs = sum(r.usage.output_tokens for r in responses)
    cached = sum(r.usage.cache_read_tokens for r in responses)
    costs = []
    for r in responses:
        rates = _rates(PRICING_PER_MILLION, r.model_name or "")
        if rates is None:
            costs.append(None)
            continue
        u = r.usage
        costs.append(((u.input_tokens - u.cache_read_tokens) * rates["input"]
                      + u.cache_read_tokens * rates.get("cached", rates["input"])
                      + u.output_tokens * rates["output"]) / 1_000_000)
    return Usage(inputs, outputs, cached, sum(costs) if all(c is not None for c in costs) else None)


def _total(turns: list[Turn]) -> Usage | None:
    calls = [turn.usage for turn in turns if turn.usage is not None]
    if not calls:
        return None
    costs = [call.cost_usd for call in calls]
    return Usage(sum(c.input_tokens for c in calls), sum(c.output_tokens for c in calls),
                 sum(c.cached_tokens for c in calls), sum(costs) if all(c is not None for c in costs) else None)


def tool_calls(messages: list[Any], returned: dict[str, str] | None = None) -> list[ToolCall]:
    """
    Each tool call of ``messages`` with its arguments and result, as the Lumen
    subject records them. ``returned`` results by tool call id take precedence over the
    history, which may hold shortened results.
    """
    results = {
        part.tool_call_id: part for message in messages if isinstance(message, ModelRequest)
        for part in message.parts if isinstance(part, (ToolReturnPart, RetryPromptPart))
    }
    calls = []
    for message in messages:
        if not isinstance(message, ModelResponse):
            continue
        for part in message.parts:
            if isinstance(part, ToolCallPart):
                result = results.get(part.tool_call_id)
                content = result.model_response() if isinstance(result, RetryPromptPart) else (
                    result.model_response_str() if result is not None else "")
                content = (returned or {}).get(part.tool_call_id, content)
                calls.append(ToolCall(part.tool_name, part.args_as_dict(), content[:MAX_TOOL_RESULT_LENGTH]))
    return calls


def chart_summary(specs: list[dict[str, Any]]) -> tuple[list[str], list[dict[str, dict[str, Any]]]]:
    """Marks and encodings of Vega-Lite specs, in the shape the Lumen subject reports them."""
    marks, encodings = [], []
    for spec in specs:
        for layer in spec.get("layer", [spec]) if isinstance(spec, dict) else []:
            mark = layer.get("mark") if isinstance(layer, dict) else None
            if isinstance(mark, dict):
                mark = mark.get("type")
            if isinstance(mark, str):
                marks.append(mark)
            encoding = layer.get("encoding", {}) if isinstance(layer, dict) else {}
            encodings.append({
                channel: {key: value for key, value in definition.items() if key in ("field", "type", "aggregate", "timeUnit")}
                for channel, definition in encoding.items() if isinstance(definition, dict)
            })
    return marks, encodings


def result_rows(frame: Any) -> tuple[list[str], list[list[Any]], int]:
    """Columns, the first rows and the row count of a result, normalized as the Lumen subject does."""
    frame = as_pandas(frame)
    return (list(frame.columns.map(str)), json.loads(frame.head(MAX_ROWS).to_json(orient="values", date_format="iso")),
            len(frame))


def answer_nodes(ops: Any, names: Iterable[str]) -> tuple[str | None, list[str]]:
    """
    Of the nodes ``names``, the SQL or metric node whose result answers a
    turn (the last one placed, or else the one the last chart, table or
    other Lumen view shows) and the charts.
    """
    names = set(names)
    touched = [(name, _graph.component_of(ops, name)) for name in _graph.nodes(ops) if name in names]
    queries = [name for name, component in touched if component in (SQL_ID, METRIC_ID)]
    charts = [name for name, component in touched if component == VIEW_ID]
    views = [name for name, component in touched if _graph.view_class(component) is not None]
    if queries:
        return queries[-1], charts
    if views:
        return _graph.upstream(ops, views[-1]), charts
    return None, charts


def node_sql(ops: Any, name: str) -> str | None:
    """
    The SQL of a SQL node, or the query a metric node compiles to: over its
    table by name for a structured metric, as a SQL node would write it.
    """
    config = _graph.config_of(ops, name)
    if "sql" in config:
        return config["sql"]
    try:
        pipeline = _graph.output(ops, name, "pipeline")
    except Exception:
        return None
    if not isinstance(pipeline, Pipeline):
        return None
    structured = any(isinstance(t, MetricAggregate) for t in pipeline.sql_transforms)
    dialect = None if pipeline.source.dialect == "any" else pipeline.source.dialect
    sql = table_sql(pipeline.table, dialect) if structured else pipeline.source.get_sql_expr(pipeline.table)
    for transform in pipeline.sql_transforms:
        sql = transform.apply(sql)
    return sql


def snapshot(session: AgentSession, turn: TurnResult) -> Turn:
    """Read a ``Turn`` off the dashboard, from the nodes the turn added or changed."""
    ops = session.ops
    data, views = answer_nodes(ops, [*turn.nodes, *turn.changed])
    sql = columns = rows = row_count = None
    errors = [turn.error] if turn.error else []
    if data is not None:
        sql = node_sql(ops, data)
        try:
            frame = _graph.output(ops, data, "data")
        except Exception as exc:
            errors.append(str(exc))
        else:
            if frame is not None:
                columns, rows, row_count = result_rows(frame)
    marks, encodings = chart_summary([_graph.config_of(ops, name).get("spec") or {} for name in views])
    calls = tool_calls(turn.messages, turn.tool_results)
    listing = next((call.result for call in reversed(calls) if call.name == "list_tables"), None)
    answer = turn.output[:MAX_ANSWER_LENGTH] if turn.output is not None else None
    return Turn(
        prompt=turn.prompt, status="error" if turn.error else "success", answer=answer,
        actors=[], task_statuses=[], sql=sql, columns=columns, rows=rows, row_count=row_count, view_types=[],
        usage=usage(turn.responses), listing=listing, chart_marks=marks, chart_encodings=encodings,
        tool_calls=calls, task_errors=errors,
    )


def _reference_source() -> DuckDBSource:
    return DuckDBSource(uri=":memory:", tables={"t": "SELECT 1 AS x"})


@functools.cache
def agent_fingerprint(instructions: str = "") -> str:
    """
    The hash of the instructions and tool schemas the agent offers, as
    ``AgentRunner`` stamps it on every response it records. Neither depends
    on the data, so it is the same for every source; extra ``instructions``,
    e.g. a suite's, are part of it. It is computed from the first request of
    a run on a model that only captures it, in a thread of its own so it can
    be called from a running event loop.
    """
    from panel_agents.adapters.model_trace import fingerprint

    captured = {}

    def capture(messages, info: AgentInfo) -> ModelResponse:
        captured["params"] = info.model_request_parameters
        return ModelResponse(parts=[TextPart("ok")])

    async def run():
        session = AgentSession(_reference_source(), FunctionModel(capture), instructions=instructions)
        try:
            await session.ask("")
        finally:
            session.close()

    with ThreadPoolExecutor(1) as pool:
        pool.submit(asyncio.run, run()).result()
    return fingerprint(captured["params"])


@register_subject("flowdash")
class FlowDashSubject:
    """
    One pydantic-ai agent building a FlowDash dashboard of Lumen nodes. Every
    prompt of a case runs through the same agent, conversation and dashboard,
    whatever ``inputs.agents`` names; a pre-seeded ``inputs.query`` is placed
    as a SQL node named ``result`` first.

    Pass ``model``, e.g. a ``FunctionModel``, or ``cassette``, the path of a
    recorded model trace, to run offline. ``model_settings`` replaces the
    agent's default model settings, ``model_settings_for`` each model;
    ``settings`` records the ones the model ``build_model`` gave gets.
    ``traces`` is a directory each case's model trace is written to, by default ``$LUMEN_AGENT_TRACES`` if set, so
    the CLI and its BIRD workers can record them too. The suite instructions
    given to ``build_model`` are appended to the agent's. ``fingerprint`` is
    the stamp of runs without them, since it may not depend on
    ``build_model``; the harness combines it with the suite instructions.
    ``metrics`` gives the metrics of a case's fixture, e.g. from its
    project file, or None for none.
    """

    name: ClassVar[str]
    supports_architecture = False

    def __init__(self, model: Any = None, cassette: str | os.PathLike | None = None,
                 traces: str | os.PathLike | None = None, model_settings: dict[str, Any] | None = None,
                 metrics: Callable[[Inputs], Any] | None = None):
        self.model = model
        self.metrics = metrics
        self.model_settings = None if model_settings is None else dict(model_settings)
        self.cassette = cassette
        traces = traces if traces is not None else os.environ.get(TRACES_ENV)
        self.traces = Path(traces) if traces else None
        self.instructions = ""
        self._built = None

    def partial_usage(self) -> Usage | None:
        """The usage of the case running now, or of the one that just ended, e.g. by a timeout or an error."""
        session = getattr(self, "_session", None)
        if session is not None:
            return usage(session.model.responses)
        return getattr(self, "_last_usage", None)

    @property
    def settings(self) -> dict[str, Any]:
        """The model settings of every request, recorded with the run settings."""
        if self.model_settings is not None:
            return {"model_settings": self.model_settings}
        if self._built is not None:
            return {"model_settings": model_settings_for(self._built)}
        return {"model_settings": {"openai_azure": MODEL_SETTINGS, "other": OTHER_MODEL_SETTINGS}}

    @property
    def requires_api_key(self) -> bool:
        return self.model is None and self.cassette is None

    def build_model(self, model: str, *, provider: str = "openai", api_key: str | None = None,
                    api: str | None = None, instructions: str = "") -> Any:
        self.instructions = instructions
        if self.model is not None:
            self._built = self.model
        elif self.cassette is not None:
            from panel_agents.adapters.cassette import Cassette

            self._built = Cassette.from_trace(self.cassette)
        else:
            self._built = build_model(model, provider=provider, api_key=api_key, api=api)
        return self._built

    def fingerprint(self) -> str | None:
        return agent_fingerprint()

    async def run(self, inputs: Inputs, model: Any, source: Any, documents: list[Any] | None = None) -> Output:
        trace = None
        if self.traces is not None:
            first = re.sub(r"\W+", "-", inputs.prompts[0].lower()).strip("-")[:40] if inputs.prompts else "case"
            digest = hashlib.sha1(json.dumps([inputs.fixture, inputs.prompts, inputs.query]).encode()).hexdigest()[:8]
            trace = self.traces / f"{first}-{digest}.jsonl"
        self._session = self._last_usage = None
        metrics = self.metrics(inputs) if self.metrics is not None else None
        session = self._session = AgentSession(source, model, instructions=self.instructions, trace=trace,
                                               model_settings=self.model_settings, metrics=metrics)
        turns, artifacts = [], []
        try:
            if inputs.query:
                session.add_sql(SEEDED_QUERY, inputs.query)
            for prompt in inputs.prompts:
                result = await session.ask(prompt)
                turn = snapshot(session, result)
                turns.append(turn)
                artifacts.append(TurnArtifacts(
                    graph=result.graph,
                    messages=ModelMessagesTypeAdapter.dump_python(result.messages, mode="json"),
                ))
                if turn.status == "error":
                    break
        finally:
            session.close()
            self._session = None
            self._last_usage = usage(session.model.responses)
        return Output(turns, _total(turns), artifacts)
