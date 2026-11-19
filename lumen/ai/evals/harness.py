from __future__ import annotations

import asyncio
import hashlib
import json
import subprocess
import warnings

from dataclasses import (
    asdict, dataclass, field, fields, replace,
)
from datetime import UTC, datetime
from importlib import import_module
from pathlib import Path
from typing import (
    Any, ClassVar, Literal, Protocol, runtime_checkable,
)

from openai import RateLimitError
from panel_material_ui import ChatInterface
from pydantic_evals import Case, Dataset
from pydantic_evals.evaluators import Evaluator, EvaluatorContext

from lumen.ai.agents import ChatAgent, SQLAgent, VegaLiteAgent
from lumen.ai.agents.document_list import DocumentListAgent
from lumen.ai.agents.document_summarizer import DocumentSummarizerAgent
from lumen.ai.agents.table_list import TableListAgent
from lumen.ai.agents.validation import ValidationAgent
from lumen.ai.coordinator import Plan
from lumen.ai.editors import VegaLiteEditor
from lumen.ai.llm import OpenAI
from lumen.ai.report import ActorTask
from lumen.ai.tool_trace import ToolCall
from lumen.ai.ui import ExplorerUI
from lumen.ai.usage import _rates
from lumen.ai.utils import describe_data
from lumen.config import SOURCE_TABLE_SEPARATOR
from lumen.data.catalog import get_metaset
from lumen.pipeline import Pipeline
from lumen.util import as_pandas

MAX_ROWS = 20
MAX_ANSWER_LENGTH = 4000
MAX_TOOL_RESULT_LENGTH = 4000
EVALUATOR_VERSION = 2
OPENROUTER_ENDPOINT = "https://openrouter.ai/api/v1"
# OpenRouter caches Anthropic prompts only below a breakpoint; a top-level one
# moves with the end of each request's prompt.
ANTHROPIC_CACHE = {"cache_control": {"type": "ephemeral"}}
# Above LumenSubject's own 120s per-prompt limit, so its timeouts still apply first.
PROMPT_TIMEOUT = 180
DEFAULT_SUBJECT = "lumen"
DEFAULT_SCORING = "all"
PRICING_PER_MILLION = {
    "gpt-4.1-mini": {"input": 0.40, "cached": 0.10, "output": 1.60},
    "gpt-4.1-nano": {"input": 0.10, "cached": 0.025, "output": 0.40},
    "gpt-5-nano": {"input": 0.05, "cached": 0.005, "output": 0.40},
    "gpt-6-luna": {"input": 0.10, "cached": 0.01, "output": 0.50},
    "gpt-6-sol": {"input": 2.00, "cached": 0.20, "output": 10.00},
    "qwen/qwen3-30b-a3b-instruct-2507": {"input": 0.10, "cached": 0.10, "output": 0.30},
    "qwen/qwen3.8-flash": {"input": 0.15, "cached": 0.016, "output": 0.47},
    "qwen/qwen3.8-27b": {"input": 0.42, "cached": 0.085, "output": 3.00},
    "google/gemma-3-27b-it": {"input": 0.08, "cached": 0.04, "output": 0.45},
    "moonshotai/kimi-k2.6": {"input": 0.95, "cached": 0.16, "output": 4.00},
    "google/gemini-3.8-flash": {"input": 0.75, "cached": 0.075, "output": 3.75},
    "xiaomi/mimo-v2.6-flash": {"input": 0.14, "cached": 0.0028, "output": 0.28},
    "z-ai/glm-5.3-flash": {"input": 0.15, "cached": 0.03, "output": 0.50},
    "anthropic/claude-sonnet-5": {"input": 2.00, "cached": 0.20, "output": 10.00, "cache_write": 2.50},
    "anthropic/claude-haiku-4.5": {"input": 1.00, "cached": 0.10, "output": 5.00, "cache_write": 1.25},
    "anthropic/claude-haiku-5.5": {"input": 0.10, "cached": 0.01, "output": 0.50, "cache_write": 0.125},
    "ibm-granite/granite-4.2-8b": {"input": 0.06, "cached": 0.015, "output": 0.15},
    "mistralai/ministral-8b-2512": {"input": 0.15, "cached": 0.015, "output": 0.15},
    "qwen/qwen3.5-9b": {"input": 0.154, "cached": 0.138, "output": 0.226},
    # OpenRouter reports and bills cache writes for gpt-6-luna.
    "openai/gpt-6-luna": {"input": 0.10, "cached": 0.01, "output": 0.50, "cache_write": 0.125},
    "deepseek/deepseek-v4.1-flash": {"input": 0.003, "cached": 0.003, "output": 2.40},
}


@dataclass
class Usage:
    """
    Tokens and cost of model requests. ``cold_cost_usd`` is what they would
    cost if the case's first request found nothing in the provider's prompt
    cache, i.e. if cases did not run back to back: the cached tokens of that
    request are priced at the cache write rate, or else the input rate.
    """

    input_tokens: int
    output_tokens: int
    cached_tokens: int
    cost_usd: float | None
    requests: int | None = None
    cold_cost_usd: float | None = None
    cache_write_tokens: int = 0

    @property
    def cached_percent(self) -> float | None:
        return 100 * self.cached_tokens / self.input_tokens if self.input_tokens else None


def request_cost(model: str, input_tokens: int, output_tokens: int, cached_tokens: int = 0,
                 cache_write_tokens: int = 0) -> float | None:
    """The cost of one request at ``PRICING_PER_MILLION``, or None for a model without a price."""
    rates = _rates(PRICING_PER_MILLION, model or "")
    if rates is None:
        return None
    return ((input_tokens - cached_tokens - cache_write_tokens) * rates["input"]
            + cached_tokens * rates.get("cached", rates["input"])
            + cache_write_tokens * rates.get("cache_write", rates["input"])
            + output_tokens * rates["output"]) / 1_000_000


def cold_surcharge(model: str, cached_tokens: int) -> float | None:
    """What ``cached_tokens`` of a request would cost more if they were not in the cache."""
    rates = _rates(PRICING_PER_MILLION, model or "")
    if rates is None:
        return None
    return cached_tokens * (rates.get("cache_write", rates["input"]) - rates.get("cached", rates["input"])) / 1_000_000


def total_usage(usages: list[Usage | None]) -> Usage | None:
    """The sum of ``usages``; a cost is only summed when every part has one."""
    usages = [u for u in usages if u is not None]
    if not usages:
        return None

    def total(name):
        values = [getattr(u, name) for u in usages]
        return sum(values) if all(v is not None for v in values) else None

    return Usage(sum(u.input_tokens for u in usages), sum(u.output_tokens for u in usages),
                 sum(u.cached_tokens for u in usages), total("cost_usd"), total("requests"),
                 total("cold_cost_usd"), sum(u.cache_write_tokens for u in usages))


class EvalOpenAI(OpenAI):
    """Add evaluation instructions and rate-limit retries."""

    def __init__(self, suite_instructions: str = "", **params):
        params.setdefault("usage_pricing", PRICING_PER_MILLION)
        super().__init__(**params)
        self.suite_instructions = suite_instructions

    def _add_system_message(self, messages, system, input_kwargs):
        if self.suite_instructions:
            system = f"{system}\n\n## Evaluation suite instructions\n{self.suite_instructions}"
        return super()._add_system_message(messages, system, input_kwargs)

    def _create_base_client(self, **kwargs):
        client = super()._create_base_client(**kwargs)
        target = client.responses if self.api == "responses" else client.chat.completions
        create = target.create

        async def eval_create(*args, **params):
            # Thinking runs go through the Responses API; chat completions keep
            # Qwen 3.8 comparable with earlier non-thinking results.
            if self.api != "responses" and self.model_kwargs["default"]["model"].startswith("qwen/qwen3.8-"):
                params["extra_body"] = {**params.get("extra_body", {}), "reasoning": {"enabled": False}}
            if self.endpoint == OPENROUTER_ENDPOINT and self.model_kwargs["default"]["model"].startswith("anthropic/"):
                params["extra_body"] = {**params.get("extra_body", {}), **ANTHROPIC_CACHE}

            async def wait_for_retry(exc, attempt):
                retry_after = exc.response.headers.get("retry-after") if exc.response else None
                try:
                    delay = float(retry_after) if retry_after is not None else 2 ** (attempt + 1)
                except ValueError:
                    delay = 2 ** (attempt + 1)
                await asyncio.sleep(min(30, max(0, delay)))

            if not params.get("stream"):
                for attempt in range(5):
                    try:
                        return await create(*args, **params)
                    except RateLimitError as exc:
                        if attempt == 4:
                            raise
                        await wait_for_retry(exc, attempt)

            async def events():
                for attempt in range(5):
                    yielded = False
                    try:
                        response = await create(*args, **params)
                        async for chunk in response:
                            yielded = True
                            yield chunk
                        return
                    except RateLimitError as exc:
                        if yielded or attempt == 4:
                            raise
                        await wait_for_retry(exc, attempt)

            return events()

        target.create = eval_create
        return client


@dataclass
class Inputs:
    prompts: list[str]
    fixture: str = "sales"
    agents: tuple[str, ...] = ()
    query: str | None = None
    seed_chat: str | None = None


@dataclass
class Turn:
    prompt: str
    status: str
    answer: str | None
    actors: list[str]
    task_statuses: list[str]
    sql: str | None
    columns: list[str] | None
    rows: list[list[Any]] | None
    row_count: int | None
    view_types: list[str]
    usage: Usage | None = None
    listing: str | None = None
    chart_marks: list[str] | None = None
    chart_encodings: list[dict[str, dict[str, Any]]] | None = None
    tool_calls: list[ToolCall] | None = None
    task_errors: list[str] | None = None
    planner_actors: list[str] | None = None
    follow_up_type: str | None = None
    document_summary: str | None = None
    validation_correct: bool | None = None
    validation_missing: list[str] | None = None


@dataclass
class TurnArtifacts:
    """
    Opaque per-turn artifacts a subject may return next to its ``Turn``:
    the serialized result graph after the turn (e.g. YAML text) and the
    JSON-serializable model message history of the turn.
    """

    graph: str | None = None
    messages: Any = None


@dataclass
class Output:
    turns: list[Turn]
    usage: Usage | None = None
    artifacts: list[TurnArtifacts] | None = None


Scoring = Literal["all", "outcome"]
def _outcome(check: str) -> dict[str, str]:
    return {"kind": "outcome", "check": check}


def _architecture(check: str) -> dict[str, str]:
    return {"kind": "architecture", "check": check}


@dataclass
class Expected:
    """
    Expectations of a case. Each field is tagged with the check that scores
    it and whether that is an outcome check, which any subject can be scored
    on, or an architecture check, which asserts how the current Lumen stack
    produced the outcome.
    """

    status: str = field(default="success", metadata=_outcome("status"))
    actors: list[str] | None = field(default=None, metadata=_architecture("actors"))
    rows: list[list[Any]] | None = field(default=None, metadata=_outcome("rows"))
    view_type: str | None = field(default=None, metadata=_architecture("view_type"))
    columns: list[str] | None = field(default=None, metadata=_outcome("columns"))
    answer_contains: str | None = field(default=None, metadata=_outcome("answer"))
    listing_contains: str | None = field(default=None, metadata=_outcome("listing"))
    chart_mark: str | None = field(default=None, metadata=_outcome("chart_mark"))
    chart_fields: dict[str, str] | None = field(default=None, metadata=_outcome("chart_fields"))
    tool_calls: list[str] | None = field(default=None, metadata=_architecture("tool_calls"))
    tool_arguments: dict[str, dict[str, Any]] | None = field(default=None, metadata=_architecture("tool_arguments"))
    tool_result_contains: dict[str, list[str]] | None = field(default=None, metadata=_architecture("tool_results"))
    turn_rows: list[list[list[Any]] | None] | None = field(default=None, metadata=_outcome("turn_rows"))
    turn_tables: list[str] | None = field(default=None, metadata=_outcome("turn_tables"))
    sql_contains: list[str] | None = field(default=None, metadata=_outcome("sql_contains"))
    turn_planner_actors: list[list[str]] | None = field(default=None, metadata=_architecture("planner_actors"))
    forbidden_actors: list[str] | None = field(default=None, metadata=_architecture("forbidden_actors"))
    follow_up_types: list[str] | None = field(default=None, metadata=_architecture("follow_up_types"))
    planned_actors: list[str] | None = field(default=None, metadata=_architecture("planned_actors"))
    document_contains: list[str] | None = field(default=None, metadata=_outcome("document_summary"))
    # Also scores the implicit check that no ValidationAgent reported a problem.
    validation_correct: bool | None = field(default=None, metadata=_architecture("validation"))
    gold_sql: str | None = field(default=None, metadata=_outcome("execution_accuracy"))
    error_contains: str | None = field(default=None, metadata=_outcome("error"))

    def expectation_kinds(self) -> set[str]:
        """Kinds of the expectations this case sets beyond the defaults."""
        return {
            item.metadata["kind"] for item in fields(self)
            if getattr(self, item.name) != item.default
        }


ARCHITECTURE_CHECKS = frozenset(
    item.metadata["check"] for item in fields(Expected) if item.metadata["kind"] == "architecture"
)
OUTCOME_CHECKS = frozenset(
    {"all_turns_completed", "output"} | {item.metadata["check"] for item in fields(Expected) if item.metadata["kind"] == "outcome"}
)


@runtime_checkable
class Subject(Protocol):
    """
    The system under evaluation.

    ``build_model`` turns the CLI's model options into whatever model type the
    subject accepts, e.g. a Lumen ``Llm`` or a pydantic-ai model. ``run``
    receives a case's inputs, that model, the source built for the case
    fixture and its documents, and returns one ``Turn`` per answered prompt.
    It may stop early after a turn that errors or times out, and may attach
    one ``TurnArtifacts`` per turn.

    ``supports_architecture`` declares whether the ``Turn`` fields used by
    architecture checks (actors, planner and tool traces, Lumen view types)
    are meaningful for this subject; if not, runs are scored on outcome
    checks only. ``fingerprint`` returns a stable string describing the
    subject's agent surface (e.g. a hash of tool schemas and instructions),
    or None to leave case fingerprints unchanged. It must not depend on
    ``build_model`` or ``run`` having been called.

    Subjects may also set ``requires_api_key = False`` (default True) when
    they run offline, e.g. on a pydantic-ai ``FunctionModel`` or a cassette,
    and ``settings``, a JSON-serializable dict of options that change how the
    subject behaves (e.g. model settings), which is recorded with the run
    settings and is part of the case fingerprint, and ``partial_usage()``,
    the ``Usage`` of the case run so far, recorded when it times out or raises.
    """

    name: str
    supports_architecture: bool

    def build_model(self, model: str, *, provider: str = "openai", api_key: str | None = None,
                    api: str | None = None, instructions: str = "") -> Any: ...

    def fingerprint(self) -> str | None: ...

    async def run(self, inputs: Inputs, model: Any, source: Any, documents: list[Any] | None = None) -> Output: ...


SUBJECTS: dict[str, type] = {}

# Subjects with optional dependencies, imported (and so registered) when first named.
OPTIONAL_SUBJECTS = {"flowdash": "lumen.agent.evals:FlowDashSubject"}


def register_subject(name: str):
    """Class decorator registering a subject under ``name`` for ``--subject``."""

    def decorator(cls):
        cls.name = name
        SUBJECTS[name] = cls
        return cls

    return decorator


def subject_spec(subject: Subject) -> str:
    """The ``package.module:Class`` spec that resolves to ``subject`` in a fresh process."""
    cls = type(subject)
    return f"{cls.__module__}:{cls.__qualname__}"


def resolve_subject(name: str | None = None, **params) -> Subject:
    """
    Instantiate a registered subject, or one given as ``package.module:Class``,
    which is imported (registering it as a side effect). An instance of a
    class without a ``name`` takes the spec as its name. Names in
    ``OPTIONAL_SUBJECTS`` import their module first, which fails if its
    dependencies are not installed.
    """
    name = name or DEFAULT_SUBJECT
    if name not in SUBJECTS and name in OPTIONAL_SUBJECTS:
        module, _ = OPTIONAL_SUBJECTS[name].split(":", 1)
        import_module(module)
    if name in SUBJECTS:
        cls = SUBJECTS[name]
    elif ":" in name:
        module, attr = name.split(":", 1)
        cls = getattr(import_module(module), attr)
    else:
        raise ValueError(f"Unknown eval subject {name!r}; registered: {', '.join(sorted(SUBJECTS))}")
    try:
        subject = cls(**params)
    except TypeError as exc:
        if not params:
            raise
        raise TypeError(f"Subject {name!r} does not support the option(s) {', '.join(params)}: {exc}") from exc
    if not isinstance(getattr(subject, "name", None), str):
        subject.name = name
    missing = [member for member in ("name", "supports_architecture", "build_model", "fingerprint", "run")
               if not hasattr(subject, member)]
    if missing:
        raise TypeError(f"{name!r} does not implement the Subject protocol; missing: {', '.join(missing)}")
    SUBJECTS.setdefault(subject.name, cls)
    return subject


def effective_scoring(subject: Subject, scoring: Scoring | None = None) -> Scoring:
    """A subject without architecture support is always scored on outcomes only."""
    return (scoring or DEFAULT_SCORING) if subject.supports_architecture else "outcome"


def _snapshot(prompt: str, plan: Any, messages: list[Any], previous_tasks: tuple[Any, ...] = ()) -> Turn:
    if plan is None:
        return Turn(prompt, "no_plan", None, [], [], None, None, None, None, [])

    tasks = [task for task in plan if task not in previous_tasks]
    data = next((task.out_context["pipeline"] for task in reversed(tasks) if "pipeline" in task.out_context), None)
    if data is None and not previous_tasks:
        data = plan.out_context.get("pipeline")
    data = data.data if data is not None else None
    if data is not None:
        frame = as_pandas(data)
        columns = list(frame.columns.map(str))
        row_count = len(frame)
        rows = json.loads(frame.head(MAX_ROWS).to_json(orient="values", date_format="iso"))
    else:
        columns = rows = row_count = None

    answer = next((msg.object for msg in reversed(messages) if msg.user not in ("User", "Planner", "Runner") and isinstance(msg.object, str)), None)
    if answer is None:
        answer = next((task.out_context["chat"] for task in reversed(tasks) if isinstance(task.out_context.get("chat"), str)), None)
    if answer is not None:
        answer = answer[:MAX_ANSWER_LENGTH]

    marks = []
    encodings = []
    for task in tasks:
        for view in task.views:
            if isinstance(view, VegaLiteEditor):
                spec = view.component.spec
                layers = spec.get("layer", [spec]) if isinstance(spec, dict) else []
                for layer in layers:
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

    errors = [str(task.out_context["__error__"]) for task in tasks if task.out_context.get("__error__")]
    validation = next((task.out_context.get("validation_result") for task in reversed(tasks) if task.out_context.get("validation_result") is not None), None)

    return Turn(
        prompt=prompt,
        status=plan.status,
        answer=answer,
        actors=[type(task.actor).__name__ for task in tasks if hasattr(task, "actor")],
        task_statuses=[task.status for task in tasks],
        sql=next((task.out_context["sql"] for task in reversed(tasks) if isinstance(task.out_context.get("sql"), str)), None),
        columns=columns,
        rows=rows,
        row_count=row_count,
        view_types=[type(view).__name__ for task in tasks for view in task.views],
        listing=next((task.out_context["listing"] for task in reversed(tasks) if isinstance(task.out_context.get("listing"), str)), None),
        chart_marks=marks,
        chart_encodings=encodings,
        task_errors=errors,
        planner_actors=list(getattr(plan, "planner_actors", [])),
        follow_up_type=getattr(plan, "follow_up_type", None),
        document_summary=next((task.out_context["document_summary"] for task in reversed(tasks) if isinstance(task.out_context.get("document_summary"), str)), None),
        validation_correct=getattr(validation, "correct", None),
        validation_missing=list(getattr(validation, "missing_elements", [])) if validation is not None else None,
    )


def _direct_plan(inputs: Inputs, llm: Any, context: dict, interface: ChatInterface, prompt: str,
                 agent_params: dict[str, Any] | None = None, sql_cleanup: bool = True) -> Plan:
    agents = {
        "SQLAgent": SQLAgent,
        "ChatAgent": ChatAgent,
        "TableListAgent": TableListAgent,
        "VegaLiteAgent": VegaLiteAgent,
        "DocumentListAgent": DocumentListAgent,
        "DocumentSummarizerAgent": DocumentSummarizerAgent,
        "ValidationAgent": ValidationAgent,
    }
    tasks = []
    for name in inputs.agents:
        agent_type = agents[name]
        agent = agent_type(llm=llm, **({"n_doc_pages": 0} if name == "VegaLiteAgent" else {}),
                           **({"clean_data": False} if name == "SQLAgent" and not sql_cleanup else {}),
                           **(agent_params or {}))
        tasks.append(ActorTask(agent, title=name))
    return Plan(*tasks, title="Direct agent evaluation", context=context,
                history=[{"role": "user", "content": prompt}], llm=llm, interface=interface)


def _error_message(exc: Exception) -> str:
    messages = []
    while exc is not None:
        messages.append(f"{type(exc).__name__}: {exc}")
        exc = exc.__cause__
    return " | caused by ".join(messages)


@register_subject(DEFAULT_SUBJECT)
class LumenSubject:
    """
    The current Lumen stack: every prompt of a case runs through ExplorerUI,
    or through ``inputs.agents`` directly, passing ``ui_params`` or
    ``agent_params`` to their constructors. The model is a Lumen ``Llm``.
    ``sql_cleanup=False`` disables SQLAgent's automatic SQL cleanup in
    direct-agent cases.
    """

    name: ClassVar[str]
    supports_architecture = True

    def __init__(self, ui_params: dict[str, Any] | None = None, agent_params: dict[str, Any] | None = None,
                 sql_cleanup: bool = True):
        self.ui_params = ui_params
        self.agent_params = agent_params
        self.sql_cleanup = sql_cleanup

    def build_model(self, model: str, *, provider: str = "openai", api_key: str | None = None,
                    api: str | None = None, instructions: str = "") -> EvalOpenAI:
        openrouter = provider == "openrouter"
        return EvalOpenAI(
            suite_instructions=instructions,
            api=api or ("chat_completions" if openrouter else "responses"),
            temperature=None, api_key=api_key,
            endpoint=OPENROUTER_ENDPOINT if openrouter else None,
            stateless_responses=openrouter,
            model_kwargs={"default": {"model": model}, "ui": {"model": model}},
        )

    def fingerprint(self) -> str | None:
        return None

    async def run(self, inputs: Inputs, model: Any, source: Any, documents: list[Any] | None = None) -> Output:
        llm = model
        if inputs.agents:
            context = {"source": source, "sources": [source]}
            interface = ChatInterface()
        else:
            ui = ExplorerUI(data=source, llm=llm, **(self.ui_params or {}))
            context = ui.context
            interface = ui.interface
            source = context["source"]
        slugs = [f"{source.name}{SOURCE_TABLE_SEPARATOR}{table}" for table in source.get_tables()]
        context["metaset"] = await get_metaset([source], slugs)
        context["visible_slugs"] = set(slugs)
        if documents is not None:
            context["metaset"].docs = documents
        if inputs.seed_chat is not None:
            context["chat"] = inputs.seed_chat
        if inputs.query:
            from lumen.sources.duckdb import DuckDBSource

            query_source = DuckDBSource(uri=":memory:", tables={**source.tables, "result": inputs.query})
            pipeline = Pipeline(source=query_source, table="result")
            context.update(pipeline=pipeline, data=await describe_data(pipeline.data), table="result")
        turns = []
        for prompt in inputs.prompts:
            before = len(interface.objects)
            previous_plan = None if inputs.agents else ui._exploration["view"].plan
            previous_tasks = tuple(previous_plan) if previous_plan is not None else ()
            plan = None
            interface.send(prompt, user="User", respond=False)
            with llm.capture_usage() as scope, llm.trace() as tool_calls:
                try:
                    if inputs.agents:
                        plan = _direct_plan(inputs, llm, context, interface, prompt, self.agent_params, self.sql_cleanup)
                        with warnings.catch_warnings():
                            warnings.filterwarnings("ignore", message="Widget.name is deprecated", category=PendingDeprecationWarning)
                            await asyncio.wait_for(plan.execute(), timeout=120)
                        context.update(plan.out_context)
                    else:
                        await asyncio.wait_for(ui._chat_invoke(prompt, "User", interface), timeout=120)
                        plan = ui._exploration["view"].plan
                    if plan is previous_plan and (plan is None or all(task in previous_tasks for task in plan)):
                        plan = None
                    turn = _snapshot(prompt, plan, interface.objects[before:], previous_tasks if plan is previous_plan else ())
                except TimeoutError:
                    turn = Turn(prompt, "timeout", None, [], [], None, None, None, None, [])
                except Exception as exc:
                    error = _error_message(exc)
                    if plan is not None:
                        turn = _snapshot(prompt, plan, interface.objects[before:], previous_tasks if plan is previous_plan else ())
                        turn.status = "error"
                        turn.task_errors = [*(turn.task_errors or []), error]
                    else:
                        turn = Turn(prompt, "error", error, [], [], None, None, None, None, [], task_errors=[error])
                finally:
                    turn.tool_calls = [ToolCall(call.name, call.arguments, call.result[:MAX_TOOL_RESULT_LENGTH]) for call in tool_calls if isinstance(call, ToolCall)]
                    turn.usage = records_usage(scope.records, first=not turns)
            turns.append(turn)
            if turn.status in ("error", "timeout"):
                break
        return Output(turns, total_usage([turn.usage for turn in turns]))


def records_usage(records: list[Any], first: bool = False) -> Usage | None:
    """
    The ``Usage`` of Lumen ``Llm`` usage records; ``first`` if they include
    the case's first request, which ``cold_cost_usd`` prices uncached.
    """
    if not records:
        return None
    costs = [record.cost_usd for record in records]
    cost = sum(costs) if all(c is not None for c in costs) else None
    surcharge = cold_surcharge(records[0].model, records[0].cached_tokens) if first else 0
    return Usage(sum(r.input_tokens for r in records), sum(r.output_tokens for r in records),
                 sum(r.cached_tokens for r in records), cost, len(records),
                 cost + surcharge if cost is not None and surcharge is not None else None,
                 sum(getattr(r, "cache_write_tokens", 0) for r in records))


async def run_case(inputs: Inputs, llm: Any, source: Any, documents: list[Any] | None = None,
                   ui_params: dict[str, Any] | None = None, agent_params: dict[str, Any] | None = None) -> Output:
    """Run a case through ``LumenSubject``."""
    return await LumenSubject(ui_params, agent_params).run(inputs, llm, source, documents)


DIAGNOSTIC_PREFIX = "diagnostic:"
APPLICABILITY = "applicability"
NOT_APPLICABLE = "not_applicable"


@dataclass
class CheckResult(Evaluator[Inputs, Output, Expected]):
    """
    Score a case's outcome and architecture checks.

    With ``scoring="all"`` and a subject that ``supports_architecture``, every
    check is an assertion. Otherwise only outcome checks are assertions;
    architecture checks become ``diagnostic:<name>`` labels ("pass" or "fail")
    if the subject supports them and are dropped if not, and a case whose only
    expectations are architecture expectations is labelled not applicable
    instead of being scored.
    """

    scoring: Scoring = DEFAULT_SCORING
    supports_architecture: bool = True

    def evaluate(self, ctx: EvaluatorContext[Inputs, Output, Expected]) -> dict[str, bool | str]:
        expected = ctx.metadata or Expected()
        checks = self._checks(ctx, expected)
        if unclassified := checks.keys() - OUTCOME_CHECKS - ARCHITECTURE_CHECKS:
            raise ValueError(f"Unclassified checks: {', '.join(sorted(unclassified))}")
        if self.scoring == "all" and self.supports_architecture:
            return checks
        results: dict[str, bool | str] = {
            name: passed for name, passed in checks.items() if name in OUTCOME_CHECKS
        }
        if self.supports_architecture:
            results.update({
                f"{DIAGNOSTIC_PREFIX}{name}": "pass" if passed else "fail"
                for name, passed in checks.items() if name in ARCHITECTURE_CHECKS
            })
        if expected.expectation_kinds() == {"architecture"}:
            results = {name: value for name, value in results.items() if name.startswith(DIAGNOSTIC_PREFIX)}
            results[APPLICABILITY] = NOT_APPLICABLE
        return results

    def _checks(self, ctx: EvaluatorContext[Inputs, Output, Expected], expected: Expected) -> dict[str, bool]:
        if not ctx.output.turns:
            return {"all_turns_completed": False, "status": False}
        last = ctx.output.turns[-1]
        checks = {
            "all_turns_completed": len(ctx.output.turns) == len(ctx.inputs.prompts),
            "status": last.status == expected.status and (expected.status != "success" or all(turn.status == "success" and all(status == "success" for status in turn.task_statuses) for turn in ctx.output.turns)),
        }
        if expected.status == "success":
            checks["output"] = any(
                turn.answer is not None or turn.rows is not None or turn.listing is not None
                or bool(turn.view_types) or bool(turn.chart_marks)
                for turn in ctx.output.turns
            ) and not any(turn.task_errors for turn in ctx.output.turns)
            if expected.validation_correct is None:
                checks["validation"] = all(turn.validation_correct is not False for turn in ctx.output.turns)
        if expected.actors is not None:
            checks["actors"] = all(actor in last.actors for actor in expected.actors)
        if expected.rows is not None:
            checks["rows"] = last.rows is not None and last.row_count == len(expected.rows) and sorted(last.rows, key=str) == sorted(expected.rows, key=str)
        if expected.columns is not None:
            checks["columns"] = last.columns == expected.columns
        if expected.view_type is not None:
            checks["view_type"] = expected.view_type in last.view_types
        if expected.answer_contains is not None:
            checks["answer"] = expected.answer_contains in (last.answer or "")
        if expected.listing_contains is not None:
            checks["listing"] = expected.listing_contains in (last.listing or "")
        if expected.chart_mark is not None:
            checks["chart_mark"] = expected.chart_mark in (last.chart_marks or [])
        if expected.chart_fields is not None:
            checks["chart_fields"] = any(
                all(any(definition.get("field") == field and definition.get("type") == kind for definition in encoding.values())
                    for kind, field in expected.chart_fields.items())
                for encoding in (last.chart_encodings or [])
            )
        if expected.tool_calls is not None:
            names = [trace.name for trace in (last.tool_calls or [])]
            checks["tool_calls"] = all(tool in names for tool in expected.tool_calls)
        if expected.tool_arguments is not None:
            checks["tool_arguments"] = all(
                any(trace.name == tool and all(trace.arguments.get(key) == value for key, value in arguments.items()) for trace in (last.tool_calls or []))
                for tool, arguments in expected.tool_arguments.items()
            )
        if expected.tool_result_contains is not None:
            checks["tool_results"] = all(
                any(trace.name == tool and all(fragment in trace.result for fragment in fragments) for trace in (last.tool_calls or []))
                for tool, fragments in expected.tool_result_contains.items()
            )
        if expected.turn_rows is not None:
            checks["turn_rows"] = len(ctx.output.turns) == len(expected.turn_rows) and all(
                expected_rows is None or (turn.rows is not None and turn.row_count == len(expected_rows) and sorted(turn.rows, key=str) == sorted(expected_rows, key=str))
                for turn, expected_rows in zip(ctx.output.turns, expected.turn_rows, strict=True)
            )
        if expected.turn_tables is not None:
            checks["turn_tables"] = len(ctx.output.turns) == len(expected.turn_tables) and all(
                table in (turn.sql or "").lower() for turn, table in zip(ctx.output.turns, expected.turn_tables, strict=True)
            )
        if expected.sql_contains is not None:
            checks["sql_contains"] = all(fragment in (last.sql or "").lower() for fragment in expected.sql_contains)
        if expected.turn_planner_actors is not None:
            checks["planner_actors"] = len(ctx.output.turns) == len(expected.turn_planner_actors) and all(
                set(turn.planner_actors or []) == set(actors)
                for turn, actors in zip(ctx.output.turns, expected.turn_planner_actors, strict=True)
            )
        if expected.forbidden_actors is not None:
            checks["forbidden_actors"] = all(actor not in last.actors for actor in expected.forbidden_actors)
        if expected.follow_up_types is not None:
            checks["follow_up_types"] = [turn.follow_up_type for turn in ctx.output.turns] == expected.follow_up_types
        if expected.planned_actors is not None:
            checks["planned_actors"] = all(actor in (last.planner_actors or []) for actor in expected.planned_actors)
        if expected.document_contains is not None:
            checks["document_summary"] = all(fragment in (last.document_summary or "") for fragment in expected.document_contains)
        if expected.validation_correct is not None:
            checks["validation"] = last.validation_correct is expected.validation_correct
        if expected.error_contains is not None:
            checks["error"] = expected.error_contains in " ".join(last.task_errors or [])
        return checks


def run_settings(settings: dict[str, Any] | None = None, subject: str | None = None,
                 scoring: Scoring | None = None, subject_settings: dict[str, Any] | None = None) -> dict[str, Any]:
    """
    Add the subject, scoring mode and the subject's own ``settings`` to the
    run settings when they differ from the defaults, so default runs keep
    their earlier fingerprints.
    """
    settings = dict(settings or {})
    if subject and subject != DEFAULT_SUBJECT:
        settings["subject"] = subject
    if scoring and scoring != DEFAULT_SCORING:
        settings["scoring"] = scoring
    if subject_settings:
        settings["subject_settings"] = subject_settings
    return settings


def case_fingerprint(cases: list[Case], fixtures: dict[str, Any] | None = None,
                     instructions: str = "", instruction_version: str = "",
                     settings: dict[str, Any] | None = None, subject_fingerprint: str | None = None) -> str:
    definitions = [
        {"name": case.name, "inputs": asdict(case.inputs), "expected": asdict(case.metadata or Expected())}
        for case in cases
    ]
    payload = {"evaluator_version": EVALUATOR_VERSION, "cases": definitions, "fixtures": fixtures,
               "instructions": instructions, "instruction_version": instruction_version}
    if settings:
        # Only when set, so results recorded without settings keep their fingerprint.
        payload["settings"] = settings
    if subject_fingerprint is not None:
        payload["subject"] = subject_fingerprint
    payload = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(payload.encode()).hexdigest()


def _partial_usage(subject: Any) -> Usage | None:
    """What a subject's run used before it timed out or raised, if the subject reports it."""
    partial = getattr(subject, "partial_usage", None)
    return partial() if callable(partial) else None


def model_metadata(model: Any) -> dict[str, Any]:
    """Describe a Lumen ``Llm`` or a pydantic-ai model for the run metadata."""
    if hasattr(model, "model_kwargs"):
        return {"model": model.model_kwargs.get("default", {}).get("model"), "api": getattr(model, "api", None),
                "provider": "openrouter" if getattr(model, "endpoint", None) == OPENROUTER_ENDPOINT else "openai"}
    return {"model": getattr(model, "model_name", None), "api": getattr(model, "api", None),
            "provider": getattr(model, "system", None)}


def _record_artifacts(output: Output) -> list[dict[str, Any]] | None:
    if not output.artifacts or all(item.graph is None and item.messages is None for item in output.artifacts):
        return None
    return [asdict(item) for item in output.artifacts]


def artifacts_path(output: Path) -> Path:
    """
    The JSON Lines sidecar holding per-turn artifacts of the cases in
    ``output``, one ``{"name": ..., "artifacts": [...]}`` line per case.
    """
    return output.with_name(f"{output.stem}.artifacts.jsonl")


def git_state() -> tuple[str, bool]:
    """The checked out commit and whether files that can move results differ from it."""
    commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=False).stdout.strip()
    # Any change to the package can move results, prompts and agents included.
    dirty = bool(subprocess.run(["git", "status", "--porcelain", "--untracked-files=all", "--", "lumen", "tests/evals", "pixi.toml", "pixi.lock", "pyproject.toml"], capture_output=True, text=True, check=False).stdout.strip())
    return commit, dirty


async def evaluate(llm: Any, dataset: Dataset, source_factory: Any, output: Path | None = None,
                   case_name: str | None = None, documents_factory: Any = None, fixtures: dict[str, Any] | None = None,
                   instruction_version: str = "", ui_params: dict[str, Any] | None = None,
                   agent_params: dict[str, Any] | None = None, settings: dict[str, Any] | None = None,
                   subject: Subject | None = None, scoring: Scoring = DEFAULT_SCORING,
                   instructions: str | None = None, prompt_timeout: float = PROMPT_TIMEOUT):
    """
    Evaluate the dataset with ``subject`` and optionally write the results to ``output``.

    ``llm`` is the model passed to the subject, which defaults to
    ``LumenSubject(ui_params, agent_params)``. ``scoring="outcome"`` scores
    outcome checks only, see ``CheckResult``; a subject without architecture
    support is always scored that way. ``settings`` describes the run
    configuration, e.g. which options produced ``ui_params``; it is recorded
    with the results and part of the fingerprint, together with the subject
    and the effective scoring mode when they are not the defaults.
    ``instructions`` are the suite instructions given to ``build_model``,
    read from the model's ``suite_instructions`` if omitted. A case whose
    subject run exceeds ``prompt_timeout`` seconds per prompt is recorded as
    a timeout. Per-turn artifacts go to ``artifacts_path(output)``.
    """
    if subject is None:
        subject = LumenSubject(ui_params, agent_params)
    elif ui_params or agent_params:
        raise ValueError("ui_params and agent_params apply to the default subject only")
    scoring = effective_scoring(subject, scoring)
    settings = run_settings(settings, subject.name, scoring, getattr(subject, "settings", None))
    if instructions is None:
        instructions = getattr(llm, "suite_instructions", "")
    subject_fingerprint = subject.fingerprint()

    async def task(inputs: Inputs) -> Output:
        prompt = inputs.prompts[0] if inputs.prompts else ""
        try:
            return await asyncio.wait_for(
                subject.run(inputs, llm, source_factory(inputs), documents_factory(inputs) if documents_factory else None),
                timeout=prompt_timeout * max(1, len(inputs.prompts)),
            )
        except TimeoutError:
            usage = _partial_usage(subject)
            return Output([Turn(prompt, "timeout", None, [], [], None, None, None, None, [], usage=usage)], usage)
        except Exception as exc:
            error = _error_message(exc)
            usage = _partial_usage(subject)
            return Output([Turn(prompt, "error", None, [], [], None, None, None, None, [], task_errors=[error], usage=usage)], usage)

    cases = [case for case in dataset.cases if case.name == case_name] if case_name else dataset.cases
    if not cases:
        raise ValueError(f"Unknown eval case: {case_name}")
    evaluators = [
        replace(evaluator, scoring=scoring, supports_architecture=subject.supports_architecture)
        if isinstance(evaluator, CheckResult) else evaluator
        for evaluator in dataset.evaluators
    ]
    selected = Dataset(name=dataset.name, cases=cases, evaluators=evaluators)
    report = await selected.evaluate(task, max_concurrency=1)
    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        commit, dirty = git_state()
        completed = {case.name: case for case in report.cases}
        failures = {failure.name: failure.error_message for failure in report.failures}
        for case in report.cases:
            if case.evaluator_failures:
                failures[case.name] = "; ".join(f"{failure.name}: {failure.error_message}" for failure in case.evaluator_failures)
        cases_out, artifacts = [], []
        for case in cases:
            result = completed.get(case.name)
            record = {"name": case.name, "turns": [], "usage": None, "assertions": {}, "duration": None,
                      "error": failures.get(case.name), "applicable": True, "diagnostics": {}}
            if result is not None:
                labels = {name: label.value for name, label in result.labels.items()}
                record.update(
                    turns=[asdict(turn) | {"cached_percent": turn.usage.cached_percent if turn.usage else None} for turn in result.output.turns],
                    usage=asdict(result.output.usage) | {"cached_percent": result.output.usage.cached_percent} if result.output.usage else None,
                    assertions={name: assertion.value for name, assertion in result.assertions.items()},
                    duration=result.task_duration,
                    applicable=labels.get(APPLICABILITY) != NOT_APPLICABLE,
                    diagnostics={name.removeprefix(DIAGNOSTIC_PREFIX): value == "pass" for name, value in labels.items() if name.startswith(DIAGNOSTIC_PREFIX)},
                )
                if (recorded := _record_artifacts(result.output)) is not None:
                    artifacts.append({"name": case.name, "artifacts": recorded})
            cases_out.append(record)
        sidecar = artifacts_path(output)
        if artifacts:
            sidecar.write_text("".join(json.dumps(line, default=str) + "\n" for line in artifacts), encoding="utf-8")
        else:
            sidecar.unlink(missing_ok=True)
        output.write_text(json.dumps({
            "run": {"timestamp": datetime.now(UTC).isoformat(), "commit": commit, "dirty": dirty, "dataset": selected.name,
                    "case_fingerprint": case_fingerprint(cases, fixtures, instructions, instruction_version, settings, subject_fingerprint),
                    "settings": settings, "evaluator_version": EVALUATOR_VERSION, "suite_instructions": instructions,
                    "instruction_version": instruction_version, **model_metadata(llm),
                    "subject": subject.name, "subject_fingerprint": subject_fingerprint, "scoring": scoring},
            "cases": cases_out,
            "failures": [{"name": failure.name, "error": failure.error_message} for failure in report.failures],
        }, default=str, indent=2), encoding="utf-8")
    return report
