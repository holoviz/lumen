from __future__ import annotations

import asyncio
import hashlib
import json
import subprocess
import warnings

from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

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
from lumen.ai.schemas import get_metaset
from lumen.ai.ui import ExplorerUI
from lumen.config import SOURCE_TABLE_SEPARATOR
from lumen.pipeline import Pipeline
from lumen.util import as_pandas

MAX_ROWS = 20
MAX_ANSWER_LENGTH = 4000
MAX_TOOL_RESULT_LENGTH = 4000
EVALUATOR_VERSION = 2
PRICING_PER_MILLION = {
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
    "z-ai/glm-5.3-flash": {"input": 0.04, "cached": 0.015, "output": 0.50},
    "anthropic/claude-sonnet-5": {"input": 2.00, "cached": 0.20, "output": 10.00},
}


@dataclass
class Usage:
    input_tokens: int
    output_tokens: int
    cached_tokens: int
    cost_usd: float | None

    @property
    def cached_percent(self) -> float | None:
        return 100 * self.cached_tokens / self.input_tokens if self.input_tokens else None


@dataclass
class ToolTrace:
    name: str
    arguments: dict[str, Any]
    result: str


class MeteredOpenAI(OpenAI):
    """Capture provider usage for every response, including streamed responses."""

    def __init__(self, **params):
        super().__init__(**params)
        self.usage: list[Usage] = []
        self.tool_calls: list[ToolTrace] = []

    async def _run_tool_calls(self, tool_instances, tool_calls, tool_contexts, messages):
        calls = [(name, dict(arguments), call_id) for name, arguments, call_id in map(self._parse_tool_call, tool_calls)]
        results = await super()._run_tool_calls(tool_instances, tool_calls, tool_contexts, messages)
        calls_by_id = {call_id: (name, arguments) for name, arguments, call_id in calls}
        for msg in results:
            name, arguments = calls_by_id.get(msg.get("tool_call_id"), (msg["name"], {}))
            self.tool_calls.append(ToolTrace(name, arguments, str(msg["content"])[:MAX_TOOL_RESULT_LENGTH]))
        return results

    def _record_usage(self, response):
        usage = getattr(response, "usage", None)
        if usage is None:
            return
        model = getattr(response, "model", None) or self._resolved_model
        prices = PRICING_PER_MILLION.get(model) or PRICING_PER_MILLION.get(self.model_kwargs["default"]["model"])
        input_tokens = getattr(usage, "input_tokens", None)
        if input_tokens is None:
            input_tokens = usage.prompt_tokens
        output_tokens = getattr(usage, "output_tokens", None)
        if output_tokens is None:
            output_tokens = usage.completion_tokens
        details = getattr(usage, "input_tokens_details", None) or getattr(usage, "prompt_tokens_details", None)
        cached_tokens = getattr(details, "cached_tokens", 0) or 0
        cost = None if prices is None else ((input_tokens - cached_tokens) * prices["input"] + cached_tokens * prices["cached"] + output_tokens * prices["output"]) / 1_000_000
        self.usage.append(Usage(input_tokens, output_tokens, cached_tokens, cost))

    def _create_base_client(self, **kwargs):
        client = super()._create_base_client(**kwargs)
        if self.api == "responses":
            create = client.responses.create

            async def metered_create(**params):
                response = await create(**params)
                if not params.get("stream"):
                    self._record_usage(response)
                    return response

                async def events():
                    async for event in response:
                        if event.type == "response.completed":
                            self._record_usage(event.response)
                        yield event

                return events()

            client.responses.create = metered_create
        else:
            create = client.chat.completions.create

            async def metered_create(*args, **params):
                if self.model_kwargs["default"]["model"].startswith("qwen/qwen3.8-"):
                    params["extra_body"] = {**params.get("extra_body", {}), "reasoning": {"enabled": False}}
                if params.get("stream"):
                    params["stream_options"] = {"include_usage": True}

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
                            response = await create(*args, **params)
                            self._record_usage(response)
                            return response
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
                                self._record_usage(chunk)
                                yield chunk
                            return
                        except RateLimitError as exc:
                            if yielded or attempt == 4:
                                raise
                            await wait_for_retry(exc, attempt)

                return events()

            client.chat.completions.create = metered_create
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
    tool_calls: list[ToolTrace] | None = None
    task_errors: list[str] | None = None
    planner_actors: list[str] | None = None
    follow_up_type: str | None = None
    document_summary: str | None = None
    validation_correct: bool | None = None
    validation_missing: list[str] | None = None


@dataclass
class Output:
    turns: list[Turn]
    usage: Usage | None = None


@dataclass
class Expected:
    status: str = "success"
    actors: list[str] | None = None
    rows: list[list[Any]] | None = None
    view_type: str | None = None
    columns: list[str] | None = None
    answer_contains: str | None = None
    listing_contains: str | None = None
    chart_mark: str | None = None
    chart_fields: dict[str, str] | None = None
    tool_calls: list[str] | None = None
    tool_arguments: dict[str, dict[str, Any]] | None = None
    tool_result_contains: dict[str, list[str]] | None = None
    turn_rows: list[list[list[Any]] | None] | None = None
    turn_tables: list[str] | None = None
    sql_contains: list[str] | None = None
    turn_planner_actors: list[list[str]] | None = None
    forbidden_actors: list[str] | None = None
    follow_up_types: list[str] | None = None
    planned_actors: list[str] | None = None
    document_contains: list[str] | None = None
    validation_correct: bool | None = None
    gold_sql: str | None = None
    error_contains: str | None = None


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


def _direct_plan(inputs: Inputs, llm: Any, context: dict, interface: ChatInterface, prompt: str) -> Plan:
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
        agent = agent_type(llm=llm, **({"n_doc_pages": 0} if name == "VegaLiteAgent" else {}))
        tasks.append(ActorTask(agent, title=name))
    return Plan(*tasks, title="Direct agent evaluation", context=context,
                history=[{"role": "user", "content": prompt}], llm=llm, interface=interface)


def _error_message(exc: Exception) -> str:
    messages = []
    while exc is not None:
        messages.append(f"{type(exc).__name__}: {exc}")
        exc = exc.__cause__
    return " | caused by ".join(messages)


async def run_case(inputs: Inputs, llm: Any, source: Any, documents: list[Any] | None = None) -> Output:
    if inputs.agents:
        context = {"source": source, "sources": [source]}
        interface = ChatInterface()
    else:
        ui = ExplorerUI(data=source, llm=llm)
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
        context.update(pipeline=pipeline, data=pipeline.data, table="result")
    turns = []
    for prompt in inputs.prompts:
        before = len(interface.objects)
        usage_start = len(getattr(llm, "usage", []))
        tools_start = len(getattr(llm, "tool_calls", []))
        previous_plan = None if inputs.agents else ui._exploration["view"].plan
        previous_tasks = tuple(previous_plan) if previous_plan is not None else ()
        plan = None
        interface.send(prompt, user="User", respond=False)
        try:
            if inputs.agents:
                plan = _direct_plan(inputs, llm, context, interface, prompt)
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
            calls = getattr(llm, "usage", [])[usage_start:]
            turn.tool_calls = list(getattr(llm, "tool_calls", [])[tools_start:])
            if calls:
                costs = [call.cost_usd for call in calls]
                turn.usage = Usage(sum(call.input_tokens for call in calls), sum(call.output_tokens for call in calls), sum(call.cached_tokens for call in calls), sum(costs) if all(cost is not None for cost in costs) else None)
        turns.append(turn)
        if turn.status in ("error", "timeout"):
            break
    calls = [turn.usage for turn in turns if turn.usage is not None]
    costs = [call.cost_usd for call in calls]
    total = Usage(sum(call.input_tokens for call in calls), sum(call.output_tokens for call in calls), sum(call.cached_tokens for call in calls), sum(costs) if all(cost is not None for cost in costs) else None) if calls else None
    return Output(turns, total)


@dataclass
class CheckResult(Evaluator[Inputs, Output, Expected]):
    def evaluate(self, ctx: EvaluatorContext[Inputs, Output, Expected]) -> dict[str, bool]:
        expected = ctx.metadata or Expected()
        if not ctx.output.turns:
            return {"all_turns_completed": False, "status": False}
        last = ctx.output.turns[-1]
        checks = {
            "all_turns_completed": len(ctx.output.turns) == len(ctx.inputs.prompts),
            "status": last.status == expected.status and (expected.status != "success" or all(turn.status == "success" and all(status == "success" for status in turn.task_statuses) for turn in ctx.output.turns)),
        }
        if expected.status == "success":
            checks["output"] = any(
                turn.answer is not None or turn.rows is not None or turn.listing is not None or bool(turn.view_types)
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


def case_fingerprint(cases: list[Case], fixtures: dict[str, Any] | None = None) -> str:
    definitions = [
        {"name": case.name, "inputs": asdict(case.inputs), "expected": asdict(case.metadata or Expected())}
        for case in cases
    ]
    payload = json.dumps({"evaluator_version": EVALUATOR_VERSION, "cases": definitions, "fixtures": fixtures}, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(payload.encode()).hexdigest()


async def evaluate(llm: Any, dataset: Dataset, source_factory: Any, output: Path | None = None,
                   case_name: str | None = None, documents_factory: Any = None, fixtures: dict[str, Any] | None = None):

    async def task(inputs: Inputs) -> Output:
        try:
            return await run_case(inputs, llm, source_factory(inputs),
                                  documents=documents_factory(inputs) if documents_factory else None)
        except Exception as exc:
            prompt = inputs.prompts[0] if inputs.prompts else ""
            error = _error_message(exc)
            return Output([Turn(prompt, "error", None, [], [], None, None, None, None, [], task_errors=[error])])

    cases = [case for case in dataset.cases if case.name == case_name] if case_name else dataset.cases
    if not cases:
        raise ValueError(f"Unknown eval case: {case_name}")
    selected = Dataset(name=dataset.name, cases=cases, evaluators=dataset.evaluators)
    report = await selected.evaluate(task, max_concurrency=1)
    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=False).stdout.strip()
        dirty = bool(subprocess.run(["git", "status", "--porcelain", "--untracked-files=all", "--", "lumen/ai/evals", "lumen/ai/agents/sql.py", "lumen/ai/agents/vega_lite.py", "lumen/ai/coordinator", "lumen/ai/llm.py", "lumen/ai/llm_dialog.py", "tests/evals", "pixi.toml"], capture_output=True, text=True, check=False).stdout.strip())
        completed = {case.name: case for case in report.cases}
        failures = {failure.name: failure.error_message for failure in report.failures}
        output.write_text(json.dumps({
            "run": {"timestamp": datetime.now(UTC).isoformat(), "commit": commit, "dirty": dirty, "dataset": selected.name, "case_fingerprint": case_fingerprint(cases, fixtures), "evaluator_version": EVALUATOR_VERSION, "model": llm.model_kwargs.get("default", {}).get("model"), "api": getattr(llm, "api", None), "provider": "openrouter" if getattr(llm, "endpoint", None) == "https://openrouter.ai/api/v1" else "openai"},
            "cases": [
                {"name": case.name, "turns": [asdict(turn) | {"cached_percent": turn.usage.cached_percent if turn.usage else None} for turn in completed[case.name].output.turns] if case.name in completed else [], "usage": asdict(completed[case.name].output.usage) | {"cached_percent": completed[case.name].output.usage.cached_percent} if case.name in completed and completed[case.name].output.usage else None, "assertions": {name: result.value for name, result in completed[case.name].assertions.items()} if case.name in completed else {}, "duration": completed[case.name].task_duration if case.name in completed else None, "error": failures.get(case.name)}
                for case in cases
            ],
            "failures": [{"name": failure.name, "error": failure.error_message} for failure in report.failures],
        }, default=str, indent=2), encoding="utf-8")
    return report
