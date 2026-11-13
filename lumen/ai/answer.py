"""
Ask a Lumen Source or Pipeline a question from Python and get an Answer.
"""
from __future__ import annotations

import asyncio
import base64
import hashlib
import html
import importlib.util
import inspect
import json
import logging
import os
import re
import sys
import threading
import time
import warnings
import weakref

from collections import OrderedDict
from collections.abc import Callable
from concurrent.futures import TimeoutError as FutureTimeoutError
from contextlib import (
    ExitStack, contextmanager, nullcontext, suppress,
)
from dataclasses import dataclass
from functools import cache
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

import hvplot.pandas  # noqa: F401  registers .hvplot, used by answer.map
import nbformat
import pandas as pd
import panel as pn
import param
import sqlglot
import yaml

from markdown_it import MarkdownIt
from panel.config import panel_extension
from panel.io.state import state
from panel.pane import Markdown, Vega
from panel.pane.vega import ds_to_records
from panel.widgets import CodeEditor
from panel_material_ui import (
    Alert, Button, Card, ChatInterface, Column, Row,
)
from panel_material_ui.chat import ChatMessage
from pydantic import TypeAdapter, ValidationError, create_model
from sqlglot import exp

from .. import __version__
from ..config import SOURCE_TABLE_SEPARATOR, config
from ..pipeline import Pipeline
from ..serializers import Serializer
from ..sources import Source
from ..sources.base import BaseSQLSource
from ..sources.data import input_kinds, joins_grids, read
from ..sources.duckdb import DuckDBSource
from ..sources.xarray_sql import XArraySQLSource
from ..transforms.sql import SQLLimit
from ..util import (
    as_pandas, geometry_columns, geometry_to_geojson, is_geodataframe, log,
)
from ..views import VegaLiteView, View, hvPlotView
from ..views.base import GRIDDED_KINDS, hvPlotBaseView, hvPlotUIView
from .actor import step_title_callback
from .agents import ChatAgent, SQLAgent, VegaLiteAgent
from .agents.hvplot import LAT_NAMES, LON_NAMES, hvPlotAgent
from .config import PROMPTS_DIR, QueryError, RetriesExceededError
from .coordinator import Plan, Planner
from .editors import LumenEditor
from .export import format_output, make_md_cell, make_preamble
from .llm import (
    Llm, get_available_llm, llm_class, quiet_retries, signed_in_cli_hint,
)
from .report import ActorTask, Report, TaskGroup
from .selection import about, ask_box, watch_clicks
from .sharing import mask_frame, sharing
from .tools import MetadataLookup
from .ui import ExplorerUI
from .usage import UsageCollector
from .utils import errors_in_result, format_error

if TYPE_CHECKING:
    from ..sources.data import DataT


# HTML off: the answer is LLM output and goes into the notebook's text/html
_MARKDOWN = MarkdownIt("gfm-like", {"html": False})


def _in_kernel() -> bool:
    """Whether code runs in a Jupyter kernel, where Panel output gets comms."""
    try:
        from IPython import get_ipython
    except ImportError:
        return False
    try:
        return get_ipython().kernel is not None
    except AttributeError:  # no IPython at all, or a terminal IPython
        return False


def _load_extensions() -> None:
    """
    Load what answer cards need, on first display rather than on import, so
    lumen.ai stays quiet in scripts and in front-ends that never show one.
    Checked per extension: a plain pn.extension() leaves Vega out, and the
    Spec section's editor needs codeeditor.
    """
    missing = [ext for ext in ("vega", "codeeditor") if ext not in panel_extension._loaded_extensions]
    if missing:
        pn.extension(*missing)


def _is_spatial(source: Source) -> bool:
    """Whether a table has geometry, or longitude and latitude columns."""
    # Only SQL sources can be probed without loading data. DuckDB returns
    # GEOMETRY as a GeoDataFrame only when geopandas is installed, which is
    # also what hvPlot needs to draw it.
    if not isinstance(source, BaseSQLSource):
        return False
    for table in source.get_tables():
        try:
            # Aliased, as Postgres and MySQL require for a subquery
            empty = source.execute(f"SELECT * FROM ({source.get_sql_expr(table)}) AS probe LIMIT 0")
        except Exception as e:
            log.debug(f"Could not probe {table!r} for geometry: {e}")
            continue
        columns = {str(c).lower() for c in empty.columns}
        if geometry_columns(empty) or (columns & LON_NAMES and columns & LAT_NAMES):
            return True
    return False


async def _default_agents(source: Source) -> list:
    agents = [ChatAgent(), SQLAgent(), VegaLiteAgent()]
    # Maps and grids can go to hvPlot, which draws them on basemap tiles and
    # rasterizes large grids; Vega-Lite stays for every other chart, and for
    # refine. The probe queries every table, which can be slow on a remote database.
    if isinstance(source, XArraySQLSource) or await asyncio.to_thread(_is_spatial, source):
        agents.append(hvPlotAgent())
    return agents


# Set through configure, and used by every question that does not pass its own
_settings: dict[str, Any] = {
    "agents": None, "llm": None,
    "cache": os.environ.get("LUMEN_AI_CACHE", "").lower() in ("1", "true", "yes"),
}

# Finished answers by question, data, model and prompt version, kept for
# configure(cache=True). A cached Answer holds its source, so its id stays
# unique for as long as the entry lives.
_answer_cache: OrderedDict[str, Answer] = OrderedDict()
_ANSWER_CACHE_SIZE = 128
# Each cached answer holds its result, so the rows across them are bounded too
_ANSWER_CACHE_ROWS = 2_000_000

_UNSET: Any = object()


def configure(
    *, llm: Llm | None = _UNSET, agents: list | None = _UNSET, max_rows: int | None = _UNSET,
    share: Literal["sample", "schema", "none"] | None = None, mask_pii: bool | None = None,
    cache: bool | None = None,
) -> dict[str, Any]:
    """
    Set the defaults for ask, aask and chat in this session.

    A question's own llm= or agents= still wins, and None goes back to
    the automatic choice. Without either set here, LUMEN_AI_PROVIDER (e.g. "kilo") and LUMEN_AI_MODEL pick the
    LLM, and otherwise the first provider with an API key set.

    max_rows caps the rows an answer's data holds (1,000,000 by
    default), as a LIMIT on its query; None lifts the cap.

    share sets what of the data prompts contain: "sample" (values, as
    before), "schema" (structure, no values) or "none" (names and row counts only).
    mask_pii withholds the values of columns that look like personal
    data. Both apply to the chat UI too; see lumen.ai.sharing.

    cache keeps answers in memory, so asking the same question of the same
    data with the same model again returns the answer without LLM calls.
    LUMEN_AI_CACHE=1 turns it on too; clear_cache() empties it.

    Returns the settings now in effect.
    """
    if llm is not _UNSET:
        _settings["llm"] = llm
    if agents is not _UNSET:
        _settings["agents"] = agents
    if cache is not None:
        _settings["cache"] = cache
    if max_rows is not _UNSET:
        # The class default, so every SQLAgent without its own value uses it
        SQLAgent.max_rows = max_rows
    sharing.param.update({k: v for k, v in dict(share=share, mask_pii=mask_pii).items() if v is not None})
    return {**_settings, "mask_pii": sharing.mask_pii, "max_rows": SQLAgent.max_rows, "share": sharing.share}


def _clone(obj: Any, **overrides) -> Any:
    """A new instance of a Lumen component with the same parameter values, or those given."""
    # None is left out so the parameter takes its default: a list parameter's default is [], not None
    values = {key: value for key, value in obj.param.values().items() if key != "name" and value is not None}
    return type(obj)(**{**values, **overrides})


def _cached_rows() -> int:
    # The data is already loaded, so its length costs no query
    return sum(len(answer.pipeline.data) for answer in _answer_cache.values() if answer.pipeline is not None)


def clear_cache() -> None:
    """Forget the answers configure(cache=True) kept."""
    _answer_cache.clear()


@cache
def _prompt_version() -> str:
    """Lumen's version and a hash of its prompt templates, read once per process."""
    sha = hashlib.sha256(__version__.encode())
    for path in sorted(PROMPTS_DIR.rglob("*.jinja2")):
        sha.update(path.read_bytes())
    return sha.hexdigest()


_SECRET_PARAM = re.compile(r"key|secret|token|password|credential", re.IGNORECASE)


def _fingerprint(obj: param.Parameterized) -> str:
    """
    The plain settings of a component, e.g. an LLM's endpoint and temperature
    or an agent's own limits, as a string. Secrets and objects such as the
    interface or another LLM are left out.
    """
    values = obj.param.values()
    settings = {
        key: value for key, value in values.items()
        # Private params, e.g. _ready, are state, not settings
        if key != "name" and not key.startswith("_") and not _SECRET_PARAM.search(key)
        and isinstance(value, (str, int, float, bool, type(None), list, dict))
    }
    # An agent can bring an LLM of its own, which answers in place of the Planner's
    llm = values.get("llm")
    own_llm = _fingerprint(llm) if isinstance(llm, Llm) else ""
    return type(obj).__name__ + json.dumps(settings, sort_keys=True, default=str) + own_llm


def _type_key(returns: Any) -> str:
    """
    The type returns= asks for, by its schema and identity: a pydantic class
    redefined in a notebook keeps its repr but is another class.
    """
    if returns is None:
        return ""
    return f"{id(returns)}:{json.dumps(TypeAdapter(returns).json_schema(), sort_keys=True)}"


def _cache_key(
    source: Source, question: str, llm: Llm, agents: list, visible_slugs: set[str] | None, returns: Any = None,
) -> str:
    """What decides an answer: the question, the data, the model, the agents, the prompts and the type."""
    parts = (
        " ".join(question.lower().split()), str(id(source)), source._get_source_hash(),
        # Columns, types, row counts and descriptions, so a schema change or describe() misses
        json.dumps(source.get_metadata(None), sort_keys=True, default=str),
        _fingerprint(llm), json.dumps(llm.model_kwargs, sort_keys=True, default=str),
        *(_fingerprint(agent) for agent in agents), str(SQLAgent.max_rows), sharing.share, str(sharing.mask_pii),
        str(sorted(visible_slugs or [])), _prompt_version(), _type_key(returns),
    )
    return hashlib.sha256("\0".join(parts).encode()).hexdigest()


def _without_values(spec: Any) -> Any:
    """A Vega-Lite spec without its inline data, at any depth (layers, concats)."""
    if isinstance(spec, list):
        return [_without_values(item) for item in spec]
    if not isinstance(spec, dict):
        return spec
    return {
        key: {k: v for k, v in value.items() if k != "values"} if key == "data" and isinstance(value, dict)
        else _without_values(value)
        for key, value in spec.items() if key != "datasets"
    }


def _row_cap_hit(pipeline: Pipeline | None) -> int | None:
    """The row cap the pipeline's data stopped at, if the cap cut it short."""
    limits = [] if pipeline is None else [
        t.limit for t in pipeline.sql_transforms if isinstance(t, SQLLimit) and t.limit is not None
    ]
    # The run has already loaded the data, so its length costs no query
    if not limits or len(pipeline.data) < limits[0]:
        return None
    # Exactly at the cap: one row more tells a cut-off result from one that
    # fits. Only this case queries again; SQLLimit writes the source's dialect.
    over = SQLLimit(limit=limits[0] + 1, write=pipeline.source.dialect)
    try:
        rows = pipeline.source.get(pipeline.table, sql_transforms=[over])
    except Exception as e:
        log.debug(f"Could not check whether the answer was cut at {limits[0]} rows: {e}")
        return None
    return limits[0] if len(rows) > limits[0] else None


def _default_llm() -> Llm:
    if _settings["llm"] is not None:
        return _settings["llm"]
    return _env_llm(os.environ.get("LUMEN_AI_PROVIDER"), os.environ.get("LUMEN_AI_MODEL"))


# One instance for every ask, so its client and the request fixes it
# learns from provider errors carry over from question to question
@cache
def _env_llm(provider: str | None, model: str | None) -> Llm:
    llm_cls = get_available_llm() if provider is None else llm_class(provider)
    if llm_cls is None:
        raise RuntimeError(
            "No LLM provider found. Pass llm=lumen.ai.llm.Kilo() for free models without an "
            f"account{signed_in_cli_hint()}, or set an API key such as OPENAI_API_KEY or "
            "GROQ_API_KEY and restart the kernel."
        )
    # The same model_kwargs the CLI builds for --model
    return llm_cls(model_kwargs={"default": {"model": model}}) if model else llm_cls()


_LOOP_LOCK = threading.Lock()
_LOOP_THREAD = "lumen-ask"


@cache
def _new_loop() -> asyncio.AbstractEventLoop:
    loop = asyncio.new_event_loop()
    threading.Thread(target=loop.run_forever, name=_LOOP_THREAD, daemon=True).start()
    return loop


def _loop() -> asyncio.AbstractEventLoop:
    """
    The event loop every question runs on, ask's and aask's alike. The
    cached LLM client and Planners hold connections and asyncio objects
    bound to the loop they first ran on, so one loop has to serve them all.
    """
    # Locked so two threads asking at once do not each start a loop
    with _LOOP_LOCK:
        return _new_loop()


# One Planner per Source, freed with it. Building a Planner and syncing its
# metadata costs about half a second, and a reused one sends the LLM exactly
# the prompts a fresh one would.
_planners: weakref.WeakKeyDictionary[Source, Planner] = weakref.WeakKeyDictionary()

# Planners answering a question right now. respond keeps per-question state
# on the Planner, so a concurrent question gets one of its own.
_busy: weakref.WeakSet[Planner] = weakref.WeakSet()


def _after_fork() -> None:
    # The loop's thread does not survive a fork, and everything cached
    # holds objects bound to that loop, so the child starts over
    _new_loop.cache_clear()
    _planners.clear()
    _env_llm.cache_clear()
    _LOOP_LOCK.release()


if os.name == "posix":  # Windows cannot fork
    # Held across the fork, so the child never inherits it locked
    os.register_at_fork(
        before=_LOOP_LOCK.acquire, after_in_parent=_LOOP_LOCK.release, after_in_child=_after_fork,
    )


async def _planner(source: Source, llm: Llm, agents: list | None) -> Planner:
    if agents is not None:
        return Planner(llm=llm, agents=agents, headless=True)
    planner = _planners.get(source)
    if planner is not None and planner.llm is llm and planner not in _busy:
        return planner
    planner = Planner(llm=llm, agents=await _default_agents(source), headless=True)
    if (cached := _planners.get(source)) is None or cached not in _busy:
        _planners[source] = planner
    return planner


def _drop_planner(source: Source) -> None:
    """Drop a source's cached Planner, and what its metadata lookup synced, so new rows are read afresh."""
    if (planner := _planners.pop(source, None)) is not None:
        MetadataLookup.forget(planner.context["vector_store"])


def _resolve(data: DataT | Pipeline) -> tuple[Source, set[str] | None]:
    """Return the Source to ask and the table slugs it is limited to."""
    if not isinstance(data, Pipeline):
        return (data if isinstance(data, Source) else read(data)), None
    if data.filters or data.transforms or data.sql_transforms:
        # The rows the pipeline selects have to be what gets queried. The
        # copy follows the pipeline's updates, so one per pipeline and table
        # is enough, and it is freed with the pipeline.
        table, source, watcher = data.__dict__.get("_ask_source", (None, None, None))
        if source is None or table != data.table:
            if source is not None:
                # The old table's copy keeps what it holds for earlier answers but stops following
                source.stop_mirroring()
                data.param.unwatch(watcher)
                _drop_planner(source)
            source = read({data.table: data})
            # The Planner's metadata describes the rows of the first
            # question, so new rows need a new Planner
            watcher = data.param.watch(lambda event, source=source: _drop_planner(source), "data")
            data._ask_source = (data.table, source, watcher)
        return source, None
    return data.source, {f"{data.source.name}{SOURCE_TABLE_SEPARATOR}{data.table}"}


def _fresh_context(planner: Planner, source: Source, visible_slugs: set[str] | None) -> dict[str, Any]:
    # Starts from the Planner's own context, which holds only its shared
    # vector stores, so nothing one question adds is seen by an unrelated one
    context: dict[str, Any] = {**planner.context, "source": source, "sources": [source]}
    if visible_slugs is not None:
        context["visible_slugs"] = visible_slugs
    return context


@contextmanager
def _asking(planner: Planner):
    """
    Mark the planner busy, so a concurrent question gets one of its own, hand
    errors back in the Answer rather than printing their tracebacks, and quiet
    instructor's per-attempt retry log, for the length of the block.
    """
    _busy.add(planner)
    token = errors_in_result.set(True)
    try:
        with quiet_retries() if log.getEffectiveLevel() > logging.DEBUG else nullcontext():
            yield
    finally:
        errors_in_result.reset(token)
        _busy.discard(planner)


@contextmanager
def _counting(planner: Planner):
    """Collect the usage of every LLM the planner and its agents use, each LLM once."""
    llms = {id(llm): llm for llm in (planner.llm, *(agent.llm for agent in planner.agents)) if llm is not None}
    usage = UsageCollector()
    with ExitStack() as stack:
        scopes = [stack.enter_context(llm.capture_usage()) for llm in llms.values()]
        try:
            yield usage
        finally:
            for scope in scopes:
                for record in scope.records:
                    usage.add(record)


async def _respond(
    planner: Planner, question: str, messages: list[dict[str, str]], context: dict[str, Any],
    data: DataT | Pipeline, agents: list | None, confirm: bool = False,
) -> tuple[Answer, dict[str, Any]]:
    """
    Run one question through the Planner, returning the Answer and the
    context it left. A failed plan gives an Answer with status "error".
    With confirm, stop after planning and return a "planned" Answer.
    """
    with _asking(planner), _counting(planner) as usage:
        await planner.sync(context)
        plan = await planner.respond(messages, context)
    if plan is None:
        raise RuntimeError(f"Lumen could not plan an answer to {question!r}.")
    if confirm and not plan.unanswerable:
        answer = Answer(
            agents=agents, data=data, llm=planner.llm, plan=plan, question=question,
            source=context["source"], status="planned", usage=usage,
            visible_slugs=context.get("visible_slugs"),
        )
        answer._pending = (planner, context)
        return answer, context
    return await _execute(planner, plan, question, context, data, agents, usage)


async def _execute(
    planner: Planner, plan: Plan, question: str, context: dict[str, Any], data: DataT | Pipeline,
    agents: list | None, usage: UsageCollector,
) -> tuple[Answer, dict[str, Any]]:
    """Run a plan and build its Answer, adding the run's usage to the planning's."""
    with _asking(planner), _counting(planner) as run_usage:
        outputs, out_context = ([], context) if plan.unanswerable else await plan.execute()
    answer = _build_answer(
        question, plan, outputs, out_context, context["source"], planner.llm, data, agents,
        context.get("visible_slugs"),
    )
    answer.usage = UsageCollector()
    for record in (*usage.records, *run_usage.records):
        answer.usage.add(record)
    return answer, out_context


def _shapes(df: pd.DataFrame | None) -> list:
    """
    The result in each shape a type could take it in: the cell of a
    one-cell result, the values of a one-column one, the row of a one-row
    one, and the rows as records. Only the shapes the result has.
    """
    if df is None or df.empty:
        return []
    # object dtype hands back Python numbers rather than numpy ones
    records = df.astype(object).where(df.notna(), None).to_dict("records")
    shapes: list = []
    if len(df.columns) == 1:
        column = [row[df.columns[0]] for row in records]
        shapes += [column[0]] if len(column) == 1 else []
        shapes.append(column)
    if len(records) == 1:
        shapes.append(records[0])
    shapes.append(records)
    return shapes


async def _typed(answer: Answer, returns: Any, llm: Llm) -> None:
    """
    Fill answer.result with the answer as the type asked for. A result
    already in a shape the type takes is validated as it is, with no LLM
    call; otherwise one structured call reads the type from the written
    answer. An answer that cannot be read as the type becomes an error.
    """
    answer._returns = returns
    if answer.status != "success":
        return
    adapter = TypeAdapter(returns)
    df = None if answer.pipeline is None else as_pandas(answer.pipeline.data)
    for shape in _shapes(df):
        with suppress(ValidationError):
            answer.result = adapter.validate_python(shape)
            return
    name = returns.__name__ if isinstance(returns, type) else str(returns)
    content = f"Question: {answer.question}\n\nAnswer: {answer.value}"
    if df is not None and sharing.share == "sample":
        # Same text on every platform, so a recorded prompt replays everywhere
        csv = mask_frame(df.head(20)).to_csv(index=False, lineterminator="\n")
        content += f"\n\nThe data the answer was written from:\n{csv}"
    try:
        with llm.capture_usage() as usage:
            out = await llm.invoke(
                [{"role": "user", "content": content}],
                system=f"Give the answer as {name}, using only the answer and data given.",
                response_model=create_model("Result", result=(returns, ...)),
            )
    except Exception as e:
        answer.param.update(status="error", error=f"The answer could not be read as {name}: {format_error(e)}")
        return
    for record in usage.records:
        answer.usage.add(record)
    answer.result = out.result


def _read_kinds(source: Source | None, sql: str | None) -> set[str]:
    """The libraries of the tables a query read, as lumen.read recorded them."""
    kinds = input_kinds(source) if source is not None else {}
    if None in kinds or not kinds:
        return {kinds.get(None, "pandas")}
    read = set(kinds)
    if sql:
        with suppress(sqlglot.errors.ParseError):
            read &= {table.name for table in sqlglot.parse_one(sql, read="duckdb").find_all(sqlglot.exp.Table)}
    return {kinds[table] for table in (read or kinds)}


def _as_input_type(df: pd.DataFrame, kinds: set[str]) -> Any:
    """A result in the library its input came in: polars, pyarrow or pandas."""
    # Input of that type means its library is already imported
    if kinds == {"polars"}:
        return sys.modules["polars"].from_pandas(df)
    if kinds == {"pyarrow"}:
        return sys.modules["pyarrow"].Table.from_pandas(df, preserve_index=False)
    return df


def _literal(node: exp.Expression | None) -> exp.Expression | None:
    """The literal a value is, inside a cast such as DATE '2024-01-01' too, or None."""
    inner = node.this if isinstance(node, exp.Cast) else node
    if isinstance(inner, exp.Literal) or (isinstance(inner, exp.Neg) and isinstance(inner.this, exp.Literal)):
        return inner
    return None


def _literals(node: exp.Expression) -> tuple[str, Any] | None:
    """
    What a comparison compares its column with, as (kind, values): one
    value, a BETWEEN pair or an IN list. None when that is not literals.
    """
    if isinstance(node, exp.In):
        literals = [_literal(e) for e in node.expressions]
        return None if not literals or None in literals else ("in", tuple(e.to_py() for e in literals))
    if isinstance(node, exp.Between):
        low, high = _literal(node.args["low"]), _literal(node.args["high"])
        return None if low is None or high is None else ("between", (low.to_py(), high.to_py()))
    literal = _literal(node.expression)
    return None if literal is None else ("value", literal.to_py())


def _parameterize(node: exp.Expression, kind: str, name: str) -> None:
    """Put the named parameter where the comparison had its literals."""
    if kind == "in":
        # One list parameter, whatever the number of values asked for later
        node.replace(exp.func("list_contains", exp.Placeholder(this=name), node.this.copy(), dialect="duckdb"))
    elif kind == "between":
        _literal(node.args["low"]).replace(exp.Placeholder(this=f"{name}_low"))
        _literal(node.args["high"]).replace(exp.Placeholder(this=f"{name}_high"))
    else:
        _literal(node.expression).replace(exp.Placeholder(this=name))


def _binder(kind: str, name: str) -> Callable[[Any], dict[str, Any]]:
    """How an argument's value becomes the query's parameters."""
    if kind == "between":
        return lambda value: {f"{name}_low": value[0], f"{name}_high": value[1]}
    if kind == "in":
        return lambda value: {name: list(value)}
    return lambda value: {name: value}


#: Rows a Vega-Lite scatter or line chart draws before the answer draws a
#: density image instead; Vega-Lite inlines every row and slows above this
LARGE_POINTS = 5_000

_POINT_KINDS = {"point": "scatter", "circle": "scatter", "square": "scatter", "tick": "scatter", "line": "line"}


def _point_chart(view: View) -> tuple[str, dict] | None:
    """For a single Vega-Lite scatter or line chart of raw rows, its hvPlot kind and encoding."""
    if not isinstance(view, VegaLiteView):
        return None
    spec = view.spec
    # A transform filters or computes fields, which the raw rows do not have
    if any(key in spec for key in ("layer", "facet", "repeat", "concat", "hconcat", "vconcat", "transform")):
        return None
    mark = spec.get("mark")
    kind = _POINT_KINDS.get(mark.get("type") if isinstance(mark, dict) else mark)
    encoding = spec.get("encoding", {})
    axes = [encoding.get(axis) for axis in ("x", "y")]
    if kind is None or not all(isinstance(a, dict) and a.get("field") for a in axes):
        return None
    if any(key in a for a in axes for key in ("aggregate", "bin")):
        return None
    return kind, encoding


def _density_view(view: View, pipeline: Pipeline | None) -> View:
    """
    A Vega-Lite scatter or line chart of more than LARGE_POINTS rows, drawn
    instead with hvPlot and datashader as a density image, which stays fast
    however many rows there are. Coloured by a category, the categories are
    blended; otherwise each pixel counts its points. Anything else, or
    without datashader, is returned as it is.
    """
    chart = _point_chart(view)
    if chart is None or pipeline is None or len(pipeline.data) <= LARGE_POINTS:
        return view
    if importlib.util.find_spec("datashader") is None:
        return view
    kind, encoding = chart
    params: dict[str, Any] = dict(
        pipeline=view.pipeline, kind=kind, x=encoding["x"]["field"], y=encoding["y"]["field"],
        responsive=True, height=400, title=view.spec.get("title", ""),
    )
    color = encoding.get("color", {})
    if color.get("field") and color.get("type") == "nominal":
        # rasterize keeps one number per pixel and loses the category
        params.update(by=color["field"], datashade=True)
    else:
        params.update(rasterize=True, cnorm="log")
    return hvPlotView(**params)


_EXPLORER_KINDS = {**_POINT_KINDS, "bar": "bar", "area": "area", "rect": "heatmap", "boxplot": "box"}


def _vega_explorer_settings(spec: dict) -> dict[str, Any]:
    """hvPlot equivalents of a Vega-Lite chart's mark and encoding."""
    spec = spec["layer"][0] if spec.get("layer") else spec
    mark = spec.get("mark")
    kind = _EXPLORER_KINDS.get(mark.get("type") if isinstance(mark, dict) else mark)
    if kind is None:
        raise ValueError("hvPlot has no kind for this chart's Vega-Lite mark, so it cannot be tweaked.")
    encoding = spec.get("encoding", {})
    x, y, color = (encoding.get(channel, {}) for channel in ("x", "y", "color"))
    if "transform" in spec or any(key in c for c in (x, y, color) for key in ("aggregate", "bin", "timeUnit")):
        # The explorer plots the rows as they are; it would draw a different chart
        raise ValueError(
            "This chart aggregates or transforms its rows in Vega-Lite, which the explorer "
            "cannot redo; ask for the aggregation in the query, then tweak that answer."
        )
    if kind == "bar" and y.get("type") in ("nominal", "ordinal") and x.get("type") == "quantitative":
        # Vega-Lite draws horizontal bars by swapping the axes; hvPlot keeps the category on x
        kind, x, y = "barh", y, x
    settings = {"kind": kind, "x": x.get("field"), "y": y.get("field")}
    if color.get("field"):
        settings["by" if color.get("type") in ("nominal", "ordinal") else "color"] = color["field"]
    return {k: v for k, v in settings.items() if v is not None}


def _explorer_settings(view: View) -> dict[str, Any]:
    """The hvPlotUIView params that redraw an answer's chart."""
    if isinstance(view, VegaLiteView):
        return _vega_explorer_settings(view.spec)
    names = set(hvPlotBaseView.param) - set(View.param)
    settings = {k: v for k, v in view.param.values().items() if k in names and v not in (None, [], False)}
    if "z" in settings and view.kind not in GRIDDED_KINDS:
        # The explorer colours points and shapes by `color`; a grid takes its variable from the data
        settings["color"] = settings.pop("z")
    return dict(settings, **view.kwargs)


def _records(data: Any) -> Any:
    """
    A frame as JSON records, with missing values as null. JSON has no
    NaN, and in a stacked chart one NaN blanks every series at that x.
    Data that is not a frame, e.g. GeoJSON features, is returned as is.
    """
    records = ds_to_records(data)
    if records is None:
        return data
    return [
        {key: None if pd.api.types.is_scalar(value) and pd.isna(value) else value for key, value in row.items()}
        for row in records
    ]


class _StatusLine:
    """One output line, updated in place with the step a question is on."""

    def __init__(self, first: str):
        self._first = first
        self._handle = None
        self._shown = None
        self._start = time.monotonic()

    def __enter__(self) -> _StatusLine:
        if _in_kernel():
            # IPython is present, since this runs in a kernel
            from IPython.display import display
            self._handle = display({"text/plain": self._first}, raw=True, display_id=True)
        return self

    def show(self, title: str) -> None:
        text = f"{title.rstrip('.')} ({time.monotonic() - self._start:.0f}s)"
        if self._handle is not None and text != self._shown:
            self._handle.update({"text/plain": text}, raw=True)
            self._shown = text

    def __exit__(self, *exc) -> None:
        if self._handle is not None:
            self._handle.update({"text/plain": ""}, raw=True)


def _run(coro, first_step: str = "Planning"):
    """
    Run a coroutine to completion from sync code, inside Jupyter or not. It
    runs on Lumen's own background loop, an interrupt cancels it, and in
    Jupyter a status line shows first_step until a step reports itself.
    """
    if threading.current_thread().name == _LOOP_THREAD:
        coro.close()
        raise RuntimeError(
            "ask would wait on the loop it runs on and never return; use await aask instead."
        )
    titles: list[str] = []

    async def tracked():
        # Set inside the task, so the steps it runs inherit it
        step_title_callback.set(titles.append)
        return await coro

    future = asyncio.run_coroutine_threadsafe(tracked(), _loop())
    with _StatusLine(first_step) as status:
        try:
            while True:
                try:
                    return future.result(timeout=0.25)
                except FutureTimeoutError:
                    status.show(titles[-1] if titles else first_step)
        except KeyboardInterrupt:
            # Cancels the task on the loop, as the report stop button does
            future.cancel()
            raise


async def _on_loop(coro):
    """
    Await a coroutine on Lumen's loop, which the cached LLM client and
    Planners are bound to. Cancelling this await cancels it there too.
    """
    loop = _loop()
    if asyncio.get_running_loop() is loop:
        return await coro
    return await asyncio.wrap_future(asyncio.run_coroutine_threadsafe(coro, loop))


def _write_tables(spec: Any, data_dir: Path) -> None:
    """Swap each DataFrame serialized into a DuckDB source spec for a Parquet file."""
    if isinstance(spec, list):
        for item in spec:
            _write_tables(item, data_dir)
        return
    if not isinstance(spec, dict):
        return
    if spec.get("type") == "duckdb":
        tables = spec.setdefault("tables", {})
        mirrors = spec.get("mirrors", {})
        frames = {**{t: v for t, v in mirrors.items() if isinstance(v, dict) and "serializer" in v},
                  **{t: v for t, v in tables.items() if isinstance(v, dict)}}
        for table, serialized in frames.items():
            Serializer.deserialize(serialized).to_parquet(data_dir / f"{table}.parquet")
            # The script resolves $LUMEN_DATA to the data folder next to it
            tables[table] = f"SELECT * FROM read_parquet('$LUMEN_DATA/{table}.parquet')"
            mirrors.pop(table, None)
    for value in spec.values():
        _write_tables(value, data_dir)


def _drop_exceptions(group: TaskGroup) -> None:
    """Drop the exceptions failed tasks keep, in nested groups too."""
    for task in group:
        task.out_context.pop("__exception__", None)
        if isinstance(task, TaskGroup):
            _drop_exceptions(task)


def _build_answer(
    question: str, plan: Plan, outputs: list, context: dict[str, Any], source: Source, llm: Llm,
    data: DataT | Pipeline, agents: list | None, visible_slugs: set[str] | None = None,
) -> Answer:
    if plan.unanswerable:
        return Answer(
            agents=agents, data=data, error=plan.unanswerable, llm=llm, plan=plan, question=question,
            source=source, status="unanswerable", value=f"Lumen cannot answer {question!r} from this data.",
        )
    if plan.status == "error":
        failed = next((task.title for task in plan if task.status == "error"), plan.title)
        # Read once and dropped: its traceback keeps every frame's data alive
        exc = context.pop("__exception__", None)
        _drop_exceptions(plan)
        error = context.get("__error__", "")
        if isinstance(exc, RetriesExceededError) and exc.__cause__ is not None:
            # The last attempt's error says what went wrong; the wrapper does not
            error = format_error(exc.__cause__)
        sql_failed = isinstance(exc, QueryError)
        return Answer(
            agents=agents,
            data=data,
            error=error,
            llm=llm,
            # A later step failed, so the queried data is still worth having
            pipeline=None if sql_failed else context.get("pipeline"),
            plan=plan,
            question=question,
            source=source,
            sql=exc.sql if sql_failed else context.get("sql"),
            sql_failed=sql_failed,
            status="error",
            value=f"Lumen could not answer {question!r}: {failed!r} failed.",
        )
    messages = [out for out in outputs if isinstance(out, ChatMessage)]
    editors = [out for out in outputs if isinstance(out, LumenEditor) and isinstance(out.component, View)]
    editor = editors[-1] if editors else None
    view = None if editor is None else editor.component
    if view is not None and (dense := _density_view(view, context.get("pipeline"))) is not view:
        # The Vega-Lite editor cannot edit the hvPlot chart that replaced it
        view, editor = dense, None
    value = str(messages[-1].object) if messages else ""
    if not value.strip() and view is None and context.get("sql") is None:
        # A plan can finish without a step that queries or answers, e.g. one
        # that only looks up the tables; that is no answer, so say so
        return Answer(
            agents=agents, data=data, llm=llm, plan=plan, question=question, source=source,
            error="The plan finished without querying the data or writing an answer.",
            status="error", value=f"Lumen could not answer {question!r}.",
        )
    return Answer(
        agents=agents,
        context=context,
        data=data,
        editor=editor,
        llm=llm,
        visible_slugs=visible_slugs,
        pipeline=context.get("pipeline"),
        plan=plan,
        question=question,
        spec=view.spec if view is not None and view.view_type == "vegalite" else None,
        source=source,
        sql=context.get("sql"),
        status=plan.status,
        truncated_at=_row_cap_hit(context.get("pipeline")),
        value=value,
        view=view,
    )


async def aask(
    data: DataT | Pipeline, question: str, *, llm: Llm | None = None, agents: list | None = None,
    confirm: bool = False, returns: Any = None,
) -> Answer:
    """
    Ask a question of some data and wait for the Answer.

    Parameters
    ----------
    data: DataT | Pipeline
        A Source, a Pipeline (limits the question to its table and filters)
        or anything lumen.read accepts.
    question: str
        The question, in plain language.
    llm: Llm | None
        The LLM to use. Defaults to the first provider with an API key set.
    agents: list | None
        The agents the Planner may use. Defaults to SQL and Vega-Lite, plus
        hvPlot for gridded data or tables with geometry.
    confirm: bool
        Stop once the question is planned and return an Answer with status
        "planned" that lists its steps; answer.run() runs them.
    returns: type | None
        The type to give the answer in as answer.result, e.g. int,
        list[str] or a pydantic model. A result already in that shape is
        used as it is; otherwise one more LLM call reads it from the answer.

    Returns
    -------
    Answer
        A question Lumen cannot answer gives an Answer with status "error"
        and the reason in ``error``, rather than raising, so it can be
        retried with ``answer.retry()``.
    """
    return await _on_loop(_aask_messages(data, question, [{"role": "user", "content": question}], llm, agents, confirm, returns=returns))


async def _aask_messages(
    data: DataT | Pipeline, question: str, messages: list[dict[str, str]],
    llm: Llm | None, agents: list | None, confirm: bool = False, use_cache: bool = True, returns: Any = None,
) -> Answer:
    source, visible_slugs = _resolve(data)
    planner = await _planner(source, llm or _default_llm(), _settings["agents"] if agents is None else agents)
    # A retry or a preview is asked for a fresh answer, so neither is cached.
    # Nor is data read here from a frame or a dict: each question reads its own
    # copy, so no later question could hit it and it would only hold that copy.
    key = None
    reusable = isinstance(data, Source | Pipeline)
    if _settings["cache"] and use_cache and reusable and len(messages) == 1 and not confirm:
        key = _cache_key(source, question, planner.llm, planner.agents, visible_slugs, returns)
        if (hit := _answer_cache.get(key)) is not None:
            _answer_cache.move_to_end(key)
            return hit._cached_copy()
    context = _fresh_context(planner, source, visible_slugs)
    answer, _ = await _respond(planner, question, messages, context, data, agents, confirm)
    if returns is not None:
        await _typed(answer, returns, planner.llm)
    if key is not None and answer.status == "success":
        # A copy, so changes to the answer handed back reach no later hit
        _answer_cache[key] = answer._cached_copy()
        while len(_answer_cache) > _ANSWER_CACHE_SIZE or _cached_rows() > _ANSWER_CACHE_ROWS:
            _answer_cache.popitem(last=False)
    return answer


def ask(
    data: DataT | Pipeline, question: str, *, llm: Llm | None = None, agents: list | None = None,
    confirm: bool = False, returns: Any = None,
) -> Answer:
    """
    Ask a question of some data and return the Answer.

    Takes the same arguments as aask and, like it, returns an Answer with
    status "error" instead of raising when the question fails. Works
    whether or not an event loop is already running, e.g. in a Jupyter notebook: the question runs
    on Lumen's own background loop. An interrupt cancels it.
    """
    return _run(_aask_messages(data, question, [{"role": "user", "content": question}], llm, agents, confirm, returns=returns))


def chat(data: DataT | Pipeline, *, llm: Llm | None = None, agents: list | None = None) -> Chat:
    """
    Start a conversation with some data, where each question can follow up
    on the answers before it. Takes the same arguments as aask.
    """
    return Chat(data, llm=llm, agents=agents)


@dataclass(frozen=True)
class Step:
    """One step of an answer's plan, as it ran."""

    title: str
    actor: str
    instruction: str
    status: str
    #: None for a step that did not run
    seconds: float | None
    llm_calls: int | None
    requires: tuple[str, ...]
    provides: tuple[str, ...]
    #: The query, for a step that wrote one
    sql: str | None


class Chat(param.Parameterized):
    """
    A conversation with a Source or Pipeline. Each question sees the earlier
    questions and answers, and runs against the data the last answer left,
    the way a follow-up in the chat UI does.
    """

    answers = param.List(doc="""
        The answers so far, oldest first.""")

    def __init__(self, data: DataT | Pipeline, *, llm: Llm | None = None, agents: list | None = None, **params):
        super().__init__(**params)
        self._data = data
        self._source, self._visible_slugs = _resolve(data)
        self._llm = llm
        self._agents = agents
        self._messages: list[dict[str, str]] = []
        self._context: dict[str, Any] | None = None
        self._interface: ChatInterface | None = None
        # The sharing level the context was built under
        self._level: tuple[str, bool] | None = None

    def __repr__(self) -> str:
        return f"Chat(answers={len(self.answers)})"

    def __panel__(self) -> ChatInterface:
        """
        The conversation as a chat widget. A question typed into it is asked
        with chat.aask, so the widget and chat.ask share one history.
        """
        if self._interface is None:
            self._interface = ChatInterface(
                callback=self._reply, callback_user="Lumen", show_rerun=False, show_undo=False,
                sizing_mode="stretch_width", height=600,
            )
            for answer in self.answers:
                self._interface.send(answer.question, user="User", respond=False)
                self._interface.send(answer.__panel__(), user="Lumen", respond=False)
        return self._interface

    async def _reply(self, contents: str, user: str, instance: ChatInterface):
        try:
            # The widget shows its own question and this reply, so _record does not
            answer = await _on_loop(self._aask(contents, in_widget=True))
        except Exception as e:
            # A failed call, e.g. a rate limit, must not leave the widget dead
            return Alert(object=format_error(e), title="Lumen could not answer", severity="error")
        return answer.__panel__()

    def _repr_mimebundle_(self, include=None, exclude=None):
        if not _in_kernel():
            return {"text/plain": repr(self), "text/html": self._html()}
        _load_extensions()
        try:
            bundle = self.__panel__()._repr_mimebundle_(include, exclude)
        except ImportError:
            # VS Code renders Panel through jupyter_bokeh, which may be missing
            bundle = None
        return bundle or {"text/plain": repr(self), "text/html": self._html()}

    def to_html(self, path: str | Path | None = None) -> str:
        """
        The conversation as one HTML page: each question over its answer,
        chart as an image, SQL and spec. Returns the page, and writes it to
        path when given.
        """
        page = (
            '<!doctype html>\n<html><head><meta charset="utf-8"><title>Lumen chat</title></head>'
            f"<body>\n{self._html()}\n</body></html>\n"
        )
        if path is not None:
            Path(path).write_text(page, encoding="utf-8")
        return page

    def to_notebook(self, path: str | Path | None = None) -> str:
        """
        The conversation as a notebook: each question and answer, then Lumen
        code that rebuilds its chart, or else its data, with no LLM. In-memory
        data is embedded, as in answer.code. Returns the notebook as JSON
        text, and writes it to path when given.
        """
        cells, extensions = [], []
        for answer in self.answers:
            text = answer.value if not answer.error else f"{answer.value}\n\n{answer.error}"
            cells.append(make_md_cell(f"## {answer.question}\n\n{text}"))
            component = answer.view if answer.view is not None else answer.pipeline
            if answer.status != "success" or component is None:
                continue
            cell, extension = format_output(component)
            if cell is not None:
                cells.append(cell)
            if extension and extension not in extensions:
                extensions.append(extension)
        notebook = nbformat.v4.new_notebook(cells=[*make_preamble("", extensions=extensions), *cells])
        text = nbformat.writes(notebook)
        if path is not None:
            Path(path).write_text(text, encoding="utf-8")
        return text

    def to_report(self, title: str | None = None) -> Report:
        """
        The conversation as a Lumen Report, with its own to_html, to_notebook
        and to_docx: each question and answer, and the chart where there is
        one. An answer with a table but no chart is in it as text only.
        """
        views: list = []
        for answer in self.answers:
            views.append(Markdown(f"## {answer.question}\n\n{answer.value}", sizing_mode="stretch_width"))
            if answer._editor is not None:
                views.append(answer._editor)
        return Report.from_views(views, title=title)

    def _html(self) -> str:
        """Each question as a heading over its answer's static card."""
        return "\n".join(
            f"<h3>{html.escape(answer.question)}</h3>\n{answer._static_bundle(answer._chart_bundle())['text/html']}"
            for answer in self.answers
        )

    async def aask(self, question: str, *, confirm: bool = False, returns: Any = None) -> Answer:
        """
        Ask the next question and wait for the Answer. With confirm, the
        Answer is "planned" and joins the conversation once it is run.
        """
        return await _on_loop(self._aask(question, confirm, returns))

    async def _aask(
        self, question: str, confirm: bool = False, returns: Any = None, in_widget: bool = False,
        content: str | None = None,
    ) -> Answer:
        """Ask question, sending content as the message if given, e.g. the question with the rows it is about."""
        if self._context is not None and self._level != (sharing.share, sharing.mask_pii):
            # The context's summaries and the replies were written under
            # another sharing level, so only the questions carry over
            self._context = None
            self._messages = [m for m in self._messages if m["role"] == "user"]
        messages = [*self._messages, {"role": "user", "content": content or question}]
        return await self._turn(question, messages, self._messages, self._context, confirm, returns, in_widget)

    async def _turn(
        self, question: str, messages: list[dict[str, str]],
        history: list[dict[str, str]], prior: dict[str, Any] | None, confirm: bool = False,
        returns: Any = None, in_widget: bool = False,
    ) -> Answer:
        """Answer messages from the conversation as it stood (history and its context), and record the turn."""
        agents = _settings["agents"] if self._agents is None else self._agents
        planner = await _planner(self._source, self._llm or _default_llm(), agents)
        context = _fresh_context(planner, self._source, self._visible_slugs) if prior is None else dict(prior)
        answer, out_context = await _respond(planner, question, messages, context, self._data, self._agents, confirm)
        # So answer.retry() asks again within this conversation
        answer._conversation, answer._history = self, (history, prior)
        if returns is not None:
            await _typed(answer, returns, planner.llm)
        if answer.status == "planned":
            answer._chat = (self, messages)
        else:
            self._record(answer, messages, out_context, in_widget)
        return answer

    def _record(
        self, answer: Answer, messages: list[dict[str, str]], out_context: dict[str, Any], in_widget: bool = False,
    ) -> None:
        self.answers = [*self.answers, answer]
        if self._interface is not None and not in_widget:
            # Asked from code: the widget shows it too, as one conversation
            self._interface.send(answer.question, user="User", respond=False)
            self._interface.send(answer.__panel__(), user="Lumen", respond=False)
        if answer.status == "error":
            # The next question starts from the last answer that worked
            return
        self._messages = [*messages, {"role": "assistant", "content": answer.value or answer.plan.title}]
        self._context = out_context
        self._level = (sharing.share, sharing.mask_pii)

    def ask(self, question: str, *, confirm: bool = False, returns: Any = None) -> Answer:
        """Ask the next question and return the Answer."""
        return _run(self._aask(question, confirm, returns))


class Answer(param.Parameterized):
    """
    The result of asking a Source or Pipeline a question.
    """

    pipeline = param.ClassSelector(class_=Pipeline, allow_None=True, doc="""
        A Pipeline over the data that answers the question, if any.""")

    plan = param.ClassSelector(class_=Plan, doc="""
        The executed Plan, with every step the Planner took.""")

    error = param.String(default="", doc="""
        Why the question could not be answered, when status is "error".""")

    question = param.String(doc="""
        The question that was asked.""")

    source = param.ClassSelector(class_=Source, allow_None=True, doc="""
        The Source the question was asked of.""")

    spec = param.Dict(default=None, allow_None=True, doc="""
        The Vega-Lite specification of the chart, if a Vega-Lite chart was made.""")

    sql = param.String(default=None, allow_None=True, doc="""
        The SQL query that produced the data, if one was run.""")

    status = param.String(doc="""
        "success", "error" when the question failed, or "unanswerable"
        when the data does not hold what it asks for; error says why.""")

    truncated_at = param.Integer(default=None, allow_None=True, doc="""
        The row cap the data stopped at, when the result had more rows.
        lumen.ai.configure(max_rows=...) sets the cap.""")

    cached = param.Boolean(default=False, doc="""
        Whether the answer came from configure(cache=True), with no LLM calls.""")

    usage = param.ClassSelector(class_=UsageCollector, default=None, allow_None=True, doc="""
        The tokens and, when the provider prices them, the cost of the LLM
        calls this answer made, e.g. answer.usage.input_tokens.""")

    value = param.String(default="", doc="""
        The written answer.""")

    selection = param.DataFrame(default=None, allow_None=True, doc="""
        The rows clicked or box-selected on the chart or map, which
        ask_about asks about; None when nothing is selected.""")

    result = param.Parameter(default=None, doc="""
        The answer as the type asked for with returns=, e.g. an int or a
        pydantic model; None when no type was asked for.""")

    view = param.ClassSelector(class_=View, allow_None=True, doc="""
        The chart as a Lumen View, if one was made.""")

    def __init__(
        self, llm: Llm | None = None, data: DataT | Pipeline | None = None,
        agents: list | None = None, editor: LumenEditor | None = None,
        context: dict[str, Any] | None = None, sql_failed: bool = False,
        visible_slugs: set[str] | None = None, **params,
    ):
        super().__init__(**params)
        self._card = None
        self._llm = llm
        self._visible_slugs = visible_slugs
        # Set by a Chat, so retry stays in the conversation
        self._conversation: Chat | None = None
        self._history: tuple[list[dict[str, str]], dict[str, Any] | None] = ([], None)
        # The plan's views with refined charts swapped in, for explore
        self._views: list | None = None
        # What was asked and with which agents, so retry asks the same way
        self._data = data
        self._agents = agents
        # The chart's editor and the context it was made in, for refine
        self._editor = editor
        self._context = context or {}
        # Whether sql is the query that failed, rather than one that worked
        self._sql_failed = sql_failed
        # For a planned answer: what running it needs, and the chat it joins
        self._pending: tuple[Planner, dict[str, Any]] | None = None
        self._chat: tuple[Chat, list[dict[str, str]]] | None = None
        self._crs_warned = False
        # The type asked for, so run and retry give the answer in it too
        self._returns = None
        # The chat this answer is part of, so a question about it follows up there
        self._conversation: Chat | None = None
        # What reports clicks on the chart, kept alive with the answer
        self._streams: list = []

    async def arefine(self, instruction: str) -> Answer:
        """
        Change the chart, e.g. its colors, title or encodings, without
        querying the data again. Returns a new Answer.
        """
        return await _on_loop(self._arefine(instruction))

    async def _arefine(self, instruction: str) -> Answer:
        actor = next((
            task.actor for task in self.plan or []
            if isinstance(task, ActorTask) and isinstance(task.actor, VegaLiteAgent)
        ), None)
        if self.status == "error":
            raise ValueError("This answer failed, so it has no chart to refine. Use answer.retry() to ask again.")
        if self.spec is None or self._editor is None or actor is None:
            raise ValueError("refine changes a Vega-Lite chart, and this answer has none.")
        # A copy, so this answer and its card keep the chart they show
        editor = type(self._editor)(component=self.view)
        messages = [{"role": "user", "content": self.question}]
        # The Planner only sets the agent's LLM while a plan runs
        llm = actor.llm or self._llm
        with actor.param.update(llm=llm), llm.capture_usage() as usage:
            editor.spec = await actor.revise(instruction, messages, self._context, view=editor)
        refined = Answer(
            agents=self._agents, context=self._context, data=self._data, editor=editor,
            llm=self._llm, pipeline=self.pipeline, plan=self.plan, question=self.question,
            source=self.source, spec=editor.component.spec, sql=self.sql, status=self.status,
            usage=usage, value=self.value, view=editor.component, visible_slugs=self._visible_slugs,
        )
        refined._conversation, refined._history = self._conversation, self._history
        views = self.plan.views if self._views is None else self._views
        refined._views = [editor if view is self._editor else view for view in views]
        if self._editor not in views:
            refined._views.append(editor)
        return refined

    def refine(self, instruction: str) -> Answer:
        """Change the chart and return the new Answer. See arefine."""
        return _run(self._arefine(instruction), "Revising the chart")

    async def arun(self) -> Answer:
        """
        Run the steps of an answer asked with confirm=True, exactly as
        planned, and fill this answer in with the result. Returns it.

        Unlike refine, which returns a new answer and keeps the old one, a
        planned answer has no result worth keeping, so it becomes the result:
        the variable you held and the card on screen both show it.
        """
        return await _on_loop(self._arun())

    async def _arun(self) -> Answer:
        if self._pending is None:
            raise ValueError("Only an answer asked with confirm=True can be run, and only once.")
        planner, context = self._pending
        # Cleared only once it ran, so a cancelled run can be run again
        done, out_context = await _execute(
            planner, self.plan, self.question, context, self._data, self._agents, self.usage,
        )
        self._pending = None
        self._take(done)
        if self._returns is not None:
            await _typed(self, self._returns, planner.llm)
        if self._chat is not None:
            chat, messages = self._chat
            chat._record(self, messages, out_context)
        return self

    def run(self) -> Answer:
        """Run a planned answer and return it. See arun."""
        return _run(self._arun(), "Running the plan")

    def _cached_copy(self) -> Answer:
        """
        This answer again, for a cache hit, with its own card, pipeline and
        chart, so a filter added to one hit shows up in no other, and no calls
        counted. The plan is shared: it only records what ran.
        """
        values = {key: value for key, value in self.param.values().items() if key != "name"}
        pipeline, view, editor = self.pipeline, self.view, self._editor
        if pipeline is not None:
            pipeline = _clone(
                pipeline, filters=[_clone(f) for f in pipeline.filters],
                transforms=[_clone(t) for t in pipeline.transforms],
                sql_transforms=[_clone(t) for t in pipeline.sql_transforms],
            )
        if view is not None:
            # Keywords that are not params, e.g. hvPlot's responsive or grid
            view = _clone(view, pipeline=pipeline, **view.kwargs)
            editor = None if editor is None else type(editor)(component=view)
        copy = Answer(
            **{**values, "pipeline": pipeline, "view": view},
            agents=self._agents, context=self._context, data=self._data, editor=editor,
            llm=self._llm, sql_failed=self._sql_failed,
        )
        copy.param.update(cached=True, usage=UsageCollector())
        copy._returns = self._returns
        return copy

    def _take(self, other: Answer) -> None:
        """Become the answer this planned one ran to, card included."""
        self.param.update({key: value for key, value in other.param.values().items() if key != "name"})
        self._editor, self._context, self._sql_failed = other._editor, other._context, other._sql_failed
        if self._card is not None:
            self._card[:] = self._card_parts()

    def with_spec(self, spec: dict | str) -> Answer:
        """
        Redraw the chart from an edited spec, without the LLM or a new
        query. Takes the spec as the card's Spec section shows it, as a
        dict or as JSON or YAML text. Returns a new Answer.
        """
        if self.view is None:
            raise ValueError("This answer has no chart to redraw.")
        if isinstance(spec, str):
            # JSON is YAML too, so one parser reads either form
            spec = yaml.safe_load(spec)
        if not isinstance(spec, dict):
            raise ValueError("The spec must be a mapping of the chart's options, as JSON or YAML.")
        pipeline = self.view.pipeline
        if self.spec is not None:
            view = VegaLiteView(pipeline=pipeline, spec=spec)
        else:
            view = type(self.view)(pipeline=pipeline, **{key: value for key, value in spec.items() if key != "type"})
        # An edit can turn a chart into a scatter of every row, as an answer can
        dense = _density_view(view, pipeline)
        vega = dense is view and self.spec is not None
        view = dense
        # Drawn now, so a spec the view cannot draw fails here and not later in the card
        view.get_panel()
        return Answer(
            agents=self._agents, context=self._context, data=self._data,
            editor=type(self._editor)(component=view) if self._editor is not None and vega else None,
            llm=self._llm, pipeline=self.pipeline, plan=self.plan, question=self.question,
            source=self.source, spec=view.spec if vega else None, sql=self.sql,
            status=self.status, truncated_at=self.truncated_at, usage=UsageCollector(),
            value=self.value, view=view,
        )

    def to_cell(self) -> None:
        """Put answer.code into a new cell below this one, in Jupyter."""
        code = self.code
        if not code:
            raise ValueError("This answer has no chart or data to turn into code.")
        if not _in_kernel():
            raise RuntimeError("to_cell needs a Jupyter kernel. answer.code holds the code as a string.")
        # IPython is present, since this runs in a kernel
        from IPython import get_ipython
        get_ipython().set_next_input(code, replace=False)

    async def aretry(self, hint: str | None = None) -> Answer:
        """
        Ask the question again, optionally with a hint on what to change,
        e.g. after an error or a wrong result. Returns a new Answer.

        In a chat, the retry follows the conversation as it stood when this
        answer was asked, so retrying an older answer drops the turns after
        it from what later questions see.
        """
        return await _on_loop(self._aretry(hint))

    async def _aretry(self, hint: str | None = None) -> Answer:
        messages = [{"role": "user", "content": self.question}]
        if hint:
            outcome = {
                "error": f"That failed: {self.error}", "unanswerable": f"{self.value} {self.error}",
            }.get(self.status, self.value)
            if self.sql:
                outcome += f"\n\nThe SQL was:\n{self.sql}"
            messages += [{"role": "assistant", "content": outcome}, {"role": "user", "content": hint}]
        if self._conversation is not None:
            history, prior = self._history
            return await self._conversation._turn(self.question, [*history, *messages], history, prior)
        data = self.source if self._data is None else self._data
        return await _aask_messages(
            data, self.question, messages, self._llm, self._agents, use_cache=False, returns=self._returns,
        )

    def retry(self, hint: str | None = None) -> Answer:
        """Ask the question again and return the new Answer. See aretry."""
        return _run(self._aretry(hint))

    @property
    def gdf(self) -> Any:
        """
        The answer's data as a GeoDataFrame with its CRS, or None when it has
        no geometry. Warns once when the geometry has no CRS but its
        coordinates look like longitude and latitude.
        """
        data = None if self.pipeline is None else self.pipeline.data
        if not is_geodataframe(data):
            return None
        if data.crs is None and not self._crs_warned and len(data):
            west, south, east, north = data.total_bounds
            if -180 <= west <= east <= 360 and -90 <= south <= north <= 90:
                warnings.warn(
                    "This geometry has no CRS but looks like longitude and latitude. "
                    "Set it with answer.gdf.set_crs('EPSG:4326') if so.", stacklevel=2,
                )
                self._crs_warned = True
        return data

    @property
    def ds(self) -> Any:
        """
        The answer's data as an xarray Dataset, for an answer about a grid,
        or None otherwise. xarray-sql builds it from the answer's query,
        so it keeps the grid's dimensions instead of a long table.
        """
        if self.pipeline is None or not isinstance(self.pipeline.source, XArraySQLSource):
            return None
        return self.pipeline.get_dataset()

    @property
    def data(self) -> Any:
        """
        The answer's data in the type it was asked with: polars or pyarrow
        for polars or pyarrow input, the GeoDataFrame of gdf for GeoPandas
        input and the Dataset of ds for xarray input, pandas otherwise.
        When the query read tables of different types, pandas.
        """
        if self.pipeline is None:
            return None
        kinds = self._input_kinds()
        if kinds == {"xarray"} and (ds := self.ds) is not None:
            return ds
        if kinds == {"geopandas"} and (gdf := self.gdf) is not None:
            return gdf
        if kinds in ({"polars"}, {"pyarrow"}):
            return self._fetch_as(*kinds)
        return as_pandas(self.pipeline.data)


    def _fetch_as(self, backend: str) -> Any:
        """
        The pipeline's data read in polars or pyarrow by DuckDB itself, with
        no pandas stop on the way. The query runs on its own, so the shared
        source is never changed and nothing watches it afterwards.
        """
        pipeline = self.pipeline
        source = pipeline.source
        if not isinstance(source, DuckDBSource) or pipeline.filters or pipeline.transforms:
            # Filters and Python transforms only run through the Pipeline, so convert its pandas data
            df = as_pandas(pipeline.data)
            if backend == "polars":
                return sys.modules["polars"].from_pandas(df)
            return sys.modules["pyarrow"].Table.from_pandas(df, preserve_index=False)
        sql = source.get_sql_expr(pipeline.table)
        for transform in pipeline.sql_transforms:
            sql = transform.apply(sql)
        return source.fetch(sql, source.table_params.get(pipeline.table), backend=backend)

    @property
    def why(self) -> str:
        """
        Why the answer took the steps it did: the planner's account of the
        question, and for an unanswerable one the reason.
        """
        if self.plan is None:
            return ""
        return "\n\n".join(part for part in (self.plan.reasoning, self.plan.unanswerable) if part)

    @property
    def trace(self) -> list[Step]:
        """
        Each step of the plan as it ran: its actor and instruction, what it
        needed and provided, how long it took, how many LLM calls it made
        and the query it wrote. answer.plan.graph() gives how the steps
        feed each other.
        """
        if self.plan is None:
            return []
        steps = []
        for i, task in enumerate(self.plan):
            seconds, calls = self.plan._runs.get(i, (None, None))
            provides = task.output_schema.__required_keys__ | task.output_schema.__optional_keys__
            steps.append(Step(
                title=task.title, actor=type(task.actor).__name__, instruction=task.instruction,
                status=task.status, seconds=seconds, llm_calls=calls,
                requires=tuple(sorted(task.input_schema.__required_keys__)), provides=tuple(sorted(provides)),
                sql=task.out_context.get("sql") if "sql" in provides else None,
            ))
        return steps

    async def aask_about(self, question: str, rows: Any = None) -> Answer:
        """
        Ask a follow-up about some rows of this answer's data: the
        selection, clicked or box-selected on the chart, unless rows says
        which (a DataFrame, a dict or list of dicts, or row positions). The
        LLM answers from the rows when they hold what is asked, and
        otherwise queries the data filtered on their values. In a chat, the
        question joins the conversation. Returns the new Answer.
        """
        rows = self._selected(rows)
        content = about(question, rows)
        if self._conversation is not None:
            return await self._conversation._aask(question, content=content)
        outcome = self.value + (f"\n\nThe SQL was:\n{self.sql}" if self.sql else "")
        messages = [
            {"role": "user", "content": self.question}, {"role": "assistant", "content": outcome},
            {"role": "user", "content": content},
        ]
        data = self.source if self._data is None else self._data
        return await _aask_messages(data, question, messages, self._llm, self._agents)

    def ask_about(self, question: str, rows: Any = None) -> Answer:
        """Ask a follow-up about the selected rows and return the Answer. See aask_about."""
        return _run(self.aask_about(question, rows))

    def _selected(self, rows: Any) -> pd.DataFrame:
        """The rows a question is about, as a DataFrame."""
        if rows is None:
            if self.selection is None or self.selection.empty:
                raise ValueError("Nothing is selected. Click a point on the chart, or pass rows=.")
            return self.selection
        if isinstance(rows, pd.DataFrame):
            return rows
        if isinstance(rows, dict):
            return pd.DataFrame([rows])
        if isinstance(rows, list) and all(isinstance(row, dict) for row in rows):
            return pd.DataFrame(rows)
        if self.pipeline is None:
            raise ValueError("This answer has no data to pick rows from.")
        return as_pandas(self.pipeline.data).iloc[list(rows)]

    async def _ask_from_box(self, question: str) -> Any:
        """Ask from the card's box; in the chat widget the answer joins the conversation."""
        answer = await self.aask_about(question)
        chat = self._conversation
        if chat is not None and chat._interface is not None:
            chat._interface.send(answer.question, user="User", respond=False)
            chat._interface.send(answer.__panel__(), user="Lumen", respond=False)
            return None
        return answer.__panel__()

    def compile(self, params: list[str] | None = None) -> Callable[..., Any]:
        """
        The answer's query as a function that runs without an LLM.

        Each name in params is a column the query compares with values,
        e.g. ``year = 2008``, ``year BETWEEN 2008 AND 2009`` or
        ``island IN ('Biscoe', 'Dream')``; it becomes a keyword argument
        that defaults to those values, a pair for BETWEEN and a list for
        IN. ``f()`` gives this answer's data and ``f(year=2009)`` the same
        query for 2009, in the type the question was asked with. Values
        are bound as query parameters, never written into the SQL.
        """
        if not self.sql or self._sql_failed or self.source is None:
            raise ValueError("This answer has no query that worked to compile.")
        # The query can read tables an earlier step made, which only the
        # pipeline's source holds
        source = self.source if self.pipeline is None else self.pipeline.source
        if not isinstance(source, DuckDBSource):
            # DuckDB named parameters only; other engines bind differently
            raise ValueError("compile needs a DuckDB source, such as one from lumen.read.")
        try:
            tree = sqlglot.parse_one(self.sql, read="duckdb")
        except sqlglot.errors.ParseError as e:
            raise ValueError(f"The answer's query could not be parsed for compile: {e}") from e
        defaults: dict[str, Any] = {}
        binders: dict[str, Callable[[Any], dict[str, Any]]] = {}
        for i, name in enumerate(params or []):
            nodes = [
                node for node in tree.find_all(exp.EQ, exp.NEQ, exp.GT, exp.GTE, exp.LT, exp.LTE, exp.In, exp.Between)
                if isinstance(node.this, exp.Column) and node.this.name == name
            ]
            values = {_literals(node) for node in nodes} - {None}
            if not values:
                compared = sorted({
                    node.this.name for node in tree.find_all(exp.EQ, exp.NEQ, exp.GT, exp.GTE, exp.LT, exp.LTE, exp.In, exp.Between)
                    if isinstance(node.this, exp.Column) and _literals(node) is not None
                })
                raise ValueError(f"The query compares no column {name!r} with a value. It compares: {', '.join(compared) or 'none'}.")
            if len(values) > 1:
                raise ValueError(f"The query compares {name!r} with different values in different places, so one argument cannot stand for them.")
            (kind, default), = values
            defaults[name] = {"in": list, "between": tuple}.get(kind, lambda value: value)(default)
            # Columns can have names a parameter cannot, e.g. with spaces
            for node in nodes:
                _parameterize(node, kind, f"p{i}")
            binders[name] = _binder(kind, f"p{i}")
        # Without parameters the query runs as written, untouched by sqlglot
        sql = tree.sql("duckdb") if params else self.sql
        if SQLAgent.max_rows is not None:
            # The same row cap as the answer's own data
            sql = SQLLimit(limit=SQLAgent.max_rows, write="duckdb").apply(sql)
        kinds = self._input_kinds()

        def run(**kwargs: Any) -> Any:
            unknown = set(kwargs) - set(defaults)
            if unknown:
                raise TypeError(f"Unknown arguments {sorted(unknown)}; this query takes {sorted(defaults)}.")
            bound: dict[str, Any] = {}
            for name, value in {**defaults, **kwargs}.items():
                bound.update(binders[name](value))
            return _as_input_type(source.execute(sql, bound), kinds)

        if all(name.isidentifier() for name in defaults):
            # A column such as "body mass" can still be passed with **{...}
            run.__signature__ = inspect.Signature([
                inspect.Parameter(name, inspect.Parameter.KEYWORD_ONLY, default=value) for name, value in defaults.items()
            ])
        run.__doc__ = f"{self.question}\n\n{sql}"
        run.sql = sql
        return run

    def _input_kinds(self) -> set[str]:
        """The libraries of the tables this answer's query read."""
        return _read_kinds(self.source, self.sql)

    def to_file(self, path: str | Path, *, overwrite: bool = False) -> Path:
        """
        Write the answer's data to a file, in the format its extension names:
        ``.parquet`` (GeoParquet when the data has geometry), ``.geojson``,
        ``.gpkg`` and ``.csv`` for tables, ``.zarr`` and ``.nc`` for grids.
        Raises FileExistsError for a path that exists unless ``overwrite``
        is set, and warns when the data was cut at ``max_rows``. Returns
        the path.
        """
        path = Path(path)
        suffix = path.suffix.lower()
        if path.exists() and not overwrite:
            raise FileExistsError(f"{path} exists. Pass overwrite=True to replace it.")
        if self.truncated_at is not None:
            warnings.warn(
                f"The answer's data stops at the first {self.truncated_at:,} rows, so the file holds only those. "
                "Raise it with lumen.ai.configure(max_rows=...).",
                stacklevel=2,
            )
        if suffix in (".zarr", ".nc"):
            ds = self.ds
            if ds is None:
                raise ValueError(f"{suffix} files hold grids, and this answer is a table. Use .parquet or .csv.")
            if suffix == ".zarr":
                # Consolidated metadata is not part of the Zarr 3 format
                ds.to_zarr(path, mode="w", consolidated=False)
            else:
                ds.to_netcdf(path)
            return path
        if self.pipeline is None:
            raise ValueError("This answer has no data to write.")
        gdf = self.gdf
        data = gdf if gdf is not None else as_pandas(self.pipeline.data)
        if suffix == ".parquet":
            data.to_parquet(path)
        elif suffix in (".geojson", ".gpkg"):
            if gdf is None:
                raise ValueError(f"{suffix} files hold geometry, and this answer has none. Use .parquet or .csv.")
            gdf.to_file(path, driver="GeoJSON" if suffix == ".geojson" else "GPKG")
        elif suffix == ".csv":
            data.to_csv(path, index=False)
        else:
            raise ValueError(f"Cannot write {suffix!r} files. Use .parquet, .geojson, .gpkg, .csv, .zarr or .nc.")
        return path

    def map(self) -> Any:
        """
        An interactive map of the answer's geometry: lonboard's when it is
        installed, otherwise hvPlot's, on a basemap when geoviews is installed.
        """
        gdf = self.gdf
        if gdf is None:
            raise ValueError("This answer has no geometry to map.")
        if importlib.util.find_spec("lonboard") is not None:
            # Deferred: lonboard is optional, and heavy to import
            from lonboard import viz
            m = viz(gdf)
            for layer in m.layers:
                layer.observe(lambda change: setattr(
                    self, "selection", None if change.new is None else gdf.iloc[[change.new]]
                ), names="selected_index")
            return m
        if importlib.util.find_spec("geoviews") is not None:
            return gdf.hvplot(geo=True, tiles="OSM")
        return gdf.hvplot()

    @property
    def chart(self) -> Any:
        """
        The chart as an object to keep working with: an Altair Chart for a
        Vega-Lite answer, the HoloViews object for an hvPlot one, so it can
        be composed with ``+`` and ``*`` or restyled with ``.opts``. None
        when the answer has no chart.
        """
        if self.view is None:
            return None
        if self.spec is None:
            return self.view.get_plot(self.view.get_data())
        try:
            # Deferred: altair is optional
            import altair as alt
        except ImportError as e:
            raise ImportError(
                "answer.chart needs altair for a Vega-Lite chart: pip install altair. "
                "answer.to_vega() gives the spec without it."
            ) from e
        spec = self.to_vega()
        # Chart.from_dict only takes single-view specs
        kinds = {"layer": "LayerChart", "facet": "FacetChart", "repeat": "RepeatChart",
                 "hconcat": "HConcatChart", "vconcat": "VConcatChart", "concat": "ConcatChart"}
        kind = next((name for key, name in kinds.items() if key in spec), "Chart")
        return getattr(alt, kind).from_dict(spec)

    def to_vega(self) -> dict:
        """
        The chart as a self-contained Vega-Lite spec, data inlined. An
        hvPlot map is given as a Vega-Lite map of its geometry.
        """
        if (spec := self._vega_lite_spec() or self._geoshape_spec()) is None:
            raise ValueError("This answer has no Vega-Lite chart; answer.chart gives an hvPlot one.")
        return spec

    def tweak(self) -> Any:
        """
        Open the chart in hvPlot's explorer, preloaded with its kind, axes,
        colours and map settings, to adjust by hand. A Vega-Lite chart opens
        as the matching hvPlot kind. ``.settings()`` on the
        result gives the hvPlot keywords, ``.hvplot()`` the plot.
        """
        if not isinstance(self.view, hvPlotView | VegaLiteView):
            raise ValueError("This answer has no chart to tweak.")
        settings = _explorer_settings(self.view)
        if self.view.pipeline.data.empty:
            raise ValueError("This answer's query returned no rows, so there is nothing to tweak.")
        return hvPlotUIView(pipeline=self.view.pipeline, **settings).get_panel()

    def explore(self, **params) -> ExplorerUI:
        """
        Open this answer in the Lumen chat UI: an ExplorerUI on the same
        Source and LLM, with the answer as its first exploration. Keyword
        arguments go to ExplorerUI.
        """
        if self._visible_slugs is not None:
            # Keep a pipeline's answer scoped to its table. ExplorerUI shows
            # every table it has not seen yet, so mark them all as seen.
            known = {f"{self.source.name}{SOURCE_TABLE_SEPARATOR}{t}" for t in self.source.get_tables()}
            params["context"] = {"visible_slugs": set(self._visible_slugs), "_known_slugs": known, **params.get("context", {})}
        ui = ExplorerUI(data=self.source, llm=self._llm, **params)

        async def seed():
            ui.interface.send(self.question, respond=False)
            await ui.add_plan_exploration(self.plan, views=self._views)

        # onload runs it inside the served session once the page is up, so
        # the browser receives the changes; state.execute ran it outside the
        # session and the page stopped updating. In a notebook it runs now.
        state.onload(seed)
        return ui

    def export(self, directory: str | Path) -> Path:
        """
        Write this answer as a script that reruns it without an LLM.

        Writes ``answer.py`` and, for data held in memory, one Parquet file
        per table under ``data/``, so the script stays small however large
        the data is. Returns the path of the script.
        """
        component = self._rebuilt()
        if component is None:
            raise ValueError("This answer has no chart or data to export.")
        directory = Path(directory)
        data_dir = directory / "data"
        data_dir.mkdir(parents=True, exist_ok=True)
        with config.param.update(serializer="csv"):
            spec = component.to_spec()
        _write_tables(spec, data_dir)
        cell, extension = format_output(
            component, spec=spec,
            load_spec='yaml.safe_load(yaml_spec.replace("$LUMEN_DATA", str(Path(__file__).parent / "data")))',
        )
        (preamble,) = make_preamble("from pathlib import Path\n", extensions=[extension] if extension else [])
        script = directory / "answer.py"
        script.write_text(f"{preamble.source}\n\n{cell.source}\n")
        return script

    def _rebuilt(self) -> View | Pipeline | None:
        """The chart, or else the data, that code and export rebuild."""
        if joins_grids(self.source) or joins_grids(self.pipeline and self.pipeline.source):
            raise ValueError(
                "This answer's data joins xarray data into DuckDB, and those tables live only "
                "in memory, so they cannot be rebuilt from a spec. Read the xarray data on its "
                "own for answer.code and answer.export."
            )
        return self.view if self.view is not None else self.pipeline

    @property
    def code(self) -> str:
        """
        Python that rebuilds the chart, or else the data, with Lumen alone.

        No LLM is called. In-memory data is embedded in the spec, so the
        code runs anywhere Lumen is installed; for a large table use
        ``export``, which keeps the data in Parquet files beside a small
        script. File-based sources keep their absolute paths. Source specs
        go in as they are, so a database URI or initializer holding a
        password ends up in the code.
        """
        component = self._rebuilt()
        if component is None:
            return ""
        cell, extension = format_output(component)
        if cell is None:
            return ""
        (preamble,) = make_preamble("", extensions=[extension] if extension else [])
        return f"{preamble.source}\n\n{cell.source}"

    def __repr__(self) -> str:
        first_line = self.value.strip().splitlines()[0] if self.value.strip() else ""
        return f"Answer(status={self.status!r}, value={first_line!r})"

    def __panel__(self):
        if self._card is None:
            self._card = Column(*self._card_parts())
        return self._card

    def _card_parts(self) -> list:
        if self.status == "planned":
            return self._planned_parts()
        if self.status in ("error", "unanswerable"):
            # The data lacking what was asked is not a failure, so it is not red
            severity = "info" if self.status == "unanswerable" else "error"
            parts = [Alert(object=self.error, severity=severity, title=self.value, sizing_mode="stretch_width")]
            if self.sql:
                parts.append(Card(
                    Markdown(f"```sql\n{self.sql}\n```"), title="Failed SQL" if self._sql_failed else "SQL",
                    collapsed=not self._sql_failed, sizing_mode="stretch_width",
                ))
        else:
            parts = [Markdown(self.value, sizing_mode="stretch_width")] if self.value else []
            if self.view is not None:
                parts.append(self._chart_part(self.view))
                parts.append(ask_box(self, self._ask_from_box))
            if self._cap_note:
                parts.append(Alert(object=self._cap_note, severity="info", sizing_mode="stretch_width"))
            if self.sql:
                parts.append(Card(
                    Markdown(f"```sql\n{self.sql}\n```"), title="SQL", collapsed=True,
                    sizing_mode="stretch_width",
                ))
            if (chart_spec := self._chart_spec()) is not None:
                parts.append(self._spec_card(*chart_spec, chart_index=1 if self.value else 0))
        if self._steps_text:
            parts.append(Card(
                Markdown(self._steps_text, sizing_mode="stretch_width"), title="Steps", collapsed=True,
                sizing_mode="stretch_width",
            ))
        if self._usage_note:
            parts.append(self._usage_pane())
        return parts

    @property
    def _steps_text(self) -> str:
        """The why and each step as it ran, as Markdown; nothing for a plan with no steps."""
        lines = []
        for i, step in enumerate(self.trace, 1):
            facts = [f"{step.seconds:.1f} s"] if step.seconds is not None else ["not run"]
            if step.llm_calls:
                facts.append(f"{step.llm_calls} LLM call{'s' if step.llm_calls > 1 else ''}")
            lines.append(f"{i}. **{step.actor}**: {step.title} · {', '.join(facts)}")
        if not lines:
            return ""
        return "\n\n".join(part for part in (f"*{self.plan.reasoning}*" if self.plan.reasoning else "", "\n".join(lines)) if part)

    @property
    def _plan_steps(self) -> str:
        """The planned steps as a numbered Markdown list."""
        return "\n".join(
            f"{i}. **{type(task.actor).__name__}**: {task.instruction}" for i, task in enumerate(self.plan, 1)
        )

    def _planned_parts(self) -> list:
        run = Button(label="Run", icon="play_arrow", variant="contained")

        async def on_run(event):
            run.loading = True
            try:
                await self.arun()
            finally:
                run.loading = False

        run.on_click(on_run)
        parts = [
            Alert(
                object=self._plan_steps, title=f"Planned: {self.plan.title}", severity="info",
                sizing_mode="stretch_width",
            ),
            run,
        ]
        if self._usage_note:
            parts.append(self._usage_pane())
        return parts

    @property
    def _usage_note(self) -> str:
        """One line on what the answer cost, or nothing when no call was counted."""
        if self.cached:
            return "From the answer cache, no LLM calls"
        if self.usage is None or not self.usage.records:
            return ""
        note = f"{self.usage.input_tokens:,} in / {self.usage.output_tokens:,} out tokens"
        if self.usage.cost_usd is not None:
            note += f" · ${self.usage.cost_usd:.4f}"
        return note

    def _usage_pane(self) -> Markdown:
        return Markdown(
            self._usage_note, sizing_mode="stretch_width", margin=(0, 10),
            styles={"font-size": "0.8em", "opacity": "0.7"},
        )

    def _chart_part(self, view: View):
        try:
            # Drawn here, because inside the card a failure only becomes a
            # notification once the chat UI has turned those on, and a
            # notebook shows none, so the chart would just be missing
            view.get_panel()
        except Exception as e:
            return Alert(
                object=f"{type(e).__name__}: {e}", title="The chart could not be drawn",
                severity="error", sizing_mode="stretch_width",
            )
        self._streams += watch_clicks(view, lambda rows: setattr(self, "selection", rows))
        if isinstance(view, hvPlotView) and view.kwargs.get("responsive"):
            # A responsive plot stretches to its container, and a notebook
            # output has no height to give it, so it would collapse
            return Column(view, min_height=400, sizing_mode="stretch_both")
        return view

    def _spec_card(self, language: str, text: str, chart_index: int) -> Card:
        """
        The chart's spec as a code block, which has a copy button, and an
        Edit button that swaps in an editor to change the spec and redraw.
        Unlike with_spec, which returns a new Answer, a redraw here changes
        this answer, since the card shows it.
        """
        code = Markdown(f"```{language}\n{text}\n```", sizing_mode="stretch_width")
        edit = Button(label="Edit", icon="edit", variant="outlined")
        editor = CodeEditor(
            value=text, language=language, sizing_mode="stretch_width", visible=False,
            theme="github_dark" if pn.config.theme == "dark" else "github_light_default",
            height=min(400, 16 * (text.count("\n") + 1) + 24),
        )
        error = Alert(severity="error", visible=False, sizing_mode="stretch_width")
        redraw = Button(label="Redraw", icon="refresh", variant="contained")
        cancel = Button(label="Cancel", variant="text")
        actions = Row(redraw, cancel, visible=False)

        def editing(on: bool):
            code.visible = edit.visible = not on
            editor.visible = actions.visible = on
            if not on:
                error.visible = False

        def start(event):
            editor.value = self._chart_spec()[1]
            editing(True)

        def apply(event):
            try:
                redrawn = self.with_spec(editor.value)
            except Exception as e:
                error.param.update(object=f"{type(e).__name__}: {e}", visible=True)
                return
            # The card shows this answer, so it takes the redrawn chart
            self.param.update(spec=redrawn.spec, view=redrawn.view)
            self._editor = redrawn._editor
            self._card[chart_index] = self._chart_part(redrawn.view)
            code.object = "```{}\n{}\n```".format(*self._chart_spec())
            editing(False)

        edit.on_click(start)
        redraw.on_click(apply)
        cancel.on_click(lambda event: editing(False))
        return Card(code, edit, editor, error, actions, title="Spec", collapsed=True, sizing_mode="stretch_width")

    def _chart_spec(self) -> tuple[str, str] | None:
        """
        The spec the chart was drawn from, as (language, text): JSON for
        Vega-Lite, the view's YAML for hvPlot. Data values are left out,
        so a large result does not flood it.
        """
        if self.view is None:
            return None
        if self.spec is not None:
            return "json", json.dumps(_without_values(self.spec), indent=2, default=str)
        spec = {key: value for key, value in self.view.to_spec().items() if key != "pipeline"}
        return "yaml", yaml.safe_dump(spec, sort_keys=False).strip()

    def _vega_lite_spec(self) -> dict | None:
        """The chart as a self-contained Vega-Lite spec, data inlined."""
        if self.spec is None:
            return None
        spec = dict(self.view.get_panel().object)
        data = spec.get("data")
        # A spec can bring its own data, e.g. a topojson url for a map
        if isinstance(data, dict) and "values" in data:
            spec["data"] = dict(data, values=_records(data["values"]))
        if isinstance(spec.get("datasets"), dict):
            spec["datasets"] = {name: _records(values) for name, values in spec["datasets"].items()}
        if any(key in spec for key in ("facet", "repeat", "concat", "hconcat", "vconcat")):
            # Vega-Lite allows no top-level size on these; their views size themselves
            spec.pop("width", None)
            spec.pop("height", None)
            return spec
        for dim, size in (("width", 600), ("height", 400)):
            if spec.get(dim, "container") == "container":
                spec[dim] = size
        return spec

    def _geoshape_spec(self) -> dict | None:
        """
        The answer's geometry as a Vega-Lite map, for answers drawn with
        hvPlot, which have no static form of their own.
        """
        gdf = self.gdf
        if gdf is None or not len(gdf):
            return None
        if gdf.crs is not None and not gdf.crs.is_geographic:
            # Vega's projections expect longitude and latitude
            gdf = gdf.to_crs("EPSG:4326")
        fields = [c for c in gdf.columns if c != gdf.geometry.name]
        numeric = [c for c in fields if pd.api.types.is_numeric_dtype(gdf[c])]
        encoding: dict[str, Any] = {"tooltip": [{"field": f"properties.{c}", "title": c} for c in fields]}
        if numeric:
            encoding["color"] = {"field": f"properties.{numeric[0]}", "type": "quantitative", "title": numeric[0]}
        return {
            "$schema": "https://vega.github.io/schema/vega-lite/v5.json",
            "data": {"values": geometry_to_geojson(gdf, clockwise=True)["features"]},
            "mark": {"type": "geoshape", "stroke": "white", "strokeWidth": 0.5},
            "projection": {"type": "mercator"},
            "encoding": encoding,
            "title": self.question,
            "width": 600,
            "height": 400,
        }

    def _chart_bundle(self) -> dict | None:
        """The chart as interactive Vega-Lite, with a PNG for viewers that cannot run it."""
        if (spec := self._vega_lite_spec() or self._geoshape_spec()) is None:
            return None
        title = spec.get("title", "")
        if isinstance(title, dict):
            title = title.get("text", "")
        return {
            "application/vnd.vegalite.v5+json": spec,
            "image/png": base64.b64encode(Vega(spec).export("png", scale=2)).decode(),
            "text/plain": f"<Vega-Lite chart: {title}>",
        }

    @property
    def _text(self) -> str:
        """The written answer, followed by the error or the row cap note."""
        if self.status == "planned":
            return f"Planned: {self.plan.title}\n\n{self._plan_steps}\n\nRun it with answer.run()."
        return "\n\n".join(part for part in (self.value, self.error, self._cap_note) if part)

    @property
    def _cap_note(self) -> str:
        """Notes on how much of the data the answer holds and draws."""
        notes = []
        if self.truncated_at is not None:
            notes.append(
                f"The data stops at the first {self.truncated_at:,} rows. "
                "Use lumen.ai.configure(max_rows=...) for more, or None for every row."
            )
        rows = 0 if self.pipeline is None or self.view is None else len(self.pipeline.data)
        if isinstance(self.view, hvPlotView) and (self.view.datashade or self.view.kwargs.get("rasterize")):
            shown = "blending the categories where they overlap" if self.view.by else "so each pixel shows how many fall there"
            notes.append(f"The chart draws {rows:,} points as a density image, {shown}.")
        elif rows > LARGE_POINTS and _point_chart(self.view) is not None:
            notes.append(
                f"The chart draws all {rows:,} points, which can be slow. "
                "Install datashader to draw them as a density image instead."
            )
        return " ".join(notes)

    def _sql_html(self) -> str:
        # A failed query is what the reader needs to see, so it starts open
        if self._sql_failed:
            return f"<details open><summary>Failed SQL</summary><pre><code>{html.escape(self.sql)}</code></pre></details>"
        return f"<details><summary>SQL</summary><pre><code>{html.escape(self.sql)}</code></pre></details>"

    def _static_bundle(self, chart: dict | None) -> dict:
        # A viewer shows one mime type per output, so the HTML carries the
        # whole card (answer, chart, SQL) and is what notebook viewers prefer
        sql = f"\n\n```sql\n{self.sql}\n```" if self.sql else ""
        chart_spec = self._chart_spec()
        if chart_spec is not None:
            sql += f"\n\n```{chart_spec[0]}\n{chart_spec[1]}\n```"
        bundle = {"text/plain": repr(self), "text/markdown": f"{self._text}{sql}"}
        prose = self._text if self.status == "planned" else self.value
        card = [_MARKDOWN.render(prose)] if prose else []
        if self.error:
            card.append(f"<pre>{html.escape(self.error)}</pre>")
        if chart is not None:
            bundle.update({k: v for k, v in chart.items() if k != "text/plain"})
            # Rendered at twice the size for sharpness, shown at the spec
            # width; a faceted or concatenated chart has none of its own
            width = chart["application/vnd.vegalite.v5+json"].get("width")
            size = f' width="{width}"' if width else ""
            card.append(f'<img src="data:image/png;base64,{chart["image/png"]}"{size} style="max-width: 100%">')
        if self._cap_note:
            card.append(f"<p><em>{html.escape(self._cap_note)}</em></p>")
        if self.sql:
            card.append(self._sql_html())
        if chart_spec is not None:
            card.append(f"<details><summary>Spec</summary><pre><code>{html.escape(chart_spec[1])}</code></pre></details>")
        if self._steps_text:
            card.append(f"<details><summary>Steps</summary>{_MARKDOWN.render(self._steps_text)}</details>")
        if self._usage_note:
            card.append(f"<p><small>{html.escape(self._usage_note)}</small></p>")
            bundle["text/markdown"] += f"\n\n*{self._usage_note}*"
        bundle["text/html"] = "\n".join(card)
        return bundle

    def _repr_mimebundle_(self, include=None, exclude=None):
        if not _in_kernel():
            return self._static_bundle(self._chart_bundle())
        _load_extensions()
        try:
            bundle = self.__panel__()._repr_mimebundle_(include, exclude)
        except ImportError:
            # VS Code renders Panel through jupyter_bokeh, which may be missing
            bundle = None
        if bundle is not None:
            return bundle
        # Panel cannot render here, but notebook front-ends draw Vega-Lite
        # themselves, so the chart goes out as its own output and keeps its
        # tooltips. IPython is present, since this runs in a kernel.
        from IPython.display import display
        if self._text:
            display({"text/markdown": self._text, "text/plain": self._text}, raw=True)
        if (chart := self._chart_bundle()) is not None:
            display(chart, raw=True)
        if not self.sql:
            return {"text/plain": ""}
        return {
            "text/markdown": f"```sql\n{self.sql}\n```",
            "text/html": self._sql_html(),
            "text/plain": self.sql,
        }
