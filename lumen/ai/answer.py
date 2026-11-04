"""
Ask a Lumen Source or Pipeline a question from Python and get an Answer.
"""
from __future__ import annotations

import asyncio
import base64
import html
import importlib.util
import json
import logging
import math
import os
import time
import warnings
import weakref

from concurrent.futures import (
    CancelledError as FutureCancelledError, ThreadPoolExecutor,
    TimeoutError as FutureTimeoutError,
)
from contextlib import suppress
from functools import cache
from pathlib import Path
from typing import TYPE_CHECKING, Any

import hvplot.pandas  # noqa: F401  registers .hvplot, used by answer.map
import pandas as pd
import panel as pn
import param
import yaml

from panel.config import panel_extension
from panel.io.state import state
from panel.pane import Markdown, Vega
from panel.pane.vega import ds_to_records
from panel.widgets import CodeEditor
from panel_material_ui import Alert, Button, Card, Column, Row
from panel_material_ui.chat import ChatMessage

from ..config import SOURCE_TABLE_SEPARATOR, config
from ..pipeline import Pipeline
from ..serializers import Serializer
from ..sources import Source
from ..sources.base import BaseSQLSource
from ..sources.xarray_sql import XArraySQLSource
from ..transforms.sql import SQLLimit
from ..util import (
    as_pandas, geometry_columns, geometry_to_geojson, is_geodataframe, log,
)
from ..views import VegaLiteView, View, hvPlotView
from .actor import step_title_callback
from .agents import ChatAgent, SQLAgent, VegaLiteAgent
from .agents.hvplot import LAT_NAMES, LON_NAMES, hvPlotAgent
from .config import QueryError, RetriesExceededError
from .coordinator import Plan, Planner
from .data import read
from .editors import LumenEditor
from .export import format_output, make_preamble
from .llm import (
    Llm, get_available_llm, llm_class, quiet_retry_log, signed_in_cli_hint,
)
from .report import ActorTask
from .sharing import sharing
from .story import _prose_to_html
from .ui import ExplorerUI
from .utils import format_error

if TYPE_CHECKING:
    from .data import DataT


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


def _is_spatial(source: Source) -> bool:
    """Whether a table has geometry, or longitude and latitude columns."""
    # Only SQL sources can be probed without loading data. DuckDB returns
    # GEOMETRY as a GeoDataFrame only when geopandas is installed, which is
    # also what hvPlot needs to draw it.
    if not isinstance(source, BaseSQLSource):
        return False
    for table in source.get_tables():
        empty = source.execute(f"SELECT * FROM ({source.get_sql_expr(table)}) LIMIT 0")
        columns = {str(c).lower() for c in empty.columns}
        if geometry_columns(empty) or (columns & LON_NAMES and columns & LAT_NAMES):
            return True
    return False


def _default_agents(source: Source) -> list:
    # Maps and grids go to hvPlot, which draws them on basemap tiles and
    # rasterizes large grids; Vega-Lite can do neither.
    if isinstance(source, XArraySQLSource) or _is_spatial(source):
        return [ChatAgent(), SQLAgent(), hvPlotAgent()]
    return [ChatAgent(), SQLAgent(), VegaLiteAgent()]


# Set through configure, and used by every question that does not pass its own
_settings: dict[str, Any] = {"agents": None, "llm": None}

_UNSET: Any = object()


def configure(
    *, llm: Llm | None = None, agents: list | None = None, max_rows: int | None = _UNSET,
    share: str | None = None, mask_pii: bool | None = None,
) -> dict[str, Any]:
    """
    Set the defaults for ask, aask and chat in this session.

    A question's own llm= or agents= still wins. Without either set
    here, LUMEN_AI_PROVIDER (e.g. "kilo") and LUMEN_AI_MODEL pick the
    LLM, and otherwise the first provider with an API key set.

    max_rows caps the rows an answer's data holds (1,000,000 by
    default), as a LIMIT on its query; None lifts the cap.

    share sets what of the data prompts contain: "sample" (values, as
    before), "schema" (structure, no values) or "none" (names only).
    mask_pii withholds the values of columns that look like personal
    data. Both apply to the chat UI too; see lumen.ai.sharing.

    Returns the settings now in effect.
    """
    if llm is not None:
        _settings["llm"] = llm
    if agents is not None:
        _settings["agents"] = agents
    if max_rows is not _UNSET:
        # The class default, so every SQLAgent without its own value uses it
        SQLAgent.max_rows = max_rows
    sharing.param.update({k: v for k, v in dict(share=share, mask_pii=mask_pii).items() if v is not None})
    return {**_settings, "mask_pii": sharing.mask_pii, "max_rows": SQLAgent.max_rows, "share": sharing.share}


def _row_cap_hit(pipeline: Pipeline | None) -> int | None:
    """The row cap the pipeline's data stopped at, if the cap cut it short."""
    limits = [] if pipeline is None else [
        t.limit for t in pipeline.sql_transforms if isinstance(t, SQLLimit) and t.limit is not None
    ]
    # The run has already loaded the data, so its length costs no query
    return limits[0] if limits and len(pipeline.data) >= limits[0] else None


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


# One Planner per Source, freed with it. Building a Planner and syncing its
# metadata costs about half a second, and a reused one sends the LLM exactly
# the prompts a fresh one would.
_planners: weakref.WeakKeyDictionary[Source, Planner] = weakref.WeakKeyDictionary()


def _planner(source: Source, llm: Llm, agents: list | None) -> Planner:
    if agents is not None:
        return Planner(llm=llm, agents=agents, headless=True)
    planner = _planners.get(source)
    if planner is None or planner.llm is not llm:
        planner = _planners[source] = Planner(llm=llm, agents=_default_agents(source), headless=True)
    return planner


def _resolve(data: DataT | Pipeline) -> tuple[Source, set[str] | None]:
    """Return the Source to ask and the table slugs it is limited to."""
    if not isinstance(data, Pipeline):
        return (data if isinstance(data, Source) else read(data)), None
    if data.filters or data.transforms or data.sql_transforms:
        # The rows the pipeline selects have to be what gets queried
        return read({data.table: data}), None
    return data.source, {f"{data.source.name}{SOURCE_TABLE_SEPARATOR}{data.table}"}


def _fresh_context(planner: Planner, source: Source, visible_slugs: set[str] | None) -> dict[str, Any]:
    # Starts from the Planner's own context (its vector stores) so nothing
    # one question adds is seen by an unrelated one
    context: dict[str, Any] = {**planner.context, "source": source, "sources": [source]}
    if visible_slugs is not None:
        context["visible_slugs"] = visible_slugs
    return context


async def _respond(
    planner: Planner, question: str, messages: list[dict[str, str]], context: dict[str, Any],
    data: DataT | Pipeline, agents: list | None,
) -> tuple[Answer, dict[str, Any]]:
    """
    Run one question through the Planner, returning the Answer and the
    context it left. A failed plan gives an Answer with status "error".
    """
    if log.getEffectiveLevel() > logging.DEBUG:
        quiet_retry_log()
    await planner.sync(context)
    plan = await planner.respond(messages, context)
    if plan is None:
        raise RuntimeError(f"Lumen could not plan an answer to {question!r}.")
    outputs, out_context = await plan.execute()
    answer = _build_answer(question, plan, outputs, out_context, context["source"], planner.llm, data, agents)
    return answer, out_context


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
        {key: None if isinstance(value, float) and math.isnan(value) else value for key, value in row.items()}
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


def _cancel(future, started: dict) -> None:
    """Cancel the question's task, as the report stop button does, and wait for it to end."""
    while "task" not in started and not future.done():
        time.sleep(0.01)
    if "task" in started:
        task = started["task"]
        task.get_loop().call_soon_threadsafe(task.cancel)
    with suppress(Exception, asyncio.CancelledError, FutureCancelledError):
        future.result()


def _run(coro, first_step: str = "Planning"):
    """
    Run a coroutine to completion from sync code, inside Jupyter or not.
    In Jupyter a status line shows first_step until a step reports itself.
    """
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        loop_running = False
    else:
        loop_running = True
    # Run outside the except block, or every error raised by the question
    # would also print a chained "no running event loop" traceback
    if not loop_running:
        # asyncio.run already cancels the question on Ctrl-C
        return asyncio.run(coro)

    titles: list[str] = []
    started: dict[str, asyncio.Task] = {}

    async def tracked():
        started["task"] = asyncio.current_task()
        # Set inside the task, so the steps it runs inherit it
        step_title_callback.set(titles.append)
        return await coro

    # A running loop (Jupyter) cannot be re-entered, so run on a fresh loop
    # in a worker thread instead of patching the loop with nest_asyncio.
    # The status line is updated from this thread, so it stays in the cell.
    # ponytail: aask has no status line, add one if async callers want it
    with ThreadPoolExecutor(1) as pool, _StatusLine(first_step) as status:
        future = pool.submit(asyncio.run, tracked())
        try:
            while True:
                try:
                    return future.result(timeout=0.25)
                except FutureTimeoutError:
                    status.show(titles[-1] if titles else first_step)
        except KeyboardInterrupt:
            _cancel(future, started)
            raise


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


def _build_answer(
    question: str, plan: Plan, outputs: list, context: dict[str, Any], source: Source, llm: Llm,
    data: DataT | Pipeline, agents: list | None,
) -> Answer:
    if plan.status == "error":
        failed = next((task.title for task in plan if task.status == "error"), plan.title)
        exc = context.get("__exception__")
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
    return Answer(
        agents=agents,
        context=context,
        data=data,
        editor=editor,
        llm=llm,
        pipeline=context.get("pipeline"),
        plan=plan,
        question=question,
        spec=view.spec if view is not None and view.view_type == "vegalite" else None,
        source=source,
        sql=context.get("sql"),
        status=plan.status,
        truncated_at=_row_cap_hit(context.get("pipeline")),
        value=str(messages[-1].object) if messages else "",
        view=view,
    )


async def aask(
    data: DataT | Pipeline, question: str, *, llm: Llm | None = None, agents: list | None = None
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

    Returns
    -------
    Answer
    """
    return await _aask_messages(data, question, [{"role": "user", "content": question}], llm, agents)


async def _aask_messages(
    data: DataT | Pipeline, question: str, messages: list[dict[str, str]],
    llm: Llm | None, agents: list | None,
) -> Answer:
    source, visible_slugs = _resolve(data)
    planner = _planner(source, llm or _default_llm(), _settings["agents"] if agents is None else agents)
    context = _fresh_context(planner, source, visible_slugs)
    answer, _ = await _respond(planner, question, messages, context, data, agents)
    return answer


def ask(
    data: DataT | Pipeline, question: str, *, llm: Llm | None = None, agents: list | None = None
) -> Answer:
    """
    Ask a question of some data and return the Answer.

    Takes the same arguments as aask, and works whether or not an event
    loop is already running, e.g. in a Jupyter notebook.
    """
    return _run(aask(data, question, llm=llm, agents=agents))


def chat(data: DataT | Pipeline, *, llm: Llm | None = None, agents: list | None = None) -> Chat:
    """
    Start a conversation with some data, where each question can follow up
    on the answers before it. Takes the same arguments as aask.
    """
    return Chat(data, llm=llm, agents=agents)


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

    def __repr__(self) -> str:
        return f"Chat(answers={len(self.answers)})"

    async def aask(self, question: str) -> Answer:
        """Ask the next question and wait for the Answer."""
        agents = _settings["agents"] if self._agents is None else self._agents
        planner = _planner(self._source, self._llm or _default_llm(), agents)
        if self._context is None:
            context = _fresh_context(planner, self._source, self._visible_slugs)
        else:
            context = dict(self._context)
        messages = [*self._messages, {"role": "user", "content": question}]
        answer, out_context = await _respond(planner, question, messages, context, self._data, self._agents)
        self.answers = [*self.answers, answer]
        if answer.status == "error":
            # The next question starts from the last answer that worked
            return answer
        self._messages = [*messages, {"role": "assistant", "content": answer.value or answer.plan.title}]
        self._context = out_context
        return answer

    def ask(self, question: str) -> Answer:
        """Ask the next question and return the Answer."""
        return _run(self.aask(question))


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
        The status of the executed Plan.""")

    truncated_at = param.Integer(default=None, allow_None=True, doc="""
        The row cap the data stopped at, when the result had more rows.
        lumen.ai.configure(max_rows=...) sets the cap.""")

    value = param.String(default="", doc="""
        The written answer.""")

    view = param.ClassSelector(class_=View, allow_None=True, doc="""
        The chart as a Lumen View, if one was made.""")

    def __init__(
        self, llm: Llm | None = None, data: DataT | Pipeline | None = None,
        agents: list | None = None, editor: LumenEditor | None = None,
        context: dict[str, Any] | None = None, sql_failed: bool = False, **params,
    ):
        super().__init__(**params)
        self._card = None
        self._llm = llm
        # What was asked and with which agents, so retry asks the same way
        self._data = data
        self._agents = agents
        # The chart's editor and the context it was made in, for refine
        self._editor = editor
        self._context = context or {}
        # Whether sql is the query that failed, rather than one that worked
        self._sql_failed = sql_failed
        self._crs_warned = False

    async def arefine(self, instruction: str) -> Answer:
        """
        Change the chart, e.g. its colors, title or encodings, without
        querying the data again. Returns a new Answer.
        """
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
        with actor.param.update(llm=actor.llm or self._llm):
            editor.spec = await actor.revise(instruction, messages, self._context, view=editor)
        return Answer(
            agents=self._agents, context=self._context, data=self._data, editor=editor,
            llm=self._llm, pipeline=self.pipeline, plan=self.plan, question=self.question,
            source=self.source, spec=editor.component.spec, sql=self.sql, status=self.status,
            value=self.value, view=editor.component,
        )

    def refine(self, instruction: str) -> Answer:
        """Change the chart and return the new Answer. See arefine."""
        return _run(self.arefine(instruction), "Revising the chart")

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
        # Drawn now, so a spec the view cannot draw fails here and not later in the card
        view.get_panel()
        return Answer(
            agents=self._agents, context=self._context, data=self._data,
            editor=type(self._editor)(component=view) if self._editor is not None else None,
            llm=self._llm, pipeline=self.pipeline, plan=self.plan, question=self.question,
            source=self.source, spec=view.spec if self.spec is not None else None, sql=self.sql,
            status=self.status, truncated_at=self.truncated_at, value=self.value, view=view,
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
        """
        messages = [{"role": "user", "content": self.question}]
        if hint:
            outcome = f"That failed: {self.error}" if self.status == "error" else self.value
            if self.sql:
                outcome += f"\n\nThe SQL was:\n{self.sql}"
            messages += [{"role": "assistant", "content": outcome}, {"role": "user", "content": hint}]
        data = self.source if self._data is None else self._data
        return await _aask_messages(data, self.question, messages, self._llm, self._agents)

    def retry(self, hint: str | None = None) -> Answer:
        """Ask the question again and return the new Answer. See aretry."""
        return _run(self.aretry(hint))

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

    def to_file(self, path: str | Path) -> Path:
        """
        Write the answer's data to a file, in the format its extension names:
        ``.parquet`` (GeoParquet when the data has geometry), ``.geojson``,
        ``.gpkg`` and ``.csv`` for tables, ``.zarr`` and ``.nc`` for grids.
        Returns the path.
        """
        path = Path(path)
        suffix = path.suffix.lower()
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
            return viz(gdf)
        if importlib.util.find_spec("geoviews") is not None:
            return gdf.hvplot(geo=True, tiles="OSM")
        return gdf.hvplot()

    def explore(self, **params) -> ExplorerUI:
        """
        Open this answer in the Lumen chat UI: an ExplorerUI on the same
        Source and LLM, with the answer as its first exploration. Keyword
        arguments go to ExplorerUI.
        """
        ui = ExplorerUI(data=self.source, llm=self._llm, **params)

        async def seed():
            ui.interface.send(self.question, respond=False)
            await ui.add_plan_exploration(self.plan)

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
        component = self.view if self.view is not None else self.pipeline
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

    @property
    def code(self) -> str:
        """
        Python that rebuilds the chart, or else the data, with Lumen alone.

        No LLM is called. In-memory data is embedded in the spec, so the
        code runs anywhere Lumen is installed; for a large table use
        ``export``, which keeps the data in Parquet files beside a small
        script. File-based sources keep their absolute paths.
        """
        component = self.view if self.view is not None else self.pipeline
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
        if self._card is None and self.status == "error":
            parts = [Alert(object=self.error, severity="error", title=self.value, sizing_mode="stretch_width")]
            if self.sql:
                parts.append(Card(
                    Markdown(f"```sql\n{self.sql}\n```"), title="Failed SQL" if self._sql_failed else "SQL",
                    collapsed=not self._sql_failed, sizing_mode="stretch_width",
                ))
            self._card = Column(*parts)
        elif self._card is None:
            parts = [Markdown(self.value, sizing_mode="stretch_width")] if self.value else []
            if self.view is not None:
                parts.append(self._chart_part(self.view))
            if self._cap_note:
                parts.append(Alert(object=self._cap_note, severity="info", sizing_mode="stretch_width"))
            if self.sql:
                parts.append(Card(
                    Markdown(f"```sql\n{self.sql}\n```"), title="SQL", collapsed=True,
                    sizing_mode="stretch_width",
                ))
            if (chart_spec := self._chart_spec()) is not None:
                parts.append(self._spec_card(*chart_spec, chart_index=1 if self.value else 0))
            self._card = Column(*parts)
        return self._card

    def _chart_part(self, view: View):
        if isinstance(view, hvPlotView) and view.kwargs.get("responsive"):
            # A responsive plot stretches to its container, and a notebook
            # output has no height to give it, so it would collapse
            return Column(view, min_height=400, sizing_mode="stretch_both")
        return view

    def _spec_card(self, language: str, text: str, chart_index: int) -> Card:
        """
        The chart's spec as a code block, which has a copy button, and an
        Edit button that swaps in an editor to change the spec and redraw.
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
            spec = {key: value for key, value in self.spec.items() if key != "datasets"}
            if isinstance(spec.get("data"), dict) and "values" in spec["data"]:
                spec["data"] = {key: value for key, value in spec["data"].items() if key != "values"}
            return "json", json.dumps(spec, indent=2, default=str)
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
            "data": {"values": geometry_to_geojson(gdf)["features"]},
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
        return "\n\n".join(part for part in (self.value, self.error, self._cap_note) if part)

    @property
    def _cap_note(self) -> str:
        if self.truncated_at is None:
            return ""
        return (
            f"The data stops at the first {self.truncated_at:,} rows. "
            "Use lumen.ai.configure(max_rows=...) for more, or None for every row."
        )

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
        card = [_prose_to_html(self.value)] if self.value else []
        if self.error:
            card.append(f"<pre>{html.escape(self.error)}</pre>")
        if chart is not None:
            bundle.update({k: v for k, v in chart.items() if k != "text/plain"})
            width = chart["application/vnd.vegalite.v5+json"]["width"]
            # Rendered at twice the size for sharpness, shown at the spec width
            card.append(f'<img src="data:image/png;base64,{chart["image/png"]}" width="{width}" style="max-width: 100%">')
        if self._cap_note:
            card.append(f"<p><em>{html.escape(self._cap_note)}</em></p>")
        if self.sql:
            card.append(self._sql_html())
        if chart_spec is not None:
            card.append(f"<details><summary>Spec</summary><pre><code>{html.escape(chart_spec[1])}</code></pre></details>")
        bundle["text/html"] = "\n".join(card)
        return bundle

    def _repr_mimebundle_(self, include=None, exclude=None):
        if not _in_kernel():
            return self._static_bundle(self._chart_bundle())
        # Loaded on first display, not on import, so lumen.ai stays
        # quiet in scripts and in front-ends that never show an Answer.
        # Checked per extension: a plain pn.extension() leaves Vega out,
        # and the Spec section's editor needs codeeditor.
        missing = [ext for ext in ("vega", "codeeditor") if ext not in panel_extension._loaded_extensions]
        if missing:
            pn.extension(*missing)
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
