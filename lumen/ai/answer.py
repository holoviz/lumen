"""
Ask a Lumen Source or Pipeline a question from Python and get an Answer.
"""
from __future__ import annotations

import asyncio
import base64
import html
import logging
import weakref

from concurrent.futures import ThreadPoolExecutor
from functools import cache
from typing import TYPE_CHECKING, Any

import panel as pn
import param

from panel.config import panel_extension
from panel.io.state import state
from panel.pane import Markdown, Vega
from panel.pane.vega import ds_to_records
from panel_material_ui import Card, Column
from panel_material_ui.chat import ChatMessage

from ..config import SOURCE_TABLE_SEPARATOR
from ..pipeline import Pipeline
from ..sources import Source
from ..sources.base import BaseSQLSource
from ..sources.xarray_sql import XArraySQLSource
from ..util import geometry_columns, log
from ..views import View
from .agents import ChatAgent, SQLAgent, VegaLiteAgent
from .agents.hvplot import hvPlotAgent
from .coordinator import Plan, Planner
from .data import read
from .editors import LumenEditor
from .export import format_output, make_preamble
from .llm import (
    Llm, get_available_llm, quiet_retry_log, signed_in_cli_hint,
)
from .story import _prose_to_html
from .ui import ExplorerUI

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


def _has_geometry(source: Source) -> bool:
    # Only SQL sources can be probed without loading data. DuckDB returns
    # GEOMETRY as a GeoDataFrame only when geopandas is installed, which is
    # also what hvPlot needs to draw it.
    if not isinstance(source, BaseSQLSource):
        return False
    return any(
        geometry_columns(source.execute(f"SELECT * FROM ({source.get_sql_expr(table)}) LIMIT 0"))
        for table in source.get_tables()
    )


def _default_agents(source: Source) -> list:
    agents = [ChatAgent(), SQLAgent(), VegaLiteAgent()]
    if isinstance(source, XArraySQLSource) or _has_geometry(source):
        agents.append(hvPlotAgent())
    return agents


# One instance for every ask, so its client and the request fixes it
# learns from provider errors carry over from question to question
@cache
def _default_llm() -> Llm:
    llm_cls = get_available_llm()
    if llm_cls is None:
        raise RuntimeError(
            "No LLM provider found. Pass llm=lumen.ai.llm.Kilo() for free models without an "
            f"account{signed_in_cli_hint()}, or set an API key such as OPENAI_API_KEY or "
            "GROQ_API_KEY and restart the kernel."
        )
    return llm_cls()


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
    planner: Planner, question: str, messages: list[dict[str, str]], context: dict[str, Any]
) -> tuple[Answer, dict[str, Any]]:
    """Run one question through the Planner, returning the Answer and the context it left."""
    if log.getEffectiveLevel() > logging.DEBUG:
        quiet_retry_log()
    await planner.sync(context)
    plan = await planner.respond(messages, context)
    if plan is None:
        raise RuntimeError(f"Lumen could not plan an answer to {question!r}.")
    outputs, out_context = await plan.execute()
    if plan.status == "error":
        failed = next((task.title for task in plan if task.status == "error"), plan.title)
        raise RuntimeError(f"Lumen could not answer {question!r}: {failed!r} failed.")
    answer = _build_answer(question, plan, outputs, out_context, context["source"], planner.llm)
    return answer, out_context


def _run(coro):
    """Run a coroutine to completion from sync code, inside Jupyter or not."""
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        loop_running = False
    else:
        loop_running = True
    # Run outside the except block, or every error raised by the question
    # would also print a chained "no running event loop" traceback
    if not loop_running:
        return asyncio.run(coro)
    # A running loop (Jupyter) cannot be re-entered, so run on a fresh loop
    # in a worker thread instead of patching the loop with nest_asyncio
    with ThreadPoolExecutor(1) as pool:
        return pool.submit(asyncio.run, coro).result()


def _build_answer(
    question: str, plan: Plan, outputs: list, context: dict[str, Any], source: Source, llm: Llm
) -> Answer:
    messages = [out for out in outputs if isinstance(out, ChatMessage)]
    views = [
        out.component for out in outputs
        if isinstance(out, LumenEditor) and isinstance(out.component, View)
    ]
    view = views[-1] if views else None
    return Answer(
        llm=llm,
        pipeline=context.get("pipeline"),
        plan=plan,
        question=question,
        spec=view.spec if view is not None and view.view_type == "vegalite" else None,
        source=source,
        sql=context.get("sql"),
        status=plan.status,
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
    source, visible_slugs = _resolve(data)
    planner = _planner(source, llm or _default_llm(), agents)
    context = _fresh_context(planner, source, visible_slugs)
    answer, _ = await _respond(planner, question, [{"role": "user", "content": question}], context)
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
        self._source, self._visible_slugs = _resolve(data)
        self._llm = llm
        self._agents = agents
        self._messages: list[dict[str, str]] = []
        self._context: dict[str, Any] | None = None

    def __repr__(self) -> str:
        return f"Chat(answers={len(self.answers)})"

    async def aask(self, question: str) -> Answer:
        """Ask the next question and wait for the Answer."""
        planner = _planner(self._source, self._llm or _default_llm(), self._agents)
        if self._context is None:
            context = _fresh_context(planner, self._source, self._visible_slugs)
        else:
            context = dict(self._context)
        messages = [*self._messages, {"role": "user", "content": question}]
        answer, out_context = await _respond(planner, question, messages, context)
        self._messages = [*messages, {"role": "assistant", "content": answer.value or answer.plan.title}]
        self._context = out_context
        self.answers = [*self.answers, answer]
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

    value = param.String(default="", doc="""
        The written answer.""")

    view = param.ClassSelector(class_=View, allow_None=True, doc="""
        The chart as a Lumen View, if one was made.""")

    def __init__(self, llm: Llm | None = None, **params):
        super().__init__(**params)
        self._card = None
        self._llm = llm

    def explore(self, **params) -> ExplorerUI:
        """
        Open this answer in the Lumen chat UI: an ExplorerUI on the same
        Source and LLM, with the answer as its first exploration. Keyword
        arguments go to ExplorerUI.
        """
        ui = ExplorerUI(data=self.source, llm=self._llm, **params)

        async def seed():
            ui.interface.send(self.question, respond=False)
            await ui._add_plan_exploration(self.plan)

        # Runs once the page is served, or on the notebook's running loop
        state.execute(seed)
        return ui

    @property
    def code(self) -> str:
        """
        Python that rebuilds the chart, or else the data, with Lumen alone.

        The data and the query travel in the spec, so the code runs
        anywhere Lumen is installed without calling an LLM.
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
        if self._card is None:
            parts = [Markdown(self.value, sizing_mode="stretch_width")] if self.value else []
            if self.view is not None:
                parts.append(self.view)
            if self.sql:
                parts.append(Card(
                    Markdown(f"```sql\n{self.sql}\n```"), title="SQL", collapsed=True,
                    sizing_mode="stretch_width",
                ))
            self._card = Column(*parts)
        return self._card

    def _vega_lite_spec(self) -> dict | None:
        """The chart as a self-contained Vega-Lite spec, data inlined."""
        if self.spec is None:
            return None
        spec = dict(self.view.get_panel().object)
        data = spec.get("data")
        # A spec can bring its own data, e.g. a topojson url for a map
        if isinstance(data, dict) and "values" in data:
            spec["data"] = dict(data, values=ds_to_records(data["values"]))
        for dim, size in (("width", 600), ("height", 400)):
            if spec.get(dim, "container") == "container":
                spec[dim] = size
        return spec

    def _chart_bundle(self) -> dict | None:
        """The chart as interactive Vega-Lite, with a PNG for viewers that cannot run it."""
        if (spec := self._vega_lite_spec()) is None:
            return None
        title = spec.get("title", "")
        if isinstance(title, dict):
            title = title.get("text", "")
        return {
            "application/vnd.vegalite.v5+json": spec,
            "image/png": base64.b64encode(Vega(spec).export("png", scale=2)).decode(),
            "text/plain": f"<Vega-Lite chart: {title}>",
        }

    def _sql_html(self) -> str:
        return f"<details><summary>SQL</summary><pre><code>{html.escape(self.sql)}</code></pre></details>"

    def _static_bundle(self, chart: dict | None) -> dict:
        # A viewer shows one mime type per output, so the HTML carries the
        # whole card (answer, chart, SQL) and is what notebook viewers prefer
        sql = f"\n\n```sql\n{self.sql}\n```" if self.sql else ""
        bundle = {"text/plain": repr(self), "text/markdown": f"{self.value}{sql}"}
        card = [_prose_to_html(self.value)] if self.value else []
        if chart is not None:
            bundle.update({k: v for k, v in chart.items() if k != "text/plain"})
            width = chart["application/vnd.vegalite.v5+json"]["width"]
            # Rendered at twice the size for sharpness, shown at the spec width
            card.append(f'<img src="data:image/png;base64,{chart["image/png"]}" width="{width}" style="max-width: 100%">')
        if self.sql:
            card.append(self._sql_html())
        bundle["text/html"] = "\n".join(card)
        return bundle

    def _repr_mimebundle_(self, include=None, exclude=None):
        if not _in_kernel():
            return self._static_bundle(self._chart_bundle())
        # Loaded on first display, not on import, so lumen.ai stays
        # quiet in scripts and in front-ends that never show an Answer.
        # Checked per extension: a plain pn.extension() leaves Vega out.
        if "vega" not in panel_extension._loaded_extensions:
            pn.extension("vega")
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
        if self.value:
            display({"text/markdown": self.value, "text/plain": self.value}, raw=True)
        if (chart := self._chart_bundle()) is not None:
            display(chart, raw=True)
        if not self.sql:
            return {"text/plain": ""}
        return {
            "text/markdown": f"```sql\n{self.sql}\n```",
            "text/html": self._sql_html(),
            "text/plain": self.sql,
        }
