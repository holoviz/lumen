"""
Ask a Lumen Source or Pipeline a question from Python and get an Answer.
"""
from __future__ import annotations

import asyncio
import base64

from concurrent.futures import ThreadPoolExecutor
from functools import cache
from typing import TYPE_CHECKING, Any

import panel as pn
import param

from panel.config import panel_extension
from panel.pane import Markdown, Vega
from panel.pane.vega import ds_to_records
from panel_material_ui import Card, Column
from panel_material_ui.chat import ChatMessage

from ..config import SOURCE_TABLE_SEPARATOR
from ..pipeline import Pipeline
from ..sources import Source
from ..sources.base import BaseSQLSource
from ..sources.xarray_sql import XArraySQLSource
from ..util import geometry_columns
from ..views import View
from .agents import ChatAgent, SQLAgent, VegaLiteAgent
from .agents.hvplot import hvPlotAgent
from .coordinator import Plan, Planner
from .data import read
from .editors import LumenEditor
from .llm import Llm, get_available_llm
from .ui import _keyless_options_message

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
        raise RuntimeError(_keyless_options_message())
    return llm_cls()


def _resolve(data: DataT | Pipeline) -> tuple[Source, set[str] | None]:
    """Return the Source to ask and the table slugs it is limited to."""
    if not isinstance(data, Pipeline):
        return (data if isinstance(data, Source) else read(data)), None
    if data.filters or data.transforms or data.sql_transforms:
        # The rows the pipeline selects have to be what gets queried
        return read({data.table: data}), None
    return data.source, {f"{data.source.name}{SOURCE_TABLE_SEPARATOR}{data.table}"}


def _build_answer(question: str, plan: Plan, outputs: list, context: dict[str, Any]) -> Answer:
    messages = [out for out in outputs if isinstance(out, ChatMessage)]
    views = [
        out.component for out in outputs
        if isinstance(out, LumenEditor) and isinstance(out.component, View)
    ]
    view = views[-1] if views else None
    return Answer(
        pipeline=context.get("pipeline"),
        plan=plan,
        question=question,
        spec=view.spec if view is not None and view.view_type == "vegalite" else None,
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
    context: dict[str, Any] = {"source": source, "sources": [source]}
    if visible_slugs is not None:
        context["visible_slugs"] = visible_slugs
    planner = Planner(
        llm=llm or _default_llm(),
        agents=agents or _default_agents(source),
        context=context,
        headless=True,
    )
    await planner.sync(context)
    plan = await planner.respond([{"role": "user", "content": question}], context)
    if plan is None:
        raise RuntimeError(f"Lumen could not plan an answer to {question!r}.")
    outputs, out_context = await plan.execute()
    if plan.status == "error":
        failed = next((task.title for task in plan if task.status == "error"), plan.title)
        raise RuntimeError(f"Lumen could not answer {question!r}: {failed!r} failed.")
    return _build_answer(question, plan, outputs, out_context)


def ask(
    data: DataT | Pipeline, question: str, *, llm: Llm | None = None, agents: list | None = None
) -> Answer:
    """
    Ask a question of some data and return the Answer.

    Takes the same arguments as aask, and works whether or not an event
    loop is already running, e.g. in a Jupyter notebook.
    """
    coro = aask(data, question, llm=llm, agents=agents)
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)
    # A running loop (Jupyter) cannot be re-entered, so run on a fresh loop
    # in a worker thread instead of patching the loop with nest_asyncio
    with ThreadPoolExecutor(1) as pool:
        return pool.submit(asyncio.run, coro).result()


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

    def __init__(self, **params):
        super().__init__(**params)
        self._card = None

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
        spec["data"] = {"values": ds_to_records(spec["data"]["values"])}
        for dim, size in (("width", 600), ("height", 400)):
            if spec.get(dim, "container") == "container":
                spec[dim] = size
        return spec

    def _repr_mimebundle_(self, include=None, exclude=None):
        if _in_kernel():
            # Loaded on first display, not on import, so lumen.ai stays
            # quiet in scripts and in front-ends that never show an Answer
            if not panel_extension._loaded:
                pn.extension("vega")
            try:
                bundle = self.__panel__()._repr_mimebundle_(include, exclude)
            except ImportError:
                # VS Code renders Panel through jupyter_bokeh, which may be missing
                bundle = None
            if bundle is not None:
                return bundle
        sql = f"\n\n```sql\n{self.sql}\n```" if self.sql else ""
        bundle = {"text/plain": repr(self), "text/markdown": f"{self.value}{sql}"}
        if (spec := self._vega_lite_spec()) is not None:
            bundle["application/vnd.vegalite.v5+json"] = spec
            bundle["image/png"] = base64.b64encode(Vega(spec).export("png", scale=2)).decode()
        return bundle
