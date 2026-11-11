"""
A first look at a Source from a notebook: its tables and columns, what
looks wrong in the data, and questions worth asking it.
"""
from __future__ import annotations

import html

import pandas as pd
import param

from panel.pane import DataFrame as DataFramePane, Markdown
from panel_material_ui import (
    Alert, Button, Column, FlexBox,
)
from pydantic import Field, create_model

from ..sources import Source
from ..sources.base import BaseSQLSource
from .answer import (
    _default_llm, _in_kernel, _load_extensions, _run,
)
from .data_quality import lint_data
from .llm import Llm
from .sharing import shares_values, sharing
from .table_stats import get_stats_store
from .usage import UsageCollector
from .utils import format_error

#: Rows the quality checks read per table
SAMPLE_ROWS = 5_000


class Overview(param.Parameterized):
    """
    A Source at a glance: per table, its columns and what looks wrong in
    the data, and questions to ask it. In a notebook each question is a
    button that asks it.
    """

    columns = param.Dict(doc="""
        Per table, a DataFrame of its columns: type, nulls (%), distinct,
        min and max where the statistics have them.""")

    flags = param.Dict(doc="""
        Per table, what looks wrong in its first rows: missing values,
        duplicates, padded text, numbers stored as text, constant columns
        and outliers.""")

    questions = param.List(item_type=str, doc="""
        Questions to ask the data, written by the LLM from the table and
        column names, types and statistics.""")

    rows = param.Dict(doc="""
        Per table, its row count, or None when it is not known.""")

    sampled = param.Dict(doc="""
        Per table whose distinct counts and ranges come from a sample of a
        large table rather than every row, the sample's size.""")

    source = param.ClassSelector(class_=Source, doc="""
        The Source this is an overview of.""")

    usage = param.ClassSelector(class_=UsageCollector, default=None, allow_None=True, doc="""
        The tokens and cost of writing the questions.""")

    def __init__(self, **params):
        super().__init__(**params)
        self._panel = None

    def __repr__(self) -> str:
        tables = ", ".join(f"{table} ({rows:,} rows)" if rows is not None else table for table, rows in self.rows.items())
        return f"Overview({tables}; {sum(map(len, self.flags.values()))} flags, {len(self.questions)} questions)"

    __str__ = __repr__

    def __panel__(self) -> Column:
        if self._panel is not None:
            return self._panel
        parts: list = []
        for table, columns in self.columns.items():
            rows = self.rows.get(table)
            heading = f"### {table}" + (f" · {rows:,} rows" if rows is not None else "")
            if table in self.sampled:
                heading += f"\n\n*Distinct counts and ranges are from a sample of {self.sampled[table]:,} rows.*"
            parts.append(Markdown(heading))
            parts.append(DataFramePane(columns, index=False, sizing_mode="stretch_width"))
            if self.flags.get(table):
                parts.append(Alert(
                    object="\n".join(f"- {flag}" for flag in self.flags[table]), severity="warning",
                    title="Worth checking", sizing_mode="stretch_width",
                ))
        answers = Column(sizing_mode="stretch_width")
        if self.questions:
            parts.append(Markdown("**Ask**"))
            parts.append(FlexBox(*(self._button(question, answers) for question in self.questions)))
        self._panel = Column(*parts, answers, sizing_mode="stretch_width")
        return self._panel

    def _button(self, question: str, answers: Column) -> Button:
        button = Button(label=question, variant="outlined")

        async def ask(event):
            button.loading = True
            try:
                answer = await self.source.aask(question)
            except Exception as e:
                # A click has no caller to raise to, e.g. on a rate limit
                answers.append(Alert(object=format_error(e), title="Lumen could not answer", severity="error"))
                return
            finally:
                button.loading = False
            answers.append(answer.__panel__())

        button.on_click(ask)
        return button

    def _repr_mimebundle_(self, include=None, exclude=None):
        if _in_kernel():
            _load_extensions()
            bundle = self.__panel__()._repr_mimebundle_(include, exclude)
            if bundle is not None:
                return bundle
        parts = []
        for table, columns in self.columns.items():
            parts.append(f"<h3>{html.escape(table)}</h3>{columns.to_html(index=False)}")
            if table in self.sampled:
                parts.append(f"<p><em>Distinct counts and ranges are from a sample of {self.sampled[table]:,} rows.</em></p>")
            if self.flags.get(table):
                parts.append("<ul>" + "".join(f"<li>{html.escape(flag)}</li>" for flag in self.flags[table]) + "</ul>")
        if self.questions:
            parts.append("<ol>" + "".join(f"<li>{html.escape(q)}</li>" for q in self.questions) + "</ol>")
        return {"text/plain": repr(self), "text/html": "\n".join(parts)}


def _first_rows(source: Source, table: str) -> pd.DataFrame:
    """Up to SAMPLE_ROWS rows of a table, read without loading the rest."""
    if isinstance(source, BaseSQLSource):
        # ponytail: the first rows, not a random sample; a sorted table can hide its problems
        return source.execute(f"SELECT * FROM ({source.get_sql_expr(table)}) AS t LIMIT {SAMPLE_ROWS}")
    return source.get(table).head(SAMPLE_ROWS)


def _describe(table: str, rows: int | None, columns: pd.DataFrame) -> str:
    """
    The table for the questions prompt, as much as sharing allows: names
    only for "none", types and row counts for "schema", ranges for "sample".
    """
    if sharing.share == "none":
        return f"Table {table}:\n" + "\n".join(columns["column"])
    keep = ["column", "type"]
    if sharing.share == "sample" and {"min", "max"} <= set(columns):
        keep += ["min", "max"]
        columns = columns.copy()
        # Ranges are values too, e.g. of a birthdate or a phone number stored as int
        hidden = [
            not shares_values(row.column, [row.min, row.max]) for row in columns.itertuples()
        ]
        columns.loc[hidden, ["min", "max"]] = None
    return f"Table {table} ({rows if rows is not None else 'unknown'} rows):\n" + columns[keep].to_csv(index=False)


async def aoverview(source: Source, *, questions: bool = True, llm: Llm | None = None) -> Overview:
    """
    Look over a Source: its tables' columns, quality flags from their first
    rows, and, with questions, five questions an LLM suggests from the
    names, types and statistics. See Source.overview.
    """
    tables = source.get_tables()
    stats = await get_stats_store().ensure(source, tables, timeout=30)
    columns, flags, rows, sampled = {}, {}, {}, {}
    for table in tables:
        if (table_stats := stats.get(table)) is not None:
            rows[table] = table_stats.rows
            if not table_stats.ranges_exact and table_stats.sample_size:
                sampled[table] = table_stats.sample_size
            columns[table] = pd.DataFrame([{
                "column": c.name, "type": c.type,
                "nulls (%)": None if c.nulls is None else round(100 * c.nulls, 1),
                "distinct": c.distinct, "min": c.min, "max": c.max,
            } for c in table_stats.columns])
        else:
            schema = source.get_schema(table)
            rows[table] = schema.get("__len__")
            columns[table] = pd.DataFrame([
                {"column": name, "type": prop.get("type")} for name, prop in schema.items() if name != "__len__"
            ])
        flags[table] = lint_data(_first_rows(source, table))
    overview = Overview(columns=columns, flags=flags, rows=rows, sampled=sampled, source=source)
    if not questions:
        return overview
    llm = llm or _default_llm()
    model = create_model("Questions", questions=(list[str], Field(min_length=1, max_length=5)))
    prompt = "\n\n".join(_describe(table, rows[table], columns[table].reindex(columns=["column", "type", "min", "max"])) for table in tables)
    with llm.capture_usage() as usage:
        out = await llm.invoke(
            [{"role": "user", "content": prompt}], response_model=model,
            system=(
                "Suggest five short questions a person exploring these tables would ask, each "
                "answerable from their columns with one query, and some answered best by a chart."
            ),
        )
    overview.param.update(questions=out.questions, usage=usage)
    return overview


def overview(source: Source, *, questions: bool = True, llm: Llm | None = None) -> Overview:
    """Look over a Source and return the Overview. See aoverview."""
    return _run(aoverview(source, questions=questions, llm=llm), "Looking over the data")
