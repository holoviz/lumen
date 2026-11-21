"""
Lumen transforms as FlowDash nodes: aggregate, sort, select and derived
columns.

Each node takes a Lumen pipeline and outputs one, so a pipeline reads as a
chain of steps: table, filter, aggregate, sort, chart. Like a filter node, a
transform node fetches nothing itself. On a table read straight from a SQL
source, or a SQL node's result, its step becomes part of the source's query,
before the row limit; on other sources, and on a pipeline chained to another
one, the same SQL runs in DuckDB on the fetched rows. Every step therefore
means the same on every source, and a derived column's expression is SQL in
the source's dialect, or DuckDB's where the step runs on fetched rows.

Each node also works out the columns and column types of its output from its
input's schema, so the nodes after it offer and check columns without
fetching rows.
"""
from __future__ import annotations

import re

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any, ClassVar

import param
import sqlglot

from panel_flowdash import register
from panel_flowdash.dataflow_engine import failure_text
from panel_flowdash.util import is_failure
from sqlglot import exp
from sqlglot.optimizer.annotate_types import annotate_types
from sqlglot.optimizer.qualify import qualify

from ..pipeline import Pipeline
from ..transforms.base import Iloc, Transform
from ..transforms.sql import SQLLimit, SQLTransform
from .checks import (
    DENIED_FUNCTIONS, _function_name, _read, pipeline_columns,
)
from .choices import pipeline_column_types
from .nodes import (
    PIPELINE_TYPE, PipelineNode, _clone, row_ceiling,
)

AGGREGATE_ID = "Lumen/aggregate"
SORT_ID = "Lumen/sort"
SELECT_ID = "Lumen/select"
DERIVE_ID = "Lumen/derive"

# The dialect of the steps that run on fetched rows.
FRAME_DIALECT = "duckdb"

# Aggregation methods by the name a node's config uses, as SQL functions.
METHODS = {"count": "COUNT", "sum": "SUM", "mean": "AVG", "min": "MIN", "max": "MAX"}
NUMERIC_METHODS = ("sum", "mean")
# The output column of a count without columns, which counts rows.
COUNT_COLUMN = "count"

_NAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_ ]*$")


def _column(name: str) -> exp.Column:
    # Names come from the input's schema, which has them as the engine spells them.
    return exp.Column(this=exp.to_identifier(name, quoted=True))


def _alias(expression: exp.Expression, name: str) -> exp.Alias:
    return exp.alias_(expression, exp.to_identifier(name, quoted=True))


class SQLAggregate(SQLTransform):
    """
    Groups the rows by ``by`` and aggregates ``columns`` with ``method``,
    each keeping its name, ordered by the groups. Without ``by`` the whole
    table is one group; a count without ``columns`` counts rows into a
    ``count`` column.
    """

    by = param.List(default=[], item_type=str)

    columns = param.List(default=[], item_type=str)

    method = param.Selector(default="sum", objects=list(METHODS))

    transform_type: ClassVar[str] = "sql_aggregate"

    def apply(self, sql_in: str) -> str:
        function = METHODS[self.method]
        if self.columns:
            aggregates = [
                _alias(exp.func(function, _column(column)), column) for column in self.columns
            ]
        else:
            aggregates = [_alias(exp.Count(this=exp.Star()), COUNT_COLUMN)]
        keys = [_column(column) for column in self.by]
        expression = exp.select(*keys, *aggregates).from_(self._to_subquery(self.parse_sql(sql_in)))
        if keys:
            expression = expression.group_by(*keys).order_by(*keys)
        return self.to_sql(expression)


class SQLOrder(SQLTransform):
    """
    Sorts the rows by ``by`` and keeps the first ``limit`` of them, if set.
    Unlike ``SQLLimit`` the limit is a step of the chain, so a filter added
    later filters the rows it kept rather than being placed before it.
    """

    by = param.List(default=[], item_type=str)

    descending = param.Boolean(default=False)

    limit = param.Integer(default=None, allow_None=True)

    transform_type: ClassVar[str] = "sql_order"

    def apply(self, sql_in: str) -> str:
        expression = exp.select("*").from_(self._to_subquery(self.parse_sql(sql_in)))
        if self.by:
            expression = expression.order_by(*(
                exp.Ordered(this=_column(column), desc=self.descending) for column in self.by
            ))
        if self.limit is not None:
            expression = expression.limit(self.limit)
        return self.to_sql(expression)


class SQLSelect(SQLTransform):
    """Keeps ``columns``, in that order."""

    columns = param.List(default=[], item_type=str)

    transform_type: ClassVar[str] = "sql_select"

    def apply(self, sql_in: str) -> str:
        subquery = self._to_subquery(self.parse_sql(sql_in))
        return self.to_sql(exp.select(*map(_column, self.columns)).from_(subquery))


class SQLDerive(SQLTransform):
    """
    Adds a column ``output`` computed by the SQL ``expression`` to the input
    ``columns``, or replaces the one of that name in its place.
    """

    columns = param.List(default=[], item_type=str)

    output = param.String(default="")

    expression = param.String(default="")

    transform_type: ClassVar[str] = "sql_derive"

    def apply(self, sql_in: str) -> str:
        value = _alias(sqlglot.parse_one(self.expression, read=self.read), self.output)
        projection = [value if column == self.output else _column(column) for column in self.columns]
        if self.output not in self.columns:
            projection.append(value)
        subquery = self._to_subquery(self.parse_sql(sql_in))
        return self.to_sql(exp.select(*projection).from_(subquery))


class FrameSQL(Transform):
    """
    Runs SQL transforms written for DuckDB on fetched rows, for the steps of
    a pipeline whose source does not run SQL.
    """

    steps = param.List(default=[], item_type=SQLTransform)

    transform_type: ClassVar[str] = "frame_sql"

    def apply(self, table: Any) -> Any:
        import duckdb

        sql = "SELECT * FROM frame"
        for step in self.steps:
            sql = step.apply(sql)
        with duckdb.connect() as connection:
            connection.register("frame", table)
            return connection.execute(sql).df()


def step_dialect(pipeline: Pipeline) -> str:
    """The SQL dialect a step on ``pipeline`` is written in."""
    source = pipeline.source
    if pipeline.pipeline is None and source._supports_sql:
        return source.dialect
    return FRAME_DIALECT


def transformed_pipeline(pipeline: Pipeline, step: SQLTransform, schema: dict[str, Any],
                         limit: int) -> Pipeline:
    """
    ``pipeline`` followed by ``step``, written in ``step_dialect(pipeline)``,
    with ``schema`` as its columns. On a pipeline read straight from a SQL
    source the step becomes part of the source's query, before the row
    limit, which the query gets at ``limit`` rows unless it has one;
    otherwise it runs on the fetched rows, before any row limit.
    """
    if is_failure(pipeline):
        raise ValueError(f"The input failed: {failure_text(pipeline)}")
    if not isinstance(pipeline, Pipeline):
        raise ValueError(f"Expected a Lumen pipeline, got {type(pipeline).__name__}.")
    if pipeline.pipeline is None and pipeline.source._supports_sql:
        sql_transforms = list(pipeline.sql_transforms)
        at = next((i for i, t in enumerate(sql_transforms) if isinstance(t, SQLLimit)), None)
        if at is None:
            at = len(sql_transforms)
            sql_transforms.append(SQLLimit(limit=limit, write=pipeline.source.dialect))
        sql_transforms.insert(at, step)
        return _clone(pipeline, sql_transforms=sql_transforms, schema=schema)
    transforms = list(pipeline.transforms)
    at = next((i for i, t in enumerate(transforms) if isinstance(t, Iloc)), len(transforms))
    transforms.insert(at, FrameSQL(steps=[step]))
    return _clone(pipeline, transforms=transforms, schema=schema)


def _typed(spec: Mapping[str, Any] | None) -> dict[str, Any]:
    """A column's schema without the statistics a step may change, e.g. its range or values."""
    return {key: value for key, value in (spec or {}).items() if key in ("type", "format")}


def _columns_of(schema: Mapping[str, Any]) -> list[str]:
    return [str(column) for column in schema if column != "__len__"]


def aggregate_schema(schema: Mapping[str, Any], by: list[str], columns: list[str], method: str) -> dict[str, Any]:
    result = {column: _typed(schema.get(column)) for column in by}
    if not columns:
        result[COUNT_COLUMN] = {"type": "integer"}
    for column in columns:
        if method == "count":
            result[column] = {"type": "integer"}
        elif method == "mean":
            result[column] = {"type": "number"}
        else:
            result[column] = _typed(schema.get(column))
    return result


# The types a derived column's expression is typed with, by the schema's type and format.
_SQL_TYPES = {"integer": "BIGINT", "number": "DOUBLE", "boolean": "BOOLEAN", "string": "VARCHAR",
              "datetime": "TIMESTAMP", "date": "DATE"}


def _schema_type(data_type: exp.DataType) -> dict[str, Any]:
    if data_type.is_type(*exp.DataType.INTEGER_TYPES):
        return {"type": "integer"}
    if data_type.is_type(*exp.DataType.REAL_TYPES):
        return {"type": "number"}
    if data_type.is_type(exp.DType.BOOLEAN):
        return {"type": "boolean"}
    if data_type.is_type(exp.DType.DATE):
        return {"type": "string", "format": "date"}
    if data_type.is_type(*exp.DataType.TEMPORAL_TYPES):
        return {"type": "string", "format": "datetime"}
    if data_type.is_type(*exp.DataType.TEXT_TYPES):
        return {"type": "string"}
    return {}


def expression_type(expression: str, schema: Mapping[str, Any], dialect: str) -> dict[str, Any]:
    """The schema of the column ``expression`` computes from the columns of ``schema``, or ``{}`` if unknown."""
    types = {}
    for column in _columns_of(schema):
        spec = schema.get(column) or {}
        kind = spec.get("format") if spec.get("format") in ("datetime", "date") else spec.get("type")
        types[column] = _SQL_TYPES.get(kind, "VARCHAR")
    try:
        tables = {"frame": types}
        select = exp.select(_alias(sqlglot.parse_one(expression, read=_read(dialect)), "value")).from_("frame")
        select = annotate_types(qualify(select, schema=tables, dialect=_read(dialect)), schema=tables)
        return _schema_type(select.selects[0].type)
    except Exception:
        return {}


def derive_schema(schema: Mapping[str, Any], name: str, expression: str, dialect: str) -> dict[str, Any]:
    value = expression_type(expression, schema, dialect)
    result = {column: (value if column == name else schema[column]) for column in schema}
    if name not in result:
        result[name] = value
    return result


def parse_expression(expression: str, dialect: str) -> exp.Expression:
    """
    The one scalar SQL expression ``expression`` parsed, or ``ValueError``
    saying why it is not one: it must not be a query or read a table, use
    an aggregate outside a window, name itself, or call a function in
    ``DENIED_FUNCTIONS``.
    """
    read = _read(dialect)
    try:
        statements = sqlglot.parse(f"SELECT {expression}", read=read)
    except sqlglot.errors.ParseError as exc:
        raise ValueError(f"The expression does not parse: {str(exc).splitlines()[0]}") from None
    select = statements[0] if len(statements) == 1 else None
    if not isinstance(select, exp.Select) or len(select.selects) != 1 or any(
        select.args.get(key) for key in select.args if key != "expressions"
    ):
        raise ValueError("Give one SQL expression, e.g. \"amount\" / 1000, not a query or several expressions.")
    parsed = select.selects[0]
    if isinstance(parsed, exp.Alias):
        raise ValueError("Leave the alias out of the expression; the column's name is its own setting.")
    if parsed.find(exp.Query, exp.Table):
        raise ValueError("The expression may only read the input's columns, not a query or a table.")
    for func in parsed.find_all(exp.Func):
        if _function_name(func) in DENIED_FUNCTIONS:
            raise ValueError(f"The expression calls {_function_name(func)}(), which reads the engine's settings or environment.")
        if isinstance(func, exp.AggFunc) and not func.find_ancestor(exp.Window):
            raise ValueError("Aggregate in an aggregate node; an expression computes one value per row.")
    return parsed


def expression_columns(expression: str, dialect: str = FRAME_DIALECT) -> set[str]:
    """The columns ``expression`` reads, or none if it does not parse."""
    try:
        parsed = sqlglot.parse_one(expression, read=_read(dialect))
    except Exception:
        return set()
    return {column.name for column in parsed.find_all(exp.Column)}


def _unknown(names: list[str], columns: list[str]) -> str | None:
    missing = [name for name in names if name not in columns]
    if not missing:
        return None
    noun = "column" if len(missing) == 1 else "columns"
    return f"The data has no {noun} {', '.join(map(repr, missing))}. Its columns are: {', '.join(columns)}."


def _input(inputs: Mapping[str, Any]) -> Pipeline | None:
    pipeline = inputs.get("pipeline")
    return pipeline if isinstance(pipeline, Pipeline) else None


# The ``check_config=`` hooks, see ``lumen.flowdash.checks``: none fetches rows.

def check_aggregate_config(config: dict[str, Any], inputs: dict[str, Any]) -> str | None:
    """Check a ``Lumen/aggregate`` config against the input's columns and their types."""
    pipeline = _input(inputs)
    if pipeline is None:
        return None
    by, columns = list(config.get("by") or []), list(config.get("columns") or [])
    method = config.get("method") or "sum"
    if problem := _unknown([*by, *columns], pipeline_columns(pipeline)):
        return problem
    if both := [c for c in columns if c in by]:
        return f"{', '.join(map(repr, both))} cannot be grouped by and aggregated at once."
    if not columns and COUNT_COLUMN in by:
        return f"Counting rows makes a column {COUNT_COLUMN!r}, which is also a grouping column."
    if method in NUMERIC_METHODS:
        types = pipeline_column_types(pipeline) or {}
        if text := [c for c in columns if types.get(c) not in ("integer", "number", "boolean")]:
            return f"Cannot take the {method} of {', '.join(map(repr, text))}, which is not numeric."
    return None


def check_sort_config(config: dict[str, Any], inputs: dict[str, Any]) -> str | None:
    """Check a ``Lumen/sort`` config: the columns to sort by must be the input's."""
    pipeline = _input(inputs)
    if pipeline is None:
        return None
    return _unknown(list(config.get("by") or []), pipeline_columns(pipeline))


def check_select_config(config: dict[str, Any], inputs: dict[str, Any]) -> str | None:
    """Check a ``Lumen/select`` config: the columns to keep must be the input's, each once."""
    columns = list(config.get("columns") or [])
    if len(set(columns)) != len(columns):
        return "Each column can be kept once."
    pipeline = _input(inputs)
    return None if pipeline is None else _unknown(columns, pipeline_columns(pipeline))


def check_derive_config(config: dict[str, Any], inputs: dict[str, Any]) -> str | None:
    """
    Check a ``Lumen/derive`` config: the column must be a plain name and
    the expression one scalar SQL expression over the input's columns.
    """
    column, expression = (config.get("column") or "").strip(), (config.get("expression") or "").strip()
    if column and not _NAME.match(column):
        return f"{column!r} is not a column name: use letters, digits, underscores and spaces."
    pipeline = _input(inputs)
    if not expression:
        return None
    dialect = step_dialect(pipeline) if pipeline is not None else FRAME_DIALECT
    try:
        parsed = parse_expression(expression, dialect)
    except ValueError as exc:
        return str(exc)
    if pipeline is None:
        return None
    columns = pipeline_columns(pipeline)
    lowered = {c.lower() for c in columns}
    unknown = [
        c.name for c in parsed.find_all(exp.Column)
        if c.name not in columns and (c.this.quoted or c.name.lower() not in lowered)
    ]
    return _unknown(list(dict.fromkeys(unknown)), columns)


@dataclass(frozen=True)
class ColumnUse:
    """
    The columns of its input a transform node reads, and the input columns
    each of its output columns comes from, for the contracts of the source
    it reads, see ``lumen.flowdash.contracts``.
    """

    reads: set[str]
    lineage: Callable[[str], set[str]]


def _aggregate_use(config: Mapping[str, Any]) -> ColumnUse:
    kept = set(config.get("by") or []) | set(config.get("columns") or [])
    return ColumnUse(kept, lambda column: {column} if column in kept else set())


def _sort_use(config: Mapping[str, Any]) -> ColumnUse:
    return ColumnUse(set(config.get("by") or []), lambda column: {column})


def _select_use(config: Mapping[str, Any]) -> ColumnUse:
    kept = set(config.get("columns") or [])
    return ColumnUse(kept, lambda column: {column} if column in kept else set())


def _derive_use(config: Mapping[str, Any]) -> ColumnUse:
    derived = config.get("column") or ""
    read = expression_columns(config.get("expression") or "")
    return ColumnUse(read, lambda column: read if column == derived else {column})


COLUMN_USES: dict[str, Callable[[Mapping[str, Any]], ColumnUse]] = {
    AGGREGATE_ID: _aggregate_use, SORT_ID: _sort_use, SELECT_ID: _select_use, DERIVE_ID: _derive_use,
}

_COLUMNS = {"flowdash": {"choices": "columns", "of": "pipeline"}}


class TransformNode(PipelineNode):
    """A node applying one step to its input pipeline, see ``transformed_pipeline``."""

    __abstract = True

    pipeline = param.Parameter(doc="The upstream pipeline.")

    _waiting = "Waiting for a pipeline"

    def _step(self, pipeline: Pipeline) -> tuple[SQLTransform, dict[str, Any]] | None:
        """The step on ``pipeline`` and the schema it outputs, or None while the config is incomplete."""
        raise NotImplementedError

    def _resolve(self) -> Pipeline | None:
        if self.pipeline is None:
            return None
        if is_failure(self.pipeline):
            raise ValueError(f"The input failed: {failure_text(self.pipeline)}")
        step = self._step(self.pipeline)
        if step is None:
            return None
        return transformed_pipeline(self.pipeline, *step, limit=row_ceiling(self.editor))

    def _check(self, problem: str | None) -> None:
        if problem:
            raise ValueError(problem)

    @param.output(param.Parameter)
    @param.depends("_object", "_error")
    def result(self):
        return self._value()


def _pipeline_node(**options: Any):
    return register(
        component=True, page=False,
        requires=[{"key": "pipeline", "type": PIPELINE_TYPE, "required": True}],
        provides=[{"key": "result", "type": PIPELINE_TYPE}],
        **options,
    )


@_pipeline_node(
    title="Lumen Aggregate", icon="functions",
    description=(
        "Groups the rows of a Lumen pipeline by the columns in by and aggregates the columns in "
        "columns with one method (count, sum, mean, min or max), each keeping its name. Without "
        "by the whole table is one row; a count without columns counts rows into a count column. "
        "On a table read from a SQL source the source runs it in its query."
    ),
    config=["by", "columns", "method"],
    check_config=check_aggregate_config,
)
class AggregateNode(TransformNode):

    by = param.List(default=[], item_type=str, doc="Columns to group by.", metadata=_COLUMNS)

    columns = param.List(default=[], item_type=str, doc="Columns to aggregate.", metadata=_COLUMNS)

    method = param.Selector(default="sum", objects=list(METHODS), doc="How to aggregate the columns.")

    _waiting = "Waiting for a pipeline and columns to aggregate"

    @param.depends("pipeline", "by", "columns", "method", watch=True, on_init=True)
    def _refresh(self, *events):
        super()._refresh()

    def _step(self, pipeline: Pipeline) -> tuple[SQLTransform, dict[str, Any]] | None:
        if not self.columns and self.method != "count":
            return None
        config = {"by": self.by, "columns": self.columns, "method": self.method}
        self._check(check_aggregate_config(config, {"pipeline": pipeline}))
        step = SQLAggregate(by=self.by, columns=self.columns, method=self.method, write=step_dialect(pipeline))
        return step, aggregate_schema(pipeline.schema, self.by, self.columns, self.method)


@_pipeline_node(
    title="Lumen Sort", icon="sort",
    description=(
        "Sorts the rows of a Lumen pipeline by the columns in by, ascending or descending, and "
        "keeps the first limit rows if a limit is set, e.g. for a top ten. On a table read from "
        "a SQL source the source runs it in its query."
    ),
    config=["by", "descending", "limit"],
    check_config=check_sort_config,
)
class SortNode(TransformNode):

    by = param.List(default=[], item_type=str, doc="Columns to sort by, in order.", metadata=_COLUMNS)

    descending = param.Boolean(default=False, doc="Sort from the highest value to the lowest.")

    limit = param.Integer(default=None, bounds=(1, None), allow_None=True, doc="""
        Rows to keep after sorting; all of them if empty.""")

    _waiting = "Waiting for a pipeline and columns to sort by"

    @param.depends("pipeline", "by", "descending", "limit", watch=True, on_init=True)
    def _refresh(self, *events):
        super()._refresh()

    def _step(self, pipeline: Pipeline) -> tuple[SQLTransform, dict[str, Any]] | None:
        if not self.by and self.limit is None:
            return None
        self._check(check_sort_config({"by": self.by}, {"pipeline": pipeline}))
        step = SQLOrder(by=self.by, descending=self.descending, limit=self.limit, write=step_dialect(pipeline))
        schema = dict(pipeline.schema)
        if self.limit is not None:
            schema.pop("__len__", None)
        return step, schema


@_pipeline_node(
    title="Lumen Select", icon="view_column",
    description=(
        "Keeps the columns of a Lumen pipeline named in columns, in that order. On a table read "
        "from a SQL source the source only returns those columns."
    ),
    config=["columns"],
    check_config=check_select_config,
)
class SelectNode(TransformNode):

    columns = param.List(default=[], item_type=str, doc="Columns to keep, in order.", metadata=_COLUMNS)

    _waiting = "Waiting for a pipeline and columns to keep"

    @param.depends("pipeline", "columns", watch=True, on_init=True)
    def _refresh(self, *events):
        super()._refresh()

    def _step(self, pipeline: Pipeline) -> tuple[SQLTransform, dict[str, Any]] | None:
        if not self.columns:
            return None
        self._check(check_select_config({"columns": self.columns}, {"pipeline": pipeline}))
        schema = pipeline.schema
        kept = {column: schema[column] for column in self.columns}
        if "__len__" in schema:
            kept["__len__"] = schema["__len__"]
        return SQLSelect(columns=self.columns, write=step_dialect(pipeline)), kept


@_pipeline_node(
    title="Lumen Derived column", icon="calculate",
    description=(
        "Adds a column, named by column, computed per row by a SQL expression over the columns of "
        "a Lumen pipeline, e.g. \"amount\" / 1000 or CASE WHEN \"amount\" > 10 THEN 'high' ELSE 'low' END, or "
        "replaces the column of that name. The expression is in the source's SQL dialect, or "
        "DuckDB's for a source that does not run SQL; it reads no tables and aggregates nothing."
    ),
    config=["column", "expression"],
    check_config=check_derive_config,
)
class DeriveNode(TransformNode):

    column = param.String(default="", doc="Name of the new column, or of the column to replace.")

    expression = param.String(default="", doc="""
        A SQL expression computing the column's value from the other
        columns of the row, e.g. "amount" / 1000.""")

    _waiting = "Waiting for a pipeline, a column name and an expression"

    @param.depends("pipeline", "column", "expression", watch=True, on_init=True)
    def _refresh(self, *events):
        super()._refresh()

    def _step(self, pipeline: Pipeline) -> tuple[SQLTransform, dict[str, Any]] | None:
        column, expression = self.column.strip(), self.expression.strip()
        if not column or not expression:
            return None
        self._check(check_derive_config({"column": column, "expression": expression}, {"pipeline": pipeline}))
        dialect = step_dialect(pipeline)
        step = SQLDerive(columns=_columns_of(pipeline.schema), output=column, expression=expression, write=dialect)
        return step, derive_schema(pipeline.schema, column, expression, dialect)


TRANSFORM_COMPONENTS = {
    AGGREGATE_ID: AggregateNode, SORT_ID: SortNode, SELECT_ID: SelectNode, DERIVE_ID: DeriveNode,
}
