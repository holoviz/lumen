"""
Save as metric: turn the query behind a KPI, a Lumen SQL node or an
aggregate into a metric of the project file, then point the node at it.

An admin picks "Save as metric…" in the node menu of such a node. The
dialog starts from a definition derived from the node's query
(``derive_draft``): the structured form (a table, measures and dimensions)
where the query maps onto it cleanly, else the SQL form with a note on what
to adjust. Before saving it shows what the node shows now beside what the
metric gives (``preview_metric``) and the SQL the metric runs, so the two
can be compared. ``save_metric`` writes the metric as the admin Metrics
section does, under the project file's lock with ``Project.set``, which
checks the whole section against the sources before writing the file
atomically, and refuses anyone who is no admin. ``point_at_metric`` then
replaces the node's query by a ``Lumen/metric`` node wired the same, keeping
its tile, as one edit that undo reverts.

The provenance labels follow from the nodes: a metric node reports
"Metric: <name>" and a SQL or aggregate node "Ad hoc" with how its number
is computed (``describe_query``), which FlowDash shows on the views that read
them, see ``panel_flowdash.node_actions``. A number aggregated from a
metric's rows on the dashboard, by an aggregate node or an edge's
``aggregate`` step, is "Derived: <how> of <name>" instead (``derived_label``),
since the metric does not define it.
"""
from __future__ import annotations

import logging
import re

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import pandas as pd
import panel as pn
import param
import sqlglot
import yaml

from panel.viewable import Viewer
from sqlglot import exp

from ..sources.base import BaseSQLSource
from .contracts import DATA_ID, METRIC_ID, SQL_ID
from .metrics import (
    METRICS_KEY, SECTION, MetricSet, Query, check_expression, read_metrics,
)
from .project import Project, ProjectError, parse_project
from .transforms import (
    AGGREGATE_ID, COUNT_COLUMN, METHODS, SQLAggregate,
)

logger = logging.getLogger(__name__)

ACTION_KEY = "lumen.save_as_metric"

KPI_ID = "Widgets/KPI"

# The nodes whose query a metric can be derived from.
QUERY_NODES = (SQL_ID, AGGREGATE_ID)

# How far up a view's inputs the query it shows is looked for.
DEPTH = 4

# Rows of a result the preview shows at most.
PREVIEW_ROWS = 5

_NAME = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
_USERINFO = re.compile(r"(\w+://)[^/\s@]+@")

GENERIC = "Computed by its own SQL query; no metric was used."

_METHOD_WORDS = {"SUM": "Sum", "AVG": "Average", "MIN": "Lowest", "MAX": "Highest", "COUNT": "Count"}


# ---------------------------------------------------------------------------
# Describing a node's query, for the "Ad hoc" label

def describe_query(sql: str, dialect: str | None = None) -> str:
    """
    How a query computes its numbers in a few words, e.g. "Sum of p_cap
    over all rows of turbines" or "Sum of p_cap by t_state over turbines",
    or a generic sentence for a query it cannot put that way.
    """
    try:
        parsed = sqlglot.parse_one(sql, read=dialect)
    except sqlglot.errors.SqlglotError:
        return GENERIC
    if not isinstance(parsed, exp.Select):
        return GENERIC
    aggregates = [a for a in parsed.expressions if a.find(exp.AggFunc) is not None]
    source = parsed.args.get("from_") or parsed.args.get("from")
    table = source.this.name if source is not None and isinstance(source.this, exp.Table) else None
    if not aggregates:
        return f"Rows of {table} picked by its own SQL query; no metric was used." if table else GENERIC
    parts = [_describe_aggregate(a) for a in aggregates[:2]]
    text = " and ".join(parts) + (" and more" if len(aggregates) > 2 else "")
    group = parsed.args.get("group")
    keys = [k.sql() for k in group.expressions] if group is not None else []
    if keys:
        text += f" by {', '.join(keys)}"
    if parsed.args.get("where") is not None:
        text += " of the rows it filters"
    elif not keys:
        text += " over all rows"
    if table:
        text += f" of {table}"
    if parsed.args.get("joins"):
        text += " with joins"
    return f"{text}, computed by its own SQL query; no metric was used."


def _describe_aggregate(expression: exp.Expression) -> str:
    inner = expression.this if isinstance(expression, exp.Alias) else expression
    if isinstance(inner, exp.AggFunc) and not inner.find(exp.Window):
        word = _METHOD_WORDS.get(inner.sql_name().upper(), inner.sql_name().capitalize())
        argument = inner.this
        if isinstance(argument, exp.Star) or argument is None:
            return "Count of rows"
        if isinstance(argument, exp.Distinct):
            return f"{word} of distinct {', '.join(e.sql() for e in argument.expressions)}"
        return f"{word} of {argument.sql()}"
    return inner.sql()


def describe_aggregate(method: str, columns: list[str], by: list[str], metric: str | None = None) -> str:
    """
    How an aggregate node computes its numbers, e.g. "Sum of p_cap over all
    rows", or with ``metric`` the metric whose rows it aggregates, e.g.
    "Sum of mw over metric state_mw".
    """
    word = _METHOD_WORDS[METHODS[method]]
    what = "Count of rows" if not columns else f"{word} of {', '.join(columns)}"
    if metric is not None:
        return (f"{what} {'by ' + ', '.join(by) + ' ' if by else ''}over metric {metric}, aggregated on the "
                f"dashboard; the metric itself does not define this number.")
    return f"{what} {'by ' + ', '.join(by) if by else 'over all rows'} of its input; no metric was used."


# How an edge's ``aggregate`` step or an aggregate node reduces rows, in a label.
_HOW_WORDS = {
    "sum": "sum", "mean": "average", "min": "lowest", "max": "highest", "count": "count",
    "nunique": "distinct count",
}

# The steps that give a one-row result's value unchanged.
_SAME_ON_ONE_ROW = ("sum", "mean", "min", "max", "last")


def derived_label(how: str, metric: str) -> str:
    """The label of a number aggregated from a metric's rows on the dashboard, e.g. "Derived: average of state_mw"."""
    if how == "last":
        return f"Derived: last row of {metric}"
    return f"Derived: {_HOW_WORDS.get(how, how)} of {metric}"


def aggregating_step(via: Sequence[Mapping[str, Any]], rows: int | None) -> tuple[str, str | None] | None:
    """
    The first ``aggregate`` step on the edges ``via`` (see
    ``panel_flowdash.node_actions``) that turns the rows of a node giving
    ``rows`` rows (None when not known) into a number the metric does not
    give, as ``(how, column)``: any step over more than one row, ``last``
    included, which shows one arbitrary row, and over one row those that
    change its value; ``last`` over rows not counted yet is given the
    benefit of the doubt. None when the edges show the metric's own value.
    """
    for edge in via:
        column = None
        for step in edge.get("transform") or []:
            if step.get("op") == "column":
                column = step.get("column")
            elif step.get("op") == "aggregate":
                how = step.get("how") or "sum"
                if (rows == 1 and how in _SAME_ON_ONE_ROW) or (rows is None and how == "last"):
                    continue
                return how, column
    return None


def describe_reading(how: str, column: str | None, metric: str) -> str:
    """How a view's number is aggregated from a metric's rows by an edge, e.g. "Average of mw over metric state_mw"."""
    if how == "last":
        what = f"The {column} of" if column else "One row of"
        return (f"{what} the last row of metric {metric}, one group's value of several; the metric itself "
                "does not define this number.")
    word = _HOW_WORDS.get(how, how)
    what = f"{word.capitalize()} of {column}" if column else word.capitalize()
    return (f"{what} over the rows of metric {metric}, aggregated on the dashboard; the metric itself does "
            "not define this number.")


# ---------------------------------------------------------------------------
# Deriving a definition from a node
#
# Everything below reads and changes the dashboard through panel-flowdash's
# NodeActionContext (``ctx``), with nodes named as in the dashboard YAML.

@dataclass
class MetricDraft:
    """A metric definition derived from a node's query, and what the node shows now."""

    name: str
    definition: dict[str, Any]
    node: str
    """The SQL or aggregate node the definition is derived from."""
    target: str
    """The node the action was opened on, e.g. a KPI reading ``node``."""
    measure: str | None = None
    """The column the target shows, e.g. the KPI's; None shows them all."""
    dimensions: list[str] = field(default_factory=list)
    structured: bool = False
    notes: list[str] = field(default_factory=list)
    current: Any = None
    """What the target shows now: a number, or rows."""


def _node(ref: str) -> str:
    return ref.rsplit(".", 1)[0]


def _port(ref: str) -> str:
    return ref.rsplit(".", 1)[1]


def _column_of(transform: list[dict[str, Any]] | None) -> str | None:
    return next((step.get("column") for step in transform or [] if step.get("op") == "column"), None)


def _output(ctx: Any, ref: str) -> Any:
    try:
        return ctx.output(_port(ref), _node(ref))
    except Exception:
        return None


def query_node(ctx: Any, name: str | None = None) -> tuple[str, str | None] | None:
    """
    The SQL or aggregate node a node's numbers come from: the node itself,
    or the nearest one up its inputs through data nodes, with the column
    the edge into the node picks; None when there is none or a metric node
    comes first.
    """
    name = name or ctx.node
    component = ctx.component(name)
    if component in QUERY_NODES:
        return name, None
    if component in (METRIC_ID, None):
        return None
    frontier: list[tuple[str, str | None]] = [(name, None)]
    for _ in range(DEPTH):
        nxt = []
        for node, column in frontier:
            for edge in ctx.inputs(node):
                upstream = _node(edge["from"])
                kind = ctx.component(upstream)
                picked = column or _column_of(edge["transform"])
                if kind in QUERY_NODES:
                    return upstream, picked
                if kind == DATA_ID:
                    nxt.append((upstream, picked))
        frontier = nxt
    return None


def _ref_of(source: Any, context: Mapping[str, Any]) -> str | None:
    from .nodes import host_sources

    return next((ref for ref, s in host_sources(context).items() if s is source), None)


def _dialect(source: Any) -> str | None:
    dialect = getattr(source, "dialect", None)
    return None if dialect in (None, "any") else dialect


def _statement(ctx: Any, name: str) -> tuple[BaseSQLSource, str]:
    """The root source of a SQL node and its query as one statement on it, see ``results.expand``."""
    from .results import expand, merge

    sql = ctx.config(name).get("sql") or ""
    inputs = [_output(ctx, e["from"]) for e in ctx.inputs(name) if _port(e["to"]) == "source"]
    try:
        source = merge(inputs[0] if len(inputs) == 1 else inputs)
    except ValueError:
        source = None
    if not isinstance(source, BaseSQLSource) or not sql.strip():
        raise ValueError("the SQL node has no source or query yet")
    return expand(sql, source)


def _base(ctx: Any, name: str) -> tuple[str, str | None, str, list[str]]:
    """
    What a pipeline input reads, as ``(ref, table, sql, notes)``: the source
    ref, the table when it is a plain table node, else None, and the full
    query on the source. Raises ``ValueError`` for an input a metric cannot
    say.
    """
    from .contracts import FILTER_ID, TABLE_ID
    from .nodes import lookup_source

    component, config = ctx.component(name), ctx.config(name)
    if component == TABLE_ID:
        wired = [_output(ctx, e["from"]) for e in ctx.inputs(name) if _port(e["to"]) == "source"]
        source = wired[0] if wired else lookup_source(config.get("ref") or "", ctx.context)
        ref, table = _ref_of(source, ctx.context), config.get("table")
        if ref is None or not table:
            raise ValueError("its table node reads no source the app serves")
        sql = exp.select("*").from_(exp.Table(this=exp.Identifier(this=table, quoted=True))).sql(
            dialect=_dialect(source))
        limit = config.get("limit")
        notes = [f"The table node's limit of {limit:,} rows is left out, so the metric reads every row of "
                 f"{table}."] if isinstance(limit, int) and not isinstance(limit, bool) else []
        return ref, table, sql, notes
    if component == SQL_ID:
        root, statement = _statement(ctx, name)
        ref = _ref_of(root, ctx.context)
        if ref is None:
            raise ValueError("its SQL node reads no source the app serves")
        return ref, None, statement, []
    if component == FILTER_ID:
        upstream = next((_node(e["from"]) for e in ctx.inputs(name) if _port(e["to"]) == "pipeline"), None)
        if upstream is None:
            raise ValueError("its filter node has no input")
        ref, table, sql, notes = _base(ctx, upstream)
        field_ = config.get("field")
        return ref, table, sql, [*notes, f"The filter on {field_!r} is left out; add it as a parameter or a "
                                         "dimension to filter by."]
    raise ValueError(f"it reads a {component} node, which a metric cannot say")


def _filtered_by(ctx: Any, names: list[str]) -> list[str]:
    """The columns of the dashboard's global filters that filter any of the nodes ``names``."""
    try:
        columns = [f["column"] for f in ctx.ops.filters() if f.get("column")]
    except Exception:
        return []
    filtered = []
    for column in columns:
        try:
            applies, _ = ctx.ops.filter_reach(column)
        except Exception:
            continue
        if any(name in applies for name in names):
            filtered.append(column)
    return filtered


def _filter_note(columns: list[str]) -> list[str]:
    if not columns:
        return []
    many = len(columns) > 1
    return [f"The dashboard's global filter{'s' * many} on {', '.join(columns)} {'are' if many else 'is'} left "
            f"out; add {'them' if many else 'it'} as dimensions to filter by."]


def structured_form(sql: str, dialect: str | None = None) -> tuple[dict[str, Any] | None, str | None]:
    """
    ``sql`` as the table, measures and dimensions of a structured metric,
    or None and why it does not map: only ``SELECT <columns>, <aggregates>
    FROM <table> GROUP BY <columns>`` does, each aggregate named.
    """
    try:
        parsed = sqlglot.parse_one(sql, read=dialect)
    except sqlglot.errors.SqlglotError:
        return None, "the query does not parse"
    if not isinstance(parsed, exp.Select):
        return None, "it is not one SELECT"
    for key, why in (("with", "it uses a WITH clause"), ("joins", "it joins tables"),
                     ("where", "it filters rows with WHERE"), ("having", "it filters groups with HAVING"),
                     ("order", "it sorts its rows"), ("limit", "it limits its rows"),
                     ("distinct", "it selects DISTINCT rows")):
        if parsed.args.get(key):
            return None, why
    source = parsed.args.get("from_") or parsed.args.get("from")
    if source is None or not isinstance(source.this, exp.Table) or not isinstance(source.this.this, exp.Identifier):
        return None, "it reads a subquery rather than a table"
    group = parsed.args.get("group")
    keys = {k.name for k in (group.expressions if group is not None else []) if isinstance(k, exp.Column)}
    if group is not None and len(keys) != len(group.expressions):
        return None, "it groups by an expression"
    measures: dict[str, Any] = {}
    dimensions: dict[str, Any] = {}
    for item in parsed.expressions:
        inner = item.this if isinstance(item, exp.Alias) else item
        name = item.alias_or_name
        if isinstance(inner, exp.Column) and inner.name in keys:
            if not _NAME.fullmatch(name):
                return None, f"column {name!r} is not a plain name"
            dimensions[name] = inner.name if name == inner.name else {"column": inner.name}
            continue
        if inner.find(exp.AggFunc) is None or inner.find(exp.Window) is not None:
            return None, f"{item.sql(dialect=dialect)} is neither a grouped column nor an aggregate"
        if not isinstance(item, exp.Alias) or not _NAME.fullmatch(name):
            return None, f"the aggregate {inner.sql(dialect=dialect)} has no plain name; give it one with AS"
        expr = inner.sql(dialect=dialect)
        try:
            check_expression(expr, aggregate=True, dialect=dialect)
        except ValueError as exc:
            return None, f"measure {name!r} {exc}"
        measures[name] = expr
    if not measures:
        return None, "it computes no aggregate"
    if set(keys) - {d if isinstance(d, str) else d["column"] for d in dimensions.values()}:
        return None, "it groups by a column it does not select"
    return {"table": source.this.name, "measures": measures, **({"dimensions": dimensions} if dimensions else {})}, None


def _slug(text: str) -> str:
    slug = re.sub(r"\W+", "_", str(text)).strip("_").lower()
    if not slug or not _NAME.fullmatch(slug):
        slug = f"metric_{slug}" if slug else "metric"
    return slug[:64]


def _title(ctx: Any, name: str) -> str:
    config = ctx.config(name)
    for key in ("label", "title"):
        if isinstance(config.get(key), str) and config[key].strip():
            return config[key]
    return name


def current_value(ctx: Any, name: str | None = None) -> Any:
    """What a node shows now: a KPI's number, a SQL node's one value, or its rows."""
    from .nodes import single_number

    name = name or ctx.node
    component = ctx.component(name)
    if component == KPI_ID:
        return ctx.input_value("value", name)
    if component == SQL_ID:
        data = _output(ctx, f"{name}.data")
        if isinstance(data, pd.DataFrame):
            try:
                return single_number(data)
            except ValueError:
                return data
        return None
    pipeline = _output(ctx, f"{name}.result")
    data = getattr(pipeline, "data", None)
    return data if isinstance(data, pd.DataFrame) else None


def derive_draft(ctx: Any) -> MetricDraft:
    """
    A metric definition derived from the query behind the context's node,
    see the module docstring. Raises ``ValueError`` when there is no query
    to derive one from.
    """
    found = query_node(ctx)
    if found is None:
        raise ValueError("Its numbers do not come from a Lumen SQL or aggregate node.")
    node, column = found
    notes: list[str] = []
    definition: dict[str, Any]
    dimensions: list[str] = []
    structured = False
    upstream = None
    if ctx.component(node) == SQL_ID:
        try:
            root, statement = _statement(ctx, node)
        except ValueError as exc:
            raise ValueError(f"Its SQL node cannot be saved as a metric: {exc}.") from None
        ref = _ref_of(root, ctx.context)
        if ref is None:
            raise ValueError("Its SQL node reads no source the app serves.")
        shape, why = structured_form(statement, _dialect(root))
        if shape is not None:
            definition, structured = {"source": ref, **shape}, True
            dimensions = list(shape.get("dimensions", {}))
        else:
            definition = {"source": ref, "sql": statement}
            notes.append(f"The query does not map to measures and dimensions: {why}. It is kept as a SQL "
                         "metric; adjust its query here, e.g. to count each entity once.")
    else:
        config = ctx.config(node)
        upstream = next((_node(e["from"]) for e in ctx.inputs(node) if _port(e["to"]) == "pipeline"), None)
        if upstream is None:
            raise ValueError("Its aggregate node has no input.")
        try:
            ref, table, sql, notes = _base(ctx, upstream)
        except ValueError as exc:
            raise ValueError(f"Its aggregate node cannot be saved as a metric: {exc}.") from None
        method, by, columns = config.get("method") or "sum", list(config.get("by") or []), list(config.get("columns") or [])
        dimensions = by
        if table is not None:
            measures = {
                c: "COUNT(*)" if not columns else exp.func(METHODS[method], exp.column(c, quoted=True)).sql()
                for c in columns or [COUNT_COLUMN]
            }
            definition = {"source": ref, "table": table, "measures": measures}
            if by:
                definition["dimensions"] = {d: d for d in by}
            structured = True
        else:
            definition = {"source": ref, "sql": SQLAggregate(by=by, columns=columns, method=method).apply(sql)}
            notes.append("It aggregates the result of a SQL node, so it is kept as a SQL metric over that query.")
    filtered = [node, *([upstream] if upstream else [])]
    notes += _filter_note(_filtered_by(ctx, filtered))
    if structured and column is None and len(definition.get("measures", {})) == 1:
        column = next(iter(definition["measures"]))
    return MetricDraft(
        name=_slug(_title(ctx, ctx.node)), definition=definition, node=node, target=ctx.node,
        measure=column, dimensions=dimensions, structured=structured, notes=notes, current=current_value(ctx),
    )


# ---------------------------------------------------------------------------
# Previewing and saving

@dataclass
class MetricPreview:
    """What a metric definition gives: its rows, the column compared, the SQL it runs, or what is wrong."""

    sql: str = ""
    data: pd.DataFrame | None = None
    column: str | None = None
    value: Any = None
    problems: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.problems and self.data is not None


def scrub(text: str) -> str:
    """``text`` without the user and password a connection URI in it may carry."""
    return _USERINFO.sub(r"\1***@", str(text))


def check_definition(name: str, definition: Mapping[str, Any], context: Mapping[str, Any]) -> MetricSet:
    """
    The metric ``name`` of ``definition`` read and checked as the
    ``metrics:`` section reads and checks the project file's, against the
    sources of ``context``. Raises ``ProjectError`` with every problem.
    """
    if not isinstance(name, str) or not _NAME.fullmatch(name):
        raise ProjectError([_problem(f"The metric's name {name!r} must be letters, digits and underscores.")])
    text = yaml.safe_dump({SECTION: {name: dict(definition)}}, sort_keys=False)
    project = parse_project(text, context=context)
    metrics = read_metrics(project)
    if name not in metrics:
        # A metric over a source the app does not serve is left out with a warning.
        warnings = "; ".join(str(w) for w in project.warnings) or f"The metric {name!r} was left out."
        raise ProjectError([_problem(warnings)])
    return metrics


def _problem(message: str) -> Any:
    from .project import Problem

    return Problem(message)


def preview_metric(name: str, definition: Mapping[str, Any], context: Mapping[str, Any],
                   measure: str | None = None, dimensions: list[str] | None = None) -> MetricPreview:
    """
    What the metric ``name`` of ``definition`` gives for ``measure`` (all
    measures when None) grouped by ``dimensions``, the SQL it runs and,
    for one row, the number to compare. A problem is reported, never raised.
    """
    from .metrics import bind_sql, table_sql
    from .nodes import fetch, lookup_source, metric_pipeline

    try:
        metric = check_definition(name, definition, context)[name]
    except ProjectError as exc:
        return MetricPreview(problems=[scrub(getattr(p, "message", p)) for p in exc.problems])
    source = lookup_source(metric.source, context)
    if metric.form == "structured":
        measures = [measure] if measure in metric.measures else list(metric.measures)
        dims = [d for d in dimensions or [] if d in metric.dimensions]
        query = Query(measures=measures, dimensions=dims)
    else:
        query = Query()
    try:
        pipeline = metric_pipeline(metric, source, query, PREVIEW_ROWS * 20)
        data = fetch(pipeline, PREVIEW_ROWS * 20)
        if metric.form == "structured":
            sql = pipeline.sql_transforms[0].apply(table_sql(metric.table, _dialect(source)))
        else:
            sql = bind_sql(metric, metric.values({}), source.dialect)
    except Exception as exc:
        return MetricPreview(problems=[scrub(str(exc).strip().splitlines()[0] if str(exc).strip() else type(exc).__name__)])
    try:
        sql = sqlglot.transpile(sql, read=_dialect(source), write=_dialect(source), pretty=True)[0]
    except sqlglot.errors.SqlglotError:
        pass
    numeric = [c for c in data.columns if pd.api.types.is_numeric_dtype(data[c])]
    column = measure if measure in data.columns else numeric[0] if len(numeric) == 1 else None
    value = _value(data[column].iloc[-1]) if column is not None and len(data) == 1 else None
    return MetricPreview(sql=sql, data=data, column=column, value=value)


def _value(value: Any) -> Any:
    if pd.isna(value):
        return None
    return value.item() if hasattr(value, "item") else value


def agree(current: Any, metric: Any) -> bool | None:
    """Whether two numbers agree to about six significant digits; None if either is not a number."""
    numbers = (int, float)
    if not isinstance(current, numbers) or not isinstance(metric, numbers) or isinstance(current, bool):
        return None
    scale = max(abs(current), abs(metric), 1e-12)
    return abs(current - metric) / scale < 1e-6


def project_of(ctx: Any) -> Project | None:
    """The project file metrics are saved to: the app's, or the one the app's metrics are read from."""
    project = ctx.project or getattr(ctx.context.get(METRICS_KEY), "project", None)
    return project if isinstance(project, Project) and project.path is not None else None


class _Admin:
    """The admin check ``metrics_admin.save_metric`` asks before writing, answered by the node action's context."""

    def __init__(self, ctx: Any):
        self._ctx = ctx
        self._identity = None

    def _still_admin(self) -> bool:
        return self._ctx.is_admin()


def save_metric(ctx: Any, name: str, definition: Mapping[str, Any]) -> None:
    """
    Write the metric ``name`` of ``definition`` to the project file, as the
    admin Metrics section saves one (``metrics_admin.save_metric``): under
    the project file's lock, keeping every other metric as the file has it
    now, checked against the sources before the file is written atomically,
    and only for someone still admin. Its SQL is written as a block.

    Raises
    ------
    PermissionError
        Unless the person is still an admin of the project.
    ValueError
        Without a project file.
    ProjectError
        With the problems that stopped it, e.g. a taken name or a column the table lacks.
    """
    from . import metrics_admin

    project = project_of(ctx)
    if project is None:
        raise ValueError("The app has no project file to save metrics to.")
    problems = metrics_admin.save_metric(project, name, metrics_admin._literal(dict(definition)), app=_Admin(ctx))
    if problems and problems[0].message == metrics_admin.NOT_ADMIN:
        raise PermissionError("Only an admin of the project may save a metric.")
    if problems:
        raise ProjectError(problems)


# ---------------------------------------------------------------------------
# Pointing the node at the metric

def _free_name(ctx: Any, stem: str) -> str:
    taken = set(ctx.nodes())
    name, n = stem[:60], 2
    while name in taken:
        name, n = f"{stem[:56]}_{n}", n + 1
    return name


def _metric_config(metric: Any, draft: MetricDraft, column: str | None) -> dict[str, Any]:
    config: dict[str, Any] = {"metric": metric.name}
    if metric.form == "structured":
        config["measures"] = [column] if column in metric.measures else list(metric.measures)
        config["dimensions"] = [d for d in draft.dimensions if d in metric.dimensions]
    return config


def _reading(transform: list[dict[str, Any]] | None, column: str | None) -> list[dict[str, Any]] | None:
    """An edge's transform with the column it picks renamed to the metric's ``column``."""
    if column is None:
        return transform
    steps = [dict(step) for step in transform or []]
    picked = next((s for s in steps if s.get("op") == "column"), None)
    if picked is None:
        return [{"op": "column", "column": column}, {"op": "aggregate", "how": "last"}, *steps]
    picked["column"] = column
    return steps


def _chain(ctx: Any, target: str, node: str) -> list[str]:
    """The nodes from ``target``'s inputs up to ``node``, nearest first: the helpers a KPI's number goes through."""
    chain, frontier = [], [_node(e["from"]) for e in ctx.inputs(target)]
    for _ in range(DEPTH):
        nxt = []
        for upstream in frontier:
            if upstream == node:
                chain.append(upstream)
            elif ctx.component(upstream) == DATA_ID and (query_node(ctx, upstream) or (None,))[0] == node:
                chain.append(upstream)
                nxt += [_node(e["from"]) for e in ctx.inputs(upstream)]
        frontier = nxt
    return chain


def point_at_metric(ctx: Any, draft: MetricDraft, metric_name: str, column: str | None = None) -> str:
    """
    Replace the query ``draft`` was derived from by a ``Lumen/metric``
    node of ``metric_name`` and return its name. The node nearest the
    target on the way to the query (the query itself, or the data node a
    KPI reads) is replaced, wired to everything it fed, in its tile slot; a
    KPI then reads the metric's ``column``. The helper nodes left reading
    nothing are removed. One step of the edit history, which one undo
    reverts; the metric stays saved.

    Raises ``ValueError`` when the metric is unknown or a node reads the
    SQL node as a source, which a metric node is not.
    """
    from .nodes import host_metrics

    metric = host_metrics(ctx.context).get(metric_name)
    if metric is None:
        raise ValueError(f"No metric {metric_name!r}.")
    chain = [draft.node] if draft.target == draft.node else _chain(ctx, draft.target, draft.node)
    if not chain:
        raise ValueError(f"{draft.target!r} no longer reads {draft.node!r}.")

    def rewire(edge: dict[str, Any]) -> tuple[str, list[dict[str, Any]] | None]:
        port = _port(edge["from"])
        if port == "result" and ctx.component(_node(edge["from"])) == SQL_ID:
            raise ValueError(f"{_node(edge['to'])!r} reads the SQL node as a source, which a metric node is not.")
        if port == "value" or _column_of(edge["transform"]) is not None:
            return "data", _reading(edge["transform"], column)
        if port == "data":
            return "data", edge["transform"]
        return "pipeline", edge["transform"]

    with ctx.edit():
        name = ctx.replace_node(chain[0], METRIC_ID, _metric_config(metric, draft, column),
                                new_name=_free_name(ctx, f"{metric_name}_metric"), rewire=rewire)
        for helper in chain[1:]:
            if helper in ctx.nodes() and not ctx.outputs(helper) and ctx.tile(helper) is None:
                ctx.ops.remove_node(helper)
    return name


# ---------------------------------------------------------------------------
# The node action and its dialog

def can_save_from(ctx: Any) -> bool:
    """Whether "Save as metric…" applies to a node: its numbers come from a SQL or aggregate node."""
    return project_of(ctx) is not None and query_node(ctx) is not None


def _shown(value: Any) -> str:
    if value is None:
        return "nothing yet"
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, int):
        return f"{value:,}"
    if isinstance(value, float):
        if abs(value) >= 1e15 or (value and abs(value) < 1e-3):
            return f"{value:.6g}"
        return f"{value:,.2f}".rstrip("0").rstrip(".")
    if isinstance(value, pd.DataFrame):
        return f"{len(value):,} row{'s' * (len(value) != 1)}"
    return str(value)


class SaveMetricDialog(Viewer):
    """The dialog of "Save as metric…": the definition, the two numbers compared, the SQL and Save."""

    metric_name = param.String(default="", doc="The new metric's name.")

    description = param.String(default="", doc="What the metric measures, shown on its label.")

    definition = param.String(default="", doc="The metric's definition as YAML, without its description.")

    def __init__(self, ctx: Any, draft: MetricDraft, **params):
        import panel_material_ui as pmui

        definition = dict(draft.definition)
        super().__init__(metric_name=draft.name, description=definition.pop("description", ""),
                         definition=yaml.safe_dump(definition, sort_keys=False), **params)
        self.ctx, self.draft = ctx, draft
        self.preview: MetricPreview | None = None
        self.saved: str | None = None
        self._name = pmui.TextInput.from_param(
            self.param.metric_name, label="Name", sizing_mode="stretch_width", css_classes=["lumen-save-metric-name"])
        self._description = pmui.TextInput.from_param(
            self.param.description, label="Description", placeholder="What it measures, e.g. capacity counted once per project",
            sizing_mode="stretch_width")
        self._code = pn.widgets.CodeEditor.from_param(
            self.param.definition, language="yaml", height=170, sizing_mode="stretch_width",
            theme="github_dark" if _dark() else "github_light_default",
        )
        self._notes = pmui.Alert(
            object=_literal(" ".join(draft.notes)), alert_type="info", visible=bool(draft.notes), sizing_mode="stretch_width",
            css_classes=["lumen-save-metric-notes"],
        )
        self._now = _stat("The node shows now", _shown(draft.current), "lumen-save-metric-now")
        self._metric = _stat("The metric gives", "…", "lumen-save-metric-value")
        self._verdict = pmui.Typography("", variant="body2", margin=(4, 0), css_classes=["lumen-save-metric-verdict"])
        self._sql = pn.widgets.CodeEditor(
            value="", language="sql", readonly=True, height=150, sizing_mode="stretch_width",
            theme="github_dark" if _dark() else "github_light_default",
        )
        self._rows = pn.pane.DataFrame(None, max_rows=PREVIEW_ROWS, visible=False, sizing_mode="stretch_width")
        self._message = pmui.Alert(object="", alert_type="error", visible=False, sizing_mode="stretch_width",
                                   css_classes=["lumen-save-metric-message"])
        self._preview_button = pmui.Button(label="Preview", icon="play_arrow", variant="outlined",
                                           css_classes=["lumen-save-metric-preview"])
        self._save = pmui.Button(label="Save metric", icon="save", variant="contained", disabled=True,
                                 css_classes=["lumen-save-metric-save"])
        self._point = pmui.Button(label="Point the node at it", icon="swap_horiz", variant="contained",
                                  visible=False, css_classes=["lumen-save-metric-point"])
        self._close = pmui.Button(label="Cancel", variant="text", css_classes=["lumen-save-metric-close"])
        self._preview_button.on_click(lambda _event: self.run_preview())
        self._save.on_click(lambda _event: self.save())
        self._point.on_click(lambda _event: self.point())
        self._close.on_click(lambda _event: self.close())
        self.param.watch(lambda _event: self._stale(), ["metric_name", "definition"])
        self._layout = pn.Column(
            self._name, self._description, self._notes,
            pmui.Typography("Definition", variant="subtitle2", margin=(8, 0, 0, 0)), self._code,
            pn.Row(self._now, self._metric, sizing_mode="stretch_width"), self._verdict, self._rows,
            pmui.Typography("SQL the metric runs", variant="subtitle2", margin=(8, 0, 0, 0)), self._sql,
            self._message,
            pn.Row(self._preview_button, pn.HSpacer(), self._close, self._save, self._point, sizing_mode="stretch_width"),
            sizing_mode="stretch_width", css_classes=["lumen-save-metric"], min_width=520,
        )
        self.run_preview()

    def _definition(self) -> dict[str, Any]:
        try:
            value = yaml.safe_load(self.definition) or {}
        except yaml.YAMLError as exc:
            raise ValueError(f"The definition is not YAML: {str(exc).splitlines()[0]}") from None
        if not isinstance(value, dict):
            raise ValueError("The definition must be a mapping, e.g. source, table and measures.")
        if self.description.strip():
            value = {"description": self.description.strip(), **value}
        return value

    def _stale(self):
        self._save.disabled = True
        self._metric[1].object = "…"
        self._verdict.object = "Preview the changed definition before saving."

    def run_preview(self) -> MetricPreview:
        self._message.visible = False
        if not self.ctx.is_admin():
            message = "Only an admin of the project may preview a metric."
            self._fail(message)
            return MetricPreview(problems=[message])
        try:
            definition = self._definition()
        except ValueError as exc:
            self._fail(str(exc))
            return MetricPreview(problems=[str(exc)])
        preview = self.preview = preview_metric(
            self.metric_name, definition, self.ctx.context, self.draft.measure, self.draft.dimensions)
        if not preview.ok:
            self._fail(" ".join(preview.problems))
            return preview
        self._sql.value = preview.sql
        multi = preview.value is None and preview.data is not None
        self._rows.param.update(object=preview.data.head(PREVIEW_ROWS) if multi else None, visible=multi)
        self._metric[1].object = _literal(_shown(preview.value if not multi else preview.data))
        same = agree(self.draft.current, preview.value)
        self._verdict.object = (
            "They agree." if same else
            "They differ: check the definition counts what the node should show." if same is False else
            "Compare the rows below with the node."
        )
        self._save.disabled = False
        return preview

    def _fail(self, message: str):
        self._message.param.update(object=_literal(message), alert_type="error", visible=True)
        self._save.disabled = True
        self._metric[1].object = "-"
        self._verdict.object = ""

    def save(self) -> bool:
        try:
            save_metric(self.ctx, self.metric_name, self._definition())
        except PermissionError as exc:
            self._fail(str(exc))
            return False
        except ProjectError as exc:
            self._fail(" ".join(scrub(getattr(p, "message", p)) for p in exc.problems))
            return False
        except ValueError as exc:
            self._fail(scrub(exc))
            return False
        self.saved = self.metric_name
        self._message.param.update(
            object=_literal(f"Saved the metric {self.metric_name} to the project file."), alert_type="success",
            visible=True)
        self._save.visible = False
        self._preview_button.visible = False
        self._close.label = "Done"
        self._point.visible = self.ctx.mode == "edit"
        return True

    def point(self) -> str | None:
        if self.saved is None:
            return None
        column = self.preview.column if self.preview is not None else self.draft.measure
        try:
            name = point_at_metric(self.ctx, self.draft, self.saved, column)
        except Exception as exc:
            self._fail(scrub(exc))
            return None
        self.close()
        self.ctx.notify("success", (
            f"{self.draft.target} now reads the metric {self.saved}. Undo points it back as before; "
            f"the metric {self.saved} stays saved in Admin, Metrics."
        ))
        return name

    def close(self):
        self.ctx.close_dialog()

    def __panel__(self):
        return self._layout


def _stat(label: str, value: str, css: str):
    import panel_material_ui as pmui

    return pn.Column(
        pmui.Typography(label, variant="caption", margin=0, sx={"color": "text.secondary"}),
        pmui.Typography(_literal(value), variant="h5", margin=0),
        margin=(8, 24, 0, 0), css_classes=[css],
    )


def _literal(text: str) -> str:
    """``text`` shown as it is by a Markdown pane or an alert, whatever a metric's name or an error holds."""
    from panel_flowdash.ui.text import escape_markdown

    return escape_markdown(text)


def _dark() -> bool:
    try:
        return pn.config.theme == "dark"
    except Exception:
        return False


def open_save_dialog(ctx: Any) -> SaveMetricDialog | None:
    """Open "Save as metric…" for the context's node, or say why its query cannot be saved."""
    try:
        draft = derive_draft(ctx)
    except ValueError as exc:
        ctx.notify("error", f"Cannot save this as a metric: {exc}")
        return None
    dialog = SaveMetricDialog(ctx, draft)
    ctx.open_dialog("Save as metric", dialog, width="md")
    return dialog


def register_action() -> None:
    """Offer "Save as metric…" in FlowDash's node menus, to admins only, in edit mode."""
    try:
        from panel_flowdash.node_actions import (
            NodeAction, NodeActionContext, register_node_action,
        )
    except ImportError:
        # A panel-flowdash without node actions offers no menu entry.
        return
    if not hasattr(NodeActionContext, "replace_node"):
        return
    register_node_action(NodeAction(
        ACTION_KEY, "Save as metric…", run=open_save_dialog, applies=can_save_from, icon="functions",
        admin_only=True,
    ))
