"""
Save as metric: turn the query behind a KPI, a Lumen SQL node or an
aggregate into a metric of the project file, then point the node at it.

An admin picks "Save as metric…" in the node menu of such a node. The
dialog starts from a definition derived from the node's query
(``derive_draft``): the structured form (a table, measures and dimensions)
where the query maps onto it cleanly, else the SQL form with a note on what
to adjust. Before saving it shows what the node shows now beside what the
metric gives (``preview_metric``) and the SQL the metric runs, so the two
can be compared. ``save_metric`` writes the metric with ``Project.set``,
which checks the whole section against the sources before writing the file
atomically, and refuses anyone who is no admin. ``point_at_metric`` then
replaces the node's query by a ``Lumen/metric`` node wired the same, keeping
its tile, as one edit that undo reverts.

The provenance labels follow from the nodes: a metric node reports
"Metric: <name>" and a SQL or aggregate node "Ad hoc" with how its number
is computed (``describe_query``), which FlowDash shows on the views that read
them, see ``panel_flowdash.node_actions``.
"""
from __future__ import annotations

import logging
import re

from collections.abc import Mapping
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

# The key of the editor context the host puts its loaded project file in, which metrics are saved to.
PROJECT_KEY = "lumen.project"

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


def describe_aggregate(method: str, columns: list[str], by: list[str]) -> str:
    """How an aggregate node computes its numbers, e.g. "Sum of p_cap over all rows"."""
    word = _METHOD_WORDS[METHODS[method]]
    what = "Count of rows" if not columns else f"{word} of {', '.join(columns)}"
    return f"{what} {'by ' + ', '.join(by) if by else 'over all rows'} of its input; no metric was used."


# ---------------------------------------------------------------------------
# Deriving a definition from a node

@dataclass
class MetricDraft:
    """A metric definition derived from a node's query, and what the node shows now."""

    name: str
    definition: dict[str, Any]
    node: str
    """The instance id of the SQL or aggregate node the definition is derived from."""
    target: str
    """The instance id of the node the action was opened on, e.g. a KPI reading ``node``."""
    measure: str | None = None
    """The column the target shows, e.g. the KPI's; None shows them all."""
    dimensions: list[str] = field(default_factory=list)
    structured: bool = False
    notes: list[str] = field(default_factory=list)
    current: Any = None
    """What the target shows now: a number, or rows."""


def _component(editor: Any, instance_id: str) -> str | None:
    item = next((i for i in editor._tile_items if i["instance_id"] == instance_id), None)
    return None if item is None or "placeholder" in item else editor._resolve_component_id(item["component_id"])


def _instance(editor: Any, instance_id: str) -> Any:
    return editor._component_instances.get(instance_id)


def _inputs(editor: Any, instance_id: str) -> list[dict[str, Any]]:
    return [e for e in editor._dataflow_graph.edges if e["target"] == instance_id]


def _outputs(editor: Any, instance_id: str) -> list[dict[str, Any]]:
    return [e for e in editor._dataflow_graph.edges if e["source"] == instance_id]


def _column_of(transform: list[dict[str, Any]] | None) -> str | None:
    return next((step.get("column") for step in transform or [] if step.get("op") == "column"), None)


def query_node(editor: Any, instance_id: str) -> tuple[str, str | None, dict[str, Any] | None] | None:
    """
    The SQL or aggregate node a node's numbers come from: the node itself,
    or the nearest one up its inputs through data nodes, with the column
    the edge into the node picks and that edge; None when there is none or
    a metric node comes first.
    """
    component = _component(editor, instance_id)
    if component in QUERY_NODES:
        return instance_id, None, None
    if component in (METRIC_ID, None):
        return None
    frontier: list[tuple[str, str | None, dict[str, Any] | None]] = [(instance_id, None, None)]
    for _ in range(DEPTH):
        nxt = []
        for node, column, first in frontier:
            for edge in _inputs(editor, node):
                upstream = edge["source"]
                kind = _component(editor, upstream)
                picked = column or _column_of(edge.get("transform"))
                entry = first or edge
                if kind in QUERY_NODES:
                    return upstream, picked, entry
                if kind == DATA_ID:
                    nxt.append((upstream, picked, entry))
        frontier = nxt
    return None


def _ref_of(source: Any, context: Mapping[str, Any]) -> str | None:
    from .nodes import host_sources

    return next((ref for ref, s in host_sources(context).items() if s is source), None)


def _base(editor: Any, instance_id: str, context: Mapping[str, Any]) -> tuple[str, str | None, str, list[str]]:
    """
    What a pipeline input reads, as ``(ref, table, sql, notes)``: the source
    ref, the table when it is a plain table node, else None, and the full
    query on the source. Raises ``ValueError`` for an input a metric cannot
    say, e.g. a filter fed by a control.
    """
    from .nodes import FilterNode, SQLNode, TableNode

    node = _instance(editor, instance_id)
    if isinstance(node, TableNode):
        source = node.source if node.source is not None else node._read_source()
        ref = _ref_of(source, context)
        if ref is None or not node.table:
            raise ValueError("its table node reads no source the app serves")
        dialect = _dialect(source)
        sql = exp.select("*").from_(exp.Table(this=exp.Identifier(this=node.table, quoted=True))).sql(dialect=dialect)
        return ref, node.table, sql, []
    if isinstance(node, SQLNode):
        root, statement = _statement(node)
        ref = _ref_of(root, context)
        if ref is None:
            raise ValueError("its SQL node reads no source the app serves")
        return ref, None, statement, []
    if isinstance(node, FilterNode):
        upstream = next((e["source"] for e in _inputs(editor, instance_id) if e["target_port"] == "pipeline"), None)
        if upstream is None:
            raise ValueError("its filter node has no input")
        ref, table, sql, notes = _base(editor, upstream, context)
        notes = [*notes, f"The filter on {node.field!r} is left out; add it as a parameter or a dimension to filter by."]
        return ref, table, sql, notes
    raise ValueError(f"it reads a {type(node).__name__}, which a metric cannot say")


def _statement(node: Any) -> tuple[BaseSQLSource, str]:
    from .results import expand

    source = node._input
    if not isinstance(source, BaseSQLSource) or not node.sql.strip():
        raise ValueError("the SQL node has no source or query yet")
    return expand(node.sql, source)


def _dialect(source: Any) -> str | None:
    dialect = getattr(source, "dialect", None)
    return None if dialect in (None, "any") else dialect


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


def _title(editor: Any, instance_id: str) -> str:
    state = editor._dataflow_graph.get_config_state(instance_id)
    for key in ("label", "title"):
        value = getattr(state, key, None) if state is not None and key in state.param else None
        if isinstance(value, str) and value.strip():
            return value
    return editor._node_name(instance_id)


def current_value(editor: Any, instance_id: str) -> Any:
    """What a node shows now: a KPI's number, a SQL node's one value, or its rows."""
    from .nodes import SQLNode, single_number

    state = editor._dataflow_graph.get_state(instance_id)
    if _component(editor, instance_id) == KPI_ID and state is not None and "value" in state.param:
        return state.value
    node = _instance(editor, instance_id)
    if isinstance(node, SQLNode):
        data = node.data()
        if isinstance(data, pd.DataFrame):
            try:
                return single_number(data)
            except ValueError:
                return data
        return None
    result = getattr(node, "result", None)
    pipeline = result() if callable(result) else None
    data = getattr(pipeline, "data", None)
    return data if isinstance(data, pd.DataFrame) else None


def derive_draft(editor: Any, instance_id: str) -> MetricDraft:
    """
    A metric definition derived from the query behind a node, see the
    module docstring. Raises ``ValueError`` when there is no query to
    derive one from.
    """
    from .nodes import SQLNode
    from .transforms import AggregateNode

    found = query_node(editor, instance_id)
    if found is None:
        raise ValueError("Its numbers do not come from a Lumen SQL or aggregate node.")
    node_id, column, _ = found
    context = editor.context
    node = _instance(editor, node_id)
    notes: list[str] = []
    definition: dict[str, Any]
    dimensions: list[str] = []
    structured = False
    if isinstance(node, SQLNode):
        root, statement = _statement(node)
        ref = _ref_of(root, context)
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
    elif isinstance(node, AggregateNode):
        upstream = next((e["source"] for e in _inputs(editor, node_id) if e["target_port"] == "pipeline"), None)
        if upstream is None:
            raise ValueError("Its aggregate node has no input.")
        try:
            ref, table, sql, notes = _base(editor, upstream, context)
        except ValueError as exc:
            raise ValueError(f"Its aggregate node cannot be saved as a metric: {exc}.") from None
        function = METHODS[node.method]
        columns = list(node.columns) or [COUNT_COLUMN]
        dimensions = list(node.by)
        if table is not None:
            measures = {
                c: "COUNT(*)" if not node.columns else exp.func(function, exp.column(c, quoted=True)).sql()
                for c in columns
            }
            definition = {"source": ref, "table": table, "measures": measures}
            if dimensions:
                definition["dimensions"] = {d: d for d in dimensions}
            structured = True
        else:
            dialect = _dialect(getattr(_instance(editor, upstream), "_input", None))
            definition = {
                "source": ref,
                "sql": SQLAggregate(by=node.by, columns=node.columns, method=node.method, write=dialect).apply(sql),
            }
            notes.append("It aggregates the result of a SQL node, so it is kept as a SQL metric over that query.")
    else:
        raise ValueError("Its numbers do not come from a Lumen SQL or aggregate node.")
    if structured and column is None and len(definition.get("measures", {})) == 1:
        column = next(iter(definition["measures"]))
    return MetricDraft(
        name=_slug(_title(editor, instance_id)), definition=definition, node=node_id, target=instance_id,
        measure=column, dimensions=dimensions, structured=structured, notes=notes,
        current=current_value(editor, instance_id),
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


def project_of(editor: Any) -> Project | None:
    """The project file metrics are saved to: the context's, or the one a live metric set reads."""
    project = editor.context.get(PROJECT_KEY) or getattr(editor.context.get(METRICS_KEY), "project", None)
    return project if isinstance(project, Project) and project.path is not None else None


def save_metric(editor: Any, name: str, definition: Mapping[str, Any], *, replace: bool = False) -> MetricSet:
    """
    Write the metric ``name`` of ``definition`` to the project file's
    ``metrics:`` section with ``Project.set`` and return the app's metrics,
    which every session's metric nodes then read.

    ``Project.set`` loads the section with the new metric, checking every
    metric against the sources, before it writes the file atomically; any
    problem raises ``ProjectError`` and leaves the file as it was.

    Raises
    ------
    PermissionError
        Unless the editor's session administers the project.
    ValueError
        Without a project file, or for a name taken while ``replace`` is off.
    """
    is_admin = getattr(editor, "is_admin", None)
    if is_admin is None or not is_admin():
        raise PermissionError("Only an admin of the project may save a metric.")
    project = project_of(editor)
    if project is None:
        raise ValueError("The app has no project file to save metrics to; serve it with --project flowdash.yaml.")
    check_definition(name, definition, editor.context)
    section = dict(project.raw(SECTION) or {})
    if name in section and not replace:
        raise ValueError(f"The project already has a metric {name!r}; pick another name.")
    section[name] = dict(definition)
    project.set(SECTION, section)
    metrics = read_metrics(project)
    shared = editor.context.get(METRICS_KEY)
    if getattr(shared, "project", None) is project:
        # A metric set that reads the project file as it is now has the metric already.
        return shared
    if isinstance(shared, MetricSet):
        shared.replace_with(metrics)
    else:
        editor.context[METRICS_KEY] = metrics
    return editor.context[METRICS_KEY]


# ---------------------------------------------------------------------------
# Pointing the node at the metric

def _taken(editor: Any) -> set[str]:
    return {editor._node_name(i["instance_id"]) for i in editor._tile_items}


def _free_name(editor: Any, stem: str) -> str:
    taken = _taken(editor)
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


def _tile_slot(editor: Any, instance_id: str) -> dict[str, Any] | None:
    from panel_flowdash.ui import tile_layout

    item = next((i for i in editor._tile_items if i["instance_id"] == instance_id), None)
    area = tile_layout.area_of(editor, item) if item is not None else None
    entry = tile_layout.tile_entry(editor, instance_id)
    if area is None or entry.get("visible") is False:
        return None
    ids = [i["instance_id"] for i in tile_layout.area_items(editor, area)]
    return {"area": area, "index": ids.index(instance_id), "width": entry.get("width"), "height": entry.get("height")}


def point_at_metric(editor: Any, draft: MetricDraft, metric_name: str, column: str | None = None) -> str:
    """
    Replace the query ``draft`` was derived from by a ``Lumen/metric``
    node of ``metric_name``, wired to what it fed, and return the new
    node's name. A view the action was opened on, e.g. a KPI, keeps its
    tile and reads the metric's ``column``; a SQL or aggregate node opened
    on itself gives its tile to the metric node. The helper nodes the query
    went through are removed when nothing else reads them. One edit, which
    undo reverts.

    Raises ``ValueError`` when the metric is unknown or a node it fed takes
    an output a metric node has none of, e.g. a SQL node's ``result``.
    """
    from panel_flowdash.agent import DashboardOps, OperationError

    from .nodes import host_metrics

    metric = host_metrics(editor.context).get(metric_name)
    if metric is None:
        raise ValueError(f"No metric {metric_name!r}.")
    ops = DashboardOps(editor)
    name_of = editor._node_name
    node, target = draft.node, draft.target
    if target == node:
        feeding = _outputs(editor, node)
    else:
        feeding = [e for e in _inputs(editor, target) if _feeds_from(editor, e["source"], node)]
    for edge in feeding:
        if edge["source_port"] == "result" and _component(editor, edge["source"]) == SQL_ID:
            raise ValueError(f"{name_of(edge['target'])!r} reads the SQL node as a source, which a metric node is not.")
    slot = _tile_slot(editor, node) if target == node else None
    chain = _chain(editor, target, node) if target != node else [node]
    name = _free_name(editor, f"{metric_name}_metric")
    ops.add_node(METRIC_ID, _metric_config(metric, draft, column), name=name)
    try:
        for edge in feeding:
            source = f"{name_of(edge['source'])}.{edge['source_port']}"
            to = f"{name_of(edge['target'])}.{edge['target_port']}"
            ops.disconnect(source, to)
            if edge["source_port"] in ("value",) or edge["target_port"] == "value":
                ops.connect(f"{name}.data", to, _reading(edge.get("transform"), column))
            elif edge["source_port"] == "data":
                ops.connect(f"{name}.data", to, edge.get("transform") or None)
            else:
                ops.connect(f"{name}.pipeline", to, edge.get("transform") or None)
        if slot is not None:
            if slot["area"] == "grid":
                ops.place_tile(name, index=slot["index"], width=slot["width"], height=slot["height"])
            else:
                ops.place_tile(name, area=slot["area"], index=slot["index"])
        else:
            ops.place_tile(name, visible=False)
        for instance_id in chain:
            if instance_id in editor._component_instances and not _outputs(editor, instance_id) and (
                    instance_id == node or _tile_slot(editor, instance_id) is None):
                editor.remove_component(instance_id)
    except (OperationError, ValueError, PermissionError):
        editor.undo()
        raise
    return name


def _feeds_from(editor: Any, start: str, node: str) -> bool:
    """Whether ``start`` is ``node`` or reads it through data nodes."""
    frontier = [start]
    for _ in range(DEPTH):
        if node in frontier:
            return True
        frontier = [e["source"] for f in frontier for e in _inputs(editor, f) if _component(editor, f) == DATA_ID]
    return node in frontier


def _chain(editor: Any, target: str, node: str) -> list[str]:
    """The nodes from ``target``'s input up to ``node``, nearest first: the helpers a KPI's number goes through."""
    chain, frontier = [], [e["source"] for e in _inputs(editor, target)]
    for _ in range(DEPTH):
        nxt = []
        for f in frontier:
            if f == node:
                chain.append(f)
            elif _component(editor, f) == DATA_ID and _feeds_from(editor, f, node):
                chain.append(f)
                nxt += [e["source"] for e in _inputs(editor, f)]
        frontier = nxt
    return chain


# ---------------------------------------------------------------------------
# The node action and its dialog

def can_save_from(editor: Any, instance_id: str) -> bool:
    """Whether "Save as metric…" applies to a node: its numbers come from a SQL or aggregate node."""
    return project_of(editor) is not None and query_node(editor, instance_id) is not None


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

    def __init__(self, editor: Any, draft: MetricDraft, **params):
        import panel_material_ui as pmui

        definition = dict(draft.definition)
        super().__init__(metric_name=draft.name, description=definition.pop("description", ""),
                         definition=yaml.safe_dump(definition, sort_keys=False), **params)
        self.editor, self.draft = editor, draft
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
        try:
            definition = self._definition()
        except ValueError as exc:
            self._fail(str(exc))
            return MetricPreview(problems=[str(exc)])
        preview = self.preview = preview_metric(
            self.metric_name, definition, self.editor.context, self.draft.measure, self.draft.dimensions)
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
            save_metric(self.editor, self.metric_name, self._definition())
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
        self._point.visible = not self.editor.read_only
        return True

    def point(self) -> str | None:
        if self.saved is None:
            return None
        column = self.preview.column if self.preview is not None else self.draft.measure
        try:
            name = point_at_metric(self.editor, self.draft, self.saved, column)
        except (ValueError, PermissionError) as exc:
            self._fail(scrub(exc))
            return None
        self.close()
        _notify(self.editor, f"The node now reads the metric {self.saved} ({name}); undo reverts it.")
        return name

    def close(self):
        from panel_flowdash.util import close_dialog

        close_dialog()

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


def _notify(editor: Any, message: str) -> None:
    try:
        editor._notify("success", message)
    except Exception:
        logger.info(message)


def open_save_dialog(editor: Any, instance_id: str) -> SaveMetricDialog | None:
    """Open "Save as metric…" for a node, or say why its query cannot be saved."""
    from panel_flowdash.util import open_dialog

    try:
        draft = derive_draft(editor, instance_id)
    except ValueError as exc:
        editor._notify("error", f"Cannot save this as a metric: {exc}")
        return None
    dialog = SaveMetricDialog(editor, draft)
    open_dialog("Save as metric", dialog, width="md", full_width=True)
    return dialog


def register_action() -> None:
    """Offer "Save as metric…" in FlowDash's node menus, to admins only."""
    try:
        from panel_flowdash.node_actions import (
            NodeAction, register_node_action,
        )
    except ImportError:
        # A panel-flowdash without node actions offers no menu entry.
        return
    register_node_action(NodeAction(
        ACTION_KEY, "Save as metric…", run=open_save_dialog, applies=can_save_from, icon="functions",
        admin_only=True,
    ))
