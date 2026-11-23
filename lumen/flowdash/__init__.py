"""
Lumen components as FlowDash nodes. Experimental: names and node configs may change.

The package is a FlowDash component library: ``FlowDashApp(component_libraries=["lumen.flowdash"])``
or ``flowdash serve --component-library lumen.flowdash`` offers its nodes under
the ids in ``COMPONENTS``. A metric node can only use the metrics the host gives under ``METRICS_KEY``
(``"lumen.metrics"``), read from the project file's ``metrics:`` section, see
``lumen.flowdash.metrics``. A source node can only use the sources the host
gives the editor in its context under ``SOURCES_KEY`` (``"lumen.sources"``),
by ref: ``lumen serve --flowdash`` passes the served data, and a ``flowdash
serve`` project sets ``editor_options = {"context": {"lumen.sources": {...}}}``
in its ``__init__.py``. Without sources, source nodes say how to provide them.

Importing ``lumen`` does not import this package or panel-flowdash.
"""
from .checks import (
    PROBE_TIMEOUT, CheckDiagnostic, check_chart, check_columns,
    check_filter_config, check_query, check_source_config, check_sql_config,
    check_table_config, check_view_config, check_view_spec,
    extract_sql_error_location, forget_probes, pipeline_columns, probe_query,
    selection_warnings,
)
from .deckgl import DECKGL_ID, DeckGLNode, check_deckgl
from .nodes import (
    COMPONENTS, DATA_ID, FILTER_ID, FILTER_OPTIONS, METRIC_ID, METRICS_KEY,
    NODE_TIMEOUT, PARAMETER_ID, PIPELINE_TYPE, PREVIEW_ROWS, QUERY_TIMEOUT,
    RESULT_LIMIT, SOURCE_ALIASES_KEY, SOURCE_ID, SOURCE_TYPE, SOURCES_KEY,
    SQL_ID, TABLE_ID, VIEW_ID, DataNode, FilterMismatch, FilterNode,
    GlobalFilterTarget, LumenViewNode, MetricNode, ParameterNode, SourceNode,
    SQLNode, TableNode, ViewNode, bind_source, bound_sources, column_values,
    compile_sql, effective_limit, fetch, filter_condition, filterable_columns,
    filtered_pipeline, forget, global_condition, global_conditions,
    globally_filtered, host_metrics, host_sources, listed_tables,
    lookup_source, metric_pipeline, missing_source, pipeline_sql,
    primed_result, python_value, register_view, result_sql, row_ceiling,
    selection_conditions, set_row_ceiling, spec_param, table_pipeline,
    unbind_source,
)
from .save_metric import PROJECT_KEY, register_action
from .table_view import TABLE_VIEW_ID, TableViewNode
from .transforms import (
    AGGREGATE_ID, DERIVE_ID, SELECT_ID, SORT_ID, TRANSFORM_COMPONENTS,
    AggregateNode, DeriveNode, SelectNode, SortNode, transformed_pipeline,
)

COMPONENTS.update(TRANSFORM_COMPONENTS)
COMPONENTS[TABLE_VIEW_ID] = TableViewNode
COMPONENTS[DECKGL_ID] = DeckGLNode

# "Save as metric…" in FlowDash's node menus, for admins.
register_action()

# What the package offers as a FlowDash component library, under stable ids
# that saved dashboards refer to.
__flowdash_components__ = COMPONENTS

__all__ = [
    "AGGREGATE_ID",
    "COMPONENTS",
    "DATA_ID",
    "DECKGL_ID",
    "DERIVE_ID",
    "FILTER_ID",
    "FILTER_OPTIONS",
    "METRICS_KEY",
    "METRIC_ID",
    "NODE_TIMEOUT",
    "PARAMETER_ID",
    "PIPELINE_TYPE",
    "PREVIEW_ROWS",
    "PROBE_TIMEOUT",
    "PROJECT_KEY",
    "QUERY_TIMEOUT",
    "RESULT_LIMIT",
    "SELECT_ID",
    "SORT_ID",
    "SOURCES_KEY",
    "SOURCE_ALIASES_KEY",
    "SOURCE_ID",
    "SOURCE_TYPE",
    "SQL_ID",
    "TABLE_ID",
    "TABLE_VIEW_ID",
    "VIEW_ID",
    "AggregateNode",
    "CheckDiagnostic",
    "DataNode",
    "DeckGLNode",
    "DeriveNode",
    "FilterMismatch",
    "FilterNode",
    "GlobalFilterTarget",
    "LumenViewNode",
    "MetricNode",
    "ParameterNode",
    "SQLNode",
    "SelectNode",
    "SortNode",
    "SourceNode",
    "TableNode",
    "TableViewNode",
    "ViewNode",
    "bind_source",
    "bound_sources",
    "check_chart",
    "check_columns",
    "check_deckgl",
    "check_filter_config",
    "check_query",
    "check_source_config",
    "check_sql_config",
    "check_table_config",
    "check_view_config",
    "check_view_spec",
    "column_values",
    "compile_sql",
    "effective_limit",
    "extract_sql_error_location",
    "fetch",
    "filter_condition",
    "filterable_columns",
    "filtered_pipeline",
    "forget",
    "forget_probes",
    "global_condition",
    "global_conditions",
    "globally_filtered",
    "host_metrics",
    "host_sources",
    "listed_tables",
    "lookup_source",
    "metric_pipeline",
    "missing_source",
    "pipeline_columns",
    "pipeline_sql",
    "primed_result",
    "probe_query",
    "python_value",
    "register_view",
    "result_sql",
    "row_ceiling",
    "selection_conditions",
    "selection_warnings",
    "set_row_ceiling",
    "spec_param",
    "table_pipeline",
    "transformed_pipeline",
    "unbind_source",
]
