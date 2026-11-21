"""
Lumen components as FlowDash nodes. Experimental: names and node configs may change.

The package is a FlowDash component library: ``FlowDashApp(component_libraries=["lumen.flowdash"])``
or ``flowdash serve --component-library lumen.flowdash`` offers its nodes under
the ids in ``COMPONENTS``. A source node can only use the sources the host
gives the editor in its context under ``SOURCES_KEY`` (``"lumen.sources"``),
by ref: ``lumen serve --flowdash`` passes the served data, and a ``flowdash
serve`` project sets ``editor_options = {"context": {"lumen.sources": {...}}}``
in its ``__init__.py``. Without sources, source nodes say how to provide them.

Importing ``lumen`` does not import this package or panel-flowdash.
"""
from .checks import (
    PROBE_TIMEOUT, CheckDiagnostic, check_chart, check_columns,
    check_filter_config, check_query, check_source_config, check_sql_config,
    check_table_config, check_view_config, extract_sql_error_location,
    forget_probes, pipeline_columns, probe_query, selection_warnings,
)
from .nodes import (
    COMPONENTS, DATA_ID, FILTER_ID, NODE_TIMEOUT, PIPELINE_TYPE, PREVIEW_ROWS,
    QUERY_TIMEOUT, RESULT_LIMIT, SOURCE_ID, SOURCE_TYPE, SOURCES_KEY, SQL_ID,
    TABLE_ID, VIEW_ID, DataNode, FilterNode, SourceNode, SQLNode, TableNode,
    ViewNode, bind_source, bound_sources, compile_sql, effective_limit, fetch,
    filter_condition, filtered_pipeline, forget, host_sources, missing_source,
    pipeline_sql, primed_result, python_value, result_sql, row_ceiling,
    selection_conditions, set_row_ceiling, table_pipeline, unbind_source,
)

# What the package offers as a FlowDash component library, under stable ids
# that saved dashboards refer to.
__flowdash_components__ = COMPONENTS

__all__ = [
    "COMPONENTS",
    "DATA_ID",
    "FILTER_ID",
    "NODE_TIMEOUT",
    "PIPELINE_TYPE",
    "PREVIEW_ROWS",
    "PROBE_TIMEOUT",
    "QUERY_TIMEOUT",
    "RESULT_LIMIT",
    "SOURCES_KEY",
    "SOURCE_ID",
    "SOURCE_TYPE",
    "SQL_ID",
    "TABLE_ID",
    "VIEW_ID",
    "CheckDiagnostic",
    "DataNode",
    "FilterNode",
    "SQLNode",
    "SourceNode",
    "TableNode",
    "ViewNode",
    "bind_source",
    "bound_sources",
    "check_chart",
    "check_columns",
    "check_filter_config",
    "check_query",
    "check_source_config",
    "check_sql_config",
    "check_table_config",
    "check_view_config",
    "compile_sql",
    "effective_limit",
    "extract_sql_error_location",
    "fetch",
    "filter_condition",
    "filtered_pipeline",
    "forget",
    "forget_probes",
    "host_sources",
    "missing_source",
    "pipeline_columns",
    "pipeline_sql",
    "primed_result",
    "probe_query",
    "python_value",
    "result_sql",
    "row_ceiling",
    "selection_conditions",
    "selection_warnings",
    "set_row_ceiling",
    "table_pipeline",
    "unbind_source",
]
