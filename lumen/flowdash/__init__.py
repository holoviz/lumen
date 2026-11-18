"""
Lumen components as FlowDash nodes. Experimental: names and node configs may change.

The package is a FlowDash component library: ``FlowDashApp(component_libraries=["lumen.flowdash"])``
or ``flowdash serve --component-library lumen.flowdash`` offers its nodes under
the ids in ``COMPONENTS``. A source node can only use sources the host binds to
the editor with ``bind_source``, which plain ``flowdash serve`` cannot do until
panel-flowdash can pass host resources to editors (panel-flowdash#166), so
there source nodes report that no source is bound; ``lumen serve --flowdash``
binds the served data and is the working entry point.

Needs panel-flowdash, which Lumen does not install; nothing else in Lumen
imports this package.
"""
try:
    import panel_flowdash  # noqa: F401
except ImportError as exc:
    raise ImportError(
        "lumen.flowdash needs panel-flowdash, which is not installed: pip install panel-flowdash"
    ) from exc

from .checks import (
    check_chart, check_filter_config, check_query, check_sql_config,
    check_table_config, check_view_config, pipeline_columns,
)
from .nodes import (
    COMPONENTS, DATA_ID, FILTER_ID, NODE_TIMEOUT, PIPELINE_TYPE, PREVIEW_ROWS,
    QUERY_TIMEOUT, RESULT_LIMIT, SOURCE_ID, SOURCE_TYPE, SQL_ID, TABLE_ID,
    VIEW_ID, DataNode, FilterNode, SourceNode, SQLNode, TableNode, ViewNode,
    bind_source, bound_sources, compile_sql, effective_limit, fetch,
    filter_condition, filtered_pipeline, forget, host_sources, pipeline_sql,
    primed_result, python_value, result_sql, row_ceiling, set_row_ceiling,
    table_pipeline, unbind_source,
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
    "QUERY_TIMEOUT",
    "RESULT_LIMIT",
    "SOURCE_ID",
    "SOURCE_TYPE",
    "SQL_ID",
    "TABLE_ID",
    "VIEW_ID",
    "DataNode",
    "FilterNode",
    "SQLNode",
    "SourceNode",
    "TableNode",
    "ViewNode",
    "bind_source",
    "bound_sources",
    "check_chart",
    "check_filter_config",
    "check_query",
    "check_sql_config",
    "check_table_config",
    "check_view_config",
    "compile_sql",
    "effective_limit",
    "fetch",
    "filter_condition",
    "filtered_pipeline",
    "forget",
    "host_sources",
    "pipeline_columns",
    "pipeline_sql",
    "primed_result",
    "python_value",
    "result_sql",
    "row_ceiling",
    "set_row_ceiling",
    "table_pipeline",
    "unbind_source",
]
