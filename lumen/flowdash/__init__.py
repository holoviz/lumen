"""
Lumen components as FlowDash nodes. Experimental: names and node configs may change.

Needs panel-flowdash, which Lumen does not install; nothing else in Lumen
imports this package.
"""
try:
    import panel_flowdash  # noqa: F401
except ImportError as exc:
    raise ImportError(
        "lumen.flowdash needs panel-flowdash, which is not installed: pip install panel-flowdash"
    ) from exc

from .checks import check_chart, check_query
from .nodes import (
    COMPONENTS, PIPELINE_TYPE, QUERY_TIMEOUT, RESULT_LIMIT, SOURCE_ID,
    SOURCE_TYPE, SQL_ID, VIEW_ID, SourceNode, SQLNode, ViewNode, bind_source,
    bound_sources, compile_sql, primed_result, result_sql, unbind_source,
)

__all__ = [
    "COMPONENTS",
    "PIPELINE_TYPE",
    "QUERY_TIMEOUT",
    "RESULT_LIMIT",
    "SOURCE_ID",
    "SOURCE_TYPE",
    "SQL_ID",
    "VIEW_ID",
    "SQLNode",
    "SourceNode",
    "ViewNode",
    "bind_source",
    "bound_sources",
    "check_chart",
    "check_query",
    "compile_sql",
    "primed_result",
    "result_sql",
    "unbind_source",
]
