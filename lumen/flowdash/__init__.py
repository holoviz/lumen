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

from .nodes import (
    COMPONENTS, PIPELINE_TYPE, SOURCE_ID, SOURCE_TYPE, SQL_ID, VIEW_ID,
    SourceNode, SQLNode, ViewNode, bind_source, compile_sql, source_config,
    unbind_source,
)

__all__ = [
    "COMPONENTS",
    "PIPELINE_TYPE",
    "SOURCE_ID",
    "SOURCE_TYPE",
    "SQL_ID",
    "VIEW_ID",
    "SQLNode",
    "SourceNode",
    "ViewNode",
    "bind_source",
    "compile_sql",
    "source_config",
    "unbind_source",
]
