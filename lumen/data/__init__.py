"""
Model-free data tools: describing data to a language model and making
model-written SQL safe to run, importable without any LLM library or
Lumen's agent framework.
"""
from .catalog import (
    Column, Metaset, TableCatalogEntry, get_metaset,
)
from .contracts import (
    ColumnProblem, QueryColumns, query_columns, schema_problems, source_schema,
)
from .quality import lint_data
from .sql import (
    build_distinct_values_sql, execute_distinct_values,
    execute_exploration_sql, format_exploration_result, inline_tables,
    validate_read_only_sql,
)
from .stats import get_stats_store
from .utils import clean_sql, format_error

__all__ = [
    "Column",
    "ColumnProblem",
    "Metaset",
    "QueryColumns",
    "TableCatalogEntry",
    "build_distinct_values_sql",
    "clean_sql",
    "execute_distinct_values",
    "execute_exploration_sql",
    "format_error",
    "format_exploration_result",
    "get_metaset",
    "get_stats_store",
    "inline_tables",
    "lint_data",
    "query_columns",
    "schema_problems",
    "source_schema",
    "validate_read_only_sql",
]
