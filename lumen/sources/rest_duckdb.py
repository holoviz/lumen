"""
RESTDuckDBSource - DuckDB source with URL parameterization support.

Enables dynamic URL query parameter updates for REST API endpoints,
making it compatible with LLM-based agents like SQLAgent.
"""
from __future__ import annotations

from typing import Any, ClassVar
from urllib.parse import urlencode, urlparse, urlunparse

import param
import sqlglot

from duckdb import InvalidInputException

from lumen.sources.base import cached

from .duckdb import DuckDBSource


class RESTDuckDBSource(DuckDBSource):
    """
    DuckDBSource subclass that supports parameterized REST API URLs.

    This source allows defining URL templates with dynamic query parameters
    that can be updated at runtime, enabling LLM agents to modify API calls
    on the fly.

    Parameters
    ----------
    materialize : bool, default True
        Whether to materialize REST tables as temp tables on initialization.
        When True, REST tables are immediately fetched and stored as temp tables,
        allowing them to be referenced in SQL expressions.
    cache : bool, default False
        Whether to cache HTTP responses using DuckDB's cache_httpfs extension.
        When True, repeated requests to the same URL return cached results.

    REST table configurations are specified as dictionaries with:
    - 'url': Base URL of the REST endpoint
    - 'url_params': Dict of query parameters (including 'format' if the API supports it)
    - 'required_params': Optional list of parameter names that must be provided
    - 'read_fn': Optional override for the DuckDB read function ('json', 'csv', 'parquet').
                 If not specified, auto-detects from url_params['format'] or URL extension.
    - 'read_options': Optional dict of DuckDB read_* function options
    """

    cache_httpfs = param.Boolean(
        default=True,
        doc="""
        Whether to cache HTTP responses using DuckDB's cache_httpfs extension.
        When True, repeated requests to the same URL return cached results.
        """,
    )

    data_format = param.String(
        default="json",
        doc="""
        Default data format for REST tables if not specified in url_params.
        Used to determine the appropriate DuckDB read function.
        """,
    )

    source_type: ClassVar[str] = 'rest_duckdb'

    # Map format to DuckDB read function (using auto variants where available)
    _format_to_reader: ClassVar[dict[str, str]] = {
        'json': 'read_json_auto',
        'csv': 'read_csv_auto',
        'parquet': 'read_parquet',
        'ndjson': 'read_ndjson_auto',
    }

    _cached_rest_tables = param.Dict(default={}, doc="Internal dict for REST tables to their last used table_params.")

    def _is_rest_table(self, table: str) -> bool:
        if table not in self.tables:
            raise ValueError(f"Table '{table}' not found in source tables.")
        return isinstance(self.tables[table], dict) and 'url' in self.tables[table]

    def _ensure_sql_expr_materialized(self, sql_expr: str, url_params: dict | None = None) -> None:
        if not isinstance(sql_expr, str):
            # Not a SQL expression, skip
            return

        table_objs = sqlglot.parse_one(sql_expr).find_all(sqlglot.exp.Table)
        tables = {table_obj.name for table_obj in table_objs if self._is_rest_table(table_obj.name)}
        for table in tables:
            if not self._is_rest_table(table):
                return
            table_params = self.tables[table]
            if self._cached_rest_tables.get(table) != table_params:
                url_params = {**table_params.get("url_params", {}), **(url_params or {})}
                df = self.get(table, url_params=url_params)
                self._connection.from_df(df).to_view(table)
                self._cached_rest_tables[table] = table_params

    def get_sql_expr(self, table: str) -> str:
        if self._is_rest_table(table):
            read_fn = self._format_to_reader[self.data_format]
            return f"SELECT * FROM {read_fn}(?)"
        return super().get_sql_expr(table)

    @cached
    def get(self, table: str, url_params: dict[str, Any] | None = None, **query):
        if not self._is_rest_table(table):
            return super().get(table, **query)

        table_params = self.tables[table].copy()
        url_params = {**table_params.get("url_params", {}), **(url_params or {})}
        last_exc = None
        url = self.render_table_url(table, url_params=url_params)
        data_format = url_params.get("format", self.data_format)
        for try_data_format in (data_format, 'csv'):
            with self.param.update(table_params={table: [url]}, data_format=try_data_format):
                try:
                    return super().get(table, **query)
                except InvalidInputException as exc:
                    last_exc = exc
                    continue

        if last_exc is not None:
            raise last_exc

    def render_table_url(self, table: str, url_params: dict[str, Any] | None = None) -> str:
        """
        Get the current full URL for a REST table.

        Parameters
        ----------
        table : str
            Name of the REST table
        url_params : dict[str, Any] | None
            Optional URL parameters to override or add to the table's url params

        Returns
        -------
        str
            Full URL with current query parameters

        Raises
        ------
        ValueError
            If table is not a REST table or if neither table nor config is provided
        """
        if not self._is_rest_table(table):
            raise ValueError(f"Table '{table}' is not a REST table.")

        table_params = self.tables[table]
        url = table_params["url"]
        if url_params is None:
            url_params = table_params["url_params"]
        parsed = urlparse(url)
        return urlunparse((
            parsed.scheme,
            parsed.netloc,
            parsed.path,
            parsed.params,
            urlencode(url_params),
            parsed.fragment,
        ))

    def execute(self, sql_query: str, params: list | dict | None = None, url_params: dict[str, Any] | None = None, *args, **kwargs):
        # First ensure all REST tables in the query are materialized
        self._ensure_sql_expr_materialized(sql_query, url_params=url_params)
        return super().execute(sql_query, *args, params=params, **kwargs)

    def to_spec(self) -> dict[str, Any]:
        spec = super().to_spec()
        spec.pop("_cached_rest_tables", None)
        return spec

    def create_sql_expr_source(self, tables: dict, materialize: bool = True, params: dict | None = None, **kwargs) -> RESTDuckDBSource:
        for sql_expr in tables.values():
            self._ensure_sql_expr_materialized(sql_expr)
        source = super().create_sql_expr_source(tables, materialize, params, **kwargs)
        return source
