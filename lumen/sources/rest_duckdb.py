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

    _created_views = param.Dict(default={}, doc="Internal dict of created views for REST tables.")

    def _is_rest_table(self, table: str) -> bool:
        if table not in self.tables:
            raise ValueError(f"Table '{table}' not found in source tables.")
        return isinstance(self.tables[table], dict) and 'url' in self.tables[table]

    def get_sql_expr(self, table: str) -> str:
        if self._is_rest_table(table):
            read_fn = self._format_to_reader[self.data_format]
            return f"SELECT * FROM {read_fn}(?)"
        return super().get_sql_expr(table)

    def get(self, table: str, url_params: dict[str, Any] | None = None, **query):
        if not self._is_rest_table(table):
            return super().get(table, **query)

        table_params = self.tables[table]
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

        Can be called with either table OR config (for internal use).

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
        table_objs = sqlglot.parse_one(sql_query).find_all(sqlglot.exp.Table)
        rest_tables = {table_obj.name for table_obj in table_objs if self._is_rest_table(table_obj.name)}
        for table in rest_tables:
            # if the table params have changed, recreate the view
            if self._created_views.get(table) != self.tables[table]:
                self._connection.from_df(self.get(table, url_params=url_params)).to_view(table)
                self._created_views[table] = {**self.tables[table], **(url_params or {})}
        return super().execute(sql_query, *args, params=params, **kwargs)

    def to_spec(self) -> dict[str, Any]:
        spec = super().to_spec()
        spec.pop("_created_views", None)
        return spec

    def create_sql_expr_source(self, tables, materialize = True, params = None, url_params: dict[str, Any] | None = None, **kwargs) -> RESTDuckDBSource:
        source = super().create_sql_expr_source(tables, materialize, params, **kwargs)
        # TODO: investigate whether we should ALWAYS copy every tables, not just REST tables
        # keep references of the original rest tables so views can be recreated
        source.tables.update(**{table: self.tables[table] for table in self.tables if self._is_rest_table(table)})
        return source
