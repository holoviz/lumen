"""
XArraySource: SQL-based querying of Xarray datasets using xarray-sql.

This source enables SQL queries on NetCDF, Zarr, and other Xarray-compatible
data formats through Apache DataFusion backend.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import param
import xarray as xr

try:
    import xarray_sql as xql
except ImportError:
    xql = None

from .base import BaseSQLSource, cached


class XArraySource(BaseSQLSource):
    chunks = param.Dict(
        default={"time": 24},
        doc="""
        Chunking strategy for Xarray datasets. xarray-sql requires
        datasets to be chunked. Keys are dimension names, values are
        chunk sizes.""",
    )

    dataset_kwargs = param.Dict(
        default={},
        doc="""
        Additional keyword arguments passed to xr.open_dataset().""",
    )

    sql_expr = param.String(
        default="SELECT * FROM {table}",
        doc="""
        The SQL expression template to execute.""",
    )

    tables = param.ClassSelector(
        class_=(list, dict),
        doc="""
        Dictionary mapping table names to file paths of Xarray datasets
        (NetCDF, Zarr, etc.). Example:

        ```
        {
            'air': '/path/to/air_temperature.nc',
            'ocean': '/path/to/ocean_data.zarr'
        }
        ```
        """,
    )

    materialize = param.Boolean(
        default=True,
        doc="""
        Whether to materialize SQL expressions as views in the DataFusion context.
        When True, SQL expressions in tables dict will be executed and registered as
        queryable tables.""",
    )

    source_type = "xarray_sql"

    # sqlglot doesn't have a 'datafusion' dialect, so we use 'postgres'
    # which is the closest compatible SQL dialect
    dialect = "postgres"

    def __init__(self, **params):
        if xql is None:
            raise ImportError("xarray-sql is required for XArraySource. Install it with: pip install xarray-sql")

        # Check if we're reusing a context from a parent source
        self._context = params.pop("_context", None)

        super().__init__(**params)

        # Create new context only if not provided
        if self._context is None:
            self._context = xql.XarrayContext()

        # Track which tables are physical datasets vs SQL expressions
        self._physical_tables = set()
        self._sql_expressions = {}

        # Load and register datasets (only for file paths, not SQL expressions)
        if isinstance(self.tables, dict):
            for name, value in self.tables.items():
                if self._is_sql_expression(value):
                    # Store SQL expression for later materialization
                    self._sql_expressions[name] = value
                else:
                    # Register as a physical dataset
                    self._register_dataset(name, value)
                    self._physical_tables.add(name)
        elif isinstance(self.tables, list):
            for path in self.tables:
                # Derive name from filename
                name = Path(path).stem
                self._register_dataset(name, path)
                self._physical_tables.add(name)

        # Materialize SQL expressions as tables in the context
        if self._sql_expressions and self.materialize:
            self._materialize_sql_expressions()

    def _materialize_sql_expressions(self):
        """
        Materialize SQL expressions as tables in the DataFusion context.

        This executes each SQL expression and registers the result as a table,
        making it available for subsequent queries.
        """
        for table_name, sql_expr in self._sql_expressions.items():
            try:
                # Execute the SQL expression
                result = self._context.sql(sql_expr)
                # Register the result as a view in the context
                self._context.register_view(table_name, result)
            except Exception as e:
                self.param.warning(f"Could not materialize SQL expression for table '{table_name}': {e}")

    def _is_sql_expression(self, value: str) -> bool:
        """
        Check if a table value is a SQL expression or a file path.

        Parameters
        ----------
        value : str
            The table value to check

        Returns
        -------
        bool
            True if value appears to be a SQL expression, False if file path
        """
        # Simple heuristic: check for SQL keywords
        value_upper = value.strip().upper()
        return any(value_upper.startswith(keyword) for keyword in ["SELECT", "WITH"])

    def _register_dataset(self, name: str, path: str):
        """
        Load and register an Xarray dataset with the SQL context.

        Parameters
        ----------
        name : str
            Table name to use in SQL queries
        path : str
            File path to the Xarray dataset
        """
        # Resolve path relative to root if not absolute
        if not Path(path).is_absolute():
            path = str(self.root / path)

        # Load the dataset
        ds = xr.open_dataset(path, **self.dataset_kwargs)

        # Register with xarray-sql context
        # Note: xarray-sql requires chunked datasets
        self._context.from_dataset(name, ds, chunks=self.chunks)

    def get_sql_expr(self, table: str | dict) -> str:
        """
        Returns a simple SQL expression for the table.

        For SQL expression tables (materialized), returns SELECT * FROM table.
        For physical dataset tables, returns SELECT * FROM "table".

        Parameters
        ----------
        table : str or dict
            The table name

        Returns
        -------
        str
            SQL expression to query the table
        """
        # If this is a SQL expression table that was materialized,
        # just return a simple SELECT from it
        if table in self._sql_expressions:
            return f'SELECT * FROM "{table}"'

        # For physical tables, return standard query
        return f'SELECT * FROM "{table}"'

    def get_tables(self) -> list[str]:
        """Return list of registered table names."""
        if isinstance(self.tables, dict):
            tables = list(self.tables.keys())
        elif isinstance(self.tables, list):
            tables = [Path(path).stem for path in self.tables]
        else:
            tables = []

        return [t for t in tables if not self._is_table_excluded(t)]

    def execute(self, sql_query: str, *args, **kwargs) -> pd.DataFrame:
        """
        Execute a SQL query against the xarray-sql context.

        Parameters
        ----------
        sql_query : str
            The SQL query to execute
        *args : list
            Positional arguments (unused, for compatibility)
        **kwargs : dict
            Keyword arguments (unused, for compatibility)

        Returns
        -------
        pd.DataFrame
            Query results as a pandas DataFrame
        """
        # Only replace file paths with aliases, skip SQL expressions
        if isinstance(self.tables, dict):
            for alias, file_table in self.tables.items():
                # Skip SQL expressions (they're already materialized as tables)
                if self._is_sql_expression(file_table):
                    continue
                # Replace file path with alias in the query
                sql_query = sql_query.replace(f"{file_table}", f"{alias}")

        # Execute query using xarray-sql
        result = self._context.sql(sql_query)

        # Convert to pandas DataFrame
        return result.to_pandas()

    @cached
    def get(self, table: str, **query) -> pd.DataFrame:
        """
        Retrieve data from a table with optional SQL transforms.

        Parameters
        ----------
        table : str
            Name of the table to query
        **query : dict
            Query parameters including sql_transforms

        Returns
        -------
        pd.DataFrame
            The queried data
        """
        query.pop("__dask", None)

        # Get base SQL expression for the table
        sql_expr = self.get_sql_expr(table)

        # Apply any SQL transforms from the query
        sql_transforms = query.pop("sql_transforms", [])
        for transform in sql_transforms:
            sql_expr = transform.apply(sql_expr)

        # Execute the query
        return self.execute(sql_expr)

    def create_sql_expr_source(self, tables: dict[str, str], materialize: bool = True, **kwargs) -> XArraySource:
        """
        Create a new XArraySource with SQL expressions.

        This creates a derived source that reuses the parent's xarray-sql context,
        allowing you to define virtual tables based on SQL queries over the
        original datasets.

        Parameters
        ----------
        tables : dict[str, str]
            Mapping from table names to SQL expressions.
            Example: {"air_sub": "SELECT * FROM air WHERE time > '2016-01-05'"}
        materialize : bool, default True
            Whether to materialize SQL expressions as tables in the context
        **kwargs : dict
            Additional parameters for the new source

        Returns
        -------
        XArraySource
            New source instance with the specified SQL expressions

        Examples
        --------
        >>> source = XArraySource(tables={'air': 'temperature.nc'}, chunks={'time': 24})
        >>> derived = source.create_sql_expr_source({
        ...     'air_sub': "SELECT * FROM air WHERE time > '2016-01-05'",
        ...     'air_avg': "SELECT lat, lon, AVG(air) as avg_temp FROM air GROUP BY lat, lon"
        ... })
        >>> df = derived.get('air_sub')
        """
        params = dict(self.param.values(), **kwargs)
        params["tables"] = tables
        params.pop("name", None)
        params["materialize"] = materialize

        # CRITICAL: Pass the parent's context to avoid re-creating datasets
        params["_context"] = self._context

        return type(self)(**params)

    def _get_table_metadata(self, tables: list[str]) -> dict[str, Any]:
        """
        Generate metadata for the specified tables.

        Works for both physical dataset tables and SQL expression tables.

        Parameters
        ----------
        tables : list[str]
            List of table names to get metadata for

        Returns
        -------
        dict
            Metadata dictionary with table information
        """
        metadata = {}

        for table_name in tables:
            try:
                # Use get_sql_expr to properly handle both physical and SQL expression tables
                base_expr = self.get_sql_expr(table_name)

                # Build queries for schema and count
                # Wrap the base expression in a subquery for SQL expression tables
                if table_name in self._sql_expressions:
                    schema_query = f"SELECT * FROM ({base_expr}) AS subquery LIMIT 1"
                    count_query = f"SELECT COUNT(*) as count FROM ({base_expr}) AS subquery"
                else:
                    # For physical tables, use direct queries
                    schema_query = f"{base_expr} LIMIT 1"
                    count_query = f'SELECT COUNT(*) as count FROM "{table_name}"'

                # Get schema by querying with limit 1
                schema_result = self.execute(schema_query)

                # Get row count
                count_result = self.execute(count_query)
                count = int(count_result.iloc[0, 0])

                # Build metadata
                description = f"SQL expression table: {table_name}" if table_name in self._sql_expressions else f"Xarray dataset table: {table_name}"
                table_metadata = {
                    "description": description,
                    "columns": {col: {"data_type": str(dtype), "description": ""} for col, dtype in zip(schema_result.columns, schema_result.dtypes, strict=False)},
                    "rows": int(count),
                    "updated_at": None,
                    "created_at": None,
                }

                metadata[table_name] = table_metadata

            except Exception as e:
                # If metadata extraction fails, provide minimal metadata
                self.param.warning(f"Could not extract full metadata for table '{table_name}': {e}")
                metadata[table_name] = {
                    "description": f"Table: {table_name}",
                    "columns": {},
                    "rows": 0,
                    "updated_at": None,
                    "created_at": None,
                }

        return metadata
