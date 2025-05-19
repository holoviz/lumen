from __future__ import annotations

import os
import re

from typing import TYPE_CHECKING, Any

import duckdb
# https://duckdb.org/docs/stable/clients/python/known_issues#numpy-import-multithreading
import numpy.core.multiarray  # noqa: F401
import pandas as pd
import param

from ..config import config
from ..serializers import Serializer
from ..transforms import Filter
from ..transforms.sql import (
    SQLCount, SQLFilter, SQLLimit, SQLSelectFrom,
)
from .base import BaseSQLSource, Source, cached

if TYPE_CHECKING:
    import pandas as pd


class DuckDBSource(BaseSQLSource):
    """
    DuckDBSource provides a simple wrapper around the DuckDB SQL
    connector.

    To specify tables to be queried provide a list or dictionary of
    tables. A SQL expression to fetch the data from the table will
    then be generated using the `sql_expr`, e.g. if we specify a local
    table flights.db the default sql_expr `SELECT * FROM {table}` will
    expand that to `SELECT * FROM flights.db`. If you want to specify
    a full SQL expression as a table you must change the `sql_expr` to
    '{table}' ensuring no further templating is applied.

    Note that certain functionality in DuckDB requires modules to be
    loaded before making a query. These can be specified using the
    `initializers` parameter, providing the ability to define DuckDb
    statements to be run when initializing the connection.
    """

    filter_in_sql = param.Boolean(default=True, doc="""
        Whether to apply filters in SQL or in-memory.""")

    initializers = param.List(default=[], doc="""
        SQL statements to run to initialize the connection.""")

    mirrors = param.Dict(default={}, doc="""
        Mirrors the tables into the DuckDB database. The mirrors
        should define a mapping from the table names to the source of
        the mirror which may be defined as a Pipeline or a tuple of
        the source and the table.""")

    load_schema = param.Boolean(default=True, doc="Whether to load the schema")

    sql_expr = param.String(default='SELECT * FROM {table}', doc="""
        The SQL expression to execute.""")

    ephemeral = param.Boolean(default=False, doc="""
        Whether the data is ephemeral, i.e. manually inserted into the
        DuckDB table or derived from real data.""")

    tables = param.ClassSelector(class_=(list, dict), doc="""
        List or dictionary of tables.""")

    uri = param.String(doc="The URI of the DuckDB database")

    source_type = 'duckdb'

    dialect = 'duckdb'

    def __init__(self, **params):
        connection = params.pop('_connection', None)
        super().__init__(**params)
        if connection:
            self._connection = connection
        else:
            self._connection = duckdb.connect(self.uri)
            for init in self.initializers:
                self._connection.execute(init)
        for table, mirror in self.mirrors.items():
            if isinstance(mirror, pd.DataFrame):
                df = mirror
            elif isinstance(mirror, tuple):
                source, src_table = mirror
                df = source.get(src_table)
            else:
                df = mirror.data
                def update(e, table=table):
                    self._connection.from_df(e.new).to_view(table)
                    self._set_cache(e.new, table)
                mirror.param.watch(update, 'data')
            try:
                self._connection.from_df(df).to_view(table)
            except (duckdb.CatalogException, duckdb.ParserException):
                continue

    @property
    def connection(self):
        return self._connection

    @classmethod
    def _recursive_resolve(
        cls, spec: dict[str, Any], source_type: type[Source]
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        mirrors = spec.pop('mirrors', {})
        spec, refs = super()._recursive_resolve(spec, source_type)
        resolved_mirrors = {}
        for table, mirror in mirrors.items():
            if isinstance(mirror, tuple):
                src_spec, src_table = mirror
                source = cls.from_spec(src_spec)
                resolved_mirrors[table] = (source, src_table)
            elif mirror.get('type') == 'pipeline':
                from ..pipeline import Pipeline
                resolved_mirrors[table] = Pipeline.from_spec(mirror)
            else:
                resolved_mirrors[table] = Serializer.deserialize(mirror)
        spec['mirrors'] = resolved_mirrors
        return spec, refs

    def _serialize_tables(self):
        tables = {}
        for t in self.get_tables():
            tdf = self.get(t)
            serializer = Serializer._get_type(config.serializer)()
            tables[t] = serializer.serialize(tdf)
        return tables

    def _process_sql_paths(self, sql_expr: dict | str) -> str:
        if isinstance(sql_expr, dict):
            return {self._process_sql_paths(k): v for k, v in sql_expr.items()}

        # Look for read_* patterns (case insensitive) like read_parquet, READ_CSV etc.
        matches = re.finditer(r"(?i)read_\w+\('([^']+)'\)", sql_expr)
        processed_sql = sql_expr

        for match in matches:
            file_path = match.group(1)
            # Ensure it isn't a URL and convert to absolute path if it's not already
            if re.match(r'^(?:http|ftp)s?://', file_path):
                continue
            if not os.path.isabs(file_path):
                abs_path = os.path.abspath(file_path)
                # Replace the old path with the absolute path in the SQL
                processed_sql = processed_sql.replace(
                    f"'{file_path}'", f"'{abs_path}'"
                )
        return processed_sql

    def to_spec(self, context: dict[str, Any] | None = None) -> dict[str, Any]:
        spec = super().to_spec(context)
        if self.ephemeral:
            spec['tables'] = self._serialize_tables()
        # Handle tables that are either a list or dictionary
        elif isinstance(self.tables, list):
            spec['tables'] = [self._process_sql_paths(table_name) for table_name in self.tables]
        else:
            # For dictionary case, process each SQL expression
            processed_tables = {}
            for table_name, sql_expr in self.tables.items():
                processed_tables[table_name] = self._process_sql_paths(sql_expr)
            spec['tables'] = processed_tables

        if 'mirrors' not in spec:
            return spec

        from ..pipeline import Pipeline
        mirrors = {}
        for table, mirror in spec['mirrors'].items():
            if isinstance(mirror, pd.DataFrame):
                serializer = Serializer._get_type(config.serializer)()
                mirrors[table] = serializer.serialize(mirror)
            elif isinstance(mirror, tuple):
                source, src_table = mirror
                src_spec = source.to_spec(context=context)
                mirrors[table] = (src_spec, src_table)
            elif isinstance(mirror, Pipeline):
                mirrors[table] = mirror.to_spec(context=context)
        spec['mirrors'] = mirrors
        return spec

    @classmethod
    def from_df(cls, tables: dict[str, pd.DataFrame], **kwargs):
        """
        Creates an ephemeral, in-memory DuckDBSource containing the
        supplied dataframe.

        Arguments
        ---------
        tables: dict[str, pandas.DataFrame]
            A dictionary mapping from table names to DataFrames
        kwargs: any
            Additional keyword arguments for the source

        Returns
        -------
        source: DuckDBSource
        """
        source = DuckDBSource(uri=':memory:', ephemeral=True, **kwargs)
        table_defs = {}
        for name, df in tables.items():
            source._connection.from_df(df).to_view(name)
            table_defs[name] = source.sql_expr.format(table=name)
        source.tables = table_defs
        return source

    @classmethod
    def from_spec(cls, spec: dict[str, Any] | str) -> Source:
        if spec.get('ephemeral') and 'tables' in spec:
            ephemeral_tables = spec['tables']
            spec['tables'] = {}
        else:
            ephemeral_tables = {}

        source = super().from_spec(spec)
        if not ephemeral_tables:
            return source
        new_tables = {}
        for t, data in ephemeral_tables.items():
            df = Serializer.deserialize(data)
            try:
                source._connection.from_df(df).to_view(t)
            except duckdb.ParserException:
                continue
            new_tables[t] = source.sql_expr.format(table=t)
        source.tables = new_tables
        return source

    def create_sql_expr_source(
        self, tables: dict[str, str], materialize: bool = True, **kwargs
    ):
        """
        Creates a new SQL Source given a set of table names and
        corresponding SQL expressions.

        Arguments
        ---------
        tables: dict[str, str]
            Mapping from table name to SQL expression.
        materialize: bool
            Whether to materialize new tables
        kwargs: any
            Additional keyword arguments.

        Returns
        -------
        source: DuckDBSource
        """
        params = dict(self.param.values(), **kwargs)
        params['tables'] = tables
        # Reuse connection unless it has changed
        if 'uri' not in kwargs and 'initializers' not in kwargs:
            params['_connection'] = self._connection
        params.pop('name', None)
        source = type(self)(**params)
        if not materialize:
            return source

        for table, sql_expr in tables.copy().items():
            if sql_expr == self.sql_expr.format(table=f'"{table}"'):
                continue
            table_expr = f'CREATE OR REPLACE TEMP TABLE "{table}" AS ({sql_expr})'
            try:
                self._connection.execute(table_expr)
            except duckdb.CatalogException as e:
                pattern = r"Table with name\s(\S+)"
                match = re.search(pattern, str(e))
                if match and isinstance(self.tables, dict):
                    name = match.group(1)
                    real = self.tables[name] if name in self.tables else self.tables[name.strip('"')]
                    table_expr = SQLSelectFrom(sql_expr=self.sql_expr, tables={name: real}).apply(table_expr)
                    self._connection.execute(table_expr)
                else:
                    raise e
        return source

    def execute(self, sql_query: str, *args, **kwargs):
        return self._connection.execute(sql_query, *args, **kwargs).fetch_df()

    def get_tables(self):
        if isinstance(self.tables, dict | list):
            return [t for t in list(self.tables) if not self._is_table_excluded(t)]

        return [
            t[0] for t in self._connection.execute('SHOW TABLES').fetchall()
            if not self._is_table_excluded(t[0])
        ]

    def normalize_table(self, table: str):
        tables = self.get_tables()
        if table not in tables and 'read_' in table:
            # Extract table name from read_* function
            table = re.search(r"read_(\w+)\('(.+?)'", table).group(2)
        return table

    @cached
    def get(self, table, **query):
        query.pop('__dask', None)
        sql_expr = self.get_sql_expr(table)
        sql_transforms = query.pop('sql_transforms', [])
        conditions = list(query.items())
        if self.filter_in_sql:
            sql_transforms = [SQLFilter(conditions=conditions)] + sql_transforms
        for st in sql_transforms:
            sql_expr = st.apply(sql_expr)
        rel = self._connection.execute(sql_expr)
        has_geom = any(d[0] == 'geometry' and d[1] == 'BINARY' for d in rel.description)
        df = rel.fetch_df(date_as_object=True)
        if has_geom:
            import geopandas as gpd
            geom = self._connection.execute(
                f'SELECT ST_AsWKB(geometry::GEOMETRY) as geometry FROM ({sql_expr})'
            ).fetch_df()
            df['geometry'] = gpd.GeoSeries.from_wkb(geom.geometry.apply(bytes))
            df = gpd.GeoDataFrame(df)
        if not self.filter_in_sql:
            df = Filter.apply_to(df, conditions=conditions)
        return df

    def _get_table_metadata(self, tables: list[str]) -> dict[str, Any]:
        """
        Generate metadata for all tables or a single table (batched=False) in DuckDB.
        Handles formats: database.schema.table_name, schema.table_name, or table_name.

        Args:
            table: Table name(s) to get metadata for

        Returns:
            Dictionary with table metadata including description, columns, row count, etc.
        """
        metadata = {}
        for table_name in tables:
            sql_expr = self.get_sql_expr(table_name)
            schema_expr = SQLLimit(limit=0).apply(sql_expr)
            count_expr = SQLCount().apply(sql_expr)
            schema_result = self.execute(schema_expr)
            count = self.execute(count_expr).iloc[0, 0]
            table_metadata = {
                "description": "",
                "columns": {
                    col: {"data_type": str(dtype), "description": ""}
                    for col, dtype in zip(schema_result.columns, schema_result.dtypes)
                },
                "rows": count,
                "updated_at": None,
                "created_at": None,
            }
            metadata[table_name] = table_metadata
        return metadata

    def close(self):
        """
        Close the DuckDB connection, releasing associated resources.

        This method should be called when the source is no longer needed to prevent
        connection leaks and properly clean up server-side resources.
        """
        if self._connection is not None:
            self._connection.close()
            self._connection = None
