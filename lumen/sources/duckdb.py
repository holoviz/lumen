from __future__ import annotations

import re

from typing import TYPE_CHECKING, Any

import duckdb
import param

from ..config import config
from ..serializers import Serializer
from ..transforms import Filter
from ..transforms.sql import (
    SQLCount, SQLDistinct, SQLFilter, SQLLimit, SQLMinMax,
)
from ..util import get_dataframe_schema
from .base import (
    BaseSQLSource, Source, cached, cached_schema,
)

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
            if isinstance(mirror, tuple):
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
        for table, (src_spec, src_table) in mirrors.items():
            source = cls.from_spec(src_spec)
            resolved_mirrors[table] = (source, src_table)
        spec['mirrors'] = resolved_mirrors
        return spec, refs

    def _serialize_tables(self):
        tables = {}
        for t in self.get_tables():
            tdf = self.get(t)
            serializer = Serializer._get_type(config.serializer)()
            tables[t] = serializer.serialize(tdf)
        return tables

    def to_spec(self, context: dict[str, Any] | None = None) -> dict[str, Any]:
        spec = super().to_spec(context)
        if self.ephemeral:
            spec['tables'] = self._serialize_tables()
        if 'mirrors' not in spec:
            return spec
        mirrors = {}
        for table, (source, src_table) in spec['mirrors'].items():
            src_spec = source.to_spec(context=context)
            mirrors[table] = (src_spec, src_table)
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
        source = type(self)(**params)
        if not materialize:
            return source
        existing = self.get_tables()
        for table, sql_expr in tables.copy().items():
            if table in existing or sql_expr == self.sql_expr.format(table=f'"{table}"'):
                continue
            table_expr = f'CREATE TEMP TABLE "{table}" AS ({sql_expr})'
            try:
                self._connection.execute(table_expr)
            except duckdb.CatalogException as e:
                pattern = r"Table with name\s(\S+)"
                match = re.search(pattern, str(e))
                if match and isinstance(self.tables, dict):
                    name = match.group(1)
                    real = self.tables[name]
                    if 'select' not in real.lower() and not real.startswith('"'):
                        real = f'"{real}"'
                    else:
                        real = f'({real})'
                    table_expr = table_expr.replace(f'FROM {name}', f'FROM {real}')
                    source.tables[table] = sql_expr.replace(f'FROM {name}', f'FROM {real}')
                    self._connection.execute(table_expr)
                else:
                    raise e
        return source

    def get_tables(self):
        if isinstance(self.tables, (dict, list)):
            return list(self.tables)
        return [t[0] for t in self._connection.execute('SHOW TABLES').fetchall()]

    def get_sql_expr(self, table: str):
        if isinstance(self.tables, dict):
            table = self.tables[table]
        if '(' not in table and ')' not in table:
            table = f'"{table}"'
        if 'select ' in table.lower():
            sql_expr = table
        else:
            sql_expr = self.sql_expr.format(table=table)
        return sql_expr.rstrip(";")

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
        df = self._connection.execute(sql_expr).fetch_df(date_as_object=True)
        if not self.filter_in_sql:
            df = Filter.apply_to(df, conditions=conditions)
        return df

    @cached_schema
    def get_schema(
        self, table: str | None = None, limit: int | None = None
    ) -> dict[str, dict[str, Any]] | dict[str, Any]:
        if table is None:
            tables = self.get_tables()
        else:
            tables = [table]

        schemas = {}
        sql_limit = SQLLimit(limit=limit or 1)
        for entry in tables:
            if not self.load_schema:
                schemas[entry] = {}
                continue
            sql_expr = self.get_sql_expr(entry)
            data = self._connection.execute(sql_limit.apply(sql_expr)).fetch_df()
            schemas[entry] = schema = get_dataframe_schema(data)['items']['properties']
            if limit:
                continue

            enums, min_maxes = [], []
            for name, col_schema in schema.items():
                if 'enum' in col_schema:
                    enums.append(name)
                elif 'inclusiveMinimum' in col_schema:
                    min_maxes.append(name)
            for col in enums:
                distinct_expr = SQLDistinct(columns=[col]).apply(sql_expr)
                distinct_expr = ' '.join(distinct_expr.splitlines())
                distinct = self._connection.execute(distinct_expr).fetch_df()
                schema[col]['enum'] = distinct[col].tolist()

            if not min_maxes:
                continue

            minmax_expr = SQLMinMax(columns=min_maxes).apply(sql_expr)
            minmax_expr = ' '.join(minmax_expr.splitlines())
            minmax_data = self._connection.execute(minmax_expr).fetch_df()
            for col in min_maxes:
                kind = data[col].dtype.kind
                if kind in 'iu':
                    cast = int
                elif kind == 'f':
                    cast = float
                elif kind == 'M':
                    cast = str
                else:
                    cast = lambda v: v
                schema[col]['inclusiveMinimum'] = cast(minmax_data[f'{col}_min'].iloc[0])
                schema[col]['inclusiveMaximum'] = cast(minmax_data[f'{col}_max'].iloc[0])

            count_expr = SQLCount().apply(sql_expr)
            count_expr = ' '.join(count_expr.splitlines())
            count_data = self._connection.execute(count_expr).fetch_df()
            schema['count'] = cast(count_data['count'].iloc[0])

        return schemas if table is None else schemas[table]
