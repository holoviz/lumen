from __future__ import annotations

from typing import Any

import pandas as pd
import param
import snowflake.connector

from ..transforms.sql import (
    SQLDistinct, SQLFilter, SQLLimit, SQLMinMax,
)
from ..util import get_dataframe_schema
from .base import BaseSQLSource, cached, cached_schema


class SnowflakeSource(BaseSQLSource):
    """
    SnowflakeSource uses the snowflake-python-connector library to load data
    from Snowflake.
    """

    account = param.String(default=None, doc="""
        The account identifier to connect to.""")

    authenticator = param.Selector(default=None, objects=['externalbrowser', 'oauth'], doc="""
        The authentication approach to use.""")

    database = param.String(default=None, doc="""
        The database to connect to.""")

    host = param.String(default=None, doc="""
        The host to authenticate with.""")

    token = param.String(default=None, doc="""
        The OAuth token if authenticator is set to "oauth".""")

    user = param.String(default=None, doc="""
        The user to authenticate as.""")

    schema = param.String(default=None, doc="""
        The database schema to load data from.""")

    warehouse = param.String(default=None, doc="""
        The warehouse to connect to.""")

    filter_in_sql = param.Boolean(default=True, doc="""
        Whether to apply filters in SQL or in-memory.""")

    sql_expr = param.String(default='SELECT * FROM {table}', doc="""
        The SQL expression to execute.""")

    tables = param.ClassSelector(class_=(list, dict), doc="""
        List or dictionary of tables.""")

    dialect = 'snowflake'

    def __init__(self, **params):
        conn = params.pop('conn', None)
        super().__init__(**params)
        conn_kwargs = {}
        if self.account is not None:
            conn_kwargs['account'] = self.account
        if self.authenticator is not None:
            conn_kwargs['authenticator'] = self.authenticator
        if self.database is not None:
            conn_kwargs['database'] = self.database
        if self.host is not None:
            conn_kwargs['host'] = self.host
        if self.token is not None:
            conn_kwargs['token'] = self.token
        if self.schema is not None:
            conn_kwargs['schema'] = self.schema
        if self.user is not None:
            conn_kwargs['user'] = self.user
        if self.warehouse is not None:
            conn_kwargs['warehouse'] = self.warehouse
        self._conn = conn or snowflake.connector.connect(**conn_kwargs)
        self._cursor = self._conn.cursor()

    def create_sql_expr_source(self, tables: dict[str, str], **kwargs):
        """
        Creates a new SQL Source given a set of table names and
        corresponding SQL expressions.
        """
        params = dict(self.param.values(), **kwargs)
        params.pop("name", None)
        params['tables'] = tables
        params['conn'] = self._conn
        return SnowflakeSource(**params)

    def get_tables(self) -> list[str]:
        if isinstance(self.tables, dict | list):
            return list(self.tables)
        tables = self._cursor.execute(f'SELECT TABLE_NAME, TABLE_SCHEMA FROM {self.database}.INFORMATION_SCHEMA.TABLES;').fetch_pandas_all()
        return [f'{self.database}.{row.TABLE_SCHEMA}.{row.TABLE_NAME}' for _, row in tables.iterrows()]

    def get_sql_expr(self, table: str):
        if isinstance(self.tables, dict):
            table = self.tables[table]
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
        return self._cursor.execute(sql_expr).fetch_pandas_all()

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
            sql_expr = self.get_sql_expr(entry)
            data = self._cursor.execute(sql_limit.apply(sql_expr)).fetch_pandas_all()
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
                distinct = self._cursor.execute(distinct_expr).fetch_pandas_all()
                schema[col]['enum'] = distinct[col].tolist()

            if not min_maxes:
                continue

            minmax_expr = SQLMinMax(columns=min_maxes).apply(sql_expr)
            minmax_expr = ' '.join(minmax_expr.splitlines())
            minmax_data = self._cursor.execute(minmax_expr).fetch_pandas_all()
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
                min_data = minmax_data[f'{col}_min'].iloc[0]
                schema[col]['inclusiveMinimum'] = min_data if pd.isna(min_data) else cast(min_data)
                max_data = minmax_data[f'{col}_max'].iloc[0]
                schema[col]['inclusiveMaximum'] = max_data if pd.isna(max_data) else cast(max_data)

        return schemas if table is None else schemas[table]
