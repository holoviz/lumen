from __future__ import annotations

import param
import snowflake.connector

from ..transforms.sql import SQLFilter
from .base import BaseSQLSource, cached


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

    def execute(self, sql_query: str):
        return self._cursor.execute(sql_query).fetch_pandas_all()

    def get_tables(self) -> list[str]:
        if isinstance(self.tables, dict | list):
            return list(self.tables)
        tables = self.execute(f'SELECT TABLE_NAME, TABLE_SCHEMA FROM {self.database}.INFORMATION_SCHEMA.TABLES;')
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
        return self.execute(sql_expr)
