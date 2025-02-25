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

    authenticator = param.Selector(default=None, objects=[
        'externalbrowser', 'oauth', 'snowflake', 'username_password_mfa'], doc="""
        The authentication approach to use.""")

    database = param.String(default=None, doc="""
        The database to connect to.""")

    host = param.String(default=None, doc="""
        The host to authenticate with.""")

    token = param.String(default=None, doc="""
        The OAuth token if authenticator is set to "oauth".""")

    user = param.String(default=None, doc="""
        The user to authenticate as.""")

    password = param.String(default=None, doc="""
        The password to authenticate with (if authenticator is set to "snowflake").""")

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
        if self.password is not None:
            conn_kwargs['password'] = self.password
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

    @cached
    def get_table_metadata(self, table_name: str) -> dict[str, dict]:
        """
        Generate metadata for a single table in Snowflake.

        Args:
            table_name: Name of the table to get metadata for

        Returns:
            A dictionary with table metadata in the format:
            {"description": ..., "columns": {"column_name": "description", ...}}
        """
        # Get table description
        table_query = f"""
            SELECT
                TABLE_NAME,
                COMMENT as TABLE_DESCRIPTION
            FROM
                {self.database}.INFORMATION_SCHEMA.TABLES
            WHERE
                TABLE_NAME = '{table_name}'
        """

        table_metadata = self.execute(table_query)

        if table_metadata.empty:
            return {"description": "", "columns": {}}

        description = table_metadata.iloc[0]['TABLE_DESCRIPTION']
        result = {
            "description": description if description else "",
            "columns": {}
        }

        # Get column metadata
        column_query = f"""
            SELECT
                COLUMN_NAME,
                COMMENT
            FROM
                {self.database}.INFORMATION_SCHEMA.COLUMNS
            WHERE
                TABLE_NAME = '{table_name}'
            ORDER BY
                ORDINAL_POSITION
        """

        columns_info = self.execute(column_query)

        for _, row in columns_info.iterrows():
            column_name = row['COLUMN_NAME']
            description = row['COMMENT']
            result["columns"][column_name] = description if description else ""

        return result

    def get_metadata(self, tables: dict[str, str] | None = None) -> dict[str, dict]:
        """
        Generate metadata dictionary for multiple tables.
        Non-cached version that processes a list of tables.

        Args:
            tables: Optional dictionary of table names to override the class tables.
                If None, uses self.tables.

        Returns:
            A dictionary with table metadata in the format:
            {"table_name": {"description": ..., "columns": {"column_name": "description", ...}}}
        """
        tables_to_use = tables if tables is not None else self.tables
        result = {}

        # If no tables provided, get all tables from the schema
        if tables_to_use is None:
            table_list = self.get_tables()
            # Convert from fully qualified names to simple table names
            tables_to_use = {t.split('.')[-1]: t for t in table_list}

        # Process one table at a time
        for table_name in tables_to_use:
            result[table_name] = self.get_table_metadata(table_name)

        return result
