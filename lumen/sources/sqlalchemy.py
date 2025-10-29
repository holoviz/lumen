from __future__ import annotations

import asyncio

from typing import TYPE_CHECKING, Any

import pandas as pd
import param

from sqlalchemy import create_engine, inspect, text
from sqlalchemy.engine.url import URL, make_url

from ..transforms.sql import SQLFilter
from .base import BaseSQLSource, cached, cached_schema

if TYPE_CHECKING:
    from sqlalchemy.engine import Connection

    DataFrame = pd.DataFrame


class SQLAlchemySource(BaseSQLSource):
    """
    SQLAlchemySource uses SQLAlchemy to connect to various SQL databases.

    Supports both synchronous and asynchronous database drivers through SQLAlchemy's
    engine system. Can connect to PostgreSQL, MySQL, SQLite, Oracle, MSSQL, and more.
    """

    # Connection string or URL
    url = param.String(default=None, doc="""
        SQLAlchemy database URL string (e.g., 'postgresql://user:pass@host:port/db').""")

    # Connection components (alternative to URL)
    drivername = param.String(default=None, doc="""
        The driver name (e.g., 'postgresql+psycopg2', 'mysql+pymysql', 'sqlite').""")

    username = param.String(default=None, doc="""
        The username for database authentication.""")

    password = param.String(default=None, doc="""
        The password for database authentication.""")

    host = param.String(default=None, doc="""
        The database host address.""")

    port = param.Integer(default=None, doc="""
        The database port number.""")

    database = param.String(default=None, doc="""
        The database name to connect to.""")

    query_params = param.Dict(default=None, doc="""
        Additional query parameters for the connection URL.""")

    # Connection configuration
    connect_args = param.Dict(default={}, doc="""
        Additional keyword arguments passed to the DBAPI's connect() method.""")

    engine_kwargs = param.Dict(default={}, doc="""
        Additional keyword arguments passed to create_engine().""")

    # Schema and table configuration
    schema = param.String(default=None, doc="""
        The default schema to use for queries.""")

    tables = param.ClassSelector(class_=(list, dict), doc="""
        List or dictionary of tables to expose.""")

    excluded_tables = param.List(default=[], doc="""
        List of table name patterns to exclude from the results.
        Supports wildcards (e.g., 'schema.table*', 'temp_*').""")

    # Query configuration
    filter_in_sql = param.Boolean(default=True, doc="""
        Whether to apply filters in SQL or in-memory.""")

    sql_expr = param.String(default='SELECT * FROM {table}', doc="""
        The SQL expression template to execute.""")

    schema_timeout_seconds = param.Integer(default=600, doc="""
        Timeout in seconds for schema retrieval operations.""")

    source_type = 'sqlalchemy'

    async_drivers = {
        'postgresql+asyncpg',
        'mysql+asyncmy',
        'mysql+aiomysql',
        'sqlite+aiosqlite',
        'oracle+oracledb_async',
    }

    def __init__(self, **params):
        engine = params.pop('engine', None)
        super().__init__(**params)

        # Build the connection URL
        if self.url:
            self._url = make_url(self.url)
        elif self.drivername:
            url_params = {}
            if self.drivername:
                url_params['drivername'] = self.drivername
            if self.username:
                url_params['username'] = self.username
            if self.password:
                url_params['password'] = self.password
            if self.host:
                url_params['host'] = self.host
            if self.port:
                url_params['port'] = self.port
            if self.database:
                url_params['database'] = self.database
            if self.query_params:
                url_params['query'] = self.query_params
            self._url = URL.create(**url_params)
        else:
            raise ValueError(
                "Either 'url' or 'drivername' must be provided to create a connection."
            )

        # Detect if driver is async
        self._driver_is_async = self._is_async_driver(self._url.drivername)

        # Create the engine
        if engine:
            self._engine = engine
        else:
            engine_kwargs = dict(
                connect_args=self.connect_args or {},
                **self.engine_kwargs
            )

            if self._driver_is_async:
                from sqlalchemy.ext.asyncio import create_async_engine
                self._engine = create_async_engine(self._url, **engine_kwargs)
            else:
                self._engine = create_engine(self._url, **engine_kwargs)

        self._inspector_cache = None

    @classmethod
    def _is_async_driver(cls, drivername: str) -> bool:
        """Check if the driver is asynchronous."""
        return drivername in cls.async_drivers or 'async' in drivername

    @staticmethod
    def _convert_params(sql_query: str, params: list | dict | None = None) -> tuple[str, dict]:
        """
        Convert parameters to SQLAlchemy-compatible format.

        Converts positional parameters (list) to named parameters (dict) and
        updates the SQL query accordingly.

        Arguments
        ---------
        sql_query : str
            The SQL query string
        params : list | dict | None
            Parameters to convert:
            - list: Positional parameters for placeholder (?) syntax
            - dict: Named parameters for :name, %(name)s, etc. syntax
            - None: No parameters

        Returns
        -------
        tuple[str, dict]
            The updated SQL query and converted parameters dict
        """
        if params is None or isinstance(params, dict):
            return sql_query, params or {}

        # Convert list params to dict
        params_dict = {f'param_{i}': v for i, v in enumerate(params)}
        # Replace ? with :param_0, :param_1, etc.
        for i in range(len(params)):
            sql_query = sql_query.replace('?', f':param_{i}', 1)
        return sql_query, params_dict

    @property
    def _connection(self) -> Connection:
        """Get or create a database connection."""
        if self._driver_is_async:
            raise RuntimeError(
                f"Driver {self._url.drivername} is asynchronous. "
                "Use execute_async() instead of execute()."
            )
        # Don't cache connection - create new one each time to avoid transaction issues
        return self._engine.connect()

    @property
    def _inspector(self):
        """Get or create a SQLAlchemy inspector for metadata operations."""
        if self._engine is None:
            return None
        if self._inspector_cache is None:
            self._inspector_cache = inspect(self._engine)
        return self._inspector_cache

    @property
    def dialect(self) -> str:
        """Detect and return the database dialect name."""
        try:
            return self._url.get_dialect().name
        except Exception:
            # Fallback to 'any' if dialect detection fails
            return 'any'

    def create_sql_expr_source(self, tables: dict[str, str], params: dict[str, list | dict] | None = None, **kwargs):
        """
        Creates a new SQL Source given a set of table names and
        corresponding SQL expressions.

        Arguments
        ---------
        tables: dict[str, str]
            Mapping from table name to SQL expression.
        params: dict[str, list | dict] | None
            Optional mapping from table name to parameters:
            - list: Positional parameters for placeholder (?) syntax
            - dict: Named parameters for :name, %(name)s, etc. syntax
        kwargs: any
            Additional keyword arguments.

        Returns
        -------
        source: SQLAlchemySource
        """
        params_dict = dict(self.param.values(), **kwargs)
        params_dict.pop("name", None)
        params_dict['tables'] = tables
        if params is not None:
            params_dict['table_params'] = params
        params_dict['engine'] = self._engine
        return SQLAlchemySource(**params_dict)

    def execute(self, sql_query: str, params: list | dict | None = None, *args, **kwargs) -> pd.DataFrame:
        """
        Executes a SQL query and returns the result as a DataFrame.

        Arguments
        ---------
        sql_query : str
            The SQL Query to execute
        params : list | dict | None
            Parameters to use in the SQL query:
            - list: Positional parameters for placeholder (?) syntax
            - dict: Named parameters for :name, %(name)s, etc. syntax
            - None: No parameters
        *args : list
            Additional positional arguments to pass to the SQL query
        **kwargs : dict
            Keyword arguments to pass to the SQL query

        Returns
        -------
        pd.DataFrame
            The result as a pandas DataFrame
        """
        if self._driver_is_async:
            raise RuntimeError(
                f"Driver {self._url.drivername} is asynchronous. "
                "Use execute_async() instead."
            )

        sql_query, params = self._convert_params(sql_query, params)

        # Use context manager to ensure connection is closed
        with self._connection as connection:
            result = connection.execute(text(sql_query), params, *args, **kwargs)
            df = pd.DataFrame(result.fetchall(), columns=result.keys())
            return df

    async def execute_async(self, sql_query: str, params: list | dict | None = None, *args, **kwargs) -> pd.DataFrame:
        """
        Executes a SQL query asynchronously and returns the result as a DataFrame.

        This default implementation runs the synchronous execute() method in a thread
        to avoid blocking the event loop. Subclasses can override this method
        to provide truly asynchronous implementations.

        Arguments
        ---------
        sql_query : str
            The SQL Query to execute
        params : list | dict | None
            Parameters to use in the SQL query:
            - list: Positional parameters for placeholder (?) syntax
            - dict: Named parameters for :name, %(name)s, etc. syntax
            - None: No parameters
        *args : list
            Additional positional arguments to pass to the SQL query
        **kwargs : dict
            Keyword arguments to pass to the SQL query

        Returns
        -------
        pd.DataFrame
            The result as a pandas DataFrame
        """
        if not self._driver_is_async:
            # Fall back to running sync in thread
            return await asyncio.to_thread(self.execute, sql_query, params, *args, **kwargs)

        sql_query, params = self._convert_params(sql_query, params)

        async with self._engine.connect() as connection:
            result = await connection.execute(text(sql_query), params, *args, **kwargs)
            rows = result.fetchall()
            df = pd.DataFrame(rows, columns=result.keys())
            return df

    def get_tables(self) -> list[str]:
        """Return the list of available tables."""
        # If specific tables were provided, use those
        if isinstance(self.tables, (dict, list)):
            return [
                t for t in list(self.tables)
                if not self._is_table_excluded(t)
            ]

        # Otherwise, inspect the database
        inspector = self._inspector
        schema_name = self.schema

        tables = []
        if schema_name:
            # Get tables from specific schema
            table_names = inspector.get_table_names(schema=schema_name)
            tables.extend([f"{schema_name}.{t}" for t in table_names])
        else:
            # Get tables from all schemas
            for schema in inspector.get_schema_names():
                table_names = inspector.get_table_names(schema=schema)
                for table in table_names:
                    # For SQLite, skip the schema prefix if it's 'main'
                    if schema == 'main' and self.dialect == 'sqlite':
                        full_name = table
                    else:
                        full_name = f"{schema}.{table}" if schema else table
                    tables.append(full_name)

        # Filter out excluded tables
        return [t for t in tables if not self._is_table_excluded(t)]

    @cached
    def get(self, table: str, **query):
        """Retrieve data from a table with optional filtering."""
        query.pop('__dask', None)
        sql_expr = self.get_sql_expr(table)
        sql_transforms = query.pop('sql_transforms', [])
        conditions = list(query.items())

        if self.filter_in_sql:
            sql_transforms = [SQLFilter(conditions=conditions)] + sql_transforms

        for st in sql_transforms:
            sql_expr = st.apply(sql_expr)

        params = self.table_params.get(table, None)
        return self.execute(sql_expr, params)

    async def get_async(self, table: str, **query):
        """
        Return a table asynchronously; optionally filtered by the given query.

        Parameters
        ----------
        table : str
             The name of the table to query
        query : dict
             A dictionary containing all the query parameters

        Returns
        -------
        DataFrame
            A DataFrame containing the queried table.
        """
        query.pop('__dask', None)
        sql_expr = self.get_sql_expr(table)
        sql_transforms = query.pop('sql_transforms', [])
        conditions = list(query.items())

        if self.filter_in_sql:
            sql_transforms = [SQLFilter(conditions=conditions)] + sql_transforms

        for st in sql_transforms:
            sql_expr = st.apply(sql_expr)

        params = self.table_params.get(table, None)
        return await self.execute_async(sql_expr, params)

    @cached_schema
    def get_schema(
        self, table: str | None = None, limit: int | None = None, shuffle: bool = False
    ) -> dict[str, dict[str, Any]] | dict[str, Any]:
        """
        Returns JSON schema describing the tables returned by the Source.

        Parameters
        ----------
        table : str | None
            The name of the table to return the schema for. If None
            returns schema for all available tables.
        limit : int | None
            Limits the number of rows considered for the schema calculation
        shuffle : bool
            Whether to shuffle rows when sampling

        Returns
        -------
        dict
            JSON schema(s) for one or all the tables.
        """
        try:
            return super().get_schema(table, limit, shuffle)
        except Exception as e:
            # Handle timeout or other errors gracefully
            self.param.warning(f"Schema retrieval failed: {e}")
            if table is None:
                return {t: {} for t in self.get_tables()}
            else:
                return {}

    def _get_table_metadata(self, tables: list[str]) -> dict[str, Any]:
        """
        Generate metadata for tables using SQLAlchemy's inspector.

        Parameters
        ----------
        tables : list[str]
            List of table names to get metadata for

        Returns
        -------
        dict
            Dictionary mapping table names to their metadata
        """
        inspector = self._inspector
        metadata = {}

        for table_name in tables:
            # Parse schema and table name
            parts = table_name.split('.')
            if len(parts) == 2:
                schema_name, table_only = parts
            else:
                schema_name = self.schema
                table_only = table_name

            try:
                # Get column information
                columns = inspector.get_columns(table_only, schema=schema_name)
                column_metadata = {}
                for col in columns:
                    column_metadata[col['name']] = {
                        'description': col.get('comment', ''),
                        'data_type': str(col['type']),
                    }

                # Get table comment if available
                table_comment = ''
                try:
                    table_info = inspector.get_table_comment(
                        table_only, schema=schema_name
                    )
                    table_comment = table_info.get('text', '')
                except Exception:
                    pass

                # Try to get row count (not all databases support this efficiently)
                row_count = None
                try:
                    count_sql = f"SELECT COUNT(*) FROM {table_name}"
                    params = self.table_params.get(table_name, None)
                    count_df = self.execute(count_sql, params)
                    row_count = int(count_df.iloc[0, 0])
                except Exception:
                    pass

                metadata[table_name] = {
                    'description': table_comment,
                    'columns': column_metadata,
                    'rows': row_count,
                    'updated_at': None,
                    'created_at': None,
                }

            except Exception as e:
                self.param.warning(
                    f"Failed to get metadata for table {table_name}: {e}"
                )
                metadata[table_name] = {
                    'description': '',
                    'columns': {},
                    'rows': None,
                    'updated_at': None,
                    'created_at': None,
                }

        return metadata

    def close(self):
        """
        Close the database connection and dispose of the engine.

        This method should be called when the source is no longer needed to prevent
        connection leaks and properly clean up server-side resources.
        """
        if self._engine is not None:
            self._engine.dispose()
            self._engine = None

        self._inspector_cache = None
