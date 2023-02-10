import duckdb
import param

from ..transforms import Filter
from ..transforms.sql import (
    SQLDistinct, SQLFilter, SQLLimit, SQLMinMax,
)
from ..util import get_dataframe_schema
from .base import Source, cached, cached_schema


class DuckDBSource(Source):
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

    load_schema = param.Boolean(default=True, doc="Whether to load the schema")

    uri = param.String(doc="The URI of the DuckDB database")

    sql_expr = param.String(default='SELECT * FROM {table}', doc="""
        The SQL expression to execute.""")

    tables = param.ClassSelector(class_=(list, dict), doc="""
        List or dictionary of tables.""")

    initializers = param.List(default=[], doc="""
        SQL statements to run to initialize the connection.""")

    source_type = 'duckdb'

    _supports_sql = True

    def __init__(self, **params):
        super().__init__(**params)
        self._connection = duckdb.connect(self.uri)
        for init in self.initializers:
            self._connection.execute(init)

    def get_tables(self):
        if isinstance(self.tables, (dict, list)):
            return list(self.tables)
        return [t[0] for t in self._connection.execute('SHOW TABLES').fetchall()]

    @cached
    def get(self, table, **query):
        query.pop('__dask', None)
        if isinstance(self.tables, dict):
            table = self.tables[table]
        sql_expr = self.sql_expr.format(table=table)
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
    def get_schema(self, table=None):
        if table is None:
            tables = self.get_tables()
        else:
            if isinstance(self.tables, dict):
                table = self.tables[table]
            tables = [table]

        schemas = {}
        limit = SQLLimit(limit=1)
        for entry in tables:
            if not self.load_schema:
                schemas[entry] = {}
                continue
            sql_expr = self.sql_expr.format(table=table)
            data = self._connection.execute(limit.apply(sql_expr)).fetch_df()
            schema = get_dataframe_schema(data)['items']['properties']
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
            schemas[entry] = schema
        return schemas if table is None else schemas[table]
