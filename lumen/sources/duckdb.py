import duckdb
import param

from ..transforms import Filter
from ..transforms.sql import (
    SQLDistinct, SQLFilter, SQLLimit, SQLMinMax,
)
from ..util import get_dataframe_schema
from .base import Source, cached, cached_schema


class DuckDBSource(Source):

    filter_in_sql = param.Boolean(default=True, doc="Whether to apply filters in SQL or in-memory.")

    load_schema = param.Boolean(default=True, doc="Whether to load the schema")

    uri = param.String(doc="The URI of the DuckDB database")

    sql_expr = param.String(default='SELECT * FROM {table}', doc="The SQL expression to execute.")

    source_type = 'duckdb'

    _supports_sql = True

    def __init__(self, **params):
        super().__init__(**params)
        self._connection = duckdb.connect(self.uri)

    def get_tables(self):
        return [t[0] for t in self._connection.execute('SHOW TABLES').fetchall()]

    @cached()
    def get(self, table, **query):
        sql_expr = self.sql_expr.format(table=table)
        sql_transforms = query.pop('sql_transforms', [])
        conditions = list(query.items())
        if self.filter_in_sql:
            sql_transforms = [SQLFilter(conditions=conditions)] + sql_transforms
        for st in sql_transforms:
            sql_expr = st.apply(sql_expr)
        df = self._connection.execute(sql_expr).fetch_df()
        if not self.filter_in_sql:
            df = Filter.apply_to(df, conditions=conditions)
        return df

    @cached_schema
    def get_schema(self, table=None):
        if table is None:
            tables = self.get_tables()
        else:
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
                distinct_expr = sql_expr
                for sql_t in (SQLDistinct(columns=[col]), SQLLimit(limit=1000)):
                    distinct_expr = sql_t.apply(distinct_expr)
                distinct = self._connection.execute(sql_expr).fetch_df()
                schema[col]['enum'] = distinct[col].to_list()

            minmax_expr = SQLMinMax(columns=min_maxes).apply(sql_expr)
            minmax_data = self._connection.execute(minmax_expr).fetch_df()
            for col in min_maxes:
                schema[col]['inclusiveMinimum'] = minmax_data[f'{col}_min'].iloc[0]
                schema[col]['inclusiveMaximum'] = minmax_data[f'{col}_max'].iloc[0]
            schemas[entry] = schema
        return schemas if table is None else schemas[table]
