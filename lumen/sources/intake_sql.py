from ..transforms.sql import SQLDistinct, SQLLimit, SQLMinMax
from ..util import get_dataframe_schema
from .base import cached
from .intake import IntakeBaseSource, IntakeSource


class IntakeBaseSQLSource(IntakeBaseSource):

    # Declare this source supports SQL transforms
    _supports_sql = True

    def _apply_transforms(self, source, sql_transforms):
        if not sql_transforms:
            return source
        sql_expr = source._sql_expr
        for sql_transform in sql_transforms:
            sql_expr = sql_transform.apply(sql_expr)
        return type(source)(**dict(source._init_args, sql_expr=sql_expr))

    def _get_source(self, table):
        try:
            source = self.cat[table]
        except KeyError:
            raise KeyError(
                f"'{table}' table could not be found in Intake catalog. "
                f"Available tables include: {list(self.cat)}."
            )
        return source

    @cached()
    def get(self, table, **query):
        '''
        Applies SQL Transforms, creating new temp catalog on the fly
        and querying the database.
        '''
        dask = query.pop('__dask', self.dask)
        sql_transforms = query.pop('sql_transforms', [])
        source = self._get_source(table)
        source = self._apply_transforms(source, sql_transforms)
        df = self._read(source)
        return df if dask or not hasattr(df, 'compute') else df.compute()

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
            source = self._get_source(entry)
            data = self._read(self._apply_transforms(source, [limit]))
            schema = get_dataframe_schema(data)['items']['properties']
            enums, min_maxes = [], []
            for name, col_schema in schema.items():
                if 'enum' in col_schema:
                    enums.append(name)
                elif 'inclusiveMinimum' in col_schema:
                    min_maxes.append(name)
            distinct_data = self._read(
                self._apply_transforms(source, [SQLDistinct(columns=enums)])
            )
            for col in enums:
                schema[col]['enum'] = distinct_data[col].to_list()
            minmax_data = self._read(
                self._apply_transforms(source, [SQLMinMax(columns=min_maxes)])
            )
            for col in min_maxes:
                schema[col]['inclusiveMinimum'] = minmax_data[f'{col}_min'].iloc[0]
                schema[col]['inclusiveMaximum'] = minmax_data[f'{col}_max'].iloc[0]
            schemas[entry] = schema
        return schemas if table is None else schemas[table]


class IntakeSQLSource(IntakeBaseSQLSource, IntakeSource):
    """
    Intake source specifically for SQL sources.
    Allows for sql transformations to be applied prior to querying the source.
    """

    source_type = 'intake_sql'
