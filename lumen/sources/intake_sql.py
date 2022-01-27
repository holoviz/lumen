from .base import cached
from .intake import IntakeBaseSource, IntakeSource


class IntakeBaseSQLSource(IntakeBaseSource):

    # Declare this source supports SQL transforms
    _sql = True

    @cached()
    def get(self, table, **query):
        '''
        Applies SQL Transforms, creating new temp catalog on the fly
        and querying the database.
        '''
        dask = query.pop('__dask', self.dask)
        sql_transforms = query.pop('sql_transforms', [])

        try:
            source = self.cat[table]
        except KeyError:
            raise KeyError(
                f"'{table}' table could not be found in Intake catalog. "
                f"Available tables include: {list(self.cat)}."
            )
        sql_expr = source._sql_expr
        for sql_transform in sql_transforms:
            sql_expr = sql_transform.apply(sql_expr)
        new_source = type(source)(**dict(source._init_args, sql_expr=sql_expr))
        df = self._read(new_source)
        return df if dask or not hasattr(df, 'compute') else df.compute()


class IntakeSQLSource(IntakeBaseSQLSource, IntakeSource):
    """
    Intake source specifically for SQL sources.
    Allows for sql transformations to be applied prior to querying the source.
    """

    source_type = 'intake_sql'
