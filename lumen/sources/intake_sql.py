from .intake import IntakeSource
from .base import cached

class IntakeSQLSource(IntakeSource):
    
    source_type = 'intake_sql'
    
    @cached(with_query=True)
    def get(self, table, **query):
        import pdb; pdb.set_trace()
        dask = query.pop('__dask', self.dask)
        sql_transforms = query.pop('sql_transforms', [])
        for sql_transform in sql_transforms:
            self.cat[table]['sql_expr'] = sql_transform.apply(self.cat[table]['sql_expr'])
        df = self._read(table)
        return df if dask or not hasattr(df, 'compute') else df.compute()