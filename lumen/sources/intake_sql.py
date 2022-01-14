import hashlib
import os
import sys

import pandas as pd

from intake_sql import SQLSource

from .base import cached
from .intake import IntakeSource


class IntakeSQLSource(IntakeSource):
    """
    Intake source specifically for SQL sources.
    Allows for sql transformations to be applied prior to querying the source.
    """
    source_type = 'intake_sql'

    def _get_cache(self, table, **query):
        query.pop('__dask', None)
        key = self._get_key(table, **query)
        if key in self._cache:
            return self._cache[key], not bool(query)
        elif self.cache_dir:
            if query:
                sha = hashlib.sha256(str(key).encode('utf-8')).hexdigest()
                filename = f'{sha}_{table}.parq'
            else:
                filename = f'{table}.parq'
            path = os.path.join(self.root, self.cache_dir, filename)
            if os.path.isfile(path) or os.path.isdir(path):
                if 'dask.dataframe' in sys.modules:
                    import dask.dataframe as dd
                    return dd.read_parquet(path), not bool(query)
                return pd.read_parquet(path), not bool(query)
        elif key[:1] in self._cache:
            return self._cache[key[:1]], True
        return None, not bool(query)

    @cached(with_query=True)
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

        new_source = SQLSource(**dict(source._init_args, sql_expr=sql_expr))
        df = self._read(new_source)
        return df if dask or not hasattr(df, 'compute') else df.compute()
