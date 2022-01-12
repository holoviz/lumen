from .intake import IntakeSource
from .base import cached
from intake_sql import SQLSource
import os
import intake
import hashlib
from functools import reduce
import sys
import pandas as pd

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
    
    def _get_key(self, table, **query):
        key = (table,)
        for k, v in sorted(query.items()):
            if k=='sql_transforms':
                v=str([str(t) for t in v])
                    
            elif isinstance(v, list):
                v=tuple(v)
            key += (k, v)
        return key
    
    def _read(self, table, cat, dask=True):
        try:
            entry = cat[table]
        except KeyError:
            raise KeyError(f"'{table}' table could not be found in Intake "
                           "catalog. Available tables include: "
                           f"{list(self.cat)}.")
        if self.dask or dask:
            try:
                return entry.to_dask()
            except Exception:
                if self.dask:
                    self.param.warning(f"Could not load {table} table with dask.")
                pass
        return entry.read()
    
    def _update_sql(self, table, new_sql_expr, cat):
        """
        Updates a table's sql statement.
        """

        return cat.add(
            SQLSource(
                uri=cat[table]._uri,
                sql_expr=new_sql_expr,
                sql_kwargs=cat[table]._sql_kwargs,
                metadata=cat[table].metadata,
            ),
            table,
            path='temp/temp.yaml'
        )

    
    def _apply_sql_transform(self, table, transform, cat):
        """
        Applies a transformation and subsequently updates a table's
        sql statement.
        """
        sql_in = cat[table]._sql_expr
        sql_out = transform.apply(sql_in)
        return self._update_sql(table, sql_out, cat)
    
    # TODO: enable caching by hash of final sql_statement
    @cached(with_query=True)
    def get(self, table, **query):
        dask = query.pop('__dask', self.dask)
        sql_transforms = query.pop('sql_transforms', [])
        
        new_cat = self.cat
        for sql_transform in sql_transforms:
            new_cat = self._apply_sql_transform(table, sql_transform, new_cat)

        df = self._read(table, new_cat)
        
        return df if dask or not hasattr(df, 'compute') else df.compute()
