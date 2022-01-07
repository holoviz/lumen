from .intake import IntakeSource
from .base import cached
from intake_sql import SQLSource
import os
import intake

class IntakeSQLSource(IntakeSource):
    """
    Intake source specifically for SQL sources.
    Allows for sql transformations to be applied prior to querying the source.
    """
    source_type = 'intake_sql'
    
    def _update_sql(self, table, new_sql_expr):
        """
        Updates a table's sql statement.
        """

        self.cat = self.cat.add(
            SQLSource(
                uri=self.cat[table]._uri,
                sql_expr=new_sql_expr,
                sql_kwargs=self.cat[table]._sql_kwargs,
                metadata=self.cat[table].metadata,
            ),
            table,
            path='temp/temp.yaml'
        )

    
    def _apply_sql_transform(self, table, transform):
        """
        Applies a transformation and subsequently updates a table's
        sql statement.
        """
        sql_in = self.cat[table]._sql_expr
        sql_out = transform.apply(sql_in)
        self._update_sql(table, sql_out)
    
    # TODO: enable caching by hash of final sql_statement
    def get(self, table, **query):
        dask = query.pop('__dask', self.dask)
        sql_transforms = query.pop('sql_transforms', [])
        
        for sql_transform in sql_transforms:
            self._apply_sql_transform(table, sql_transform)

        df = self._read(table)
        
        # Hack to reset the catalog obj. after the data pull
        self.cat = intake.open_catalog(self.uri)
        return df if dask or not hasattr(df, 'compute') else df.compute()
