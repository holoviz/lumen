from __future__ import annotations

from typing import Any

import param  # type: ignore

from ..transforms.base import Filter
from ..transforms.sql import (
    SQLDistinct, SQLFilter, SQLLimit, SQLMinMax,
)
from ..util import get_dataframe_schema
from ..validation import ValidationError
from .base import BaseSQLSource, cached, cached_schema
from .intake import IntakeBaseSource, IntakeSource


class IntakeBaseSQLSource(BaseSQLSource, IntakeBaseSource):

    cache_per_query = param.Boolean(default=True, doc="""
        Whether to query the whole dataset or individual queries.""")

    filter_in_sql = param.Boolean(default=True, doc="")

    dialect = 'any'

    # Declare this source supports SQL transforms
    _supports_sql = True

    __abstract = True

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
        except KeyError as e:
            raise ValidationError(
                f"'{table}' table could not be found in Intake catalog. "
                f"Available tables include: {list(self.cat)}."
            ) from e
        return source

    def get_sql_expr(self, table):
        return self._get_source(table)._sql_expr

    @cached
    def get(self, table, **query):
        '''
        Applies SQL Transforms, creating new temp catalog on the fly
        and querying the database.
        '''
        dask = query.pop('__dask', self.dask)
        sql_transforms = query.pop('sql_transforms', [])
        source = self._get_source(table)
        if not hasattr(source, '_sql_expr'):
            if sql_transforms:
                raise ValueError(
                    'SQLTransforms cannot be applied to non-SQL based Intake source.'
                )
            return super().get(table, **query)
        conditions = list(query.items())
        if self.filter_in_sql:
            sql_transforms = [SQLFilter(conditions=conditions)] + sql_transforms
        source = self._apply_transforms(source, sql_transforms)
        df = self._read(source)
        if not self.filter_in_sql:
            df = Filter.apply_to(df, conditions=conditions)
        return df if dask or not hasattr(df, 'compute') else df.compute()

    @cached_schema
    def get_schema(
        self, table: str | None = None, limit: int | None = None
    ) -> dict[str, dict[str, Any]] | dict[str, Any]:
        if table is None:
            tables = self.get_tables()
        else:
            tables = [table]

        schemas = {}
        sql_limit = SQLLimit(limit=limit or 1)
        for entry in tables:
            if not self.load_schema:
                schemas[entry] = {}
                continue
            source = self._get_source(entry)
            if not hasattr(source, '_sql_expr'):
                schemas[entry] = super().get_schema(table)
                continue
            data = self._read(self._apply_transforms(source, [sql_limit]))
            schemas[entry] = schema = get_dataframe_schema(data)['items']['properties']
            if limit:
                continue

            enums, min_maxes = [], []
            for name, col_schema in schema.items():
                if 'enum' in col_schema:
                    enums.append(name)
                elif 'inclusiveMinimum' in col_schema:
                    min_maxes.append(name)

            # Calculate enum schemas
            for col in enums:
                distinct_transforms = [SQLDistinct(columns=[col])]
                distinct = self._read(
                    self._apply_transforms(source, distinct_transforms)
                )
                schema[col]['enum'] = distinct[col].to_list()

            if not min_maxes:
                continue

            # Calculate numeric schemas
            minmax_data = self._read(
                self._apply_transforms(source, [SQLMinMax(columns=min_maxes)])
            )
            for col in min_maxes:
                kind = data[col].dtype.kind
                if kind in 'iu':
                    cast = int
                elif kind == 'f':
                    cast = float
                else:
                    cast = lambda v: v
                schema[col]['inclusiveMinimum'] = cast(minmax_data[f'{col}_min'].iloc[0])
                schema[col]['inclusiveMaximum'] = cast(minmax_data[f'{col}_max'].iloc[0])
        return schemas if table is None else schemas[table]


class IntakeSQLSource(IntakeBaseSQLSource, IntakeSource):
    """
    `IntakeSQLSource` extends the `IntakeSource` with support for SQL data.

    In addition to the standard intake support for reading catalogs
    the `IntakeSQLSource` computes the schema by querying the database
    instead of loading all the data into memory and allows for
    `SQLTransform` to be applied when querying the SQL database.
    """

    source_type = 'intake_sql'
