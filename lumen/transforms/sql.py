import datetime as dt

import numpy as np
import param

from jinja2 import Template

from .base import Transform


class SQLTransform(Transform):
    """
    Base class for SQL transforms.
    Mainly for informational purposes.
    """

    __abstract = True

    @classmethod
    def apply_to(cls, sql_in, **kwargs):
        """
        Calls the apply method based on keyword arguments passed to define transform.

        Parameters
        ----------
        sql_in: SQL Select statement to input to transformation.

        Returns
        -------
        SQL statement after application of transformation.
        """
        return cls(**kwargs).apply(sql_in)

    def apply(self, sql_in):
        """
        Given an SQL statement, manipulate it, and return a new SQL statement.

        Parameters
        ----------
        sql_in: string
            The initial SQL query to be manipulated.

        Returns
        -------
        string
            New SQL query derived from the above query.
        """
        return sql_in


class SQLGroupBy(SQLTransform):
    """
    Performs a Group-By and aggregation
    """

    by = param.List(doc="""
        Columns to Group by""")

    aggregates = param.Dict(doc="""
        mapping of Aggregate Functions to use to which column to use them on""")

    transform_type = 'sql_group_by'

    def apply(self, sql_in):
        template = """
            SELECT
                {{by_cols}},
                {{aggs}}
            FROM ( {{sql_in}} )
            GROUP BY {{by_cols}}
        """
        by_cols = ', '.join(self.by)
        aggs = ', '.join([
            f'{agg}({col}) AS {col}' for agg, col in self.aggregates.items()
        ])
        return Template(template, trim_blocks=True, lstrip_blocks=True).render(
            by_cols=by_cols, aggs=aggs, sql_in=sql_in
        )


class SQLLimit(SQLTransform):
    """
    Performs a LIMIT SQL operation on the query
    """

    limit = param.Integer(default=1000, doc="Limit on the number of rows to return")

    transform_type = 'sql_limit'

    def apply(self, sql_in):
        template = """
            SELECT
                *
            FROM ( {{sql_in}} )
            LIMIT {{limit}}
        """
        return Template(template, trim_blocks=True, lstrip_blocks=True).render(
            limit=self.limit, sql_in=sql_in
        )


class SQLDistinct(SQLTransform):

    columns = param.List(default=[], doc="Columns to return distinct values for.")

    transform_type = 'sql_distinct'

    def apply(self, sql_in):
        template = """
            SELECT DISTINCT
                {{columns}}
            FROM ( {{sql_in}} )
        """
        return Template(template, trim_blocks=True, lstrip_blocks=True).render(
            columns=', '.join(self.columns), sql_in=sql_in
        )


class SQLMinMax(SQLTransform):

    columns = param.List(default=[], doc="Columns to return min/max values for.")

    transform_type = 'sql_minmax'

    def apply(self, sql_in):
        aggs = []
        for col in self.columns:
            aggs.append(f'MIN({col}) as {col}_min')
            aggs.append(f'MAX({col}) as {col}_max')
        template = """
            SELECT
                {{columns}}
            FROM ( {{sql_in}} )
        """
        return Template(template, trim_blocks=True, lstrip_blocks=True).render(
            columns=', '.join(aggs), sql_in=sql_in
        )


class SQLColumns(SQLTransform):

    columns = param.List(default=[], doc="Columns to return.")

    transform_type = 'sql_columns'

    def apply(self, sql_in):
        template = """
            SELECT
                {{columns}}
            FROM ( {{sql_in}} )
        """
        return Template(template, trim_blocks=True, lstrip_blocks=True).render(
            columns=', '.join(self.columns), sql_in=sql_in
        )


class SQLFilter(SQLTransform):
    """
    Translates Lumen Filter query into a SQL WHERE statement.
    """

    conditions = param.List(doc="""
      List of filter conditions expressed as tuples of the column
      name and the filter value.""")

    transform_type = 'sql_filter'

    @classmethod
    def _range_filter(cls, col, v1, v2):
        start = str(v1) if isinstance(v1, dt.date) else v1
        end = str(v2) if isinstance(v2, dt.date) else v2
        if isinstance(v1, dt.date) and not isinstance(v1, dt.datetime):
            start += ' 00:00:00'
        if isinstance(v2, dt.date) and not isinstance(v2, dt.datetime):
            end += ' 00:00:00'
        return f'{col} BETWEEN {start!r} AND {end!r}'

    def apply(self, sql_in):
        conditions = []
        for col, val in self.conditions:
            if val is None:
                condition = f'{col} IS NULL'
            elif np.isscalar(val):
                condition = f'{col} = {val!r}'
            elif isinstance(val, dt.datetime):
                condition = f'{col} = {str(val)!r}'
            elif isinstance(val, dt.date):
                condition = f"{col} BETWEEN '{str(val)} 00:00:00' AND '{str(val)} 23:59:59'"
            elif (isinstance(val, list) and all(
                    isinstance(v, tuple) and len(v) == 2 for v in val
            )):
                val = [v for v in val if v is not None]
                if not val:
                    continue
                condition = ' OR '.join([
                    self._range_filter(col, v1, v2) for v1, v2 in val
                ])
            elif isinstance(val, list):
                if not val:
                    continue
                non_null = [v for v in val if v is not None]
                condition = f"{col} IN ({', '.join(map(repr, non_null))})"
                if not non_null:
                    condition = f'{col} IS NULL'
                elif len(val) != len(non_null):
                    condition = f'({condition}) OR ({col} IS NULL)'
            elif isinstance(val, tuple):
                condition = self._range_filter(col, *val)
            else:
                self.param.warning(
                    'Condition {val!r} on {col!r} column not understood. '
                    'Filter query will not be applied.'
                )
                continue
            conditions.append(condition)
        if not conditions:
            return sql_in

        template = """
            SELECT
                *
            FROM ( {{sql_in}} )
            WHERE ( {{conditions}} )
        """
        return Template(template, trim_blocks=True, lstrip_blocks=True).render(
            conditions=' AND '.join(conditions), sql_in=sql_in
        )
