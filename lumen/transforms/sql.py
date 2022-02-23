import param

from jinja2 import Template

from .base import Transform


class SQLTransform(Transform):
    """
    Base class for SQL transforms.
    Mainly for informational purposes.
    """

    __abstract = True

    def __hash__(self):
        """
        Implements hashing to allow a Source to compute a hash key.
        """
        hash_vals = (type(self).__name__,)
        hash_vals += tuple(sorted([
            (k, v) for k, v in self.param.values().items()
            if k not in Transform.param
        ]))
        return hash(str(hash_vals))

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

    columns = param.List(default=[], doc="Columns to return distinct values for.")

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
