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
