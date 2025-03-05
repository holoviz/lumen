from __future__ import annotations

import datetime as dt

from typing import ClassVar

import param  # type: ignore
import sqlglot

from sqlglot import parse_one
from sqlglot.expressions import (
    LT, Column, Expression, Literal, Null, Select, Star, TableSample, and_,
    func, or_, select,
)
from sqlglot.optimizer import optimize

from .base import Transform


class SQLTransform(Transform):
    """
    Base class for SQL transforms using sqlglot.
    """

    error_level = param.ClassSelector(
        class_=sqlglot.ErrorLevel,
        default=sqlglot.ErrorLevel.RAISE,
        doc="Error level for parsing",
    )

    identify = param.Boolean(default=False, doc="""
        Delimit all identifiers, e.g. turn `FROM database.table` into `FROM "database"."table"`.
        This is useful for dialects that don't support unquoted identifiers.""")

    optimize = param.Boolean(
        default=False, doc="""
        Whether to optimize the generated SQL query; may produce invalid results, especially with
        duckdb's read_* functions.""")

    pretty = param.Boolean(default=False, doc="Prettify output SQL, i.e. add newlines and indentation")

    read = param.String(default=None, doc="Source dialect for parsing; if None, automatically detects")

    unsupported_level = param.ClassSelector(
        class_=sqlglot.ErrorLevel,
        default=sqlglot.ErrorLevel.WARN,
        doc="When using `to_sql`, how to handle unsupported dialect features.",
    )

    write = param.String(
        default=None, doc="Target dialect for output; if None, defaults to read dialect"
    )

    __abstract = True

    @classmethod
    def apply_to(cls, sql_in: str, **kwargs) -> str:
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

    def apply(self, sql_in: str) -> str:
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
        return self.to_sql(self.parse_sql(sql_in))

    def parse_sql(self, sql_in: str) -> Expression:
        """
        Parse SQL string into sqlglot AST.

        Parameters
        ----------
        sql_in: string
            SQL string to parse

        Returns
        -------
        sqlglot.Expression
            Parsed SQL expression
        """
        return parse_one(
            sql_in,
            read=self.read,
            error_level=self.error_level,
        )

    def to_sql(self, expression: Expression) -> str:
        """
        Convert sqlglot expression back to SQL string.

        Parameters
        ----------
        expression: sqlglot.Expression
            Expression to convert to SQL

        Returns
        -------
        string
            SQL string representation
        """
        if self.optimize:
            expression = optimize(expression, dialect=self.read)

        return expression.sql(
            dialect=self.write,
            identify=self.identify,
            pretty=self.pretty,
            unsupported_level=self.unsupported_level,
        )


class SQLGroupBy(SQLTransform):
    """
    Performs a Group-By and aggregation
    """

    by = param.List(doc="Columns to group by.")

    aggregates = param.Dict(doc="""
        Mapping of aggregate functions to use to which column(s) to use them on,
        e.g. {"AVG": "col1", "SUM": ["col1", "col2"]}.""")

    transform_type: ClassVar[str] = 'sql_group_by'

    def apply(self, sql_in: str) -> str:
        if not self.aggregates:
            raise ValueError(
                'SQLGroupBy transform requires at least one aggregate function to be specified.'
            )
        elif not self.by:
            return sql_in

        subquery = self.parse_sql(sql_in).subquery()
        aggs = []
        for agg, cols in self.aggregates.items():
            for col in [cols] if isinstance(cols, str) else cols:
                aggs.append(f'{agg}({col}) AS {col}')
        expression = select(*self.by, *aggs).from_(subquery).group_by(*self.by)
        return self.to_sql(expression)


class SQLLimit(SQLTransform):
    """
    Performs a LIMIT SQL operation on the query
    """

    limit = param.Integer(default=1000, allow_None=True, doc="Limit on the number of rows to return")

    transform_type: ClassVar[str] = 'sql_limit'

    def apply(self, sql_in: str) -> str:
        if self.limit is None:
            return sql_in

        subquery = self.parse_sql(sql_in).subquery()
        expression = select("*").from_(subquery).limit(self.limit)
        return self.to_sql(expression)


class SQLDistinct(SQLTransform):

    columns = param.List(default=[], doc="Columns to return distinct values for.")

    transform_type: ClassVar[str] = 'sql_distinct'

    def apply(self, sql_in: str) -> str:
        if not self.columns:
            return sql_in

        subquery = self.parse_sql(sql_in).subquery()
        expression = select(self.columns).from_(subquery).distinct()
        return self.to_sql(expression)


class SQLCount(SQLTransform):

    transform_type: ClassVar[str] = 'sql_count'

    def apply(self, sql_in: str) -> str:
        subquery = self.parse_sql(sql_in).subquery()
        expression = select("COUNT(*) as count").from_(subquery)
        return self.to_sql(expression)


class SQLMinMax(SQLTransform):

    columns = param.List(default=[], doc="Columns to return min/max values for.")

    transform_type: ClassVar[str] = 'sql_minmax'

    def apply(self, sql_in: str) -> str:
        if not self.columns:
            return sql_in

        subquery = self.parse_sql(sql_in).subquery()
        minmax = [
            f'{func}({col}) AS {col}_{func.lower()}'
            for col in self.columns for func in ('MIN', 'MAX')
        ]
        expression = select(*minmax).from_(subquery)
        return self.to_sql(expression)


class SQLColumns(SQLTransform):

    columns = param.List(default=[], doc="Columns to return.")

    transform_type: ClassVar[str] = 'sql_columns'

    def apply(self, sql_in: str) -> str:
        if not self.columns:
            return sql_in

        subquery = self.parse_sql(sql_in).subquery()
        expression = select(*self.columns).from_(subquery)
        return self.to_sql(expression)


class SQLFilter(SQLTransform):

    conditions = param.List(doc="""
      List of filter conditions expressed as tuples of the column
      name and the filter value.""")

    transform_type: ClassVar[str] = 'sql_filter'

    def apply(self, sql_in: str) -> str:
        if not self.conditions:
            return sql_in

        subquery = self.parse_sql(sql_in).subquery()
        filters = []

        for col, val in self.conditions:
            column_expr = Column(this=col)

            if val is None:
                filters.append(column_expr.is_(Null()))
            elif isinstance(val, (int, float)):
                filters.append(column_expr.eq(Literal.number(val)))
            elif isinstance(val, str):
                filters.append(column_expr.eq(Literal.string(val)))
            elif isinstance(val, dt.datetime):
                filters.append(column_expr.eq(Literal.string(str(val))))
            elif isinstance(val, dt.date):
                start = Literal.string(f"{val} 00:00:00")
                end = Literal.string(f"{val} 23:59:59")
                filters.append(column_expr.between(start, end))
            elif isinstance(val, tuple) and len(val) == 2:
                start, end = val
                if isinstance(start, dt.date) and not isinstance(start, dt.datetime):
                    start_str = f"{start} 00:00:00"
                else:
                    start_str = str(start)

                if isinstance(end, dt.date) and not isinstance(end, dt.datetime):
                    end_str = f"{end} 23:59:59"
                else:
                    end_str = str(end)

                filters.append(column_expr.between(Literal.string(start_str), Literal.string(end_str)))
            elif isinstance(val, list):
                if all(isinstance(v, tuple) and len(v) == 2 for v in val):
                    range_filters = []
                    for v1, v2 in val:
                        if isinstance(v1, dt.date) and not isinstance(v1, dt.datetime):
                            v1_str = f"{v1} 00:00:00"
                        else:
                            v1_str = str(v1)

                        if isinstance(v2, dt.date) and not isinstance(v2, dt.datetime):
                            v2_str = f"{v2} 23:59:59"
                        else:
                            v2_str = str(v2)

                        range_filters.append(column_expr.between(Literal.string(v1_str), Literal.string(v2_str)))
                    filters.append(or_(*range_filters))
                else:
                    filters.append(column_expr.isin(*[Literal.string(str(v)) for v in val]))
            else:
                self.param.warning(f"Condition {val!r} on {col!r} column not understood. Filter query will not be applied.")
                continue

        expression = select("*").from_(subquery).where(and_(*filters))
        return self.to_sql(expression)


class SQLOverride(SQLTransform):

    override = param.String()

    def apply(self, sql_in: str) -> str:
        return self.override


class SQLSample(SQLTransform):
    """
    Samples rows from a SQL query using TABLESAMPLE or similar functionality,
    depending on the dialect's support.
    """

    percent = param.Number(default=10.0, bounds=(0.0, 100.0), doc="""
        percent of rows to sample. Must be between 0 and 100.""")

    seed = param.Integer(default=None, allow_None=True, doc="""
        Random seed for reproducible sampling.""")

    size = param.Integer(default=None, allow_None=True, doc="""
        Absolute number of rows to sample. If specified, takes precedence over percent.""")

    sample_kwargs = param.Dict(default={}, doc="""
        Other keyword arguments, like method, bucket_numerator, bucket_denominator, bucket_field.""")

    transform_type: ClassVar[str] = 'sql_sample'

    def apply(self, sql_in: str) -> str:
        if (self.percent <= 0 or self.percent >= 100) and self.size is None:
            return sql_in

        expression = self.parse_sql(sql_in)
        dialect = self.write or self.read

        if dialect in ('postgres', 'redshift', 'snowflake', 'bigquery', 'sqlserver', 'duckdb'):
            return self._apply_tablesample_dialect(expression)
        elif dialect in ('mysql', 'mariadb'):
            return self._apply_mysql_dialect(expression)
        elif dialect == 'sqlite':
            return self._apply_sqlite_dialect(expression)
        else:
            return self._apply_generic_dialect(expression)

    def _apply_tablesample_dialect(self, expression: Expression) -> str:
        subquery = expression.subquery("subquery")

        if self.size is not None:
            sample_clause = TableSample(size=Literal(this=str(self.size), is_string=False))
        else:
            sample_clause = TableSample(percent=Literal(this=str(self.percent), is_string=False))

        if self.seed is not None:
            sample_clause.args["seed"] = Literal(this=str(self.seed), is_string=False)

        if "method" in self.sample_kwargs:
            sample_clause.args["method"] = Literal(this=str(self.sample_kwargs["method"]), is_string=True)

        if "bucket_numerator" in self.sample_kwargs:
            sample_clause.args["bucket_numerator"] = Literal(this=str(self.sample_kwargs["bucket_numerator"]), is_string=False)

        if "bucket_denominator" in self.sample_kwargs:
            sample_clause.args["bucket_denominator"] = Literal(this=str(self.sample_kwargs["bucket_denominator"]), is_string=False)

        if "bucket_field" in self.sample_kwargs:
            sample_clause.args["bucket_field"] = Column(this=self.sample_kwargs["bucket_field"], is_string=True)

        sample_expression = Select(
            expressions=["*"],
            sample=sample_clause,
        ).from_(subquery)
        return self.to_sql(sample_expression)

    def _apply_mysql_dialect(self, expression: Expression) -> str:
        """Handle MySQL sampling using ORDER BY RAND() with an indexed random column."""
        subquery = expression.subquery("subquery")

        if self.size is not None:
            rand_low = func("RAND")
            rand_high = rand_low + Literal.number(3 * self.size / 1000000)

            sample_expression = (
                select("*")
                .from_(subquery)
                .where(func("frozen_rand").between(rand_low, rand_high))
                .order_by(func("RAND"))
                .limit(self.size)
            )
        else:
            sample_expression = (
                select("*")
                .from_(subquery)
                .order_by(func("RAND"))
                .limit(
                    func("FLOOR",
                        func("COUNT", Star()) * (self.percent / 100.0)
                    )
                )
            )

        return self.to_sql(sample_expression)

    def _apply_sqlite_dialect(self, expression: Expression) -> str:
        """Handle SQLite sampling which doesn't have native TABLESAMPLE support.

        Uses a more efficient approach with a subquery for sample size-based sampling.
        For percent-based sampling, uses RANDOM() < X filter.
        """
        main_subquery = expression.subquery("main_query")
        sampling_subquery = expression.subquery("sampling_query")
        if self.seed is not None:
            random_func = func("RANDOM", Literal.number(self.seed))
        else:
            random_func = func("RANDOM")
        if self.size is not None:
            inner_query = (
                select(Column(this="rowid"))
                .from_(sampling_subquery)
                .order_by(random_func)
                .limit(self.size)
            )
            sample_expr = (
                select("*")
                .from_(main_subquery)
                .where(Column(this="rowid").isin(inner_query))
            )
            return self.to_sql(sample_expr)
        else:
            filter_expr = LT(
                this=random_func,
                expression=Literal.number(self.percent/100.0)
            )
            sample_expr = (
                select("*")
                .from_(main_subquery)
                .where(filter_expr)
            )
            return self.to_sql(sample_expr)

    def _apply_generic_dialect(self, expression: Expression) -> str:
        """Handle sampling for generic SQL dialects."""
        subquery = expression.subquery("subquery")

        if self.seed is not None:
            rand_func = func("RAND", Literal.number(self.seed))
        else:
            rand_func = func("RAND")

        if self.size is not None:
            sample_expression = (
                select("*")
                .from_(subquery)
                .order_by(rand_func)
                .limit(self.size)
            )
        else:
            filter_expr = LT(
                this=rand_func,
                expression=Literal.number(self.percent / 100.0)
            )
            sample_expression = (
                select("*")
                .from_(subquery)
                .where(filter_expr)
            )

        return self.to_sql(sample_expression)


__all__ = [name for name, obj in locals().items() if isinstance(obj, type) and issubclass(obj, SQLTransform)]
