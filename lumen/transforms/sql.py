from __future__ import annotations

import datetime as dt
import pathlib
import re

from copy import deepcopy
from typing import ClassVar

import param  # type: ignore
import sqlglot

from sqlglot import parse
from sqlglot.expressions import (
    LT, Column, Expression, Identifier, Literal as SQLLiteral, Max, Min, Null,
    ReadCSV, Select, Star, Table, TableSample, and_, func, or_,
    replace_placeholders, replace_tables, select,
)
from sqlglot.optimizer import optimize

from ..config import SOURCE_TABLE_SEPARATOR
from ..util import detect_file_encoding
from .base import Transform


class SQLTransform(Transform):
    """
    Base class for SQL transforms using sqlglot.
    """

    comments = param.Boolean(
        default=False, doc="Whether to include comments in the output SQL")

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

    def __init__(self, **params):
        for key in ["read", "write"]:
            if params.get(key) == "any":
                params[key] = None

        if params.get("read") and not params.get("write"):
            params["write"] = params["read"]
        elif not params.get("read") and params.get("write"):
            params["read"] = params["write"]
        super().__init__(**params)

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
        try:
            expressions = parse(
                sql_in,
                read=self.read,
                error_level=self.error_level,
            )
        except sqlglot.ParseError:
            # Find table name after FROM and quote it if it contains special characters
            pattern = r'FROM\s+([a-zA-Z0-9_\-\.]+)'
            quoted_sql_in = re.sub(
                pattern, lambda m: f'FROM "{m.group(1)}"', sql_in, flags=re.IGNORECASE
            )
            expressions = parse(
                quoted_sql_in,
                read=self.read,
                error_level=self.error_level,
            )

        if len(expressions) > 1:
            raise ValueError(
                "Multiple SQL statements found. Please provide only a single SQL statement."
            )
        expression = expressions[0]
        return expression

    def _add_encoding_to_read_csv(self, expression: Expression) -> Expression:
        """
        Add file encoding when reading CSV files using DuckDB.

        Parameters
        ----------
        expression : Expression
            An sqlglot expression object.

        Returns
        -------
        Expression
            A modified expression that includes the file encoding.
        """
        expr = deepcopy(expression)
        if isinstance(expr, ReadCSV):
            read_csv = expr.find(ReadCSV) or ReadCSV()
            literal = read_csv.find(SQLLiteral) or SQLLiteral()
            if pathlib.Path(literal.this).suffix.lower() == ".csv" and "encoding" not in literal.this:
                encoding = detect_file_encoding(file_obj=literal.this)
                expr.find(ReadCSV).find(SQLLiteral).replace(Identifier(this=f"'{literal.this}', encoding='{encoding}'", is_string=literal.is_string))

        return expr

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

        expression = self._add_encoding_to_read_csv(expression=expression)

        return expression.sql(
            comments=self.comments,
            dialect=self.write,
            identify=self.identify,
            pretty=self.pretty,
            unsupported_level=self.unsupported_level,
        )


class SQLFormat(SQLTransform):
    """
    Format SQL expressions with parameterized replacements.

    This transform allows for replacing placeholders in SQL queries using
    either Python string format-style placeholders {name} or sqlglot-style
    placeholders :name.

    Parameters
    ----------
    sql_expr : str
        The SQL expression template with placeholders to be formatted.
    parameters : dict
        Dictionary of parameter names and values to replace in the SQL template.
    py_format : bool
        Whether to convert Python format-style placeholders {name} to sqlglot
        placeholders :name before parsing. Default is True.
    """

    parameters = param.Dict(default={}, doc="""
        Dictionary of parameter names and values to replace in the SQL template.""")

    transform_type: ClassVar[str] = 'sql_format'

    def apply(self, sql_in: str) -> str:
        """
        Apply the formatting to the input SQL, replacing placeholders with values.

        Parameters
        ----------
        sql_in : str
            The input SQL query to format. This is used as a base query that will
            have the formatted sql_expr applied to it, typically as a subquery.

        Returns
        -------
        str
            The formatted SQL query with all placeholders replaced.
        """
        sql_template = re.sub(r'\{(\w+)\}', r':\1', sql_in)
        expression = self.parse_sql(sql_template)
        if self.parameters:
            parameters = {}
            for k, v in self.parameters.items():
                if isinstance(v, str):
                    parameters[k] = Identifier(this=v, quoted=self.identify)
                else:
                    parameters[k] = v
            replaced_expression = replace_placeholders(expression, **parameters)
        return self.to_sql(replaced_expression,)


class SQLSelectFrom(SQLFormat):

    sql_expr = param.String(
        default="SELECT * FROM {table}",
        doc="""
        The SQL expression to useif the sql_in does NOT
        already contain a SELECT statement.""",
    )

    tables = param.ClassSelector(
        default=None,
        class_=(list, dict),
        doc="""
        Dictionary of tables to replace or use in the SQL expression.
        If None, the original table will be used.""",
    )

    transform_type: ClassVar[str] = "sql_select_from"

    def apply(self, sql_in: str) -> str:
        try:
            expression = self.parse_sql(sql_in)
        except sqlglot.ParseError:
            # The expression has no SELECT statement, and invalid characters
            # e.g. read_parquet("/path/to/file.parquet"); so we need to quote
            expression = Table(this=Identifier(this=sql_in, quoted=True))

        # if 'data/life-expectancy.csv' becomes 'data / life-expectancy.csv'
        if not list(expression.find_all(Select)) and " / " in expression.sql():
            expression = Table(this=Identifier(this=sql_in, quoted=True))

        tables = {}

        # one word table names may be classified as a Column
        existing_tables = list(expression.find_all(Table)) or expression.find_all(
            Column
        )
        if isinstance(self.tables, list):
            for existing_table, renamed_table in zip(existing_tables, self.tables):
                if isinstance(renamed_table, Table):
                    tables[existing_table] = renamed_table
                else:
                    tables[existing_table] = Table(
                        this=Identifier(this=renamed_table),
                        alias=Identifier(this=existing_table.alias),
                    )

        else:
            for renamed_key, renamed_value in (self.tables or {}).items():
                if isinstance(renamed_key, Table) and isinstance(renamed_value, Table):
                    tables[renamed_key] = renamed_value
                elif isinstance(renamed_key, Table):
                    tables[renamed_key] = Table(
                        Identifier(this=renamed_value),
                        alias=Identifier(this=renamed_key.alias),
                    )
                elif isinstance(renamed_value, Table):
                    tables[
                        Table(
                            this=renamed_key, alias=Identifier(this=renamed_value.alias)
                        )
                    ] = renamed_value
                else:
                    alias = next(
                        (
                            existing_table.alias
                            for existing_table in existing_tables
                            if renamed_key == str(existing_table.this)
                        ),
                        "",
                    )
                    tables[
                        Table(
                            this=Identifier(this=renamed_key),
                            alias=Identifier(this=alias),
                        )
                    ] = Table(
                        this=Identifier(this=renamed_value),
                        alias=Identifier(this=alias),
                    )

        if expression.find(Select) is not None:
            # if no tables to replace, just return the SQL
            if not tables:
                return sql_in
            # if Select is found, replace tables
            replaced_expression = replace_tables(expression, tables, dialect=self.read)
            return self.to_sql(replaced_expression)

        # if Select is NOT found, use the default sql_expr
        # and if tables are provided, use that table
        # else use the sql_in turned into expression (which should be a table)
        table = next(iter(tables.values()), expression)
        self.parameters = {"table": table}
        return super().apply(self.sql_expr)


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
    Performs a LIMIT SQL operation on the query.
    If the query already has a LIMIT clause, it will only be applied if the
    existing limit is less than the new limit.
    """

    limit = param.Integer(default=1000, allow_None=True, doc="Limit on the number of rows to return")

    transform_type: ClassVar[str] = 'sql_limit'

    def apply(self, sql_in: str) -> str:
        if self.limit is None:
            return sql_in

        parsed_expression = self.parse_sql(sql_in)
        if existing_limit := parsed_expression.args.get("limit"):
            if int(existing_limit.expression.this) < self.limit:
                # if existing limit is less than the new limit
                # do not modify the original query
                return sql_in

        subquery = parsed_expression.subquery()
        expression = select("*").from_(subquery).limit(self.limit)
        return self.to_sql(expression)


class SQLDistinct(SQLTransform):

    columns = param.List(default=[], doc="Columns to return distinct values for.")

    transform_type: ClassVar[str] = 'sql_distinct'

    def apply(self, sql_in: str) -> str:
        if not self.columns:
            return sql_in

        subquery = self.parse_sql(sql_in).subquery()
        expression = select(*[Identifier(this=col, quoted=True) for col in self.columns]).from_(subquery).distinct()
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
        minmax = []
        for col in self.columns:
            quoted = self.identify or bool(re.search(r'\W', col))
            for agg_func in (Min, Max):
                alias = Identifier(this=(col + f"_{agg_func.__name__.lower()}"), quoted=quoted)
                minmax.append(
                    agg_func(this=Column(this=Identifier(this=col, quoted=quoted))).as_(alias)
                )
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
                filters.append(column_expr.eq(SQLLiteral.number(val)))
            elif isinstance(val, str):
                filters.append(column_expr.eq(SQLLiteral.string(val)))
            elif isinstance(val, dt.datetime):
                filters.append(column_expr.eq(SQLLiteral.string(str(val))))
            elif isinstance(val, dt.date):
                start = SQLLiteral.string(f"{val} 00:00:00")
                end = SQLLiteral.string(f"{val} 23:59:59")
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

                filters.append(column_expr.between(SQLLiteral.string(start_str), SQLLiteral.string(end_str)))
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

                        range_filters.append(column_expr.between(SQLLiteral.string(v1_str), SQLLiteral.string(v2_str)))
                    filters.append(or_(*range_filters))
                else:
                    # Handle None values separately in lists
                    non_none_values = [v for v in val if v is not None]
                    has_none = any(v is None for v in val)

                    if has_none and not non_none_values:
                        # All values are None
                        filters.append(column_expr.is_(Null()))
                    elif has_none:
                        # Mix of None and non-None values
                        none_filter = column_expr.is_(Null())
                        in_filter = column_expr.isin(*[SQLLiteral.string(str(v)) for v in non_none_values])
                        filters.append(or_(none_filter, in_filter))
                    else:
                        # No None values
                        filters.append(column_expr.isin(*[SQLLiteral.string(str(v)) for v in val]))
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

        if dialect in ('sqlserver', 'duckdb') or (dialect in ('postgres', 'redshift', 'snowflake', 'bigquery')):
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
            sample_clause = TableSample(size=SQLLiteral(this=str(self.size), is_string=False))
        else:
            sample_clause = TableSample(percent=SQLLiteral(this=str(self.percent), is_string=False))

        if self.seed is not None:
            sample_clause.args["seed"] = SQLLiteral(this=str(self.seed), is_string=False)

        if "method" in self.sample_kwargs:
            sample_clause.args["method"] = self.sample_kwargs["method"]

        if "bucket_numerator" in self.sample_kwargs:
            sample_clause.args["bucket_numerator"] = SQLLiteral(this=str(self.sample_kwargs["bucket_numerator"]), is_string=False)

        if "bucket_denominator" in self.sample_kwargs:
            sample_clause.args["bucket_denominator"] = SQLLiteral(this=str(self.sample_kwargs["bucket_denominator"]), is_string=False)

        if "bucket_field" in self.sample_kwargs:
            sample_clause.args["bucket_field"] = Column(this=self.sample_kwargs["bucket_field"], is_string=True)

        sample_expression = Select(
            sample=sample_clause,
        ).select("*").from_(subquery)
        return self.to_sql(sample_expression)

    def _apply_mysql_dialect(self, expression: Expression) -> str:
        """Handle MySQL sampling using ORDER BY RAND() with an indexed random column."""
        subquery = expression.subquery("subquery")

        if self.size is not None:
            rand_low = func("RAND")
            rand_high = rand_low + SQLLiteral.number(3 * self.size / 1000000)

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
            random_func = func("RANDOM", SQLLiteral.number(self.seed))
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
                expression=SQLLiteral.number(self.percent/100.0)
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
            rand_func = func("RAND", SQLLiteral.number(self.seed))
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
                expression=SQLLiteral.number(self.percent / 100.0)
            )
            sample_expression = (
                select("*")
                .from_(subquery)
                .where(filter_expr)
            )

        return self.to_sql(sample_expression)


class SQLRemoveSourceSeparator(SQLTransform):
    """
    Class to exclude the source and separator.
    """

    separator = param.String(default=SOURCE_TABLE_SEPARATOR, doc="""
        Separator used to split the source and table name in the SQL query.""")

    _parsable_separator = "___AT___"

    def apply(self, sql_in: str) -> str:
        """
        Exclude the source and separator from the SQL query.

        Parameters
        ----------
        sql_in: string
            The initial SQL query to be manipulated.

        Returns
        -------
        string
            New SQL query derived from the above query.
        """
        if self.separator not in sql_in:
            return sql_in
        sql_in = sql_in.replace(SOURCE_TABLE_SEPARATOR, self._parsable_separator)
        return self.to_sql(self.parse_sql(sql_in).transform(self._remove_source_separator))

    def _remove_source_separator(self, node):
        if isinstance(node, Table):
            node_str = node.this
            while hasattr(node_str, "this"):
                node_str = node_str.this
            after = node_str.split(self._parsable_separator)[-1]
            filename = node.this.sql().replace(node_str, after)
            return filename
        elif isinstance(node, sqlglot.exp.Literal) and isinstance(node.this, str):
            filename = node.this.split(self._parsable_separator)[-1]
            return filename
        else:
            return node


__all__ = [name for name, obj in locals().items() if isinstance(obj, type) and issubclass(obj, SQLTransform)]
