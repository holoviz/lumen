"""
Model-free helpers for running model-written SQL against Lumen sources:
read-only validation, bounded exploration queries, distinct-value listings
and the compact text previews a language model reads back.
"""
import math
import typing as t

import narwhals.stable.v2 as nw
import pandas as pd
import sqlglot

from ..config import SOURCE_TABLE_SEPARATOR
from ..sources.base import BaseSQLSource
from ..transforms.sql import SQLLimit
from ..util import as_narwhals, as_pandas, is_lazyframe
from .catalog import Metaset, resolve_table_slug
from .sharing import shares_values, sharing
from .utils import (
    clean_sql, format_error, format_unknown_name, normalize_object_dtypes,
    truncate_to_tokens,
)

if t.TYPE_CHECKING:
    from narwhals.stable.v2.typing import Frame, IntoFrame

# Rows/columns shown by format_exploration_result. Exploration reveals a frame's
# shape, dtypes and value domains; a handful of rows does that, and the previous
# 50-row aligned dump cost ~3.4k tokens for a 53x7 frame — 8x this one.
EXPLORATION_PREVIEW_ROWS = 5
EXPLORATION_PREVIEW_COLS = 25
EXPLORATION_MAX_TOKENS = 1200
EXPLORATION_MAX_ROWS = 1000
DISTINCT_VALUES_LIMIT = 50


def validate_read_only_sql(sql_query: str, dialect: str) -> None:
    # A comment after the final ";" parses as an empty Semicolon statement.
    statements = [
        statement for statement in sqlglot.parse(sql_query, read=None if dialect == "any" else dialect)
        if statement is not None and not isinstance(statement, sqlglot.exp.Semicolon)
    ]
    # DuckDB's top-level PIVOT/UNPIVOT parses as a Pivot rather than a Query.
    if len(statements) != 1 or not isinstance(statements[0], (sqlglot.exp.Query, sqlglot.exp.Pivot)) or any(
        statements[0].find(kind) for kind in (
            sqlglot.exp.Insert, sqlglot.exp.Update, sqlglot.exp.Delete,
            sqlglot.exp.Create, sqlglot.exp.Drop, sqlglot.exp.Command, sqlglot.exp.Into,
        )
    ):
        raise ValueError("Only one read-only SELECT query is allowed.")


def _with_of(query: sqlglot.exp.Expression) -> sqlglot.exp.With | None:
    return next((arg for arg in query.iter_expressions() if isinstance(arg, sqlglot.exp.With)), None)


def _binding(table: sqlglot.exp.Table) -> sqlglot.exp.CTE | None:
    """The CTE that ``table`` reads, following SQL's scoping: a CTE sees the ones before it, a recursive one itself too."""
    name = table.name.lower()
    child, node = table, table.parent
    while node is not None:
        if isinstance(node, sqlglot.exp.With):
            index = node.expressions.index(child)
            visible = node.expressions[:index + 1] if node.args.get("recursive") else node.expressions[:index]
            if cte := next((c for c in visible if c.alias_or_name.lower() == name), None):
                return cte
            # The query owning the WITH clause is skipped: its body's scope does not reach into the clause.
            child, node = node.parent, node.parent.parent if node.parent is not None else None
            continue
        if (with_ := _with_of(node)) is not None and child is not with_:
            if cte := next((c for c in with_.expressions if c.alias_or_name.lower() == name), None):
                return cte
        child, node = node, node.parent
    return None


def _table_refs(tree: sqlglot.exp.Expression) -> list[tuple[sqlglot.exp.Table, sqlglot.exp.CTE | None]]:
    """Each unqualified table name ``tree`` reads, with the CTE it is bound to, if any."""
    return [
        (table, _binding(table)) for table in tree.find_all(sqlglot.exp.Table)
        if isinstance(table.this, sqlglot.exp.Identifier) and not table.args.get("db") and not table.args.get("catalog")
    ]


def _rename_ref(table: sqlglot.exp.Table, name: str) -> None:
    if table.name == name:
        return
    if not table.alias:
        # Columns qualified with the old name keep resolving.
        table.set("alias", sqlglot.exp.TableAlias(this=table.this.copy()))
    table.set("this", sqlglot.exp.to_identifier(name))


def _unique(name: str, taken: set[str]) -> str:
    candidate, n = name, 1
    while candidate.lower() in taken:
        candidate, n = f"{name}_{n}", n + 1
    taken.add(candidate.lower())
    return candidate


def inline_tables(sql: str, tables: t.Mapping[str, str], dialect: str | None = None, *, pretty: bool = False) -> str:
    """
    Rewrite ``sql`` into one self-contained statement by defining each table
    of ``tables`` it reads, directly or through another of them, as a CTE.

    ``tables`` maps a table name to the query that defines it, which may
    read other tables of ``tables``. The CTEs come in dependency order,
    before the query's own. A definition's own WITH clause is hoisted into
    the statement's, since not every engine accepts a WITH inside a CTE.
    Names that would collide, with the query's own CTEs, another
    definition's CTEs or a table the statement reads, are renamed, and a
    renamed table keeps its old name as its alias. A name bound to a CTE
    of the query, or qualified with a schema, is never inlined.

    Returns ``sql`` unchanged when it reads none of ``tables``; raises
    ``ValueError`` when the definitions read each other in a cycle.
    """
    read = None if dialect in (None, "any") else dialect
    definitions = {name.lower(): (name, text) for name, text in tables.items()}

    def parse(text: str) -> sqlglot.exp.Expression:
        tree = sqlglot.parse_one(text, read=read)
        if not isinstance(tree, sqlglot.exp.Query):
            # DuckDB's top-level PIVOT takes no WITH clause; as a subquery it can follow one.
            tree = sqlglot.select("*").from_(tree.subquery())
        return tree

    def inlined(table, cte) -> bool:
        return cte is None and table.name.lower() in definitions

    query = parse(sql)
    query_refs = _table_refs(query)
    if not any(inlined(table, cte) for table, cte in query_refs):
        return sql

    trees: dict[str, sqlglot.exp.Expression] = {}
    refs: dict[str, list] = {}
    order: list[str] = []
    active: list[str] = []

    def visit(key: str) -> None:
        if key in trees:
            if key in active:
                cycle = [definitions[k][0] for k in active[active.index(key):] + [key]]
                raise ValueError(f"The tables read each other in a cycle: {' -> '.join(cycle)}.")
            return
        active.append(key)
        trees[key] = parse(definitions[key][1])
        refs[key] = _table_refs(trees[key])
        for table, cte in refs[key]:
            if inlined(table, cte):
                visit(table.name.lower())
        active.pop()
        order.append(key)

    for table, cte in query_refs:
        if inlined(table, cte):
            visit(table.name.lower())

    own = _with_of(query)
    taken = {cte.alias_or_name.lower() for cte in (own.expressions if own else [])}
    taken |= {
        table.name.lower()
        for found in [query_refs, *refs.values()] for table, cte in found
        if cte is None and table.name.lower() not in definitions
    }
    names = {key: _unique(definitions[key][0], taken) for key in order}

    ctes, recursive = [], bool(own and own.args.get("recursive"))
    for key in order:
        tree, with_ = trees[key], _with_of(trees[key])
        hoisted = {}
        if with_ is not None:
            recursive = recursive or bool(with_.args.get("recursive"))
            for cte in with_.expressions:
                hoisted[id(cte)] = _unique(cte.alias_or_name, taken)
        for table, cte in refs[key]:
            if inlined(table, cte):
                _rename_ref(table, names[table.name.lower()])
            elif cte is not None and id(cte) in hoisted:
                _rename_ref(table, hoisted[id(cte)])
        if with_ is not None:
            for cte in list(with_.expressions):
                cte.args["alias"].set("this", sqlglot.exp.to_identifier(hoisted[id(cte)]))
                ctes.append(cte.pop())
            with_.pop()
        alias = sqlglot.exp.TableAlias(this=sqlglot.exp.to_identifier(names[key]))
        ctes.append(sqlglot.exp.CTE(this=tree, alias=alias))

    for table, cte in query_refs:
        if inlined(table, cte):
            _rename_ref(table, names[table.name.lower()])
    if own is not None:
        ctes.extend(cte.pop() for cte in list(own.expressions))
        own.pop()
    key = "with_" if "with_" in query.arg_types else "with"
    query.set(key, sqlglot.exp.With(expressions=ctes, recursive=recursive))
    return query.sql(dialect=read, pretty=pretty)


def format_exploration_result(df: "IntoFrame | Frame", *, capped: bool = False) -> str:
    """
    Render an exploration query result as a compact preview for the LLM.

    Reports the fetched row count (or a lower bound if capped), column dtypes,
    and a few example rows. Markdown omits index and alignment padding.

    Only the handful of cells actually rendered is converted to pandas, so a
    polars or pyarrow result reaches the model without the whole frame being
    copied. The dtypes reported are the preview's for that reason, which is
    also what keeps a pandas caller's output exactly as it was.
    """
    frame = as_narwhals(df)
    if is_lazyframe(frame):
        frame = frame.collect()
    n_rows, n_cols = frame.shape
    parts = [f"{'at least ' if capped else ''}{n_rows} rows x {n_cols} columns"]

    # Normalised because which columns land on object varies by library:
    # pandas converts a DECIMAL to float itself, polars and pyarrow keep it,
    # and an object column is reported as str by the dtype line below.
    names = frame.columns[:EXPLORATION_PREVIEW_COLS]
    preview = normalize_object_dtypes(as_pandas(
        frame.select(*[nw.col(name) for name in names]).head(EXPLORATION_PREVIEW_ROWS)
    ))
    if n_cols > EXPLORATION_PREVIEW_COLS:
        parts[0] += f" (showing the first {EXPLORATION_PREVIEW_COLS} columns)"

    def _dtype_name(dtype):
        return "str" if pd.api.types.is_string_dtype(dtype) else str(dtype)

    dtypes = ", ".join(f"{col}: {_dtype_name(dtype)}" for col, dtype in preview.dtypes.items())
    parts.append(f"Columns — {dtypes}")

    if n_rows:
        shown = min(n_rows, EXPLORATION_PREVIEW_ROWS)
        label = "all rows" if shown == n_rows else f"first {shown} of {n_rows} rows"
        parts.append(f"Sample ({label}):\n" + preview.to_markdown(index=False))
    else:
        parts.append("The query returned no rows.")

    return truncate_to_tokens("\n\n".join(parts), EXPLORATION_MAX_TOKENS)


def _source_slugs(sources: dict[tuple[str, str], BaseSQLSource]) -> dict[str, tuple[str, str]]:
    return {f"{source}{SOURCE_TABLE_SEPARATOR}{table}": (source, table) for source, table in sources}


def resolve_source_table(
    reference: str, sources: dict[tuple[str, str], BaseSQLSource]
) -> tuple[BaseSQLSource, str]:
    """Resolve a model-supplied table reference to its source and table name (see :func:`resolve_table_slug`)."""
    slugs = _source_slugs(sources)
    key = slugs[resolve_table_slug(reference, slugs)]
    return sources[key], key[1]


def resolve_exploration_source(
    source: str | None, sources: dict[tuple[str, str], BaseSQLSource]
) -> BaseSQLSource:
    """
    Resolve the ``source`` argument of an exploration tool call.

    Weak models often pass a table name, a generic ``source/<table>`` or a
    differently cased name instead of the source name, so any reference that
    identifies one source unambiguously is accepted.
    """
    names = sorted({s for s, _ in sources})
    if len(names) == 1 and (not source or source.strip().lower() in {"source", names[0].lower()}):
        return next(iter(sources.values()))
    if not source:
        raise ValueError(f"Several sources are available; pass `source` as one of: {', '.join(names)}.")
    by_name = {name.lower(): name for name in names}
    if (name := by_name.get(source.strip().lower())) is not None:
        return next(obj for (s, _), obj in sources.items() if s == name)
    try:
        return resolve_source_table(source, sources)[0]
    except ValueError:
        if len(names) == 1:
            return next(iter(sources.values()))
    raise ValueError(format_unknown_name("source", source, names))


async def execute_exploration_sql(
    source: str | None,
    sql_query: str,
    *,
    sources: dict[tuple[str, str], BaseSQLSource],
    timeout: float | None = None,
) -> str:
    """
    Run a read-only SQL statement on the named Lumen source and return a text preview of results.

    Intended for LLM tool calls: the model supplies the logical ``source`` name and ``sql_query``;
    the ``sources`` map must be provided by the caller (e.g. closed over when building the tool).

    Parameters
    ----------
    source : str | None
        Name of the data source (key prefix before the source/table separator in slugs).
        May be omitted when there is a single source.
    sql_query : str
        SQL to execute (SELECT or WITH only).
    sources : dict[tuple[str, str], Source]
        Mapping from ``(source_name, table_name)`` to :class:`~lumen.sources.base.Source` instances.
    timeout : float, optional
        Seconds after which the query is abandoned and reported as an error.

    Returns
    -------
    str
        Tabular preview, or an error message string if execution fails.
    """
    try:
        base = resolve_exploration_source(source, sources)
    except ValueError as e:
        return f"{e} Tables: {', '.join(sorted({t for _, t in sources}))}."

    try:
        sql_clean = clean_sql(sql_query.strip(), base.dialect, prettify=False)
        validate_read_only_sql(sql_clean, base.dialect)
        limited = SQLLimit(limit=EXPLORATION_MAX_ROWS + 1, write=base.dialect).apply(sql_clean)
    except (ValueError, sqlglot.errors.ParseError, sqlglot.errors.TokenError) as e:
        return f"SQL parse/clean error: {format_error(e)}"

    try:
        df = await base.execute_with_timeout(limited, timeout, fetch=True)
    except Exception as e:
        return format_error(e)

    # A pyarrow Table has no .head(); narwhals gives every backend one.
    frame = as_narwhals(df)
    capped = len(frame) > EXPLORATION_MAX_ROWS
    return format_exploration_result(frame.head(EXPLORATION_MAX_ROWS) if capped else frame, capped=capped)


def _known_columns(metaset: Metaset | None, source: BaseSQLSource, table: str) -> list[str]:
    slug = f"{source.name}{SOURCE_TABLE_SEPARATOR}{table}"
    entry = metaset.catalog.get(slug) if metaset is not None else None
    if entry is not None and entry.columns:
        return [col.name for col in entry.columns]
    schema = (metaset.schemas or {}).get(slug) if metaset is not None else None
    return [key for key in (schema or {}) if key != "__len__"]


def build_distinct_values_sql(
    table_expr: str, column: str, dialect: str, like: str | None = None, limit: int = DISTINCT_VALUES_LIMIT
) -> str:
    """Top ``limit + 1`` values of ``column`` by frequency; the extra row reveals truncation."""
    read = None if dialect == "any" else dialect
    col = sqlglot.exp.column(column, quoted=True)
    count = sqlglot.exp.Count(this=sqlglot.exp.Star())
    query = (
        sqlglot.select(col.as_("value", quoted=True), count.as_("count", quoted=True))
        .from_(sqlglot.parse_one(table_expr, read=read).subquery("_t"))
        .group_by(col.copy())
        .order_by(
            sqlglot.exp.Ordered(this=sqlglot.exp.column("count", quoted=True), desc=True),
            sqlglot.exp.Ordered(this=sqlglot.exp.column("value", quoted=True)),
        )
        .limit(limit + 1)
    )
    if like:
        text = sqlglot.exp.Cast(this=col.copy(), to=sqlglot.exp.DataType.build("text"))
        query = query.where(sqlglot.exp.ILike(this=text, expression=sqlglot.exp.Literal.string(f"%{like}%")))
    return query.sql(dialect=read)


def _format_value(value: t.Any) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "NULL"
    # repr keeps padding and quotes visible, which is what the model needs to copy.
    return repr(value) if isinstance(value, str) else str(value)


async def execute_distinct_values(
    table: str,
    column: str,
    like: str | None = None,
    *,
    sources: dict[tuple[str, str], BaseSQLSource],
    metaset: Metaset | None = None,
    limit: int = DISTINCT_VALUES_LIMIT,
    timeout: float | None = None,
) -> str:
    try:
        source, table_name = resolve_source_table(table, sources)
    except ValueError as e:
        return str(e)
    known = _known_columns(metaset, source, table_name)
    if known and column not in known:
        matches = [name for name in known if name.lower() == column.strip().strip('"').lower()]
        if len(matches) != 1:
            return format_unknown_name("column", column, known)
        column = matches[0]
    withheld = f"The values of {column!r} are withheld: the column looks like personal data."
    if not shares_values(column):
        return withheld
    try:
        expr = source.get_sql_expr(table_name)
        if like and sharing.mask_pii:
            # Judge the column by its own values, not the ones matching like,
            # or "no values" against "withheld" would tell what it contains
            sample = as_pandas(await source.execute_with_timeout(
                build_distinct_values_sql(expr, column, source.dialect, None, limit), timeout
            ))
            if not shares_values(column, sample.iloc[:, 0].tolist()):
                return withheld
        sql = build_distinct_values_sql(expr, column, source.dialect, like, limit)
        df = as_pandas(await source.execute_with_timeout(sql, timeout))
    except Exception as e:
        return format_error(e)
    rows = list(df.itertuples(index=False, name=None))
    if not shares_values(column, [value for value, _ in rows]):
        return withheld
    truncated = len(rows) > limit
    rows = rows[:limit]
    matching = f" containing {like!r}" if like else ""
    if not rows:
        return f"No values of {column!r} in {table_name!r}{matching}."
    lines = [f"Values of {column!r} in {table_name!r}{matching}, most frequent first (value: rows):"]
    lines += [f"- {_format_value(value)}: {n}" for value, n in rows]
    if truncated:
        lines.append(f"More than {limit} values match; pass `like` to narrow the list.")
    return "\n".join(lines)
