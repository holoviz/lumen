"""Source + SQL tools: connect a DuckDB workspace, inspect it, and run SQL.

The session workspace is an in-memory Lumen ``DuckDBSource``. Connecting a file loads its tables into
that workspace (the original file is never modified). Every SQL result is materialized as a named
table via ``create_sql_expr_source(materialize=True)`` and referenced by name; results are read back
through ``source.get(table)`` (which resolves both real and derived tables).

Note: files are read fully into the in-memory workspace. Fine for typical analysis sizes; switching
large on-disk sources to DuckDB ``ATTACH`` is a later enhancement.
"""

from __future__ import annotations

import json
import os
from typing import Optional

import duckdb
import pandas as pd
from lumen.sources.duckdb import DuckDBSource

from .session import session

_DB_SUFFIXES = (".db", ".duckdb", ".ddb")


def _stem(uri: str) -> str:
    return os.path.splitext(os.path.basename(uri))[0]


def _records(df: pd.DataFrame, limit: Optional[int] = None) -> list:
    """JSON-safe list of row dicts (handles dates and numpy dtypes)."""
    if limit is not None:
        df = df.head(limit)
    return json.loads(df.to_json(orient="records", date_format="iso"))


def _load_db(uri: str) -> DuckDBSource:
    con = duckdb.connect(uri, read_only=True)
    try:
        names = [row[0] for row in con.execute("SHOW TABLES").fetchall()]
        frames = {name: con.execute(f'SELECT * FROM "{name}"').fetch_df() for name in names}
    finally:
        con.close()
    if not frames:
        raise ValueError(f"No tables found in {uri!r}.")
    return DuckDBSource.from_df(frames)


def connect_source(uri: str, name: Optional[str] = None) -> dict:
    """Connect a data source and make it the session workspace.

    ``uri`` may be a DuckDB database (``.db``/``.duckdb``), a data file (``.csv``/``.parquet``/
    ``.json``), or ``":memory:"``. Returns the available table names.
    """
    if uri == ":memory:":
        source = DuckDBSource(uri=":memory:", ephemeral=True)
    elif uri.endswith(_DB_SUFFIXES):
        source = _load_db(uri)
    elif uri.endswith(".parquet"):
        source = DuckDBSource.from_df({name or _stem(uri): pd.read_parquet(uri)})
    elif uri.endswith(".csv"):
        source = DuckDBSource.from_df({name or _stem(uri): pd.read_csv(uri)})
    elif uri.endswith(".json"):
        source = DuckDBSource.from_df({name or _stem(uri): pd.read_json(uri)})
    else:
        raise ValueError(
            f"Unsupported source {uri!r}. Use a .db/.duckdb, .csv, .parquet or .json path, "
            "or ':memory:'."
        )
    session.source = source
    return {"connected": uri, "tables": source.get_tables()}


def list_tables() -> dict:
    """List the tables in the current workspace."""
    return {"tables": session.require_source().get_tables()}


def describe_table(table: str) -> dict:
    """Return columns, dtypes, row count, and a small sample (context for writing SQL)."""
    df = session.require_source().get(table)
    dtypes = {col: str(dtype) for col, dtype in zip(df.columns, df.dtypes)}
    return {
        "table": table,
        "columns": list(df.columns),
        "dtypes": dtypes,
        "row_count": int(len(df)),
        "sample": _records(df, limit=5),
    }


def run_sql(sql: str, name: Optional[str] = None) -> dict:
    """Execute SQL against the workspace; the result becomes a named table.

    Returns the table name, row count, columns, and a sample (first 20 rows). Chart it by passing
    the returned ``table`` to ``render_vegalite``.
    """
    source = session.require_source()
    table = name or session.next_result_name()
    session.source = source.create_sql_expr_source({table: sql}, materialize=True)
    df = session.source.get(table)
    return {
        "table": table,
        "row_count": int(len(df)),
        "columns": list(df.columns),
        "sample": _records(df, limit=20),
    }
