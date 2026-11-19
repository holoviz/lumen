from types import SimpleNamespace

import pytest

pytest.importorskip("duckdb")

from lumen.flowdash import (
    CheckDiagnostic, check_columns, check_query, check_sql_config,
    check_view_config, extract_sql_error_location, nodes as flowdash_nodes,
)
from lumen.sources.duckdb import DuckDBSource


def test_check_diagnostic_is_str_subclass():
    diag = CheckDiagnostic(
        "Bad column",
        line=4,
        column=7,
        end_column=13,
        field="sql",
        suggestion="p_year",
    )

    assert isinstance(diag, str)
    assert str(diag) == "Bad column"
    assert diag == "Bad column"
    assert (diag.line, diag.column, diag.end_column) == (4, 7, 13)
    assert diag.field == "sql"
    assert diag.suggestion == "p_year"
    assert "line=4" in repr(diag)


async def test_sqlglot_parse_error_reports_multiline_position(source, monkeypatch):
    bad_sql = "SELECT *\nFROM sales\nWHERE foo = bar baz"

    diag = await check_sql_config({"sql": bad_sql}, {"source": source})

    assert isinstance(diag, CheckDiagnostic)
    assert isinstance(diag, str)
    assert diag.field == "sql"
    assert diag.line == 3
    assert diag.column == 17
    assert diag.end_column == 20

    monkeypatch.setattr(flowdash_nodes, "host_sources", lambda: {"data": source})
    unbound = await check_sql_config({"sql": bad_sql}, {})
    assert isinstance(unbound, CheckDiagnostic)
    assert (unbound.line, unbound.column, unbound.end_column, unbound.field) == (3, 17, 20, "sql")


async def test_duckdb_binder_error_reports_position_and_suggestion():
    years = DuckDBSource(
        uri=":memory:",
        tables={"sales": "SELECT 2024 AS p_year, 10 AS amount"},
    )
    sql = "SELECT\n  p_year,\n  amount\nFROM sales\nWHERE p_yaer IS NOT NULL"

    diag = await check_sql_config({"sql": sql}, {"source": years})

    assert isinstance(diag, CheckDiagnostic)
    assert isinstance(diag, str)
    assert diag.field == "sql"
    assert diag.line == 5
    assert diag.column == 7
    assert diag.end_column == 13
    assert diag.suggestion == "p_year"
    assert "LINE " not in diag
    assert "LIMIT 0" not in diag


async def test_sql_check_adjusts_for_leading_whitespace():
    years = DuckDBSource(
        uri=":memory:",
        tables={"sales": "SELECT 2024 AS p_year, 10 AS amount"},
    )
    sql = "\n\n  SELECT p_yaer\nFROM sales"

    diag = await check_sql_config({"sql": sql}, {"source": years})

    assert isinstance(diag, CheckDiagnostic)
    assert diag.line == 3
    assert diag.column == 10
    assert diag.end_column == 16
    assert diag.suggestion == "p_year"


async def test_check_query_semantic_errors_report_locations(source):
    undeclared_sql = (
        "SELECT *\n"
        "FROM sales\n"
        "JOIN missing_tbl\n"
        "  ON sales.category = missing_tbl.category"
    )
    undeclared = await check_sql_config({"sql": undeclared_sql}, {"source": source})
    assert isinstance(undeclared, CheckDiagnostic)
    assert undeclared.field == "sql"
    assert (undeclared.line, undeclared.column, undeclared.end_column) == (3, 6, 17)

    denied_sql = (
        "SELECT\n"
        "  category,\n"
        "  getenv('HOME') AS home\n"
        "FROM sales"
    )
    denied = await check_sql_config({"sql": denied_sql}, {"source": source})
    assert isinstance(denied, CheckDiagnostic)
    assert denied.field == "sql"
    assert (denied.line, denied.column, denied.end_column) == (3, 3, 9)

    table_fn_sql = "SELECT *\nFROM\n  read_csv('/etc/hosts')"
    table_fn = await check_sql_config({"sql": table_fn_sql}, {"source": source})
    assert isinstance(table_fn, CheckDiagnostic)
    assert table_fn.field == "sql"
    assert (table_fn.line, table_fn.column, table_fn.end_column) == (3, 3, 11)

    with pytest.raises(ValueError) as exc_info:
        check_query(undeclared_sql, source)
    assert exc_info.value.diagnostic.line == 3


async def test_check_columns_reports_unresolved_column_location():
    class UncancellableSource(DuckDBSource):
        cancels_queries = False

    uncancellable = UncancellableSource(
        uri=":memory:",
        tables={"sales": "SELECT 'A' AS category, 10 AS amount"},
    )
    uncancellable.get_schema("sales")
    sql = "SELECT\n  category,\n  price\nFROM sales"

    diag = await check_sql_config({"sql": sql}, {"source": uncancellable})

    assert isinstance(diag, CheckDiagnostic)
    assert isinstance(diag, str)
    assert diag.field == "sql"
    assert (diag.line, diag.column, diag.end_column) == (3, 3, 8)

    with pytest.raises(ValueError, match="price"):
        check_columns(sql, uncancellable)


def test_check_view_config_reports_a_spec_diagnostic():
    pipeline = SimpleNamespace(schema={"category": {}, "total": {}})
    bad_field = check_view_config(
        {"spec": {"mark": "bar", "encoding": {"x": {"field": "revenue", "type": "nominal"}}}},
        {"pipeline": pipeline},
    )
    assert isinstance(bad_field, CheckDiagnostic)
    assert isinstance(bad_field, str)
    assert bad_field.field == "spec"


def test_extract_sql_error_location_handles_comments_duplicates_and_truncation():
    sql = (
        "-- p_yaer in comment\n"
        "SELECT\n"
        "  'p_yaer' AS label,\n"
        "  p_year AS p_yaer\n"
        "FROM sales\n"
        "WHERE p_yaer = 1"
    )
    probe = (
        "SELECT 'p_yaer' AS label, p_year AS p_yaer FROM sales "
        "WHERE p_yaer = 1 LIMIT 0"
    )
    raw = (
        'Binder Error: Referenced column "p_yaer" not found in FROM clause!\n'
        'Candidate bindings: "p_year"\n\n'
        "LINE 1: ...bel, p_year AS p_yaer FROM sales WHERE p_yaer = 1 LIMIT 0\n"
        "                                                  ^^^^^^"
    )

    diag = extract_sql_error_location(raw, sql=sql, probe=probe, dialect="duckdb")

    assert isinstance(diag, CheckDiagnostic)
    assert (diag.line, diag.column, diag.end_column) == (6, 7, 13)
    assert diag.suggestion == "p_year"
    assert diag.field == "sql"
    assert "LINE 1:" not in diag

    direct_raw = (
        'Binder Error: Table "s" does not have a column named "p_yaer"\n\n'
        'Candidate bindings: : "p_year"\n\n'
        "LINE 2: SELECT s.p_yaer FROM sales AS s\n"
        "               ^^^^^^^^"
    )
    direct = extract_sql_error_location(direct_raw)
    assert (direct.line, direct.column, direct.end_column) == (2, 8, 16)
    assert direct.suggestion == "p_year"
    assert "LINE 2:" not in direct
