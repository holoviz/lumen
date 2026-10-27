import asyncio
import importlib
import warnings

import pandas as pd
import param
import pytest

from lumen.sources.base import BaseSQLSource, FileSource, InMemorySource
from lumen.sources.guard import (
    RAW_ALIAS, GuardPlan, guard_query, hash_value,
)

try:
    from lumen.sources.duckdb import DuckDBSource
except ImportError:
    DuckDBSource = None

try:
    from lumen.ai.config import GuardrailViolation
    from lumen.ai.guardrails import Guardrail
except ImportError:
    Guardrail = None

needs_duckdb = pytest.mark.skipif(DuckDBSource is None, reason="duckdb is not installed")
needs_ai = pytest.mark.skipif(Guardrail is None, reason="lumen.ai could not be imported")

PLAN = GuardPlan(columns=("id", "ssn"), masked={"ssn": ["mask_columns"]})

SQL_SOURCES = [
    ("lumen.sources.duckdb", "DuckDBSource"),
    ("lumen.sources.sqlalchemy", "SQLAlchemySource"),
    ("lumen.sources.snowflake", "SnowflakeSource"),
    ("lumen.sources.bigquery", "BigQuerySource"),
    ("lumen.sources.xarray_sql", "XArraySQLSource"),
]


if Guardrail is not None:
    class AtGuardrail(Guardrail):
        """Flags any text containing "@", standing in for a PII detector."""

        fail = param.Boolean(default=False)

        def __init__(self, **params):
            super().__init__(**params)
            self.seen = []

        async def check(self, text, mode):
            self.seen.append((mode, text))
            if self.fail:
                raise ConnectionError("API down")
            return ["pii"] if "@" in text else []


@pytest.fixture
def customers():
    return pd.DataFrame({
        "id": [1, 2, 3],
        "email": ["a@x.com", "b@x.com", "a@x.com"],
        "ssn": ["900-00-0001", "900-00-0002", "900-00-0003"],
        "plan": ["basic", "pro", "basic"],
    })


@pytest.fixture
def guardrail():
    if Guardrail is None:
        pytest.skip("lumen.ai could not be imported")
    return AtGuardrail()


@pytest.fixture
def guarded(customers, guardrail):
    if DuckDBSource is None:
        pytest.skip("duckdb is not installed")
    return DuckDBSource.from_df({"customers": customers}, guardrails=[guardrail], mask_columns=["SSN"])


def test_get_masks_flagged_and_listed_columns(guarded, customers):
    df = guarded.get("customers")
    assert df["email"].tolist() == [hash_value("a@x.com"), hash_value("b@x.com"), hash_value("a@x.com")]
    assert df["ssn"].tolist() == [hash_value(v) for v in customers["ssn"]]
    assert df["id"].tolist() == [1, 2, 3]
    assert df["plan"].tolist() == customers["plan"].tolist()
    assert guarded.guarded_columns("customers") == {"email": ["pii"], "ssn": ["mask_columns"]}


def test_sql_hash_matches_python_hash(guarded):
    """Equal values hash equally across get, SQL and the pandas path."""
    df = guarded.execute("SELECT email FROM customers WHERE id = 1")
    assert df["email"].iloc[0] == hash_value("a@x.com")


def test_aliased_and_joined_columns_stay_masked(guarded):
    df = guarded.execute(
        "SELECT a.email AS contact, b.ssn AS id_number, count(*) AS n "
        "FROM customers a JOIN customers AS b ON a.id = b.id GROUP BY 1, 2 ORDER BY 1, 2"
    )
    assert "a@x.com" not in df["contact"].tolist()
    assert set(df["contact"]) == {hash_value("a@x.com"), hash_value("b@x.com")}
    assert not df["id_number"].str.startswith("900").any()


def test_subqueries_and_ctes_read_masked_values(guarded):
    df = guarded.execute(
        "WITH c AS (SELECT email FROM customers) "
        "SELECT email FROM c WHERE email IN (SELECT email FROM customers)"
    )
    assert set(df["email"]) == {hash_value("a@x.com"), hash_value("b@x.com")}


def test_schema_enums_are_masked(guarded):
    schema = guarded.get_schema("customers")
    assert set(schema["email"]["enum"]) == {hash_value("a@x.com"), hash_value("b@x.com")}
    assert all(not v.startswith("900") for v in schema["ssn"]["enum"])
    assert set(schema["plan"]["enum"]) == {"basic", "pro"}


def test_async_paths_are_masked(guarded):
    df = asyncio.run(guarded.execute_async("SELECT email FROM customers"))
    assert "a@x.com" not in df["email"].tolist()
    df = asyncio.run(guarded.execute_with_timeout("SELECT ssn FROM customers", 10, fetch=True))
    assert not df["ssn"].str.startswith("900").any()
    df = asyncio.run(guarded.get_async("customers"))
    assert "a@x.com" not in df["email"].tolist()


def test_columns_are_checked_once(guarded, guardrail):
    for _ in range(3):
        guarded.get_schema("customers")
        guarded.execute("SELECT * FROM customers")
    assert len(guardrail.seen) == 4  # one text per column, from one sample


def test_derived_tables_are_not_masked_twice(guarded):
    derived = guarded.create_sql_expr_source(
        {"by_email": "SELECT email, count(*) AS n FROM customers GROUP BY email"}
    )
    df = derived.get("by_email").sort_values("n")
    assert df["email"].tolist() == [hash_value("b@x.com"), hash_value("a@x.com")]
    # Joining a derived table back to its source still matches.
    joined = derived.execute("SELECT c.id FROM by_email b JOIN customers c ON b.email = c.email")
    assert sorted(joined["id"]) == [1, 2, 3]


def test_derived_table_named_like_a_guarded_one_stays_guarded(guarded):
    guarded._derive_guard(["Customers", "by_plan"])
    assert guarded._guard_state.derived == {"by_plan"}
    assert "a@x.com" not in guarded.execute("SELECT email FROM customers")["email"].tolist()


def test_derived_view_reads_masked_tables(guarded):
    derived = guarded.create_sql_expr_source(
        {"by_email": "SELECT email, count(*) AS n FROM customers GROUP BY email"}, materialize=False
    )
    df = derived.connection.execute("SELECT email FROM by_email").fetchdf()
    assert "a@x.com" not in df["email"].tolist()


@needs_duckdb
def test_table_functions_are_masked(tmp_path, customers):
    """Reading the file behind a table does not get around the guard."""
    path = tmp_path / "customers.csv"
    customers.to_csv(path, index=False)
    source = DuckDBSource(tables={"customers": f"SELECT * FROM read_csv('{path}')"}, mask_columns=["ssn"])
    assert not source.get("customers")["ssn"].str.startswith("900").any()
    direct = source.execute(f"SELECT ssn FROM read_csv('{path}')")
    assert not direct["ssn"].str.startswith("900").any()


@needs_duckdb
@pytest.mark.parametrize("mask", ["redact", "drop"])
def test_mask_modes(customers, mask):
    source = DuckDBSource.from_df({"customers": customers}, mask_columns=["ssn"], mask=mask)
    df = source.get("customers")
    if mask == "drop":
        assert "ssn" not in df.columns
        assert "ssn" not in source.get_schema("customers")
    else:
        assert df["ssn"].isna().all()


@needs_duckdb
def test_unmask_columns_wins(customers, guardrail):
    source = DuckDBSource.from_df({"customers": customers}, guardrails=[guardrail], unmask_columns=["email"])
    assert source.get("customers")["email"].tolist() == customers["email"].tolist()


@needs_duckdb
@needs_ai
def test_unreachable_guardrail_blocks(customers):
    source = DuckDBSource.from_df({"customers": customers}, guardrails=[AtGuardrail(fail=True)])
    with pytest.raises(GuardrailViolation, match="guardrail unavailable"):
        source.get("customers")


@needs_duckdb
@needs_ai
def test_unreachable_guardrail_allowed_masks_listed_columns(customers):
    source = DuckDBSource.from_df(
        {"customers": customers}, guardrails=[AtGuardrail(fail=True, on_error="allow")], mask_columns=["ssn"]
    )
    df = source.get("customers")
    assert df["email"].tolist() == customers["email"].tolist()
    assert not df["ssn"].str.startswith("900").any()


@needs_duckdb
def test_unguarded_source_runs_sql_as_given(customers, monkeypatch):
    from lumen.sources import base
    monkeypatch.setattr(base, "guard_query", lambda *a, **k: pytest.fail("guard ran"))
    source = DuckDBSource.from_df({"customers": customers})
    assert source.execute("SELECT ssn FROM customers")["ssn"].tolist() == customers["ssn"].tolist()


@needs_duckdb
def test_guarded_source_refuses_unparseable_sql(customers):
    source = DuckDBSource.from_df({"customers": customers}, mask_columns=["ssn"])
    with pytest.raises(ValueError, match="can parse"):
        source.execute("SELECT ssn FROM customers WHERE (")


def test_guard_query_masks_what_ddl_reads_not_its_target():
    sql = guard_query("CREATE TABLE t AS SELECT * FROM t", "duckdb", lambda key, table: PLAN, "hash")
    assert sql.startswith('CREATE TABLE t AS SELECT * FROM (SELECT "id", SUBSTRING(MD5(')


def test_guard_query_leaves_cte_references():
    sql = "WITH t AS (SELECT 1 AS ssn) SELECT ssn FROM t"
    assert guard_query(sql, "duckdb", lambda key, table: PLAN, "hash") == sql


def test_guard_query_masks_a_cte_shadowing_its_table():
    sql = guard_query("WITH t AS (SELECT * FROM t) SELECT ssn FROM t", "duckdb", lambda key, table: PLAN, "hash")
    assert sql.count("MD5") == 1 and sql.endswith("SELECT ssn FROM t")


def test_guard_query_is_idempotent():
    once = guard_query("SELECT ssn FROM t", "duckdb", lambda key, table: PLAN, "hash")
    assert RAW_ALIAS in once and "MD5" in once
    assert guard_query(once, "duckdb", lambda key, table: PLAN, "hash") == once


def test_guard_query_leaves_system_tables():
    sql = "SELECT table_name FROM information_schema.tables"
    assert guard_query(sql, "duckdb", lambda key, table: pytest.fail(key), "hash") == sql


@pytest.mark.parametrize(
    ("value", "expected"),
    [(None, None), (float("nan"), None), (pd.NaT, None), ("a@x.com", hash_value("a@x.com")), ([1, 2], hash_value("[1, 2]"))],
    ids=["none", "nan", "nat", "string", "list"],
)
def test_hash_value_keeps_missing_values_missing(value, expected):
    """Nulls stay null so counts of missing values survive; non-scalar cells are hashed, not an error."""
    assert hash_value(value) == expected


def test_in_memory_source_masks(customers):
    source = InMemorySource(tables={"customers": customers}, mask_columns=["ssn"], mask="drop")
    assert "ssn" not in source.get("customers").columns
    assert "ssn" not in source.get_schema("customers")


def test_file_source_masks_before_caching(tmp_path, customers):
    path = tmp_path / "customers.csv"
    customers.to_csv(path, index=False)
    source = FileSource(tables={"customers": str(path)}, mask_columns=["ssn"], root=tmp_path)
    df = source.get("customers")
    assert df["ssn"].tolist() == [hash_value(v) for v in customers["ssn"]]
    assert not any(str(v).startswith("900") for v in source.get_schema("customers")["ssn"].get("enum", []))


def test_sqlalchemy_source_masks(tmp_path, customers):
    pytest.importorskip("sqlalchemy")
    from lumen.sources.sqlalchemy import SQLAlchemySource
    url = f"sqlite:///{tmp_path / 'customers.db'}"
    customers.to_sql("customers", url, index=False)
    # SQLite has no MD5, so redact.
    source = SQLAlchemySource(url=url, mask_columns=["ssn"], mask="redact")
    assert source.get("customers")["ssn"].isna().all()
    assert source.execute("SELECT ssn AS x FROM customers")["x"].isna().all()


@pytest.mark.parametrize("module,name", SQL_SOURCES)
def test_every_query_method_is_guarded(module, name):
    """A SQL source method that runs a query without the guard would leak raw values."""
    try:
        with warnings.catch_warnings():
            # Some connectors warn about their vendored dependencies on import.
            warnings.simplefilter("ignore")
            cls = getattr(importlib.import_module(module), name)
    except ImportError:
        pytest.skip(f"{module} could not be imported")
    assert issubclass(cls, BaseSQLSource)
    for method in ("execute", "execute_async", "fetch", "execute_with_timeout", "get_sql_expr"):
        assert getattr(getattr(cls, method), "__lumen_guarded__", False), f"{name}.{method} is not guarded"
