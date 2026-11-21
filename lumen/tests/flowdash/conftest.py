import pytest

TABLES = {"sales": "SELECT * FROM (VALUES ('A', 10), ('B', 20), ('A', 5)) AS t(category, amount)"}


@pytest.fixture
def executed(monkeypatch):
    """The statements DuckDB sources fetch results for, whitespace collapsed."""
    duckdb_source = pytest.importorskip("lumen.sources.duckdb")
    queries = []
    fetch = duckdb_source.DuckDBSource._fetch_df

    def logged(self, cursor, sql, *args, **kwargs):
        queries.append(" ".join(str(sql).split()))
        return fetch(self, cursor, sql, *args, **kwargs)

    monkeypatch.setattr(duckdb_source.DuckDBSource, "_fetch_df", logged)
    return queries


@pytest.fixture
def source():
    duckdb_source = pytest.importorskip("lumen.sources.duckdb")
    return duckdb_source.DuckDBSource(uri=":memory:", tables=TABLES)
