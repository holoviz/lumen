import pytest

try:
    import intake_dremio
except Exception:
    pytest.skip('intake-dremio not available.', allow_module_level=True)

from lumen.sources.intake_dremio import IntakeDremioSQLSource


def test_intake_dremio_resolve_module_type():
    assert IntakeDremioSQLSource._get_type('lumen.sources.intake_dremio.IntakeDremioSQLSource') is IntakeDremioSQLSource
    assert IntakeDremioSQLSource.source_type == 'intake_dremio_sql'


def test_intake_dremio_init():
    credentials = dict(
        cert="./test.ipynb",
        uri="uri",
        tls=True,
        username="username",
        password="password",
    )
    source = IntakeDremioSQLSource(
        tables={"table": "SELECT * FROM table", "table2": "SELECT * FROM table2 LIMIT 3;"},
        **credentials
    )
    assert source.tables == {"table": "SELECT * FROM table", "table2": "SELECT * FROM table2 LIMIT 3;"}
    assert source.cat == {
        "table": intake_dremio.DremioSource(
            sql_expr="SELECT * FROM table",
            **credentials
        ),
        "table2": intake_dremio.DremioSource(
            sql_expr="SELECT * FROM table2 LIMIT 3;",
            **credentials
        )
    }
