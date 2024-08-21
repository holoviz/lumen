from __future__ import annotations

import param  # type: ignore

from .intake_sql import IntakeBaseSQLSource


class IntakeBaseDremioSource(IntakeBaseSQLSource):
    """
    Base class with common parameters for Dremio sources.
    """

    cert = param.String(default="Path to certificate file")

    dask = param.Boolean(default=False, doc="""
        Whether to return a dask DataFrame.""")

    uri = param.String(doc="URI of the Dremio server.")

    tls = param.Boolean(default=False, doc="Enable encryption")

    username = param.String(default=None, doc="Dremio username")

    password = param.String(default=None, doc="Dremio password or token")

    dialect = 'dremio'

    def _get_source(self, table):
        # Fuzzy matching to ignore quoting issues
        normalized_table = table.replace('"', '').lower()
        tables = self.get_tables()
        normalized_tables = [t.replace('"', '').lower() for t in tables]
        if table not in tables and normalized_table in normalized_tables:
            table = tables[normalized_tables.index(normalized_table)]
        return super()._get_source(table)

    def __contains__(self, table):
        try:
            self._get_source(table)
            return True
        except KeyError:
            return False

    def create_sql_expr_source(self, tables: dict[str, str], **kwargs):
        """
        Creates a new SQL Source given a set of table names and
        corresponding SQL expressions.
        """
        params = dict(self.param.values(), **kwargs)
        params['tables'] = tables
        return IntakeDremioSQLSource(**params)


class IntakeDremioSource(IntakeBaseDremioSource):
    """
    `IntakeDremioSource` allows querying Dremio catalog tables and views.

    When provided with the `uri` of the Dremio server and credentials
    to authenticate with the Dremio instance all available tables can
    be queried via this `Source`.

    Requires the intake-dremio package to be installed.
    """

    source_type = 'intake_dremio'

    def __init__(self, **params):
        from intake_dremio.dremio_cat import DremioCatalog  # type: ignore
        super().__init__(**params)
        self.cat = DremioCatalog(
            self.uri, cert=self.cert, tls=self.tls, username=self.username,
            password=self.password
        )


class IntakeDremioSQLSource(IntakeBaseDremioSource):
    """
    `IntakeDremioSQLSource` allows querying a subset of
    Dremio catalog tables and views via custom SQL expressions.

    When provided with the `uri` of the Dremio server and credentials
    to authenticate with the Dremio instance, unlike `IntakeDremioSource`,
    only the tables specified in the `tables` parameter can be used.

    Requires the intake-dremio package to be installed.
    """

    tables = param.Dict(default={}, doc="""
        Mapping of table names to desired SQL expressions""")

    source_type = 'intake_dremio_sql'

    def __init__(self, **params):
        from intake_dremio.intake_dremio import DremioSource
        super().__init__(**params)
        self.cat = {
            table: DremioSource(
                sql_expr=sql_expr,
                cert=self.cert,
                uri=self.uri,
                tls=self.tls,
                username=self.username,
                password=self.password,
            )
            for table, sql_expr in self.tables.items()
        }
