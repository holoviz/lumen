import param  # type: ignore

from .intake_sql import IntakeBaseSQLSource


class IntakeDremioSource(IntakeBaseSQLSource):
    """
    `IntakeDremioSource` allows querying Dremio tables and views.

    When provided with the `uri` of the Dremio server and credentials
    to authenticate with the Dremio instance all available tables can
    be queried via this `Source`.

    Requires the intake-dremio package to be installed.
    """

    cert = param.String(default="Path to certificate file")

    dask = param.Boolean(default=False, doc="""
        Whether to return a dask DataFrame.""")

    uri = param.String(doc="URI of the Dremio server.")

    tls = param.Boolean(default=False, doc="Enable encryption")

    username = param.String(default=None, doc="Dremio username")

    password = param.String(default=None, doc="Dremio password or token")

    source_type = 'intake_dremio'

    def __init__(self, **params):
        from intake_dremio.dremio_cat import DremioCatalog  # type: ignore
        super().__init__(**params)
        self.cat = DremioCatalog(
            self.uri, cert=self.cert, tls=self.tls, username=self.username,
            password=self.password
        )

    def _get_source(self, table):
        # Fuzzy matching to ignore quoting issues
        normalized_table = table.replace('"', '').lower()
        tables = self.get_tables()
        normalized_tables = [t.replace('"', '').lower() for t in tables]
        if table not in tables and normalized_table in normalized_tables:
            table = tables[normalized_tables.index(normalized_table)]
        return super()._get_source(table)
