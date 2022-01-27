import param

from .intake_sql import IntakeBaseSQLSource


class IntakeDremioSource(IntakeBaseSQLSource):

    cert = param.String(default="Path to certificate file")

    dask = param.Boolean(default=False, doc="""
        Whether to return a dask DataFrame.""")

    uri = param.String(doc="URI of the catalog file.")

    tls = param.Boolean(default=False, doc="Enable encryption")

    username = param.String(default=None, doc="Dremio username")

    password = param.String(default=None, doc="Dremio password or token")

    source_type = 'intake_dremio'

    def __init__(self, **params):
        from intake_dremio.dremio_cat import DremioCatalog
        super().__init__(**params)
        self.cat = DremioCatalog(
            self.uri, cert=self.cert, tls=self.tls, username=self.username,
            password=self.password
        )
