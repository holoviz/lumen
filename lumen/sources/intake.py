import intake
import param

from ..util import get_dataframe_schema
from .base import Source, cached


class IntakeBaseSource(Source):

    load_schema = param.Boolean(default=True, doc="""
        Whether to load the schema""")

    __abstract = True

    def _read(self, table, dask=True):
        try:
            entry = self.cat[table]
        except KeyError:
            raise KeyError(f"'{table}' table could not be found in Intake "
                           "catalog. Available tables include: "
                           f"{list(self.cat)}.")
        if self.dask or dask:
            try:
                return entry.to_dask()
            except Exception:
                if self.dask:
                    self.param.warning(f"Could not load {table} table with dask.")
                pass
        return entry.read()

    def get_tables(self):
        return list(self.cat)

    def get_schema(self, table=None):
        schemas = {}
        for entry in list(self.cat):
            if table is not None and entry != table:
                continue
            data = self.get(entry, __dask=True)
            if self.load_schema:
                schemas[entry] = {}
            else:
                schemas[entry] = get_dataframe_schema(data)['items']['properties']
        return schemas if table is None else schemas[table]

    @cached(with_query=False)
    def get(self, table, **query):
        dask = query.pop('__dask', self.dask)
        df = self._read(table)
        return df if dask or not hasattr(df, 'compute') else df.compute()


class IntakeSource(IntakeBaseSource):
    """
    An IntakeSource loads data from an Intake catalog.
    """

    catalog = param.Dict(doc="An inlined Catalog specification.")

    dask = param.Boolean(default=False, doc="""
        Whether to return a dask DataFrame.""")

    uri = param.String(doc="URI of the catalog file.")

    source_type = 'intake'

    def __init__(self, **params):
        super().__init__(**params)
        if self.uri and self.catalog:
            raise ValueError("Either specify a Catalog uri or an "
                             "inlined catalog, not both.")
        elif self.uri:
            self.cat = intake.open_catalog(self.uri)
        elif self.catalog:
            context = {'root': self.root}
            result = intake.catalog.local.CatalogParser(self.catalog, context=context)
            if result.errors:
                raise intake.catalog.exceptions.ValidationError(
                    "Catalog '{}' has validation errors:\n\n{}"
                    "".format(self.catalog, "\n".join(result.errors)), result.errors)
            cfg = result.data
            del cfg['plugin_sources']
            entries = {entry.name: entry for entry in cfg.pop('data_sources')}
            self.cat = intake.catalog.Catalog(**cfg)
            self.cat._entries = entries


class IntakeDremioSource(IntakeBaseSource):

    cert = param.String(default="Path to certificate file")

    dask = param.Boolean(default=False, doc="""
        Whether to return a dask DataFrame.""")

    uri = param.String(doc="URI of the catalog file.")

    tls = param.Boolean(default=False, doc="Enable encryption")

    source_type = 'intake_dremio'

    def __init__(self, **params):
        from intake_dremio.dremio_cat import DremioCatalog
        super().__init__(**params)
        self.cat = DremioCatalog(self.uri, cert=self.cert, tls=self.tls)
