import intake  # type: ignore
import param  # type: ignore

from ..util import get_dataframe_schema
from .base import Source, cached, cached_schema


class IntakeBaseSource(Source):

    cache_per_query = param.Boolean(default=False, doc="""
        Whether to query the whole dataset or individual queries.""")

    load_schema = param.Boolean(default=True, doc="""
        Whether to load the schema""")

    __abstract = True

    def _read(self, entry, dask=True):
        if self.dask or dask:
            try:
                return entry.to_dask()
            except Exception:
                if self.dask:
                    self.param.warning(
                        f"Could not load {entry.name!r} table with dask."
                    )
        return entry.read()

    def get_tables(self):
        return list(self.cat)

    @cached_schema
    def get_schema(self, table=None):
        schemas = {}
        for entry in list(self.cat):
            if table is not None and entry != table:
                continue
            elif not self.load_schema:
                schemas[entry] = {}
                continue
            data = self.get(entry, __dask=True)
            schemas[entry] = get_dataframe_schema(data)['items']['properties']
        return schemas if table is None else schemas[table]

    @cached
    def get(self, table, **query):
        dask = query.pop('__dask', self.dask)
        try:
            entry = self.cat[table]
        except KeyError:
            raise KeyError(f"'{table}' table could not be found in Intake "
                           "catalog. Available tables include: "
                           f"{list(self.cat)}.")
        df = self._read(entry, dask)
        return df if dask or not hasattr(df, 'compute') else df.compute()


class IntakeSource(IntakeBaseSource):
    """
    An `IntakeSource` loads data from an Intake catalog.

    Intake is a lightweight set of tools for loading and sharing data
    in data science projects using convenient catalog specifications.

    The `IntakeSource` can be given a dictionary `catalog` specification
    OR a URI pointing to a catalog.yaml file on disk.
    """

    catalog = param.Dict(doc="An inlined Catalog specification.")

    dask = param.Boolean(default=False, doc="""
        Whether to return a dask DataFrame.""")

    uri = param.String(doc="URI of the catalog file.")

    source_type = 'intake'

    def __init__(self, **params):
        super().__init__(**params)
        if self.uri and self.catalog:
            raise ValueError(
                "Either specify a Catalog uri or an inlined catalog, not both."
            )
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
