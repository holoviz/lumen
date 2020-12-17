import intake
import param

from ..util import get_dataframe_schema
from .base import Source, cached


class IntakeSource(Source):
    """
    An IntakeSource loads data from an Intake catalog.
    """

    dask = param.Boolean(default=False, doc="""
        Whether to return a dask DataFrame.""")

    filename = param.String(doc="Deprecated alias for the URI")

    uri = param.String(doc="URI of the catalog file.")

    specification = param.Dict(default="An inlined Catalog specification.")

    source_type = 'intake'

    def __init__(self, **params):
        super().__init__(**params)
        if self.filename:
            self.param.warning("IntakeSource filename parameter is "
                               "deprecated in favor of uri parameter.")
            self.uri = self.filename
        if self.uri and self.specification:
            raise ValueError("Either specify a Catalog uri or an "
                             "inlined catalog specification, not both.")
        elif self.uri:
            self.cat = intake.open_catalog(self.uri)
        elif self.specification:
            result = intake.catalog.local.CatalogParser(self.specification)
            if result.errors:
                raise intake.catalog.exceptions.ValidationError(
                    "Catalog '{}' has validation errors:\n\n{}"
                    "".format(self.path, "\n".join(result.errors)), result.errors)
            cfg = result.data
            self.cat = Catalog(cfg.pop('data_sources'), **cfg)

    def _read(self, table, dask=True):
        if self.dask or dask:
            try:
                return self.cat[table].to_dask()
            except Exception:
                if self.dask:
                    self.param.warning(f"Could not load {table} table with dask.")
                pass
        return self.cat[table].read()

    def get_schema(self, table=None):
        if table:
            cat = self.get(table, dask=True)
            return get_dataframe_schema(cat)['items']['properties']
        else:
            return {name: get_dataframe_schema(self.get(name, dask=True))['items']['properties']
                    for name in list(self.cat)}

    @cached()
    def get(self, table, **query):
        dask = query.pop('dask', self.dask)
        df = self._read(table)
        df = self._filter_dataframe(df, **query)
        return df if dask or not hasattr(df, 'compute') else df.compute()
