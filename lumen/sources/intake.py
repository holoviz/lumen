import intake
import param

from ..util import get_dataframe_schema
from .base import Source, cached


class IntakeSource(Source):
    """
    An IntakeSource loads data from an Intake catalog.
    """

    catalog = param.String(doc="""
        The name of the catalog; must be provided.""")

    dask = param.Boolean(default=False, doc="""
        Whether to return a dask DataFrame.""")

    filename = param.String(doc="Filename of the catalog file.")

    source_type = 'intake'

    def __init__(self, **params):
        super().__init__(**params)
        self.cat = intake.open_catalog(self.filename)
        if not self.catalog:
            cats = list(self.cat)
            if len(cats) == 1:
                self.catalog = cats[0]
            else:
                raise ValueError("Could not determine catalog to load.")
        elif self.catalog not in self.cat:
            raise ValueError(f"'{self.catalog}' not found in catalog.'")

    def get_schema(self, table=None):
        if table:
            cat = self.cat[table].to_dask()
            return get_dataframe_schema(cat)['items']['properties']
        else:
            return {name: get_dataframe_schema(self.cat[name].to_dask())['items']['properties']
                    for name in list(self.cat)}

    @cached()
    def get(self, table, **query):
        df = self.cat[table].to_dask()
        df = self._filter_dataframe(df, **query)
        return df if self.dask else df.compute()
