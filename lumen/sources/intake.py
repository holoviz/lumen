import intake
import numpy as np
import param

from ..util import get_dataframe_schema
from .base import Source


class IntakeSource(Source):
    """
    An IntakeSource loads data from an Intake catalog.
    """

    catalog = param.String(doc="""
        The name of the catalog; must be provided.""")

    dask = param.Boolean(default=False, doc="""
        Whether to return a dask DataFrame.""")

    filename = param.String(doc="Filename of the catalog file.")

    index = param.List(default=[], doc="List of index columns.")

    variables = param.List(default=[], doc="List of data variable columns.")

    source_type = 'intake'

    def __init__(self, **params):
        super().__init__(**params)
        self.cat = intake.Catalog(self.filename)
        if not self.catalog:
            cats = list(self.cat)
            if len(cats) == 1:
                self.catalog = cats[0]
            else:
                raise ValueError("Could not determine catalog to load.")
        elif self.catalog not in self.cat:
            raise ValueError(f"'{self.catalog}' not found in catalog.'")

    def get_schema(self, variable=None):
        cat = self.cat[self.catalog].to_dask()
        if variable:
            return get_dataframe_schema(cat, self.index, variable)
        else:
            return {var: get_dataframe_schema(cat, self.index, var) for var in self.variables}

    def get(self, variable, **query):
        df = self.cat[self.catalog].to_dask()[self.index+[variable]]
        filters = []
        for k, val in query.items():
            if np.isscalar(val):
                filters.append(df[k] == val)
            elif isinstance(val, list):
                filters.append(df[k].isin(val))
            elif isinstance(val, tuple):
                filters.append((df[k]>val[0]) & (df[k]<=val[1]))
        if filters:
            mask = filters[0]
            for f in filters:
                mask &= f
            df = df[mask]
        return df if self.dask else df.compute()
