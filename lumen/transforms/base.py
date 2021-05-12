"""
The Transform components allow transforming tables in arbitrary ways.
"""

import datetime as dt

import numpy as np
import pandas as pd
import param

from ..util import resolve_module_reference


class Transform(param.Parameterized):
    """
    A Transform provides the ability to transform a table supplied by
    a Source.
    """

    transform_type = None

    __abstract = True

    @classmethod
    def _get_type(cls, transform_type):
        if '.' in transform_type:
            return resolve_module_reference(transform_type, Transform)
        try:
            __import__(f'lumen.transforms.{transform_type}')
        except Exception:
            pass
        for transform in param.concrete_descendents(cls).values():
            if transform.transform_type == transform_type:
                return transform
        raise ValueError(f"No Transform for transform_type '{transform_type}' could be found.")

    @classmethod
    def from_spec(cls, spec):
        """
        Resolves a Transform specification.

        Parameters
        ----------
        spec: dict
            Specification declared as a dictionary of parameter values.

        Returns
        -------
        The resolved Transform object.
        """
        spec = dict(spec)
        transform_type = Transform._get_type(spec.pop('type', None))
        return transform_type(**spec)

    def apply(self, table):
        """
        Given a table transform it in some way and return it.

        Parameters
        ----------
        table : DataFrame
            The queried table as a DataFrame.

        Returns
        -------
        DataFrame
            A DataFrame containing the transformed data.
        """
        return table


class HistoryTransform(Transform):
    """
    The HistoryTransform accumulates a history of the queried data in
    a buffer up to the supplied length and (optionally) adds a
    date_column to the data.
    """

    date_column = param.String(doc="""
        If defined adds a date column with the supplied name.""")

    length = param.Integer(default=10, bounds=(1, None), doc="""
        Accumulates a history of data.""")

    transform_type = 'history'

    def __init__(self, **params):
        super().__init__(**params)
        self._buffer = []

    def apply(self, table):
        """
        Accumulates a history of the data in a buffer up to the
        declared `length` and optionally adds the current datetime to
        the declared `date_column`.

        Parameters
        ----------
        data : DataFrame
            The queried table as a DataFrame.

        Returns
        -------
        DataFrame
            A DataFrame containing the buffered history of the data.
        """
        if self.date_column:
            table = table.copy()
            table[self.date_column] = dt.datetime.now()
        self._buffer.append(table)
        self._buffer[:] = self._buffer[-self.length:]
        return pd.concat(self._buffer)


class Aggregate(Transform):
    """
    Aggregate one or more columns or indexes, see `pandas.DataFrame.groupby`.

    df.groupby(<by>)[<columns>].<method>()[.reset_index()]
    """

    by = param.ClassSelector(class_=(list, str, int), doc="""
        Columns or indexes to group by.""")

    columns = param.List(doc="""
        Columns to aggregate.""")

    with_index = param.Boolean(default=True, doc="""
        Whether to make the groupby columns indexes.""")

    method = param.String(default=None, doc="""
        Name of aggregation method.""")

    kwargs = param.Dict(default={}, doc="""
        Keyword arguments to the aggregation method.""")

    transform_type = 'aggregate'

    def apply(self, table):
        grouped = table.groupby(self.by)
        if self.columns:
            grouped = grouped[self.columns]
        agg = getattr(grouped, self.method)(**self.kwargs)
        return agg if self.with_index else agg.reset_index()


class Sort(Transform):
    """
    Sort on one or more columns, see `pandas.DataFrame.sort_values`.

    df.sort_values(<by>, ascending=<ascending>)
    """

    by = param.ClassSelector(class_=(list, str, int), doc="""
       Columns or indexes to sort by.""")

    ascending = param.ClassSelector(default=True, class_=(bool, list), doc="""
       Sort ascending vs. descending. Specify list for multiple sort
       orders. If this is a list of bools, must match the length of
       the by.""")

    transform_type = 'sort'

    def apply(self, table):
        return table.sort_values(self.by, ascending=self.ascending)


class Query(Transform):
    """
    Applies the `pandas.DataFrame.query` method.

    df.query(<query>)
    """

    query = param.String(doc="""
        The query to apply to the table.""")

    transform_type = 'query'

    def apply(self, table):
        return table.query(self.query)


class Columns(Transform):
    """
    Selects a subset of columns.

    df[<columns>]
    """

    columns = param.List(doc="""
        The subset of columns to select.""")

    transform_type = 'columns'

    def apply(self, table):
        return table[self.columns]


class Stack(Transform):
    """
    Stacks the declared level, see `pandas.DataFrame.stack`.

    df.stack(<level>)
    """

    level = param.ClassSelector(class_=(int, list, str), doc="""
        The indexes to stack.""")

    transform_type = 'stack'

    def apply(self, table):
        return table.stack(self.level)


class Unstack(Transform):
    """
    Unstacks the declared level(s), see `pandas.DataFrame.stack`.

    df.unstack(<level>)
    """

    level = param.ClassSelector(class_=(int, list, str), doc="""
        The indexes to unstack.""")

    transform_type = 'unstack'

    def apply(self, table):
        return table.unstack(self.level)


class Iloc(Transform):
    """
    Applies integer slicing to the data, see `pandas.DataFrame.iloc`.

    df.iloc[<start>:<end>]
    """

    start = param.Integer(default=None)

    end = param.Integer(default=None)

    transform_type = 'iloc'

    def apply(self, table):
        return table.iloc[self.start:self.end]


class Sample(Transform):
    """
    Random sample of items.

    df.sample(n=<n>, frac=<frac>, replace=<replace>)
    """

    n = param.Integer(default=None, doc="""
        Number of items to return.""")

    frac = param.Number(default=None, bounds=(0, 1), doc="""
        Fraction of axis items to return.""")

    replace = param.Boolean(default=False, doc="""
        Sample with or without replacement.""")

    transform_type = 'sample'

    def apply(self, table):
        return table.sample(n=self.n, frac=self.frac, replace=self.replace)


class Compute(Transform):
    """
    Turns a Dask DataFrame into a pandas DataFrame.
    """

    transform_type = 'compute'

    def apply(self, table):
        return table.compute()


class Melt(Transform):
    """
    Melts a DataFrame given the id_vars and value_vars. 
    """

    id_vars = param.List(default=[], doc="""
        Column(s) to use as identifier variables.""")

    value_vars = param.List(default=None, doc="""
        Column(s) to unpivot. If not specified, uses all columns that
        are not set as `id_vars`.""")

    var_name = param.String(default=None, doc="""
         Name to use for the 'variable' column. If None it uses
         ``frame.columns.name`` or 'variable'.""")

    value_name = param.String(default='value', doc="""
         Name to use for the 'value' column.""")

    transform_type = 'melt'

    def apply(self, table):
        return pd.melt(table, id_vars=self.id_vars, value_vars=self.value_vars,
                       var_name=self.var_name, value_name=self.value_name)


class project_lnglat(Transform):
    """
    Projects the given (longitude, latitude) values into Web Mercator
    coordinates (meters East of Greenwich and meters North of the Equator).
    """

    longitude = param.String(default='longitude', doc="Longitude column")
    latitude = param.String(default='longitude', doc="Latitude column")

    transform_type = 'project_lnglat'

    def apply(self, table):
        table = table.copy()
        longitude = table[self.longitude]
        latitude = table[self.latitude]

        origin_shift = np.pi * 6378137
        table[self.longitude] = longitude * origin_shift / 180.0
        table[self.latitude] = np.log(np.tan((90 + latitude) * np.pi / 360.0)) * origin_shift / np.pi
        return table
