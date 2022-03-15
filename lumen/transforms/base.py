"""
The Transform components allow transforming tables in arbitrary ways.
"""

import datetime as dt

import numpy as np
import pandas as pd
import panel as pn
import param

from ..base import Component
from ..state import state
from ..util import is_ref


class Transform(Component):
    """
    A Transform provides the ability to transform a table supplied by
    a Source.
    """

    controls = param.List(default=[], doc="""
        Parameters that should be exposed as widgets in the UI.""")

    transform_type = None

    _field_params = []

    __abstract = True

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
        new_spec, refs = {}, {}
        for k, v in spec.items():
            if (k in transform_type.param and
                isinstance(transform_type.param[k], param.ListSelector) and
                not isinstance(v, list)):
                v = [v]
            if is_ref(v):
                refs[k] = v
                v = state.resolve_reference(v)
            new_spec[k] = v

        # Resolve any specs for the controls
        controls, control_kwargs = [], {}
        for control in new_spec.get('controls', []):
            if isinstance(control, dict):
                ckws = {}
                if 'options' in control:
                    options = control['options']
                    if isinstance(options, str):
                        options = state.resolve_reference(options)
                    ckws['objects'] = options
                if 'start' in control or 'end' in control:
                    ckws['bounds'] = (control.get('start'), control.get('end'))
                control = control['name']
                control_kwargs[control] = ckws
            controls.append(control)
        new_spec['controls'] = controls

        # Instantiate the transform
        transform = transform_type(refs=refs, **new_spec)

        # Modify the parameters for the controls
        for p, vs in control_kwargs.items():
            p = transform.param[p]
            for attr, val in vs.items():
                if hasattr(p, attr):
                    setattr(p, attr, val)
                else:
                    attr = 'options' if attr == 'objects' else attr
                    cls.param.warning(
                        f"{transform_type.__name__} is of type {type(p).__name} "
                        f"and has not attribute {attr!r}. Ensure the controls "
                        "parameter supports the provided options, e.g. if "
                        "you are declaring 'options' ensure that the parameter "
                        "is a param.Selector type."
                    )
        return transform

    @classmethod
    def apply_to(cls, table, **kwargs):
        """
        Calls the apply method based on keyword arguments passed to define transform.
        
        Parameters
        ----------
        table: `pandas.DataFrame`
        
        Returns
        -------
        A DataFrame with the results of the transformation.
        """
        return cls(**kwargs).apply(table)

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

    @property
    def control_panel(self):
        return pn.Param(
            self.param, parameters=self.controls, sizing_mode='stretch_width',
            margin=(-10, 0, 5, 0)
        )

    @property
    def refs(self):
        refs = super().refs
        for c in self.controls:
            if c not in refs:
                refs.append(c)
        return refs


class Filter(Transform):
    """
    Transform that applies the query values from Lumen Filter
    components to a dataframe. The filter values can be one of the
    following:

      * scalar: A scalar value will be matched using equality operators
      * tuple:  A tuple value specifies a numeric or date range.
      * list:   A list value specifies a set of categories to match against.
      * list(tuple): A list of tuples specifies a list of ranges.
    """

    conditions = param.List(doc="""
      List of filter conditions expressed as tuples of the column
      name and the filter value.""")

    @classmethod
    def _range_filter(cls, column, start, end):
        if column.dtype.kind == 'M':
            if isinstance(start, dt.date) and not isinstance(start, dt.datetime):
                start = dt.datetime(*start.timetuple()[:3], 0, 0, 0)
            if isinstance(end, dt.date) and not isinstance(end, dt.datetime):
                end = dt.datetime(*end.timetuple()[:3], 23, 59, 59)
        if start is None and end is None:
            return None
        elif start is None:
            mask = column<=end
        elif end is None:
            mask = column>=start
        else:
            print(column.compute(), (column>=start).compute(), (column<=end).compute())
            mask = (column>=start) & (column<=end)
        return mask

    def apply(self, df):
        filters = []
        for k, val in self.conditions:
            if k not in df.columns:
                continue
            column = df[k]
            if np.isscalar(val) or isinstance(val, dt.date):
                if (column.dtype.kind == 'M' and isinstance(val, dt.date)
                    and not isinstance(val, dt.datetime)):
                    val = dt.datetime(*val.timetuple()[:3], 0, 0, 0)
                mask = column == val
            elif isinstance(val, list) and all(isinstance(v, tuple) and len(v) == 2 for v in val):
                val = [v for v in val if v is not None]
                if not val:
                    continue
                mask = self._range_filter(column, *val[0])
                for v in val[1:]:
                    mask |= self._range_filter(column, *v)
                if mask is not None:
                    filters.append(mask)
                continue
            elif isinstance(val, list):
                if not val:
                    continue
                mask = column.isin(val)
            elif isinstance(val, tuple):
                mask = self._range_filter(column, *val)
            else:
                self.param.warning(
                    'Condition {val!r} on {col!r} column not understood. '
                    'Filter query will not be applied.'
                )
                continue
            if mask is not None:
                filters.append(mask)
        if filters:
            mask = filters[0]
            for f in filters[1:]:
                mask &= f
            df = df[mask]
        return df


class HistoryTransform(Transform):
    """
    The HistoryTransform accumulates a history of the queried data in
    a buffer up to the supplied length and (optionally) adds a
    date_column to the data.
    """

    date_column = param.Selector(doc="""
        If defined adds a date column with the supplied name.""")

    length = param.Integer(default=10, bounds=(1, None), doc="""
        Accumulates a history of data.""")

    transform_type = 'history'

    _field_params = ['date_column']

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

    by = param.ListSelector(doc="""
        Columns or indexes to group by.""")

    columns = param.ListSelector(doc="""
        Columns to aggregate.""")

    with_index = param.Boolean(default=True, doc="""
        Whether to make the groupby columns indexes.""")

    method = param.String(default=None, doc="""
        Name of aggregation method.""")

    kwargs = param.Dict(default={}, doc="""
        Keyword arguments to the aggregation method.""")

    transform_type = 'aggregate'

    _field_params = ['by', 'columns']

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

    by = param.ListSelector(default=[], doc="""
       Columns or indexes to sort by.""")

    ascending = param.ClassSelector(default=True, class_=(bool, list), doc="""
       Sort ascending vs. descending. Specify list for multiple sort
       orders. If this is a list of bools, must match the length of
       the by.""")

    transform_type = 'sort'

    _field_params = ['by']

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

    columns = param.ListSelector(doc="""
        The subset of columns to select.""")

    transform_type = 'columns'

    _field_params = ['columns']

    def apply(self, table):
        return table[self.columns]


class Astype(Transform):
    """
    Transforms the type of one or more columns.
    """

    dtypes = param.Dict(doc="Mapping from column name to new type.")

    def apply(self, table):
        table = table.copy()
        for col, dtype in self.dtypes.items():
            table[col] = table[col].astype(dtype)
        return table


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

    id_vars = param.ListSelector(default=[], doc="""
        Column(s) to use as identifier variables.""")

    value_vars = param.ListSelector(default=None, doc="""
        Column(s) to unpivot. If not specified, uses all columns that
        are not set as `id_vars`.""")

    var_name = param.String(default=None, doc="""
         Name to use for the 'variable' column. If None it uses
         ``frame.columns.name`` or 'variable'.""")

    value_name = param.String(default='value', doc="""
         Name to use for the 'value' column.""")

    transform_type = 'melt'

    _field_params = ['id_vars', 'value_vars']

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
