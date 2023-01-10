"""
The Transform components allow transforming tables in arbitrary ways.
"""
from __future__ import annotations

import datetime as dt
import hashlib

from typing import (
    TYPE_CHECKING, Any, Callable, ClassVar, Dict, List, Tuple, Union,
)

import numpy as np
import pandas as pd
import panel as pn
import param  # type: ignore

from panel.io.cache import _generate_hash

from ..base import MultiTypeComponent
from ..state import state
from ..util import is_ref

if TYPE_CHECKING:
    from dask.dataframe import DataFrame as dDataFrame, Series as dSeries
    from panel.viewable import Viewable
    DataFrame = Union[pd.DataFrame, dDataFrame]
    Series = Union[pd.Series, dSeries]


class Transform(MultiTypeComponent):
    """
    `Transform` components implement transforms of `DataFrame` objects.
    """

    controls = param.List(default=[], doc="""
        Parameters that should be exposed as widgets in the UI.""")

    transform_type: ClassVar[str | None] = None

    _field_params: ClassVar[List[str]] = []

    __abstract = True

    @classmethod
    def from_spec(cls, spec: Dict[str, Any] | str) -> 'Transform':
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
        if isinstance(spec, str):
            raise ValueError(
                "Transform cannot be materialized by reference. Please pass "
                "full specification for the transform."
            )
        spec = dict(spec)
        transform_type = Transform._get_type(spec.pop('type', None))
        new_spec, refs = {}, {}
        for k, v in spec.items():
            if is_ref(v):
                refs[k] = v
                v = state.resolve_reference(v)
            elif isinstance(v, dict):
                resolved = {}
                for sk, sv in v.items():
                    if is_ref(sv):
                        refs[f'{k}.{sk}'] = sv
                        sv = state.resolve_reference(sv)
                    resolved[sk] = sv
                v = resolved
            if (k in transform_type.param and
                isinstance(transform_type.param[k], param.ListSelector) and
                not isinstance(v, list)):
                v = [v]
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
                        f"{transform_type.__name__} is of type {type(p).__name__} "
                        f"and has not attribute {attr!r}. Ensure the controls "
                        "parameter supports the provided options, e.g. if "
                        "you are declaring 'options' ensure that the parameter "
                        "is a param.Selector type."
                    )
        return transform

    @classmethod
    def apply_to(cls, table: DataFrame, **kwargs) -> DataFrame:
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

    def __hash__(self) -> int:
        """
        Implements hashing to allow a Source to compute a hash key.
        """
        sha = hashlib.sha256()
        hash_vals: Tuple[Any, ...] = (type(self).__name__.encode('utf-8'),)
        hash_vals += tuple(sorted([
            (k, v) for k, v in self.param.values().items()
            if k not in Transform.param
        ]))
        sha.update(_generate_hash(hash_vals))
        return int(sha.hexdigest(), base=16)

    def apply(self, table: DataFrame) -> DataFrame:
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
    def control_panel(self) -> Viewable:
        return pn.Param(
            self.param, parameters=self.controls, sizing_mode='stretch_width',
            margin=(-10, 0, 5, 0)
        )

    @property
    def refs(self) -> List[str]:
        refs = super().refs
        for c in self.controls:
            if c not in refs:
                refs.append(c)
        return refs


class Filter(Transform):
    """
    `Filter` transform implement the filtering behavior of `Filter` components.

    The filter `conditions` must be declared as a list of tuple containing
    the name of the column to be filtered and one of the following:

      * scalar: A scalar value will be matched using equality operators
      * tuple:  A tuple value specifies a numeric or date range.
      * list:   A list value specifies a set of categories to match against.
      * list(tuple): A list of tuples specifies a list of ranges.
    """

    conditions = param.List(doc="""
      List of filter conditions expressed as tuples of the column
      name and the filter value.""")

    @classmethod
    def _range_filter(cls, column: Series, start: Any, end: Any) -> Series | None:
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
            mask = (column>=start) & (column<=end)
        return mask

    def apply(self, df: DataFrame) -> DataFrame:
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
    `HistoryTransform` accumulates a history of the queried data.

    The internal buffer accumulates data up to the supplied `length`
    and (optionally) adds a date_column to the data.
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

    def apply(self, table: DataFrame) -> DataFrame:
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
    `Aggregate` one or more columns or indexes, see `pandas.DataFrame.groupby`.

    `df.groupby(<by>)[<columns>].<method>()[.reset_index()]`
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

    transform_type: ClassVar[str] = 'aggregate'

    _field_params: ClassVar[List[str]] = ['by', 'columns']

    def apply(self, table: DataFrame) -> DataFrame:
        grouped = table.groupby(self.by)
        if self.columns:
            grouped = grouped[self.columns]
        agg = getattr(grouped, self.method)(**self.kwargs)
        return agg if self.with_index else agg.reset_index()


class Sort(Transform):
    """
    `Sort` on one or more columns, see `pandas.DataFrame.sort_values`.

    `df.sort_values(<by>, ascending=<ascending>)`
    """

    by = param.ListSelector(default=[], doc="""
       Columns or indexes to sort by.""")

    ascending = param.ClassSelector(default=True, class_=(bool, list), doc="""
       Sort ascending vs. descending. Specify list for multiple sort
       orders. If this is a list of bools, must match the length of
       the by.""")

    transform_type: ClassVar[str] = 'sort'

    _field_params: ClassVar[List[str]] = ['by']

    def apply(self, table: DataFrame) -> DataFrame:
        return table.sort_values(self.by, ascending=self.ascending)


class Query(Transform):
    """
    `Query` applies the `pandas.DataFrame.query` method.

    `df.query(<query>)`
    """

    query = param.String(doc="""
        The query to apply to the table.""")

    transform_type: ClassVar[str] = 'query'

    def apply(self, table: DataFrame) -> DataFrame:
        return table.query(self.query)


class Columns(Transform):
    """
    `Columns` selects a subset of columns.

    `df[<columns>]`
    """

    columns = param.ListSelector(doc="""
        The subset of columns to select.""")

    transform_type: ClassVar[str] = 'columns'

    _field_params: ClassVar[List[str]] = ['columns']

    def apply(self, table: DataFrame) -> DataFrame:
        return table[self.columns]


class Astype(Transform):
    """
    `Astype` transforms the type of one or more columns.
    """

    dtypes = param.Dict(doc="Mapping from column name to new type.")

    transform_type: ClassVar[str] = 'as_type'

    def apply(self, table: DataFrame) -> DataFrame:
        table = table.copy()
        for col, dtype in self.dtypes.items():
            if col in table.columns:
                table[col] = table[col].astype(dtype)
        return table


class Stack(Transform):
    """
    `Stack` applies `pandas.DataFrame.stack` to the declared `level`.

    `df.stack(<level>)`
    """

    dropna = param.Boolean(default=True, doc="""
        Whether to drop rows in the resulting Frame/Series with missing values.
        Stacking a column level onto the index axis can create combinations of
        index and column values that are missing from the original
        dataframe.""")

    level = param.ClassSelector(default=-1, class_=(int, list, str), doc="""
        The indexes to stack.""")

    transform_type: ClassVar[str] = 'stack'

    def apply(self, table: DataFrame) -> DataFrame:
        return table.stack(level=self.level, dropna=self.dropna)


class Unstack(Transform):
    """
    `Unstack` applies `pandas.DataFrame.unstack` to the declared `level`.

    `df.unstack(<level>)`
    """

    fill_value = param.ClassSelector(default=None, class_=(int, str, dict), doc="""
        Replace NaN with this value if the unstack produces missing values.""")

    level = param.ClassSelector(default=-1, class_=(int, list, str), doc="""
        The indexes to unstack.""")

    transform_type: ClassVar[str] = 'unstack'

    def apply(self, table: DataFrame) -> DataFrame:
        return table.unstack(level=self.level, fill_value=self.fill_value)


class Iloc(Transform):
    """
    `Iloc` applies integer slicing to the data, see `pandas.DataFrame.iloc`.

    `df.iloc[<start>:<end>]`
    """

    start = param.Integer(default=None)

    end = param.Integer(default=None)

    transform_type: ClassVar[str] = 'iloc'

    def apply(self, table: DataFrame) -> DataFrame:
        return table.iloc[self.start:self.end]


class Sample(Transform):
    """
    `Sample` returns a random sample of items.

    `df.sample(n=<n>, frac=<frac>, replace=<replace>)`
    """

    n = param.Integer(default=None, doc="""
        Number of items to return.""")

    frac = param.Number(default=None, bounds=(0, 1), doc="""
        Fraction of axis items to return.""")

    replace = param.Boolean(default=False, doc="""
        Sample with or without replacement.""")

    transform_type: ClassVar[str] = 'sample'

    def apply(self, table: DataFrame) -> DataFrame:
        return table.sample(n=self.n, frac=self.frac, replace=self.replace)


class Compute(Transform):
    """
    `Compute` turns a `dask.dataframe.DataFrame` into a `pandas.DataFrame`.
    """

    transform_type: ClassVar[str] = 'compute'

    def apply(self, table: DataFrame) -> DataFrame:
        return table.compute()


class Pivot(Transform):
    """
    `Pivot` applies `pandas.DataFrame.pivot` given an index, columns, and values.
    """

    index = param.String(default=None, doc="""
        Column to use to make new frame's index.
        If None, uses existing index.""")

    columns = param.String(default=None, doc="""
        Column to use to make new frame’s columns.""")

    values = param.ListSelector(default=None, doc="""
        Column(s) to use for populating new frame’s values.
        If not specified, all remaining columns will be used
        and the result will have hierarchically indexed columns.""")

    transform_type: ClassVar[str] = 'pivot'

    def apply(self, table: DataFrame) -> DataFrame:
        return table.pivot(index=self.index, columns=self.columns, values=self.values)


class Melt(Transform):
    """
    `Melt` applies the `pandas.melt` operation given the `id_vars` and `value_vars`.
    """

    id_vars = param.ListSelector(default=[], doc="""
        Column(s) to use as identifier variables.""")

    ignore_index = param.Boolean(default=True, doc="""
        If True, original index is ignored. If False, the original
        index is retained. Index labels will be repeated as
        necessary.""")

    value_vars = param.ListSelector(default=None, doc="""
        Column(s) to unpivot. If not specified, uses all columns that
        are not set as `id_vars`.""")

    var_name = param.String(default=None, doc="""
         Name to use for the 'variable' column. If None it uses
         ``frame.columns.name`` or 'variable'.""")

    value_name = param.String(default='value', doc="""
         Name to use for the 'value' column.""")

    transform_type: ClassVar[str] = 'melt'

    _field_params: ClassVar[List[str]] = ['id_vars', 'value_vars']

    def apply(self, table: DataFrame) -> DataFrame:
        melt: Callable
        if isinstance(table, pd.DataFrame):
            melt = pd.melt
        else:
            import dask.dataframe as dd
            melt = dd.melt
        return melt(
            table, id_vars=self.id_vars, value_vars=self.value_vars,
            var_name=self.var_name, value_name=self.value_name,
            ignore_index=self.ignore_index
        )


class SetIndex(Transform):
    """
    `SetIndex` promotes DataFrame columns to indexes, see `pandas.DataFrame.set_index`.

    `df.set_index(<keys>, drop=<drop>, append=<append>, verify_integrity=<verify_integrity>)`
    """

    append = param.Boolean(default=False, doc="""
        Whether to append columns to existing index.""")

    drop = param.Boolean(default=True, doc="""
        Delete columns to be used as the new index.""")

    keys = param.ClassSelector(default=None, class_=(str, list), doc="""
        This parameter can be either a single column key or a list
        containing column keys.""")

    verify_integrity = param.Boolean(default=False, doc="""
        Check the new index for duplicates. Otherwise defer the check
        until necessary. Setting to False will improve the performance
        of this method.""")

    transform_type: ClassVar[str] = 'set_index'

    _field_params: ClassVar[List[str]] = ['keys']

    def apply(self, table: DataFrame) -> DataFrame:
        return table.set_index(
            self.keys, drop=self.drop, append=self.append,
            verify_integrity=self.verify_integrity
        )


class ResetIndex(Transform):
    """
    `ResetIndex` resets DataFrame indexes to columns or drops them, see `pandas.DataFrame.reset_index`

    `df.reset_index(drop=<drop>, col_fill=<col_fill>, col_level=<col_level>, level=<level>)`
    """

    col_fill = param.String(default="", doc="""
        If the columns have multiple levels, determines how the other
        levels are named. If None then the index name is repeated.""")

    col_level = param.ClassSelector(default=0, class_=(int, str), doc="""
        If the columns have multiple levels, determines which level the
        labels are inserted into. By default it is inserted into the
        first level.""")

    drop = param.Boolean(default=False, doc="""
        Do not try to insert index into dataframe columns. This resets
        the index to the default integer index.""")

    level = param.ClassSelector(default=None, class_=(int, str, list), doc="""
        Only remove the given levels from the index. Removes all levels
        by default.""")

    transform_type: ClassVar[str] = 'reset_index'

    def apply(self, table: DataFrame) -> DataFrame:
        return table.reset_index(
            drop=self.drop, col_fill=self.col_fill, col_level=self.col_level,
            level=self.level
        )  # type: ignore


class Rename(Transform):
    """
    `Rename` renames columns or indexes, see `pandas.DataFrame.rename`.

    df.rename(mapper=<mapper>, columns=<columns>, index=<index>,
              level=<level>, axis=<axis>, copy=<copy>)
    """

    axis = param.ClassSelector(default=None, class_=(int, str), doc="""
        The axis to rename. 0 or 'index', 1 or 'columns'""")

    columns = param.Dict(default=None, doc="""
        Alternative to specifying axis (`mapper, axis=1` is equivalent to
        `columns=mapper`).""")

    copy = param.Boolean(default=False, doc="""
        Also copy underlying data.""")

    index = param.Dict(default=None, doc="""
        Alternative to specifying axis (`mapper, axis=0` is equivalent to
        `index=mapper`).""")

    mapper = param.Dict(default=None, doc="""
        Dict to apply to that axis' values. Use either `mapper` and `axis` to
        specify the axis to target with `mapper`, or `index` and `columns`.""")

    level = param.ClassSelector(default=None, class_=(int, str), doc="""
        In case of a MultiIndex, only rename labels in the specified level.""")

    transform_type: ClassVar[str] = 'rename'

    def apply(self, table: DataFrame) -> DataFrame:
        return table.rename(
            axis=self.axis, columns=self.columns, copy=self.copy,
            index=self.index, mapper=self.mapper, level=self.level,
        )  # type: ignore


class RenameAxis(Transform):
    """
    Set the name of the axis for the index or columns,
    see `pandas.DataFrame.rename_axis`.

    df.rename_axis(mapper=<mapper>, columns=<columns>, index=<index>,
                  axis=<axis>, copy=<copy>)
    """

    axis = param.ClassSelector(default=0, class_=(int, str), doc="""
        The axis to rename. 0 or 'index', 1 or 'columns'""")

    columns = param.ClassSelector(default=None, class_=(str, list, dict), doc="""
        A scalar, list-like, dict-like to apply to that axis' values.
        Note that the columns parameter is not allowed if the object
        is a Series. This parameter only apply for DataFrame type objects.
        Use either mapper and axis to specify the axis to target with
        mapper, or index and/or columns.""")

    copy = param.Boolean(default=True, doc="""
        Also copy underlying data.""")

    index = param.ClassSelector(default=None, class_=(str, list, dict), doc="""
        A scalar, list-like, dict-like to apply to that axis' values.
        Note that the columns parameter is not allowed if the object
        is a Series. This parameter only apply for DataFrame type objects.
        Use either mapper and axis to specify the axis to target with
        mapper, or index and/or columns.""")

    mapper = param.ClassSelector(default=None, class_=(str, list), doc="""
        Value to set the axis name attribute.""")

    transform_type: ClassVar[str] = 'rename_axis'

    def apply(self, table: DataFrame) -> DataFrame:
        return table.rename_axis(
            axis=self.axis, columns=self.columns, copy=self.copy,
            index=self.index, mapper=self.mapper,
        )


class Count(Transform):
    """
    Counts non-nan values in each column of the DataFrame and returns
    a new DataFrame with a single row with a count for each original
    column, see `pandas.DataFrame.count`.

    df.count(axis=<axis>, level=<level>, numeric_only=<numeric_only>).to_frame().T
    """

    axis = param.ClassSelector(default=0, class_=(int, str), doc="""
        The axis to rename. 0 or 'index', 1 or 'columns'""")

    level = param.ClassSelector(default=None, class_=(int, list, str), doc="""
        The indexes to stack.""")

    numeric_only = param.Boolean(default=False, doc="""
        Include only float, int or boolean data.""")

    transform_type: ClassVar[str] = 'count'

    def apply(self, table: DataFrame) -> DataFrame:
        return table.count(
            axis=self.axis, level=self.level, numeric_only=self.numeric_only
        ).to_frame().T


class Sum(Transform):
    """
    Sums numeric values in each column of the DataFrame and returns a
    new DataFrame with a single row containing the sum for each
    original column, see `pandas.DataFrame.sum`.

    df.count(axis=<axis>, level=<level>).to_frame().T
    """

    axis = param.ClassSelector(default=0, class_=(int, str), doc="""
        The axis to rename. 0 or 'index', 1 or 'columns'""")

    level = param.ClassSelector(default=None, class_=(int, list, str), doc="""
        The indexes to stack.""")

    transform_type: ClassVar[str] = 'sum'

    def apply(self, table: DataFrame) -> DataFrame:
        return table.sum(
            axis=self.axis, level=self.level
        ).to_frame().T


class Eval(Transform):
    """
    Applies an eval assignment expression to a DataFrame. The
    expression can reference columns on the original table by
    referencing `table.<column>` and must assign to a variable that
    will become a new column in the DataFrame, e.g. to divide a
    `value` column by one thousand and assign the result to a new column
    called `kilo_value` you can write an `expr` like:

        kilo_value = table.value / 1000

    See `pandas.eval` for more information.
    """

    expr = param.String(doc="""
        The expression to apply to the table.""")

    transform_type: ClassVar[str] = 'eval'

    def apply(self, table: DataFrame) -> DataFrame:
        return pd.eval(self.expr, target=table)


class project_lnglat(Transform):
    """
    `project_lnglat` projects the given longitude/latitude columns to Web Mercator.

    Converts latitude and longitude values into WGS84 (Web Mercator)
    coordinates (meters East of Greenwich and meters North of the
    Equator).
    """

    longitude = param.String(default='longitude', doc="Longitude column")
    latitude = param.String(default='longitude', doc="Latitude column")

    transform_type: ClassVar[str] = 'project_lnglat'

    def apply(self, table: DataFrame) -> DataFrame:
        table = table.copy()
        longitude = table[self.longitude]
        latitude = table[self.latitude]

        origin_shift = np.pi * 6378137
        table[self.longitude] = longitude * origin_shift / 180.0
        table[self.latitude] = np.log(np.tan((90 + latitude) * np.pi / 360.0)) * origin_shift / np.pi
        return table


__all__ = [name for name, obj in locals().items() if isinstance(obj, type) and issubclass(obj, Transform)]
