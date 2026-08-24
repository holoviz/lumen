"""
The Transform components allow transforming tables in arbitrary ways.
"""
from __future__ import annotations

import datetime as dt
import hashlib
import math

from collections.abc import Callable
from functools import reduce
from operator import and_, or_
from typing import (
    TYPE_CHECKING, Any, ClassVar, Literal,
)

import narwhals.stable.v2 as nw
import numpy as np
import pandas as pd
import panel as pn
import param  # type: ignore

from packaging.version import Version
from panel.io.cache import _generate_hash

from ..base import MultiTypeComponent
from ..state import state
from ..util import (
    as_narwhals, as_pandas, is_lazyframe, is_narwhals, is_ref, log,
)

pd_version = Version(pd.__version__)

# Aggregation names whose narwhals result is verified equal to pandas on every
# backend. Anything absent takes the pandas path, because a shared name is not
# a shared meaning: pyarrow's median is approximate, and narwhals `first` does
# not skip missing values the way pandas groupby.first does.
_NARWHALS_AGGREGATIONS = frozenset({
    'max', 'mean', 'min', 'std', 'sum', 'var',
})

if TYPE_CHECKING:
    from dask.dataframe import DataFrame as dDataFrame, Series as dSeries
    from panel.viewable import Viewable
    DataFrame = pd.DataFrame | dDataFrame
    Series = pd.Series | dSeries


def _is_missing(value):
    """True for every spelling of a missing value a schema enum can carry.

    The pandas schema path emits NaN for a missing string where the narwhals
    path emits None, and either can arrive back here as a filter value.
    """
    return value is None or (isinstance(value, float) and math.isnan(value))


class Transform(MultiTypeComponent):
    """
    `Transform` components implement transforms of `DataFrame` objects.
    """

    controls = param.List(default=[], doc="""
        Parameters that should be exposed as widgets in the UI.""")

    transform_type: ClassVar[str | None] = None

    _field_params: ClassVar[list[str]] = []

    # Whether apply() can work on a lazy frame. False means _narwhals_frame
    # collects one first, because slicing, sampling and len() need real rows.
    _lazy: ClassVar[bool] = True

    # Whether apply() handles narwhals frames. False means _coerce materializes
    # the data to pandas first, which keeps third-party subclasses working.
    _narwhals: ClassVar[bool] = False

    _valid_keys: ClassVar[list[str] | Literal['params'] | None] = 'params'

    __abstract = True

    @classmethod
    def from_spec(cls, spec: dict[str, Any] | str) -> Transform:
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
    def _coerce(cls, table: Any) -> Any:
        """Materialize to pandas for transforms with no narwhals implementation.

        Most transforms encode a pandas concept narwhals has no counterpart for,
        such as the index or the query expression language. Rather than fail deep
        inside another dataframe library, they convert and say what it cost.
        """
        if cls._narwhals:
            return table
        return cls._coerce_to_pandas(table)

    @classmethod
    def _coerce_to_pandas(cls, table: Any) -> Any:
        """Convert a non-pandas frame to pandas, saying what it cost."""
        if isinstance(table, pd.DataFrame):
            return table
        narwhals_table = as_narwhals(table)
        if not is_narwhals(narwhals_table):
            return table
        cls.param.warning(
            f'{cls.__name__} has no narwhals implementation, so the data was '
            'converted to pandas, loading the whole frame into memory.'
        )
        return as_pandas(narwhals_table)

    def _narwhals_frame(self, table: Any, lazy: bool = False):
        """Return table as a narwhals frame, or None to take the pandas path."""
        if isinstance(table, pd.DataFrame):
            return None
        narwhals_table = as_narwhals(table)
        if not is_narwhals(narwhals_table):
            return None
        if not (self._lazy or lazy) and is_lazyframe(narwhals_table):
            return narwhals_table.collect()
        return narwhals_table

    def _to_native(self, frame, source):
        """Return frame as the same kind of object the caller handed in."""
        if is_lazyframe(as_narwhals(source)) and not is_lazyframe(frame):
            return frame.lazy().to_native()
        return frame.to_native()

    def _try_narwhals(self, table, build, lazy=False):
        """Run build() on a narwhals frame, or hand back a pandas table.

        Returns (result, table). A result of None means take the pandas path
        using the returned table, which has been converted if it was not
        pandas already.

        Every narwhals gap routes through here: an option narwhals has no
        counterpart for, an expression method it does not implement, a dtype a
        backend rejects. Catching them in one place is what stops each ported
        transform needing its own hand-written predicate for what narwhals can
        and cannot do, which is where the divergences kept coming from.
        """
        frame = self._narwhals_frame(table, lazy=lazy)
        if frame is None:
            return None, table
        try:
            return self._to_native(build(frame), table), table
        except (MemoryError, RecursionError):
            # Recovering by materializing the same data as pandas would only
            # exhaust the same resource again, with the first failure hidden.
            raise
        except Exception as e:
            log.debug(
                '%s could not be expressed in narwhals, using pandas: %r',
                type(self).__name__, e, exc_info=True
            )
            type(self).param.warning(
                f'{type(self).__name__} could not run this configuration on '
                f'{type(table).__name__}, so the data was converted to pandas.'
            )
            return None, as_pandas(table)

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
        return cls(**kwargs).apply(cls._coerce(table))

    def __hash__(self) -> int:
        """
        Implements hashing to allow a Source to compute a hash key.
        """
        sha = hashlib.sha256()
        hash_vals: tuple[Any, ...] = (type(self).__name__.encode('utf-8'),)
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

    def requires_columns(self) -> set[str] | None:
        """
        Return the columns this transform reads, or None if it may read any.

        A Pipeline uses this to narrow the query it sends to a SQL source to
        the columns something actually needs. None is the safe answer and the
        default: it means the query cannot be narrowed, because a projection
        might drop a column this transform reads. Only override it where
        every column the transform touches is named by its parameters, and
        return None for the parameter values that mean "all the rest".
        """
        return None

    @property
    def control_panel(self) -> Viewable:
        return pn.Param(
            self.param, parameters=self.controls, sizing_mode='stretch_width',
            margin=(-10, 0, 5, 0)
        )

    def _drop_none_values(self, **kwargs: dict[str, Any]) -> dict[str, Any]:
        return {k: v for k, v in kwargs.items() if v is not None}


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

    _narwhals: ClassVar[bool] = True

    @staticmethod
    def _widen_dates(start: Any, end: Any) -> tuple[Any, Any]:
        """Grow a date range to cover the whole of its first and last day."""
        if isinstance(start, dt.date) and not isinstance(start, dt.datetime):
            start = dt.datetime(*start.timetuple()[:3], 0, 0, 0)
        if isinstance(end, dt.date) and not isinstance(end, dt.datetime):
            end = dt.datetime(*end.timetuple()[:3], 23, 59, 59)
        return start, end

    @classmethod
    def _range_expr(cls, temporal: bool, name: str, start: Any, end: Any):
        """Narwhals counterpart of _range_filter, expressed against a column name."""
        if temporal:
            start, end = cls._widen_dates(start, end)
        if start is None and end is None:
            return None
        column = nw.col(name)
        if start is None:
            return column <= end
        if end is None:
            return column >= start
        return (column >= start) & (column <= end)

    def _apply_narwhals(self, df):
        schema = df.collect_schema()
        exprs = []
        for k, val in self.conditions:
            if k not in schema.names():
                continue
            temporal = isinstance(schema[k], (nw.Datetime, nw.Date))
            if np.isscalar(val) or isinstance(val, dt.date):
                if temporal:
                    val, _ = self._widen_dates(val, None)
                expr = nw.col(k) == val
            elif isinstance(val, list) and all(isinstance(v, tuple) and len(v) == 2 for v in val):
                ranges = [
                    self._range_expr(temporal, k, *v) for v in val if v is not None
                ]
                ranges = [r for r in ranges if r is not None]
                if not ranges:
                    continue
                expr = reduce(or_, ranges)
            elif isinstance(val, list):
                if not val:
                    continue
                # A None in the list matches nulls for pandas isin on an object
                # column, but polars drops those rows, so ask for nulls
                # separately. The schema emits such lists for nullable columns.
                # Numeric columns are excluded because pandas isin([None]) does
                # not match NaN there, and matching it would be a difference.
                # A missing marker matches nulls for pandas isin on an object
                # column, but polars drops those rows, so ask for nulls
                # separately. Numeric columns are excluded because pandas
                # isin([None]) does not match NaN there.
                members = [v for v in val if not _is_missing(v)]
                expr = nw.col(k).is_in(members) if members else nw.col(k).is_null() & ~nw.col(k).is_null()
                if len(members) != len(val) and not schema[k].is_numeric():
                    expr = expr | nw.col(k).is_null()
            elif isinstance(val, tuple):
                expr = self._range_expr(temporal, k, *val)
            else:
                self.param.warning(
                    f'Condition {val!r} on {k!r} column not understood. '
                    'Filter query will not be applied.'
                )
                continue
            if expr is not None:
                exprs.append(expr)
        if exprs:
            df = df.filter(reduce(and_, exprs))
        return df

    @classmethod
    def _range_filter(cls, column: Series, start: Any, end: Any) -> Series | None:
        if column.dtype.kind == 'M':
            start, end = cls._widen_dates(start, end)
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
        result, df = self._try_narwhals(df, self._apply_narwhals)
        if result is not None:
            return result
        filters = []
        for k, val in self.conditions:
            if k not in df.columns:
                continue
            column = df[k]
            if np.isscalar(val) or isinstance(val, dt.date):
                if (column.dtype.kind == 'M' and isinstance(val, dt.date)
                    and not isinstance(val, dt.datetime)):
                    val, _ = self._widen_dates(val, None)
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
                    f'Condition {val!r} on {k!r} column not understood. '
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
        table : DataFrame
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

    `by` must be provided.

    `df.groupby(<by>)[<columns>].<method>()[.reset_index()]`
    """

    by = param.ListSelector(doc="""
        Columns or indexes to group by.""")

    columns = param.ListSelector(allow_None=True, doc="""
        Columns to aggregate.""")

    with_index = param.Boolean(default=True, doc="""
        Whether to make the groupby columns indexes.""")

    method = param.String(default="mean", doc="""
        Name of the pandas aggregation method, e.g. max, min, count.""")

    kwargs = param.Dict(default={}, doc="""
        Keyword arguments to the aggregation method.""")

    transform_type: ClassVar[str] = 'aggregate'

    _field_params: ClassVar[list[str]] = ['by', 'columns']

    _narwhals: ClassVar[bool] = True

    def requires_columns(self) -> set[str] | None:
        if not self.columns:
            # Without an explicit selection every numeric column is aggregated.
            return None
        return set(self.by or []) | set(self.columns)

    def apply(self, table: DataFrame) -> DataFrame:
        def build(frame):
            if self.method not in _NARWHALS_AGGREGATIONS:
                raise NotImplementedError(self.method)
            schema = frame.collect_schema()
            cols = self.columns or [
                name for name, dtype in schema.items()
                if dtype.is_numeric() and name not in self.by
            ]
            # pandas aggregations skip NaN; polars propagates it. narwhals
            # skips nulls, so map NaN onto null for the float columns first.
            floats = [
                c for c in cols
                if schema[c].is_numeric() and not schema[c].is_integer()
            ]
            aggs = [getattr(nw.col(c), self.method)(**self.kwargs) for c in cols]
            # Two things pandas groupby does that narwhals does not: it drops
            # rows whose key is null or NaN, and it sorts by the key. polars
            # treats NaN as a value, so exclude it explicitly.
            keys = reduce(
                and_, [
                    (~nw.col(c).is_nan() & ~nw.col(c).is_null())
                    if schema[c].is_numeric() and not schema[c].is_integer()
                    else ~nw.col(c).is_null()
                    for c in self.by
                ]
            )
            if floats:
                frame = frame.with_columns(*[
                    nw.when(~nw.col(c).is_nan()).then(nw.col(c)).alias(c)
                    for c in floats
                ])
            return frame.filter(keys).group_by(self.by).agg(*aggs).sort(self.by)

        if not self.with_index:
            result, table = self._try_narwhals(table, build)
            if result is not None:
                return result
        # with_index moves the group keys into an index narwhals does not have.
        table = type(self)._coerce_to_pandas(table)
        grouped = table.groupby(self.by)
        if self.columns:
            cols = self.columns
        else:
            cols = [
                c for c in table.select_dtypes(include='number').columns
                if c not in self.by
            ]
        grouped = grouped[cols]
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

    _field_params: ClassVar[list[str]] = ['by']

    _narwhals: ClassVar[bool] = True

    def requires_columns(self) -> set[str] | None:
        return set(self.by or [])

    def apply(self, table: DataFrame) -> DataFrame:
        def build(frame):
            if not self.by:
                # sort_values([]) is a no-op in pandas but raises in narwhals,
                # and converting a whole frame to do nothing would be absurd.
                raise NotImplementedError('by is empty')
            descending = (
                [not a for a in self.ascending]
                if isinstance(self.ascending, list) else [not self.ascending] * len(self.by)
            )
            # narwhals sorts nulls first by default and polars orders NaN above
            # every value; pandas puts both last whichever way the sort runs.
            schema = frame.collect_schema()
            flags, keys, order = {}, [], []
            for column, desc in zip(self.by, descending, strict=True):
                if schema[column].is_numeric() and not schema[column].is_integer():
                    # The flag has to sort immediately before its own column,
                    # so NaN lands last within each group of the keys to its
                    # left rather than being pushed to the end of the frame.
                    flag = f'_nan_{column}'
                    flags[flag] = nw.col(column).is_nan()
                    keys.append(flag)
                    order.append(False)
                keys.append(column)
                order.append(desc)
            if not flags:
                return frame.sort(self.by, descending=descending, nulls_last=True)
            return (
                frame.with_columns(**flags)
                .sort(keys, descending=order, nulls_last=True)
                .drop(list(flags))
            )

        result, table = self._try_narwhals(table, build)
        if result is not None:
            return result
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

    _field_params: ClassVar[list[str]] = ['columns']

    _narwhals: ClassVar[bool] = True

    def requires_columns(self) -> set[str] | None:
        return set(self.columns or [])

    def apply(self, table: DataFrame) -> DataFrame:
        result, table = self._try_narwhals(table, lambda f: f.select(self.columns))
        if result is not None:
            return result
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
    `Iloc` allows selecting the data with integer indexing, see `pandas.DataFrame.iloc`.

    `df.iloc[<start>:<end>]`
    """

    start = param.Integer(default=None)

    end = param.Integer(default=None)

    transform_type: ClassVar[str] = 'iloc'

    _lazy: ClassVar[bool] = False

    _narwhals: ClassVar[bool] = True

    def apply(self, table: DataFrame) -> DataFrame:
        if is_lazyframe(as_narwhals(table)) and not self.start and self.end and self.end > 0:
            # A leading slice is head(), which a lazy backend pushes down
            # rather than materializing the whole frame to throw most away.
            result, table = self._try_narwhals(
                table, lambda f: f.head(self.end), lazy=True
            )
            if result is not None:
                return result
        result, table = self._try_narwhals(table, lambda f: f[self.start:self.end])
        if result is not None:
            return result
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

    _lazy: ClassVar[bool] = False

    _narwhals: ClassVar[bool] = True

    def apply(self, table: DataFrame) -> DataFrame:
        result, table = self._try_narwhals(table, lambda f: f.sample(
            **self._drop_none_values(
                n=self.n, fraction=self.frac, with_replacement=self.replace
            )
        ))
        if result is not None:
            return result
        return table.sample(
            **self._drop_none_values(n=self.n, frac=self.frac, replace=self.replace)
        )


class Compute(Transform):
    """
    `Compute` turns a `dask.dataframe.DataFrame` into a `pandas.DataFrame`.
    """

    transform_type: ClassVar[str] = 'compute'

    def apply(self, table: DataFrame) -> DataFrame:
        if hasattr(table, 'compute'):
            return table.compute()
        return table


class Pivot(Transform):
    """
    `Pivot` applies `pandas.DataFrame.pivot` given an index, columns, and values.
    """

    index = param.String(default=None, doc="""
        Column to use to make new frame's index.
        If None, uses existing index.""")

    columns = param.String(default=None, doc="""
        Column to use to make new frame's columns.""")

    values = param.ListSelector(default=None, doc="""
        Column(s) to use for populating new frame's values.
        If not specified, all remaining columns will be used
        and the result will have hierarchically indexed columns.""")

    transform_type: ClassVar[str] = 'pivot'

    def apply(self, table: DataFrame) -> DataFrame:
        pivot_table = table.pivot(
            **self._drop_none_values(
                index=self.index,
                columns=self.columns,
                values=self.values
            )
        )
        pivot_table.columns = pivot_table.columns.to_flat_index()
        return pivot_table


class PivotTable(Transform):
    """
    `PivotTable` applies pandas.pivot_table` to the data.
    """

    values = param.ListSelector(default=[], doc="""
        Column or columns to aggregate.""")

    index = param.ListSelector(default=[], doc="""
        Column, Grouper, array, or list of the previous
        Keys to group by on the pivot table index. If a list is passed,
        it can contain any of the other types (except list). If an array is
        passed, it must be the same length as the data and will be used in
        the same manner as column values.""")

    columns = param.ListSelector(default=[], doc="""
        Column, Grouper, array, or list of the previous
        Keys to group by on the pivot table column. If a list is passed,
        it can contain any of the other types (except list). If an array is
        passed, it must be the same length as the data and will be used in
        the same manner as column values.""")

    aggfunc = param.String(default="mean", doc="""
        Function, list of functions, dict, default 'mean'""")

    _field_params: ClassVar[list[str]] = ['values', 'index', 'columns']

    def apply(self, table: DataFrame) -> DataFrame:
        values = self.values if len(self.values) > 1 else self.values[0]
        columns = self.columns if len(self.columns) > 1 else self.columns[0]
        return pd.pivot_table(
            table, values=values, index=self.index, columns=columns,
            aggfunc=self.aggfunc
        )


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

    _field_params: ClassVar[list[str]] = ['id_vars', 'value_vars']

    _narwhals: ClassVar[bool] = True

    def apply(self, table: DataFrame) -> DataFrame:
        def build(frame):
            if self.value_vars == []:
                # pandas reads an empty value_vars differently depending on
                # whether id_vars was given. Not worth reimplementing.
                raise NotImplementedError
            # pandas never melts a column that is already an id_var.
            on = self.value_vars or [
                c for c in frame.collect_schema().names() if c not in self.id_vars
            ]
            on = [c for c in on if c not in self.id_vars]
            # ignore_index has no counterpart: unpivot never keeps an index.
            return frame.unpivot(
                on=on, index=self.id_vars,
                variable_name='variable' if self.var_name is None else self.var_name,
                value_name=self.value_name,
            )

        result, table = self._try_narwhals(table, build)
        if result is not None:
            return result
        melt: Callable
        if isinstance(table, pd.DataFrame):
            melt = pd.melt
        else:
            import dask.dataframe as dd  # noqa: PLC0415
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

    _field_params: ClassVar[list[str]] = ['keys']

    def requires_columns(self) -> set[str] | None:
        if self.keys is None:
            return None
        return {self.keys} if isinstance(self.keys, str) else set(self.keys)

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

    _narwhals: ClassVar[bool] = True

    def apply(self, table: DataFrame) -> DataFrame:
        renaming_columns = self.columns or (self.mapper and self.axis in (1, 'columns'))
        if renaming_columns and self.level is None and not (self.columns and self.mapper):
            # Only the columns case maps over: index and level are pandas index
            # concepts, and narwhals has no index to rename. Unknown columns are
            # dropped from the mapping because pandas rename ignores them.
            def build(frame):
                mapping = self.columns or self.mapper
                known = set(frame.collect_schema().names())
                return frame.rename({k: v for k, v in mapping.items() if k in known})

            result, table = self._try_narwhals(table, build)
            if result is not None:
                return result
        table = type(self)._coerce_to_pandas(table)
        kwargs: dict[str, Any] = dict(
            axis=self.axis, columns=self.columns,
            index=self.index, mapper=self.mapper, level=self.level,
        )
        if pd_version < Version('3.0.0'):
            kwargs['copy'] = self.copy
        return table.rename(**kwargs)  # type: ignore


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
        kwargs: dict[str, Any] = dict(
            axis=self.axis, columns=self.columns,
            index=self.index, mapper=self.mapper,
        )
        if pd_version < Version('3.0.0'):
            kwargs['copy'] = self.copy
        return table.rename_axis(**kwargs)


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
        kwargs = {}
        if pd_version < Version('2.0.0'):
            kwargs['level'] = self.level
        return table.count(
            axis=self.axis, numeric_only=self.numeric_only, **kwargs
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
        kwargs = {}
        if pd_version < Version('2.0.0'):
            kwargs['level'] = self.level
        return table.sum(
            axis=self.axis, **kwargs
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


class DropNA(Transform):
    """
    `DropNA` drops rows with any missing values.

    `df.dropna(axis=<axis>, how=<how>, thresh=<thresh>, subset=<subset>)`
    """

    axis = param.ClassSelector(default=0, class_=(int, str), doc="""
        The axis to rename. 0 or 'index', 1 or 'columns'""")

    how = param.Selector(default='any', objects=['any', 'all'], doc="""
        Determine if row or column is removed from DataFrame, when we have
        at least one NA or all NA.""")

    thresh = param.Integer(default=None, doc="""
        Require that many non-NA values.""")

    subset = param.ListSelector(default=None, doc="""
        Labels along other axis to consider, e.g. if you are dropping rows
        these would be a list of columns to include.""")

    transform_type: ClassVar[str] = 'dropna'

    _narwhals: ClassVar[bool] = True

    def apply(self, table: DataFrame) -> DataFrame:
        droppable = self.axis in (0, 'index') and self.thresh is None
        if droppable and self.how in (None, 'any'):
            def build(frame):
                # drop_nulls is row-wise how='any' only; axis=1, how='all' and
                # thresh have no counterpart and take the pandas path below.
                # It also keeps NaN, which pandas dropna removes, so float
                # columns need the extra predicate.
                schema = frame.collect_schema()
                considered = self.subset or schema.names()
                floats = [
                    c for c in considered
                    if schema[c].is_numeric() and not schema[c].is_integer()
                ]
                frame = frame.drop_nulls(subset=self.subset)
                if floats:
                    frame = frame.filter(
                        reduce(and_, [~nw.col(c).is_nan() for c in floats])
                    )
                return frame

            result, table = self._try_narwhals(table, build)
            if result is not None:
                return result
        table = type(self)._coerce_to_pandas(table)
        kwargs = {'axis': self.axis, 'subset': self.subset}
        if self.how and self.thresh is None:
            kwargs['how'] = self.how
        if self.thresh is not None:
            kwargs['thresh'] = self.thresh
        return table.dropna(**kwargs)


class Corr(Transform):
    """
    ``Corr`` computes pairwise correlation of columns, excluding NA/null values.
    """

    method = param.Selector(default='pearson', objects=[
        'pearson', 'kendall', 'spearman'], doc="""
        Method of correlation.""")

    min_periods = param.Integer(default=1, doc="""
        Minimum number of observations required per pair of columns
        to have a valid result. Currently only available for Pearson
        and Spearman correlation.""")

    numeric_only = param.Boolean(default=False, doc="""
        Include only `float`, `int` or `boolean` data.""")

    transform_type: ClassVar[str] = 'corr'

    def apply(self, table: DataFrame) -> DataFrame:
        return table.corr(
            method=self.method, min_periods=self.min_periods, numeric_only=self.numeric_only
        )


class project_lnglat(Transform):
    """
    `project_lnglat` projects the given longitude/latitude columns to Web Mercator.

    Converts latitude and longitude values into WGS84 (Web Mercator)
    coordinates (meters East of Greenwich and meters North of the
    Equator).
    """

    longitude = param.String(default='longitude', doc="Longitude column")
    latitude = param.String(default='latitude', doc="Latitude column")

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
