"""
Scientific transforms for xarray-derived DataFrames.

These transforms operate on pandas DataFrames with coordinate columns
(time, lat, lon) as produced by XArraySQLSource. They provide common
scientific data operations: time resampling, rolling windows, anomaly
computation, and spatial filtering.
"""

from __future__ import annotations

from typing import ClassVar

import pandas as pd
import param

from ..transforms.base import Transform

from pandas import DataFrame


class TimeResample(Transform):
    """
    Resample time series data at a different frequency.

    Groups the DataFrame by a datetime column resampled to the given
    frequency, then applies an aggregation method.

    ``df.set_index(<date_column>).resample(<rule>)[<columns>].<method>()``
    """

    date_column = param.String(default='time', doc="""
        Column containing datetime values.""")

    rule = param.Selector(default='D', objects=[
        'h', 'D', 'W', 'ME', 'MS', 'QE', 'QS', 'YE', 'YS',
    ], doc="""
        Resample frequency.""")

    method = param.Selector(default='mean', objects=[
        'mean', 'sum', 'min', 'max', 'count', 'std', 'median',
    ], doc="""
        Aggregation method.""")

    columns = param.ListSelector(allow_None=True, doc="""
        Columns to aggregate. None means all numeric columns.""")

    transform_type: ClassVar[str] = 'time_resample'

    _field_params: ClassVar[list[str]] = ['date_column', 'columns']

    def apply(self, table: DataFrame) -> DataFrame:
        if table[self.date_column].dtype.kind != 'M':
            raise ValueError(
                f"Column '{self.date_column}' has dtype "
                f"'{table[self.date_column].dtype}', expected datetime."
            )
        table = table.set_index(self.date_column)
        resampled = table.resample(self.rule)
        if self.columns:
            resampled = resampled[self.columns]
        result = getattr(resampled, self.method)()
        return result.reset_index()


class RollingWindow(Transform):
    """
    Apply rolling window operations for smoothing or running statistics.

    Computes a rolling calculation on numeric columns while preserving
    all other columns unchanged.

    ``df[<columns>].rolling(<window>).<method>()``
    """

    window = param.Integer(default=7, bounds=(1, None), doc="""
        Window size in number of rows.""")

    method = param.Selector(default='mean', objects=[
        'mean', 'sum', 'std', 'min', 'max', 'median',
    ], doc="""
        Rolling method.""")

    columns = param.ListSelector(allow_None=True, doc="""
        Columns to apply rolling to. None means all numeric columns.""")

    center = param.Boolean(default=False, doc="""
        Whether to center the window.""")

    min_periods = param.Integer(default=None, allow_None=True, doc="""
        Minimum observations in window required for a value.""")

    transform_type: ClassVar[str] = 'rolling_window'

    _field_params: ClassVar[list[str]] = ['columns']

    def apply(self, table: DataFrame) -> DataFrame:
        cols = self.columns or list(
            table.select_dtypes(include='number').columns
        )
        rolling = table[cols].rolling(
            window=self.window,
            center=self.center,
            min_periods=self.min_periods,
        )
        result = getattr(rolling, self.method)()
        table = table.copy()
        table[cols] = result
        return table


class Anomaly(Transform):
    """
    Compute anomaly (deviation from climatological mean).

    Groups by a datetime accessor (month, dayofyear, hour, etc.),
    computes the group mean, and subtracts it from the original values.
    The result has the same shape as the input.
    """

    date_column = param.String(default='time', doc="""
        Column containing datetime values.""")

    columns = param.ListSelector(allow_None=True, doc="""
        Columns to compute anomaly for. None means all numeric columns.""")

    groupby_freq = param.Selector(default='month', objects=[
        'month', 'dayofyear', 'hour', 'weekday', 'quarter', 'year',
    ], doc="""
        Datetime accessor for grouping.""")

    transform_type: ClassVar[str] = 'anomaly'

    _field_params: ClassVar[list[str]] = ['date_column', 'columns']

    def apply(self, table: DataFrame) -> DataFrame:
        if table[self.date_column].dtype.kind != 'M':
            raise ValueError(
                f"Column '{self.date_column}' has dtype "
                f"'{table[self.date_column].dtype}', expected datetime."
            )
        cols = self.columns or list(
            table.select_dtypes(include='number').columns
        )
        grouper = getattr(table[self.date_column].dt, self.groupby_freq)
        climatology = table.groupby(grouper)[cols].transform('mean')
        table = table.copy()
        table[cols] = table[cols] - climatology
        return table


class SpatialBBox(Transform):
    """
    Filter data to a spatial bounding box.

    Keeps rows where latitude and longitude columns fall within the
    specified bounds. Useful for clipping global datasets to a region.
    """

    lat_column = param.String(default='lat', doc="""
        Column containing latitude values.""")

    lon_column = param.String(default='lon', doc="""
        Column containing longitude values.""")

    lat_bounds = param.NumericTuple(default=(-90, 90), length=2, doc="""
        Latitude range (min, max).""")

    lon_bounds = param.NumericTuple(default=(-180, 180), length=2, doc="""
        Longitude range (min, max).""")

    transform_type: ClassVar[str] = 'spatial_bbox'

    _field_params: ClassVar[list[str]] = ['lat_column', 'lon_column']

    def apply(self, table: DataFrame) -> DataFrame:
        lat_min, lat_max = self.lat_bounds
        lon_min, lon_max = self.lon_bounds
        lat_mask = (
            (table[self.lat_column] >= lat_min)
            & (table[self.lat_column] <= lat_max)
        )
        if lon_min <= lon_max:
            lon_mask = (
                (table[self.lon_column] >= lon_min)
                & (table[self.lon_column] <= lon_max)
            )
        else:
            # Anti-meridian crossing (e.g. lon_bounds=(170, -170))
            lon_mask = (
                (table[self.lon_column] >= lon_min)
                | (table[self.lon_column] <= lon_max)
            )
        return table[lat_mask & lon_mask].reset_index(drop=True)


__all__ = [
    'TimeResample',
    'RollingWindow',
    'Anomaly',
    'SpatialBBox',
]
