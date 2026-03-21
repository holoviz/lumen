"""Tests for scientific transforms."""

import numpy as np
import pandas as pd
import pytest

from lumen.transforms.base import Transform
from lumen.transforms.scientific import (
    Anomaly,
    RollingWindow,
    SpatialBBox,
    TimeResample,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def climate_df():
    """Daily climate data over 2 years with lat/lon."""
    np.random.seed(42)
    n = 730
    return pd.DataFrame({
        'time': pd.date_range('2020-01-01', periods=n, freq='D'),
        'lat': np.tile(np.linspace(20, 60, 10), n // 10),
        'lon': np.tile(np.linspace(-120, -70, 10), n // 10),
        'temperature': 20 + 10 * np.sin(np.arange(n) * 2 * np.pi / 365) + np.random.randn(n),
        'pressure': 1013 + np.random.randn(n) * 5,
    })


@pytest.fixture
def hourly_df():
    """Hourly data for fine-grained resampling tests."""
    n = 240  # 10 days of hourly data
    return pd.DataFrame({
        'time': pd.date_range('2020-06-01', periods=n, freq='h'),
        'value': np.random.randn(n).cumsum(),
    })


# ---------------------------------------------------------------------------
# TimeResample
# ---------------------------------------------------------------------------

class TestTimeResample:

    def test_daily_to_monthly(self, climate_df):
        t = TimeResample(date_column='time', rule='ME', method='mean')
        result = t.apply(climate_df)
        assert 'time' in result.columns
        assert len(result) < len(climate_df)
        assert len(result) <= 24  # at most 24 months

    def test_hourly_to_daily(self, hourly_df):
        t = TimeResample(date_column='time', rule='D', method='mean')
        result = t.apply(hourly_df)
        assert len(result) == 10

    def test_sum_method(self, hourly_df):
        t = TimeResample(date_column='time', rule='D', method='sum')
        result = t.apply(hourly_df)
        # Sum of 24 hourly values per day
        assert len(result) == 10

    def test_specific_columns(self, climate_df):
        t = TimeResample(
            date_column='time', rule='ME', method='mean',
            columns=['temperature'],
        )
        result = t.apply(climate_df)
        assert 'temperature' in result.columns
        # Non-selected numeric columns should not be in result
        assert 'pressure' not in result.columns

    def test_preserves_time_column(self, climate_df):
        t = TimeResample(date_column='time', rule='YE', method='mean')
        result = t.apply(climate_df)
        assert result['time'].dtype.kind == 'M'

    def test_weekly_resample(self, climate_df):
        t = TimeResample(date_column='time', rule='W', method='max')
        result = t.apply(climate_df)
        assert len(result) > 0
        assert len(result) < len(climate_df)


# ---------------------------------------------------------------------------
# RollingWindow
# ---------------------------------------------------------------------------

class TestRollingWindow:

    def test_basic_rolling_mean(self, climate_df):
        t = RollingWindow(window=7, method='mean')
        result = t.apply(climate_df)
        assert len(result) == len(climate_df)
        # First 6 values should be NaN (window=7, min_periods=default)
        assert result['temperature'].isna().sum() >= 6

    def test_preserves_non_numeric(self, climate_df):
        t = RollingWindow(window=3, method='mean')
        result = t.apply(climate_df)
        assert 'time' in result.columns
        assert result['time'].equals(climate_df['time'])

    def test_specific_columns(self, climate_df):
        t = RollingWindow(window=5, method='mean', columns=['temperature'])
        result = t.apply(climate_df)
        # Temperature should be smoothed
        assert result['temperature'].isna().sum() >= 4
        # Pressure should be unchanged
        assert result['pressure'].equals(climate_df['pressure'])

    def test_center(self, climate_df):
        t = RollingWindow(window=5, method='mean', center=True)
        result = t.apply(climate_df)
        # Centered window: NaNs at both edges
        assert result['temperature'].isna().any()

    def test_min_periods(self, hourly_df):
        t = RollingWindow(window=10, method='mean', min_periods=1)
        result = t.apply(hourly_df)
        # With min_periods=1, no NaN values
        assert result['value'].isna().sum() == 0

    def test_rolling_std(self, hourly_df):
        t = RollingWindow(window=5, method='std')
        result = t.apply(hourly_df)
        assert len(result) == len(hourly_df)


# ---------------------------------------------------------------------------
# Anomaly
# ---------------------------------------------------------------------------

class TestAnomaly:

    def test_monthly_anomaly(self, climate_df):
        t = Anomaly(date_column='time', groupby_freq='month')
        result = t.apply(climate_df)
        assert len(result) == len(climate_df)
        # Mean anomaly should be close to 0
        assert abs(result['temperature'].mean()) < 2.0

    def test_dayofyear_anomaly(self, climate_df):
        t = Anomaly(date_column='time', groupby_freq='dayofyear')
        result = t.apply(climate_df)
        assert len(result) == len(climate_df)

    def test_preserves_non_target_columns(self, climate_df):
        t = Anomaly(
            date_column='time', columns=['temperature'],
            groupby_freq='month',
        )
        result = t.apply(climate_df)
        # Pressure should be unchanged
        assert result['pressure'].equals(climate_df['pressure'])
        # Time should be unchanged
        assert result['time'].equals(climate_df['time'])

    def test_specific_columns(self, climate_df):
        t = Anomaly(
            date_column='time', columns=['temperature'],
            groupby_freq='month',
        )
        result = t.apply(climate_df)
        # Temperature should be anomalized
        assert not result['temperature'].equals(climate_df['temperature'])


# ---------------------------------------------------------------------------
# SpatialBBox
# ---------------------------------------------------------------------------

class TestSpatialBBox:

    def test_basic_bbox(self, climate_df):
        t = SpatialBBox(lat_bounds=(30, 50), lon_bounds=(-100, -80))
        result = t.apply(climate_df)
        assert len(result) <= len(climate_df)
        assert all(result['lat'] >= 30)
        assert all(result['lat'] <= 50)
        assert all(result['lon'] >= -100)
        assert all(result['lon'] <= -80)

    def test_full_extent(self, climate_df):
        t = SpatialBBox(lat_bounds=(-90, 90), lon_bounds=(-180, 180))
        result = t.apply(climate_df)
        assert len(result) == len(climate_df)

    def test_empty_result(self, climate_df):
        t = SpatialBBox(lat_bounds=(80, 90), lon_bounds=(150, 160))
        result = t.apply(climate_df)
        assert len(result) == 0

    def test_custom_column_names(self):
        df = pd.DataFrame({
            'latitude': [10, 20, 30, 40, 50],
            'longitude': [-100, -90, -80, -70, -60],
            'value': [1, 2, 3, 4, 5],
        })
        t = SpatialBBox(
            lat_column='latitude', lon_column='longitude',
            lat_bounds=(20, 40), lon_bounds=(-95, -65),
        )
        result = t.apply(df)
        assert len(result) == 3
        assert list(result['value']) == [2, 3, 4]

    def test_antimeridian_crossing(self):
        df = pd.DataFrame({
            'lat': [0, 0, 0, 0, 0],
            'lon': [160, 175, -175, -160, 0],
            'value': [1, 2, 3, 4, 5],
        })
        # Crossing anti-meridian: 170 to -170 (wraps around 180)
        t = SpatialBBox(lat_bounds=(-10, 10), lon_bounds=(170, -170))
        result = t.apply(df)
        assert len(result) == 2
        assert set(result['lon']) == {175, -175}


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

class TestValidation:

    def test_resample_non_datetime_raises(self):
        df = pd.DataFrame({'time': [1, 2, 3], 'value': [10, 20, 30]})
        t = TimeResample(date_column='time', rule='D')
        with pytest.raises(ValueError, match="expected datetime"):
            t.apply(df)

    def test_anomaly_non_datetime_raises(self):
        df = pd.DataFrame({'time': [1, 2, 3], 'value': [10, 20, 30]})
        t = Anomaly(date_column='time', groupby_freq='month')
        with pytest.raises(ValueError, match="expected datetime"):
            t.apply(df)


# ---------------------------------------------------------------------------
# Transform chaining
# ---------------------------------------------------------------------------

class TestChaining:

    def test_resample_then_rolling(self, climate_df):
        df = TimeResample(date_column='time', rule='ME', method='mean').apply(climate_df)
        df = RollingWindow(window=3, method='mean', columns=['temperature']).apply(df)
        assert len(df) <= 24
        assert 'temperature' in df.columns

    def test_bbox_then_anomaly(self, climate_df):
        df = SpatialBBox(lat_bounds=(30, 50), lon_bounds=(-100, -80)).apply(climate_df)
        df = Anomaly(date_column='time', groupby_freq='month').apply(df)
        assert abs(df['temperature'].mean()) < 5.0


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------

class TestRegistration:

    def test_transform_types_registered(self):
        from lumen.transforms import TimeResample, RollingWindow, Anomaly, SpatialBBox
        assert TimeResample.transform_type == 'time_resample'
        assert RollingWindow.transform_type == 'rolling_window'
        assert Anomaly.transform_type == 'anomaly'
        assert SpatialBBox.transform_type == 'spatial_bbox'

    def test_type_resolution(self):
        assert Transform._get_type('time_resample') is TimeResample
        assert Transform._get_type('rolling_window') is RollingWindow
        assert Transform._get_type('anomaly') is Anomaly
        assert Transform._get_type('spatial_bbox') is SpatialBBox


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:

    def test_time_resample_empty_df(self):
        df = pd.DataFrame({
            'time': pd.Series(dtype='datetime64[ns]'),
            'value': pd.Series(dtype='float64'),
        })
        t = TimeResample(date_column='time', rule='D')
        result = t.apply(df)
        assert len(result) == 0

    def test_rolling_window_empty_df(self):
        df = pd.DataFrame({'value': pd.Series(dtype='float64')})
        t = RollingWindow(window=3)
        result = t.apply(df)
        assert len(result) == 0

    def test_anomaly_empty_df(self):
        df = pd.DataFrame({
            'time': pd.Series(dtype='datetime64[ns]'),
            'value': pd.Series(dtype='float64'),
        })
        t = Anomaly(date_column='time')
        result = t.apply(df)
        assert len(result) == 0

    def test_spatial_bbox_empty_df(self):
        df = pd.DataFrame({
            'lat': pd.Series(dtype='float64'),
            'lon': pd.Series(dtype='float64'),
            'value': pd.Series(dtype='float64'),
        })
        t = SpatialBBox(lat_bounds=(0, 10), lon_bounds=(0, 10))
        result = t.apply(df)
        assert len(result) == 0


# ---------------------------------------------------------------------------
# Serialization roundtrip
# ---------------------------------------------------------------------------

class TestSerialization:

    def test_time_resample_roundtrip(self):
        t = TimeResample(date_column='time', rule='W', method='sum')
        spec = t.to_spec()
        assert spec['type'] == 'time_resample'
        t2 = Transform.from_spec(spec)
        assert isinstance(t2, TimeResample)
        assert t2.rule == 'W'
        assert t2.method == 'sum'

    def test_rolling_window_roundtrip(self):
        t = RollingWindow(window=10, method='std', center=True)
        spec = t.to_spec()
        assert spec['type'] == 'rolling_window'
        t2 = Transform.from_spec(spec)
        assert isinstance(t2, RollingWindow)
        assert t2.window == 10
        assert t2.center is True

    def test_anomaly_roundtrip(self):
        t = Anomaly(date_column='time', groupby_freq='dayofyear')
        spec = t.to_spec()
        assert spec['type'] == 'anomaly'
        t2 = Transform.from_spec(spec)
        assert isinstance(t2, Anomaly)
        assert t2.groupby_freq == 'dayofyear'

    def test_spatial_bbox_roundtrip(self):
        t = SpatialBBox(lat_bounds=(30, 60), lon_bounds=(-100, -80))
        spec = t.to_spec()
        assert spec['type'] == 'spatial_bbox'
        t2 = Transform.from_spec(spec)
        assert isinstance(t2, SpatialBBox)
        assert t2.lat_bounds == (30, 60)
        assert t2.lon_bounds == (-100, -80)
