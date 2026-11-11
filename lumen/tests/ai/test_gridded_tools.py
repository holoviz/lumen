import numpy as np
import pandas as pd
import pytest

try:
    import lumen.ai  # noqa
except ModuleNotFoundError:
    pytest.skip("lumen.ai could not be imported, skipping tests.", allow_module_level=True)

xr = pytest.importorskip("xarray")
pytest.importorskip("xarray_sql")

from lumen.ai.tools.gridded import gridded_sql_tools, gridded_tools
from lumen.sources.duckdb import DuckDBSource
from lumen.sources.xarray_sql import XArraySQLSource


def _grid(lat="lat", lon="lon", time="time", units="K", calendar=None):
    times = pd.date_range("2000-01-01", periods=4, freq="15D") if calendar is None else xr.date_range(
        "2000-01-01", periods=4, freq="15D", calendar=calendar, use_cftime=True)
    rng = np.random.default_rng(0)
    return xr.Dataset(
        {"air": ([time, lat, lon], 250 + 40 * rng.random((4, 3, 2)), {"units": units, "long_name": "Air"})},
        coords={
            time: times,
            lat: (lat, [-60.0, 0.0, 60.0], {"units": "degrees_north"}),
            lon: (lon, [10.0, 20.0], {"units": "degrees_east"}),
        },
    )


def _tools(ds):
    source = XArraySQLSource(_dataset=ds)
    return source, {tool.function.__name__: tool.function for tool in gridded_tools([source])}


def test_area_weighted_mean_matches_xarray():
    ds = _grid()
    source, tools = _tools(ds)
    sql = tools["area_weighted_mean"]("air")
    got = source.execute(sql).sort_values("time")["air_weighted_mean"].to_numpy()
    expected = ds.air.weighted(np.cos(np.deg2rad(ds.lat))).mean(("lat", "lon")).to_numpy()
    np.testing.assert_allclose(got, expected)
    overall = source.execute(tools["area_weighted_mean"]("air", per_time=False)).iloc[0, 0]
    np.testing.assert_allclose(overall, ds.air.weighted(np.cos(np.deg2rad(ds.lat))).mean().item())


def test_climatology_and_anomaly_match_xarray():
    ds = _grid()
    source, tools = _tools(ds)
    clim = source.execute(tools["climatology"]("air")).set_index(["month", "lat", "lon"])["air_climatology"]
    expected = ds.air.groupby("time.month").mean().to_series()
    np.testing.assert_allclose(clim.sort_index().to_numpy(), expected.sort_index().to_numpy())
    anomaly = source.execute(tools["anomaly"]("air")).set_index(["time", "lat", "lon"])["air_anomaly"].sort_index()
    expected = (ds.air.groupby("time.month") - ds.air.groupby("time.month").mean()).to_series().sort_index()
    np.testing.assert_allclose(anomaly.to_numpy(), expected.to_numpy())


def test_time_slice_keeps_the_dates_asked_for():
    ds = _grid()
    source, tools = _tools(ds)
    data = source.execute(tools["time_slice"]("air", "2000-01-10", "2000-02-10"))
    assert sorted(data["time"].unique()) == list(ds.time.values[1:3])
    assert len(data) == 2 * 3 * 2


def test_convert_units_reads_the_cf_units():
    ds = _grid(units="degK")
    source, tools = _tools(ds)
    data = source.execute(tools["convert_units"]("air", "degC"))
    np.testing.assert_allclose(np.sort(data["air_degC"].to_numpy()), np.sort(ds.air.to_numpy().ravel() - 273.15))
    assert "converting to furlongs is not one of" in tools["convert_units"]("air", "furlongs")


def test_the_sql_uses_the_grids_own_dimension_names():
    ds = _grid(lat="latitude", lon="longitude", time="valid_time")
    source, tools = _tools(ds)
    sql = tools["area_weighted_mean"]("air")
    assert 'COS(RADIANS("latitude"))' in sql and 'GROUP BY "valid_time"' in sql
    assert len(source.execute(sql)) == 4


@pytest.mark.parametrize("calendar", ["360_day", "noleap"])
def test_a_calendar_sql_cannot_date_is_named_instead_of_guessed(calendar):
    pytest.importorskip("cftime")
    _source, tools = _tools(_grid(calendar=calendar))
    message = tools["climatology"]("air")
    assert f"{calendar} calendar" in message and "SELECT" not in message


def test_an_unknown_table_lists_the_gridded_tables():
    _source, tools = _tools(_grid())
    assert tools["climatology"]("nope") == "No gridded table 'nope'; the gridded tables are air."


def test_the_tools_are_offered_only_with_a_gridded_source():
    tabular = DuckDBSource(tables={"t": "SELECT 1 AS x"})
    assert gridded_sql_tools({"sources": [tabular]}) is None
    names = [tool.function.__name__ for tool in gridded_sql_tools({"sources": [XArraySQLSource(_dataset=_grid())]})]
    assert names == ["time_slice", "area_weighted_mean", "climatology", "anomaly", "convert_units"]


def test_the_sql_agent_offers_the_tools_for_a_grid():
    from lumen.ai.actor import _merge_prompt_tools
    from lumen.ai.agents.sql import SQLAgent
    agent = SQLAgent()
    source = XArraySQLSource(_dataset=_grid())
    names = {getattr(tool, "function", tool).__name__ for tool in _merge_prompt_tools(agent.llm_tools, None, {"sources": [source]})}
    assert {"area_weighted_mean", "climatology", "anomaly", "convert_units", "time_slice"} <= names
    tabular = DuckDBSource(tables={"t": "SELECT 1 AS x"})
    assert _merge_prompt_tools(agent.llm_tools, None, {"sources": [tabular]}) is None


def test_a_derived_table_is_not_a_grid():
    source = XArraySQLSource(_dataset=_grid())
    derived = source.create_sql_expr_source({"warm": "SELECT * FROM air WHERE air > 270"})
    assert "warm" in derived.get_tables() and derived.column_units("warm") == {}
    tools = {tool.function.__name__: tool.function for tool in gridded_tools([derived])}
    assert tools["climatology"]("warm").startswith("No gridded table 'warm'")
    assert "air" in tools["climatology"]("air")


def test_a_climatology_by_quarter():
    source, tools = _tools(_grid())
    assert "EXTRACT(QUARTER FROM" in tools["climatology"]("air", period="quarter")
