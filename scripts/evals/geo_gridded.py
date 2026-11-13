"""
Spatial and gridded questions with known answers, asked through lumen.ai.

    python scripts/evals/geo_gridded.py --llm kilo --era5 era5_t2m_2022.nc

Each case asks a question of a small fixture and checks the answer's data
against a value computed directly with geopandas, pyproj or xarray, never
by Lumen. The checks are tight enough that the usual mistakes fail: a mean
without latitude weights, an area measured in degrees or in Web Mercator,
a distance in degrees, longitude and latitude swapped.

The ERA5 file is any year of 2 m temperature named ``t2m`` with ``time``,
``lat`` and ``lon`` dimensions, e.g. a slice of WeatherBench2's 64x32 ERA5.
Without --era5 the ERA5 cases are skipped.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import time

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import xarray as xr

from pyproj import Geod
from shapely.geometry import LineString, Point, box

import lumen

from lumen.ai.answer import Answer, aask
from lumen.ai.llm import ClaudeCode, Kilo

GEOD = Geod(ellps="WGS84")
EARTH_RADIUS_KM = 6371.0088  # the mean radius ST_Distance_Sphere uses

CITIES = {
    "Berlin": (13.405, 52.520, 3_850_000), "Paris": (2.352, 48.857, 2_100_000),
    "Madrid": (-3.704, 40.417, 3_330_000), "Rome": (12.496, 41.903, 2_760_000),
    "Warsaw": (21.012, 52.230, 1_860_000), "Vienna": (16.373, 48.208, 1_980_000),
    "Prague": (14.438, 50.076, 1_360_000), "Hamburg": (9.993, 53.551, 1_900_000),
    "Munich": (11.582, 48.135, 1_510_000), "Copenhagen": (12.568, 55.676, 660_000),
    "Stockholm": (18.069, 59.329, 980_000), "Oslo": (10.752, 59.914, 710_000),
    "Lisbon": (-9.139, 38.722, 550_000), "Athens": (23.728, 37.984, 640_000),
    "Budapest": (19.040, 47.498, 1_700_000), "Brussels": (4.352, 50.847, 1_220_000),
    "Amsterdam": (4.904, 52.368, 920_000), "Zurich": (8.541, 47.377, 430_000),
    "Milan": (9.190, 45.464, 1_370_000), "Helsinki": (24.938, 60.170, 660_000),
}
REGIONS = {
    "North": box(-10, 55, 35, 71), "Central": box(-10, 47, 35, 55), "South": box(-10, 35, 35, 47),
}
ROADS = {
    "Berlin-Prague": ["Berlin", "Prague"], "Paris-Brussels-Amsterdam": ["Paris", "Brussels", "Amsterdam"],
    "Madrid-Lisbon": ["Madrid", "Lisbon"],
}


def great_circle_km(lon1, lat1, lon2, lat2):
    lon1, lat1, lon2, lat2 = map(np.radians, (lon1, lat1, lon2, lat2))
    a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return 2 * EARTH_RADIUS_KM * np.arcsin(np.sqrt(a))


def spatial_fixture() -> dict[str, gpd.GeoDataFrame]:
    places = gpd.GeoDataFrame(
        {"name": list(CITIES), "population": [pop for *_, pop in CITIES.values()]},
        geometry=[Point(lon, lat) for lon, lat, _ in CITIES.values()], crs="EPSG:4326",
    )
    regions = gpd.GeoDataFrame({"region": list(REGIONS)}, geometry=list(REGIONS.values()), crs="EPSG:4326")
    roads = gpd.GeoDataFrame(
        {"road": list(ROADS)},
        geometry=[LineString([CITIES[c][:2] for c in cities]) for cities in ROADS.values()], crs="EPSG:4326",
    )
    # Ten 10 km squares near Oslo as drawn in Web Mercator, where 10 km of map
    # is only about 5 km of ground at 60 degrees north
    x0, y0 = gpd.GeoSeries([Point(10.75, 59.9)], crs="EPSG:4326").to_crs(3857).iloc[0].coords[0]
    parcels = gpd.GeoDataFrame(
        {"parcel": [f"P{i}" for i in range(10)]},
        geometry=[box(x0 + i * 10_000, y0, x0 + i * 10_000 + 10_000, y0 + 10_000) for i in range(10)],
        crs="EPSG:3857",
    )
    return {"places": places, "regions": regions, "roads": roads, "parcels": parcels}


def weighted_mean(da: xr.DataArray) -> float:
    return float(da.weighted(np.cos(np.deg2rad(da.lat))).mean())


# ---------------------------------------------------------------- checks

def numbers(df: pd.DataFrame) -> list[float]:
    return [float(v) for col in df.select_dtypes("number").columns for v in df[col].dropna()]


def close(value: float, expected: float, rel: float) -> bool:
    return abs(value - expected) <= rel * max(abs(expected), 1e-9)


def any_close(expected: float, rel: float) -> Callable[[pd.DataFrame, Answer], bool]:
    """A small result holds the expected number (any column, any row)."""
    return lambda df, _: len(df) <= 3 and any(close(v, expected, rel) for v in numbers(df))


def per_key(expected: dict[str, float], rel: float) -> Callable[[pd.DataFrame, Answer], bool]:
    """Every key maps to its expected number, in some string and numeric column pair."""
    def check(df, _):
        for key_col in df.select_dtypes(exclude="number").columns:
            keys = df[key_col].astype(str)
            if not set(expected) <= set(keys):
                continue
            for val_col in df.select_dtypes("number").columns:
                values = dict(zip(keys, df[val_col], strict=False))
                if all(close(float(values[k]), v, rel) for k, v in expected.items()):
                    return True
        return False
    return check


def names(expected: str) -> Callable[[pd.DataFrame, Answer], bool]:
    """The first row names the expected item, so a ranked list counts too."""
    return lambda df, _: len(df) > 0 and expected in df.astype(str).to_numpy()[0].tolist()


def first_month(expected: int) -> Callable[[pd.DataFrame, Answer], bool]:
    """The first row names the expected month, as a number or its name."""
    def check(df, _):
        if not len(df):
            return False
        first = df.head(1)
        return (expected in [int(v) for v in numbers(first) if float(v).is_integer()]
                or pd.Timestamp(2022, expected, 1).strftime("%B") in first.astype(str).to_numpy()[0].tolist())
    return check


def declines(missing: str) -> Callable[[pd.DataFrame, Answer], bool]:
    """No numbers invented for something the data does not hold, and the answer says so."""
    def check(df, answer):
        if answer.status == "unanswerable":
            return True
        invented = any(missing in str(col).lower() for col in df.columns) and bool(numbers(df))
        declined = ("no ", "not ", "n't", "cannot", "unavailable")
        text = answer.value.lower().replace("’", "'")
        # A decline has to be about the missing thing, not any "no" or "not" in the text
        return not invented and missing in text and any(word in text for word in declined)
    return check


@dataclass
class Case:
    suite: str
    name: str
    fixture: str
    question: str
    check: Callable[[pd.DataFrame, Answer], bool]
    #: SQL that answers correctly, and SQL making the usual mistake; --selfcheck
    #: runs both so a check that cannot tell them apart is caught
    reference: str | None = None
    trap: str | None = None


def spatial_cases(f: dict[str, gpd.GeoDataFrame]) -> list[Case]:
    places, regions, roads, parcels = f["places"], f["regions"], f["roads"], f["parcels"]
    joined = gpd.sjoin(places, regions, predicate="within")
    counts = joined.groupby("region").size().to_dict()
    lon, lat = places.geometry.x, places.geometry.y
    berlin = CITIES["Berlin"]
    dist = great_circle_km(berlin[0], berlin[1], lon, lat)
    radius = 750
    # Keep the radius well clear of any city, so rounding cannot flip a case
    assert np.min(np.abs(dist[dist > 0] - radius)) > 40
    within = int(((dist > 0) & (dist <= radius)).sum())
    nearest = places.name[np.argmin(great_circle_km(10, 50, lon, lat))]
    areas = dict(zip(regions.region, regions.to_crs(6933).area / 1e6, strict=False))
    road_km = {r: GEOD.geometry_length(g) / 1000 for r, g in zip(roads.road, roads.geometry, strict=False)}
    parcel_km2 = float(parcels.to_crs(6933).area.sum() / 1e6)
    north_pop = int(joined.loc[joined.region == "North", "population"].sum())
    berlin_geom = "(SELECT geometry FROM places WHERE name = 'Berlin')"
    # DuckDB's sphere and spheroid functions read points as (lat, lon)
    sphere = "ST_Distance_Sphere(ST_FlipCoordinates({a}), ST_FlipCoordinates({b}))"
    to_area = "ST_Transform(geometry, '{crs}', 'EPSG:6933', always_xy := true)"
    return [
        Case("spatial", "points_in_polygons", "spatial", "How many places are inside each region?", per_key(counts, 0),
             "SELECT r.region, COUNT(*) AS n FROM places p JOIN regions r ON ST_Within(p.geometry, r.geometry) GROUP BY r.region",
             "SELECT r.region, COUNT(*) AS n FROM places p CROSS JOIN regions r GROUP BY r.region"),
        Case("spatial", "within_distance", "spatial",
             f"How many places are within {radius} km of Berlin, not counting Berlin itself?", any_close(within, 0),
             f"SELECT COUNT(*) AS n FROM places WHERE name <> 'Berlin' AND {sphere.format(a='geometry', b=berlin_geom)} <= {radius * 1000}",
             f"SELECT COUNT(*) AS n FROM places WHERE name <> 'Berlin' AND ST_Distance_Sphere(geometry, {berlin_geom}) <= {radius * 1000}"),
        Case("spatial", "region_area", "spatial", "What is the area of each region in square kilometres?", per_key(areas, 0.01),
             f"SELECT region, ST_Area({to_area.format(crs='EPSG:4326')}) / 1e6 AS km2 FROM regions",
             "SELECT region, ST_Area(geometry) AS km2 FROM regions"),
        Case("spatial", "nearest_point", "spatial",
             "Which place is closest to the point at longitude 10, latitude 50?", names(nearest),
             f"SELECT name FROM places ORDER BY {sphere.format(a='geometry', b='ST_Point(10, 50)')} LIMIT 1",
             "SELECT name FROM places ORDER BY ST_Distance(geometry, ST_Point(50, 10)) LIMIT 1"),
        Case("spatial", "population_in_region", "spatial",
             "What is the total population of the places in the North region?", any_close(north_pop, 0),
             "SELECT SUM(p.population) AS total FROM places p JOIN regions r ON ST_Within(p.geometry, r.geometry) WHERE r.region = 'North'",
             "SELECT SUM(population) AS total FROM places"),
        Case("spatial", "line_length", "spatial", "How long is each road in kilometres?", per_key(road_km, 0.02),
             "SELECT road, ST_Length_Spheroid(ST_FlipCoordinates(geometry)) / 1000 AS km FROM roads",
             "SELECT road, ST_Length(geometry) AS km FROM roads"),
        Case("spatial", "projected_area", "spatial",
             "What is the total ground area of all parcels in square kilometres?", any_close(parcel_km2, 0.02),
             f"SELECT SUM(ST_Area({to_area.format(crs='EPSG:3857')})) / 1e6 AS km2 FROM parcels",
             "SELECT SUM(ST_Area(geometry)) / 1e6 AS km2 FROM parcels"),
        Case("spatial", "unanswerable", "spatial", "What is the elevation in metres of Berlin?", declines("elev")),
        *_more_spatial_cases(f),
    ]


def _more_spatial_cases(f: dict[str, gpd.GeoDataFrame]) -> list[Case]:
    """Traps the review and the CRS rules are for: degree buffers, map metres, joins, densities."""
    places, regions, roads, parcels = f["places"], f["regions"], f["roads"], f["parcels"]
    paris = CITIES["Paris"]
    lon, lat = places.geometry.x, places.geometry.y
    near_paris = int(((great_circle_km(paris[0], paris[1], lon, lat) <= 450) & (places.name != "Paris")).sum())
    crossing_south = int(gpd.sjoin(roads, regions[regions.region == "South"], predicate="intersects").road.nunique())
    centres = parcels.centroid.to_crs(4326)
    p0, p9 = centres.iloc[0], centres.iloc[9]
    parcel_gap_km = float(great_circle_km(p0.x, p0.y, p9.x, p9.y))
    joined = gpd.sjoin(places, regions, predicate="within")
    area = dict(zip(regions.region, regions.to_crs(6933).area / 1e6, strict=False))
    density = {r: float(pop) / area[r] for r, pop in joined.groupby("region").population.sum().items()}
    paris_geom = "(SELECT geometry FROM places WHERE name = 'Paris')"
    sphere = "ST_Distance_Sphere(ST_FlipCoordinates({a}), ST_FlipCoordinates({b}))"
    to_area = "ST_Transform(r.geometry, 'EPSG:4326', 'EPSG:6933', always_xy := true)"
    centre = "ST_Transform(ST_Centroid(geometry), 'EPSG:3857', 'EPSG:4326', always_xy := true)"
    return [
        Case("spatial", "degree_buffer", "spatial",
             "How many places are within 450 km of Paris, not counting Paris itself?", any_close(near_paris, 0),
             f"SELECT COUNT(*) AS n FROM places WHERE name <> 'Paris' AND {sphere.format(a='geometry', b=paris_geom)} <= 450000",
             f"SELECT COUNT(*) AS n FROM places WHERE name <> 'Paris' AND ST_DWithin(geometry, {paris_geom}, 450000)"),
        Case("spatial", "lines_cross_polygon", "spatial",
             "How many roads pass through the South region?", any_close(crossing_south, 0),
             "SELECT COUNT(DISTINCT r.road) AS n FROM roads r JOIN regions g ON ST_Intersects(r.geometry, g.geometry) WHERE g.region = 'South'",
             "SELECT COUNT(*) AS n FROM roads r CROSS JOIN regions g WHERE g.region = 'South'"),
        Case("spatial", "map_metres", "spatial",
             "How far apart on the ground, in kilometres, are the centres of parcels P0 and P9?", any_close(parcel_gap_km, 0.02),
             f"SELECT ST_Distance_Sphere(ST_FlipCoordinates(a.c), ST_FlipCoordinates(b.c)) / 1000 AS km FROM "
             f"(SELECT {centre} AS c FROM parcels WHERE parcel = 'P0') a, (SELECT {centre} AS c FROM parcels WHERE parcel = 'P9') b",
             "SELECT ST_Distance(ST_Centroid(a.geometry), ST_Centroid(b.geometry)) / 1000 AS km FROM parcels a, parcels b "
             "WHERE a.parcel = 'P0' AND b.parcel = 'P9'"),
        Case("spatial", "population_density", "spatial",
             "What is the population density of the places in each region, in people per square kilometre of the region?",
             per_key(density, 0.02),
             f"SELECT r.region, SUM(p.population) / (ST_Area({to_area}) / 1e6) AS density FROM places p "
             "JOIN regions r ON ST_Within(p.geometry, r.geometry) GROUP BY r.region, r.geometry",
             "SELECT r.region, SUM(p.population) / ST_Area(r.geometry) AS density FROM places p "
             "JOIN regions r ON ST_Within(p.geometry, r.geometry) GROUP BY r.region, r.geometry"),
        Case("spatial", "unanswerable_coast", "spatial", "Which of these places lie on the coast?", declines("coast")),
    ]


def gridded_cases(air: xr.Dataset, era5: xr.Dataset | None) -> list[Case]:
    a = air.air
    box = a.sel(lat=slice(40, 30), lon=slice(250, 260))
    # A plain and an area-weighted box mean are both fair readings
    box_means = [float(box.mean()) - 273.15, weighted_mean(box) - 273.15]
    hottest = a.where(a == a.max(), drop=True)
    weighted = "SUM({v} * COS(RADIANS(lat))) / SUM(COS(RADIANS(lat)))"
    # The grid holds five days; weighted, the warmest is not the plain mean's warmest
    daily = a.weighted(np.cos(np.deg2rad(a.lat))).mean(("lat", "lon")).resample(time="1D").mean()
    warmest_day = str(daily.idxmax().dt.strftime("%Y-%m-%d").item())
    day = "date_trunc('day', time)"
    cases = [
        Case("gridded", "box_mean_celsius", "air",
             "What is the mean air temperature in degrees Celsius over latitudes 30 to 40 and longitudes 250 to 260?",
             lambda df, answer: any(any_close(m, 0.01)(df, answer) for m in box_means),
             "SELECT AVG(air) - 273.15 AS c FROM air WHERE lat BETWEEN 30 AND 40 AND lon BETWEEN 250 AND 260",
             "SELECT AVG(air) AS c FROM air WHERE lat BETWEEN 30 AND 40 AND lon BETWEEN 250 AND 260"),
        Case("gridded", "weighted_mean", "air",
             "What is the area-weighted mean air temperature over the whole grid, in Kelvin?",
             any_close(weighted_mean(a), 0.002),
             f"SELECT {weighted.format(v='air')} AS k FROM air", "SELECT AVG(air) AS k FROM air"),
        Case("gridded", "hottest_point", "air", "At which latitude and longitude was the air temperature highest?",
             lambda df, _: len(df) <= 3 and all(any(close(v, float(hottest[c].item()), 1e-6) for v in numbers(df)) for c in ("lat", "lon")),
             "SELECT lat, lon FROM air ORDER BY air DESC LIMIT 1", "SELECT lat, lon FROM air ORDER BY air ASC LIMIT 1"),
        Case("gridded", "warmest_day", "air", "Which day had the highest area-weighted mean air temperature?",
             lambda df, _: len(df) > 0 and any(str(v).startswith(warmest_day) for v in df.iloc[0]),
             f"SELECT {day} AS day, {weighted.format(v='air')} AS w FROM air GROUP BY 1 ORDER BY w DESC LIMIT 1",
             f"SELECT {day} AS day, AVG(air) AS w FROM air GROUP BY 1 ORDER BY w DESC LIMIT 1"),
    ]
    cases += _more_air_cases(air)
    if era5 is None:
        return cases
    t = era5.t2m
    # Every step weighs the same, so this is the weighted mean over each month
    monthly = t.weighted(np.cos(np.deg2rad(t.lat))).mean(("lat", "lon")).groupby("time.month").mean()
    warmest = int(monthly.idxmax())
    july_box = float(t.sel(time="2022-07").where((t.lat >= 30) & (t.lat <= 45) & (t.lon >= 0) & (t.lon <= 40)).max()) - 273.15
    month_mean = f"SELECT EXTRACT(MONTH FROM time) AS month, {weighted.format(v='t2m')} AS w FROM t2m GROUP BY 1"
    # Longitudes run 0 to 360, so a box from -30 to 30 wraps across 0
    meridian = t.where((t.lat >= 35) & (t.lat <= 60) & ((t.lon >= 330) | (t.lon <= 30)))
    meridian_means = [float(meridian.mean()) - 273.15, weighted_mean(meridian) - 273.15]
    point = t.sel(lat=52.5, lon=13.4, method="nearest")
    point_mean = float(point.mean()) - 273.15
    cases += [
        Case("gridded", "warmest_month", "era5",
             "Which month of 2022 had the highest area-weighted global mean 2 m temperature?",
             # The first row, so a list ranked warmest first counts too
             first_month(warmest),
             f"{month_mean} ORDER BY w DESC LIMIT 1", f"{month_mean} ORDER BY w ASC LIMIT 1"),
        Case("gridded", "july_box_max", "era5",
             "What was the highest 2 m temperature in degrees Celsius in July 2022 between latitudes 30 and 45 and longitudes 0 and 40?",
             any_close(july_box, 0.01),
             "SELECT MAX(t2m) - 273.15 AS c FROM t2m WHERE EXTRACT(MONTH FROM time) = 7 AND lat BETWEEN 30 AND 45 AND lon BETWEEN 0 AND 40",
             "SELECT MAX(t2m) AS c FROM t2m WHERE EXTRACT(MONTH FROM time) = 7 AND lat BETWEEN 30 AND 45 AND lon BETWEEN 0 AND 40"),
        Case("gridded", "global_weighted_celsius", "era5",
             "What was the area-weighted global mean 2 m temperature in 2022, in degrees Celsius?",
             any_close(weighted_mean(t) - 273.15, 0.02),
             f"SELECT {weighted.format(v='t2m')} - 273.15 AS c FROM t2m", "SELECT AVG(t2m) - 273.15 AS c FROM t2m"),
        Case("gridded", "seasonal_difference", "era5",
             "How much warmer, in degrees Celsius, was the area-weighted global mean 2 m temperature in July 2022 than in January 2022?",
             any_close(float(monthly.sel(month=7) - monthly.sel(month=1)), 0.03),
             f"SELECT MAX(w) FILTER (WHERE month = 7) - MAX(w) FILTER (WHERE month = 1) AS d FROM ({month_mean})",
             "SELECT AVG(t2m) FILTER (WHERE EXTRACT(MONTH FROM time) = 7) - AVG(t2m) FILTER (WHERE EXTRACT(MONTH FROM time) = 1) AS d FROM t2m"),
        Case("gridded", "box_across_meridian", "era5",
             "What was the mean 2 m temperature in degrees Celsius in 2022 between latitudes 35 and 60 and longitudes -30 and 30?",
             lambda df, answer: any(any_close(m, 0.01)(df, answer) for m in meridian_means),
             "SELECT AVG(t2m) - 273.15 AS c FROM t2m WHERE lat BETWEEN 35 AND 60 AND (lon >= 330 OR lon <= 30)",
             "SELECT AVG(t2m) - 273.15 AS c FROM t2m WHERE lat BETWEEN 35 AND 60 AND lon BETWEEN -30 AND 30"),
        Case("gridded", "nearest_grid_point", "era5",
             "What was the mean 2 m temperature in degrees Celsius in 2022 at the grid point nearest to latitude 52.5, longitude 13.4?",
             any_close(point_mean, 0.01),
             f"SELECT AVG(t2m) - 273.15 AS c FROM t2m WHERE lat = {float(point.lat)} AND lon = {float(point.lon)}",
             "SELECT AVG(t2m) - 273.15 AS c FROM t2m WHERE lat = 52.5 AND lon = 13.4"),
    ]
    return cases


def _more_air_cases(air: xr.Dataset) -> list[Case]:
    """Units, anomalies and latitude bands on the NCEP air temperature grid."""
    a = air.air
    weights = np.cos(np.deg2rad(a.lat))
    above_20c = int((a > 293.15).sum())
    band = a.sel(lat=slice(50, 30))
    per_step = a.weighted(weights).mean(("lat", "lon"))
    last_anomaly = float(per_step.isel(time=-1) - per_step.mean())
    weighted = "SUM(air * COS(RADIANS(lat))) / SUM(COS(RADIANS(lat)))"
    return [
        Case("gridded", "kelvin_threshold", "air",
             "How many grid cells, counting every time step, were warmer than 20 degrees Celsius?", any_close(above_20c, 0),
             "SELECT COUNT(*) AS n FROM air WHERE air - 273.15 > 20", "SELECT COUNT(*) AS n FROM air WHERE air > 20"),
        Case("gridded", "latitude_band", "air",
             "What is the area-weighted mean air temperature in Kelvin between latitudes 30 and 50?",
             any_close(float(band.weighted(np.cos(np.deg2rad(band.lat))).mean()), 0.0005),
             f"SELECT {weighted} AS k FROM air WHERE lat BETWEEN 30 AND 50",
             "SELECT AVG(air) AS k FROM air WHERE lat BETWEEN 30 AND 50"),
        Case("gridded", "last_step_anomaly", "air",
             "How far, in Kelvin, was the area-weighted mean air temperature at the last time step from its mean over all time steps?",
             any_close(last_anomaly, 0.03),
             f"SELECT w - AVG(w) OVER () AS anomaly FROM (SELECT time, {weighted} AS w FROM air GROUP BY time) ORDER BY time DESC LIMIT 1",
             "SELECT AVG(air) - (SELECT AVG(air) FROM air) AS anomaly FROM air WHERE time = (SELECT MAX(time) FROM air)"),
        Case("gridded", "unanswerable_rain", "air", "How much rain fell over this grid?", declines("rain")),
    ]


def era5_like() -> xr.Dataset:
    """A grid named as ARCO-ERA5 names it: latitude, longitude, valid_time, in Kelvin."""
    rng = np.random.default_rng(1)
    lat = np.linspace(60, -60, 7)
    values = 288 + 15 * np.cos(np.deg2rad(lat))[None, :, None] + rng.normal(0, 1, (4, 7, 8))
    return xr.Dataset(
        {"t2m": (["valid_time", "latitude", "longitude"], values, {"units": "K", "long_name": "2 metre temperature"})},
        coords={"valid_time": pd.date_range("2020-07-01", periods=4, freq="6h"),
                "latitude": ("latitude", lat, {"units": "degrees_north"}),
                "longitude": ("longitude", np.arange(0, 360, 45.0), {"units": "degrees_east"})},
    )


def noleap_grid() -> xr.Dataset:
    """A year of daily temperature on a CMIP noleap calendar, warmest in July."""
    times = xr.date_range("2050-01-01", periods=365, freq="D", calendar="noleap", use_cftime=True)
    doy = np.arange(365)
    values = 285 + 12 * np.cos(2 * np.pi * (doy - 196) / 365)
    return xr.Dataset(
        {"tas": (["time", "lat", "lon"], np.broadcast_to(values[:, None, None], (365, 2, 2)).copy(), {"units": "K"})},
        coords={"time": times, "lat": [10.0, 20.0], "lon": [0.0, 10.0]},
    )


def extra_grid_cases() -> list[Case]:
    """Grids the defaults do not fit: ERA5's own names, and a model calendar."""
    e = era5_like().t2m
    weighted_c = float(e.weighted(np.cos(np.deg2rad(e.latitude))).mean()) - 273.15
    monthly = noleap_grid().tas.mean(("lat", "lon")).groupby("time.month").mean()
    return [
        Case("gridded", "era5_names", "era5_like",
             "What is the area-weighted mean 2 m temperature in degrees Celsius?", any_close(weighted_c, 0.01),
             'SELECT SUM(t2m * COS(RADIANS(latitude))) / SUM(COS(RADIANS(latitude))) - 273.15 AS c FROM t2m',
             "SELECT AVG(t2m) - 273.15 AS c FROM t2m"),
        # xarray-sql stores noleap dates as time elapsed in that calendar, so SQL
        # reads 2050-01-01 as 2049-12-12 and EXTRACT(MONTH) gives June; naming
        # the calendar problem is as right as finding July some other way
        Case("gridded", "noleap_warmest_month", "noleap",
             "Which month has the highest mean temperature?",
             lambda df, answer: first_month(int(monthly.idxmax()))(df, answer)
             # Naming the calendar problem counts, but not alongside a wrong month
             or (any(word in answer.value.lower() for word in ("noleap", "calendar"))
                 and not any(first_month(month)(df, answer) for month in range(1, 13)))),
    ]


def selfcheck(cases: list[Case], sources: dict) -> bool:
    """Each reference SQL must pass its check and each trap SQL must fail it."""
    ok = True
    blank = Answer(question="", status="success")
    for case in cases:
        if case.reference is None:
            continue
        source = sources[case.fixture]
        good = case.check(pd.DataFrame(source.execute(case.reference)), blank)
        bad = case.check(pd.DataFrame(source.execute(case.trap)), blank)
        print(f"{'ok  ' if good and not bad else 'BAD '} {case.name:24} reference={'pass' if good else 'FAIL'} trap={'FAIL' if not bad else 'pass'}")
        ok &= good and not bad
    return ok


async def run_case(case: Case, source, llm) -> dict:
    start = time.monotonic()
    try:
        answer = await aask(source, case.question, llm=llm)
        df = answer.pipeline.data if answer.pipeline is not None else pd.DataFrame()
        passed = bool(case.check(pd.DataFrame(df), answer))
        sql, error, value = answer.sql, answer.error, answer.value
    except Exception as e:
        passed, sql, error, value = False, None, f"{type(e).__name__}: {e}", ""
    return {"suite": case.suite, "case": case.name, "passed": passed, "sql": sql, "error": error,
            "value": value[:500], "seconds": round(time.monotonic() - start, 1)}


async def main(args):
    import hvsampledata
    llm_cls = {"kilo": Kilo, "claude-code": ClaudeCode}[args.llm]
    llm = llm_cls(model_kwargs={"default": {"model": args.model}}) if args.model else llm_cls()
    # A provider's default model can be None in the settings
    model = args.model or llm.model_kwargs.get("default", {}).get("model") or "default"
    era5 = xr.open_dataset(args.era5) if args.era5 else None
    air = hvsampledata.air_temperature("xarray")
    sources = {
        "spatial": lumen.read(spatial_fixture()),
        "air": lumen.read({"air": air}),
        **({"era5": lumen.read({"t2m": era5})} if era5 is not None else {}),
        "era5_like": lumen.read({"t2m": era5_like()}),
        "noleap": lumen.read({"tas": noleap_grid()}),
    }
    cases = spatial_cases(spatial_fixture()) + gridded_cases(air, era5) + extra_grid_cases()
    if args.selfcheck:
        raise SystemExit(0 if selfcheck(cases, sources) else 1)
    if args.only:
        cases = [c for c in cases if c.suite == args.only]
    results = []
    for repeat in range(args.repeat):
        for case in cases:
            result = {**await run_case(case, sources[case.fixture], llm), "repeat": repeat}
            print(f"{'PASS' if result['passed'] else 'FAIL'} {case.suite:8} {case.name:24} {result['seconds']:6}s")
            results.append(result)
    for suite in sorted({r["suite"] for r in results}):
        rows = [r for r in results if r["suite"] == suite]
        print(f"{suite}: {sum(r['passed'] for r in rows)}/{len(rows)} passed")
    passed = sum(r["passed"] for r in results)
    # One line per run, to paste into a table of pass rates by model
    # What was run, so two rows are only compared when they ran the same cases
    options = f"{args.only or 'all'}, x{args.repeat}, {'era5' if args.era5 else 'no era5'}"
    print(f"| {args.llm} | {model} | {options} | {passed}/{len(results)} | {100 * passed / max(len(results), 1):.0f}% |")
    if args.out:
        Path(args.out).write_text(json.dumps({
            "llm": args.llm, "model": model, "era5": args.era5, "only": args.only, "repeat": args.repeat,
            "results": results,
        }, indent=1))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--llm", choices=["kilo", "claude-code"], default="kilo")
    parser.add_argument("--model", help="The model the LLM provider runs, recorded with the results")
    parser.add_argument("--era5", help="NetCDF or Zarr with t2m(time, lat, lon)")
    parser.add_argument("--only", choices=["spatial", "gridded"])
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--out", help="Write the results as JSON")
    parser.add_argument("--selfcheck", action="store_true", help="Check the checks with reference and trap SQL, no LLM")
    asyncio.run(main(parser.parse_args()))
