"""
Small synthetic tables the scenarios run on, generated deterministically so
nothing large is committed. Each one keeps the trap a playtest fell into on
the real data: a project-level column repeated on every turbine row, place
names stored as ``, CA`` rather than ``California``, and a date column whose
last month is not the current one.
"""
from __future__ import annotations

import functools

from datetime import datetime, timedelta

import numpy as np
import pandas as pd

from ...sources.duckdb import DuckDBSource

STATES = [
    ("TX", "Texas", "South"), ("IA", "Iowa", "Midwest"), ("OK", "Oklahoma", "South"),
    ("KS", "Kansas", "Midwest"), ("IL", "Illinois", "Midwest"), ("CA", "California", "West"),
    ("MN", "Minnesota", "Midwest"), ("CO", "Colorado", "West"), ("WY", "Wyoming", "West"),
    ("OR", "Oregon", "West"), ("WA", "Washington", "West"), ("IN", "Indiana", "Midwest"),
    ("ND", "North Dakota", "Midwest"), ("NM", "New Mexico", "West"), ("NY", "New York", "Northeast"),
    ("PA", "Pennsylvania", "Northeast"), ("ME", "Maine", "Northeast"), ("SD", "South Dakota", "Midwest"),
    ("NE", "Nebraska", "Midwest"), ("MT", "Montana", "West"), ("ID", "Idaho", "West"),
    ("MI", "Michigan", "Midwest"), ("OH", "Ohio", "Midwest"), ("VT", "Vermont", "Northeast"),
]
MANUFACTURERS = [
    "GE Wind", "Vestas", "Siemens", "Gamesa", "Nordex", "Suzlon", "Mitsubishi", "Acciona", "Clipper",
    "REpower", "Goldwind", "Senvion", "Enercon", "Bonus", "NEG Micon", "Zond", "DeWind", "Kenetech",
]
# Centroids the turbine coordinates scatter around, by state.
CENTROIDS = {
    "TX": (-99.3, 31.5), "IA": (-93.5, 42.0), "OK": (-97.5, 35.6), "KS": (-98.4, 38.5), "IL": (-89.2, 40.0),
    "CA": (-119.5, 36.5), "MN": (-94.3, 46.0), "CO": (-105.5, 39.0), "WY": (-107.5, 43.0), "OR": (-120.5, 44.0),
    "WA": (-120.5, 47.4), "IN": (-86.3, 40.0), "ND": (-100.5, 47.5), "NM": (-106.1, 34.4), "NY": (-75.5, 42.9),
    "PA": (-77.6, 40.9), "ME": (-69.2, 45.3), "SD": (-100.2, 44.4), "NE": (-99.8, 41.5), "MT": (-109.6, 47.0),
    "ID": (-114.6, 44.4), "MI": (-84.7, 43.6), "OH": (-82.8, 40.3), "VT": (-72.7, 44.0),
}


@functools.cache
def turbines() -> pd.DataFrame:
    """
    One row per turbine, like the USGS wind turbine database. ``p_cap`` is
    the project's capacity in MW, the same on every turbine row of the
    project, so summing it over rows overcounts by the project's turbine
    count; ``t_cap`` is the turbine's own capacity in kW.
    """
    rng = np.random.default_rng(7)
    rows = []
    case_id = 3000000
    weights = np.linspace(3.0, 0.4, len(STATES))
    weights /= weights.sum()
    for project in range(90):
        state = STATES[rng.choice(len(STATES), p=weights)][0]
        year = int(rng.integers(1999, 2023))
        count = int(rng.integers(3, 18))
        t_cap = int(rng.choice([1000, 1500, 1650, 2000, 2300, 2500, 3000]))
        manufacturer = MANUFACTURERS[int(rng.integers(0, len(MANUFACTURERS)))]
        hub = round(float(rng.uniform(60, 110)), 1)
        x0, y0 = CENTROIDS[state]
        x0, y0 = x0 + rng.uniform(-1.5, 1.5), y0 + rng.uniform(-1.0, 1.0)
        name = f"{['North', 'South', 'East', 'West', 'Big', 'High', 'Red', 'Blue', 'Lone'][project % 9]} " \
               f"{['Ridge', 'Mesa', 'Prairie', 'Creek', 'Hills', 'Plains', 'Valley', 'Peak', 'Flats', 'Bluff'][project // 9]}"
        for _ in range(count):
            case_id += int(rng.integers(1, 40))
            rows.append({
                "case_id": case_id, "t_state": state, "p_name": name, "p_year": year, "p_tnum": count,
                "p_cap": count * t_cap / 1000, "t_manu": manufacturer, "t_cap": t_cap, "t_hh": hub,
                "xlong": round(x0 + float(rng.normal(0, 0.05)), 4), "ylat": round(y0 + float(rng.normal(0, 0.05)), 4),
            })
    return pd.DataFrame(rows)


@functools.cache
def states() -> pd.DataFrame:
    """US states with their census region, for joins to the turbines."""
    return pd.DataFrame(STATES, columns=["state", "name", "region"])


@functools.cache
def penguins() -> pd.DataFrame:
    """Palmer-penguins-like measurements; a few rows miss their sex or body mass, which a count must keep."""
    rng = np.random.default_rng(11)
    shapes = {
        "Adelie": (50, ["Torgersen", "Biscoe", "Dream"], 38.8, 18.3, 190, 3700),
        "Gentoo": (40, ["Biscoe"], 47.5, 15.0, 217, 5080),
        "Chinstrap": (30, ["Dream"], 48.8, 18.4, 196, 3730),
    }
    rows = []
    for species, (count, islands, bill, depth, flipper, mass) in shapes.items():
        for i in range(count):
            sex = "male" if i % 2 else "female"
            shift = 1 if sex == "male" else -1
            rows.append({
                "species": species, "island": islands[i % len(islands)],
                "bill_length_mm": round(bill + shift + float(rng.normal(0, 2.0)), 1),
                "bill_depth_mm": round(depth + shift * 0.6 + float(rng.normal(0, 0.8)), 1),
                "flipper_length_mm": float(round(flipper + shift * 3 + float(rng.normal(0, 5)))),
                "body_mass_g": float(round(mass + shift * 300 + float(rng.normal(0, 250)), -1)),
                "sex": sex, "year": 2007 + i % 3,
            })
    frame = pd.DataFrame(rows)
    frame.loc[[3, 61, 100], "sex"] = None
    frame.loc[[17, 88], ["body_mass_g", "flipper_length_mm"]] = np.nan
    return frame


PLACES = [
    ("{d} km NE of Ridgecrest, CA", 0.13), ("{d} km W of Petrolia, CA", 0.05), ("{d} km SE of Anza, CA", 0.05),
    ("Northern California", 0.01), ("offshore Northern California", 0.01),
    ("{d} km S of Adak, Alaska", 0.18), ("{d} km NW of Anchorage, Alaska", 0.07),
    ("{d} km N of Pahala, Hawaii", 0.08), ("{d} km E of Hawthorne, NV", 0.06),
    ("Puerto Rico region", 0.06), ("{d} km SW of Tonopah, Nevada", 0.05),
    ("{d} km WNW of Ferndale, CA", 0.04), ("{d} km S of Volcano, Hawaii", 0.05),
    ("Fiji region", 0.06), ("{d} km ESE of Lima, Peru", 0.04), ("south of the Kermadec Islands", 0.06),
]
NETWORKS = {"CA": "ci", "California": "nc", "Alaska": "ak", "Hawaii": "hv", "NV": "nn", "Nevada": "nn"}


@functools.cache
def quakes() -> pd.DataFrame:
    """
    Earthquakes of 2024 and 2025 with USGS-style place strings: most
    California events are stored as ``..., CA``, a few as ``California``;
    the last event is on 2025-12-31.
    """
    rng = np.random.default_rng(23)
    weights = np.array([w for _, w in PLACES])
    weights /= weights.sum()
    start = datetime(2024, 1, 1)
    span = (datetime(2025, 12, 31, 23, 0) - start).total_seconds()
    rows = []
    for i in range(900):
        template = PLACES[rng.choice(len(PLACES), p=weights)][0]
        place = template.format(d=int(rng.integers(2, 80)))
        at = start + timedelta(seconds=float(rng.uniform(0, span)))
        net = next((code for key, code in NETWORKS.items() if place.endswith(key) or key in place), "us")
        roll = rng.random()
        kind = "earthquake" if roll < 0.94 else ("quarry blast" if roll < 0.98 else "explosion")
        rows.append({
            "time": at.replace(microsecond=0), "latitude": round(float(rng.uniform(-30, 62)), 3),
            "longitude": round(float(rng.uniform(-180, -60)), 3), "depth": round(float(rng.gamma(2.0, 12.0)), 2),
            "mag": round(4.0 + float(rng.exponential(0.45)), 2), "net": net, "place": place, "type": kind,
            "id": f"{net}{70000000 + i}",
        })
    rows.append({**rows[-1], "time": datetime(2025, 12, 31, 22, 15), "id": "us79999999"})
    frame = pd.DataFrame(rows).sort_values("time", ignore_index=True)
    frame["time"] = pd.to_datetime(frame["time"]).astype("datetime64[us]")
    return frame


# Each fixture's tables, by table name.
FIXTURES = {
    "turbines": lambda: {"turbines": turbines()},
    "turbines_states": lambda: {"turbines": turbines(), "states": states()},
    "penguins": lambda: {"penguins": penguins()},
    "quakes": lambda: {"quakes": quakes()},
}

# The turbines' metrics, as a FlowDash project file declares them; the session's source node is ``data``.
TURBINE_METRICS = """\
metrics:
  capacity:
    description: Installed wind capacity, from the turbine table.
    source: data
    table: turbines
    measures:
      capacity_mw: {expr: "SUM(t_cap) / 1000.0", format: ",.1f", description: Rated capacity in MW}
      turbines: {expr: "COUNT(*)", format: ",", description: Number of turbines}
    dimensions:
      state: {column: t_state, description: Two-letter state code}
      year: {column: p_year, description: Year the project came online}
"""
METRICS = {"turbines": TURBINE_METRICS}


def fixture_source(name: str) -> DuckDBSource:
    """A fresh in-memory DuckDB source holding the fixture's tables, single-threaded so row order is stable."""
    try:
        tables = FIXTURES[name]()
    except KeyError:
        raise ValueError(f"Unknown scenario fixture {name!r}; one of {sorted(FIXTURES)}") from None
    source = DuckDBSource.from_df({table: frame.copy() for table, frame in tables.items()})
    source._connection.execute("SET threads TO 1")
    return source


def fixture_metrics(name: str | None) -> object | None:
    """The metrics of a project file in ``METRICS``, by name, or None."""
    if name is None:
        return None
    from ...flowdash.metrics import read_metrics
    from ...flowdash.project import parse_project

    return read_metrics(parse_project(METRICS[name], "flowdash.yaml"))
