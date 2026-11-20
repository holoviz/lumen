import pytest

try:
    import lumen.ai  # noqa
except ModuleNotFoundError:
    pytest.skip("lumen.ai could not be imported, skipping tests.", allow_module_level=True)

from lumen.ai.sql_review import TableFacts, review_sql

PLACES = {"places": TableFacts(geometry={"geometry": True}), "parks": TableFacts(geometry={"geometry": True})}
PROJECTED = {"places": TableFacts(geometry={"geometry": False})}
GRID = {"air": TableFacts(units={"air": "degK", "lat": "degrees_north"})}


@pytest.mark.parametrize("sql, expected", [
    ("SELECT ST_Buffer(geometry, 500) FROM places", "ST_Buffer(geometry, ..., 500) works in degrees"),
    ("SELECT * FROM places WHERE ST_DWithin(geometry, ST_Point(13, 52), 1000)", "ST_DWithin(geometry, ..., 1000) works in degrees"),
    ("SELECT ST_Area(geometry) FROM parks", "ST_Area(geometry) returns square degrees"),
    ("SELECT ST_Length(geometry) FROM parks", "ST_Length(geometry) returns degrees"),
    ("SELECT ST_Distance(a.geometry, b.geometry) FROM places a JOIN parks b ON ST_Intersects(a.geometry, b.geometry)",
     "ST_Distance(geometry) returns degrees"),
    ("SELECT * FROM places, parks", "places and parks both have geometry but are joined without a spatial predicate"),
    ("SELECT * FROM places CROSS JOIN parks", "joined without a spatial predicate"),
    ("SELECT * FROM air WHERE air > 30", "air is in degk, so 30 looks like degrees Celsius (303.15 K)"),
    ("SELECT * FROM air WHERE 30 < air", "air is in degk, so 30 looks like degrees Celsius (303.15 K)"),
    ("SELECT ST_Area_Spheroid(geometry) FROM parks", "reads points as (latitude, longitude)"),
    ("SELECT ST_Area(geometry) FROM parks", "ST_Area(ST_Transform(geometry, 'EPSG:4326', 'EPSG:6933', always_xy := true))"),
    ("SELECT ST_Length(geometry) FROM parks", "ST_Length_Spheroid(ST_FlipCoordinates(geometry))"),
])
def test_review_finds_mistakes_that_run_without_error(sql, expected):
    (finding,) = review_sql(sql, {**PLACES, **GRID})
    assert expected in finding


@pytest.mark.parametrize("sql", [
    "SELECT ST_Area_Spheroid(ST_FlipCoordinates(geometry)) FROM parks",
    "SELECT ST_Area(ST_Transform(geometry, 'EPSG:4326', 'EPSG:3857')) FROM parks",
    "SELECT ST_Buffer(geometry, 0.01) FROM places",
    "SELECT * FROM places JOIN parks ON ST_Within(places.geometry, parks.geometry)",
    "SELECT * FROM places, parks WHERE ST_Intersects(places.geometry, parks.geometry)",
    "SELECT * FROM places JOIN parks ON places.n = parks.p",
    "SELECT * FROM places a JOIN parks b ON ST_Distance_Sphere(ST_FlipCoordinates(a.geometry), ST_FlipCoordinates(b.geometry)) <= 750000",
    "SELECT * FROM air WHERE air > 300",
    "SELECT * FROM air WHERE lat > 30",
    "not sql at all (",
])
def test_review_leaves_correct_queries_alone(sql):
    assert review_sql(sql, {**PLACES, **GRID}) == []


def test_a_projected_crs_is_not_in_degrees():
    assert review_sql("SELECT ST_Area(geometry), ST_Buffer(geometry, 500) FROM places", PROJECTED) == []


def test_a_celsius_column_compared_with_kelvin():
    (finding,) = review_sql("SELECT * FROM t2m WHERE t2m < 273", {"t2m": TableFacts(units={"t2m": "degC"})})
    assert "looks like Kelvin (-0.15 °C)" in finding


@pytest.mark.parametrize("column_type, expected", [
    ("GEOMETRY('EPSG:4326')", True), ("GEOMETRY('EPSG:3857')", False), ("GEOMETRY", False), ("GEOMETRY('nonsense')", False),
])
def test_geographic_crs_reads_the_column_type(column_type, expected):
    pytest.importorskip("pyproj")
    from lumen.ai.sql_review import geographic_crs
    assert geographic_crs(column_type) is expected


WRAPPED = {
    "air": TableFacts(units={"air": "degK"}, lon_range={"lon": (200.0, 330.0)}),
    "states": TableFacts(geometry={"geometry": True}),
}


@pytest.mark.parametrize("sql", [
    "SELECT s.name FROM air a JOIN states s ON ST_Within(ST_Point(a.lon, a.lat), s.geometry)",
    "SELECT name FROM air, states WHERE ST_Within(ST_Point(lon, lat), geometry)",
    "SELECT s.name FROM air a JOIN states s ON ST_Intersects(ST_MakeEnvelope(a.lon - 1.25, a.lat - 1.25, a.lon + 1.25, a.lat + 1.25), s.geometry)",
    # Passed through a CTE first, still unshifted
    "WITH c AS (SELECT lat, lon, air - 273.15 AS t FROM air) "
    "SELECT s.name FROM states s JOIN c ON ST_Intersects(s.geometry, ST_MakeEnvelope(c.lon - 1, c.lat - 1, c.lon + 1, c.lat + 1))",
])
def test_review_finds_a_0_to_360_longitude_against_geometry(sql):
    (finding,) = review_sql(sql, WRAPPED)
    assert "runs from 0 to 360" in finding and "lon - 360" in finding


@pytest.mark.parametrize("sql", [
    "SELECT s.name FROM air a JOIN states s ON ST_Within(ST_Point(CASE WHEN a.lon > 180 THEN a.lon - 360 ELSE a.lon END, a.lat), s.geometry)",
    "WITH c AS (SELECT lat, CASE WHEN lon > 180 THEN lon - 360 ELSE lon END AS lon FROM air) "
    "SELECT s.name FROM c JOIN states s ON ST_Intersects(ST_MakeEnvelope(c.lon - 1, c.lat - 1, c.lon + 1, c.lat + 1), s.geometry)",
    "SELECT ST_Point(lon, lat) FROM air",
])
def test_review_leaves_shifted_longitudes_and_grids_alone(sql):
    assert review_sql(sql, WRAPPED) == []


def test_a_grid_already_in_minus_180_to_180_is_fine():
    facts = {**WRAPPED, "air": TableFacts(lon_range={"lon": (-180.0, 177.5)})}
    assert review_sql("SELECT 1 FROM air a JOIN states s ON ST_Within(ST_Point(a.lon, a.lat), s.geometry)", facts) == []
