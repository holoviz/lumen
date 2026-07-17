import json
import subprocess
import sys

import pandas as pd
import panel as pn
import pytest

from panel_material_ui import Checkbox, FloatSlider, Select

from lumen.schema import JSONSchema
from lumen.util import get_dataframe_schema

try:
    import geopandas as gpd

    from shapely.geometry import Polygon
except ImportError:
    gpd = None


def test_boolean_schema():
    json_schema = JSONSchema(schema={'bool': {'type': 'boolean'}}, multi=False)
    assert isinstance(json_schema._widgets['bool'], Checkbox)

def test_number_schema():
    json_schema = JSONSchema(multi=False, schema={
        'number': {'type': 'number', 'inclusiveMinimum': 0, 'inclusiveMaximum': 3.14}
    })
    widget = json_schema._widgets['number']
    assert isinstance(widget, FloatSlider)
    assert widget.start == 0
    assert widget.end == 3.14

def test_enum_schema():
    json_schema = JSONSchema(multi=False, schema={
        'enum': {'type': 'enum', 'enum': ['A', 'B', 'C']}
    })
    widget = json_schema._widgets['enum']
    assert isinstance(widget, Select)
    assert widget.options == ['A', 'B', 'C']


def test_get_dataframe_schema_geometry():
    """A geometry column is emitted as a compact, JSON-serializable marker."""
    if gpd is None:
        pytest.skip("geopandas is not installed")
    gdf = gpd.GeoDataFrame(
        {
            "name": ["a", "b"],
            "pop": [100, 200],
            "geometry": [
                Polygon([(0, 0), (1, 0), (1, 1)]),
                Polygon([(2, 0), (3, 0), (3, 1)]),
            ],
        },
        crs="EPSG:4326",
    )
    schema = get_dataframe_schema(gdf)
    props = schema["items"]["properties"]
    assert props["geometry"]["format"] == "geometry"
    assert props["geometry"]["geometry_type"] == "Polygon"
    # a geographic (lat/lon) CRS is reported so consumers can route/style
    assert props["geometry"]["geographic"] is True
    assert props["geometry"]["crs"] == "EPSG:4326"
    # numeric/string columns are unaffected
    assert props["pop"]["type"] == "integer"
    assert props["name"]["enum"] == ["a", "b"]
    # the whole schema must be JSON-serializable (it is sent to the LLM)
    json.dumps(schema)


def test_get_dataframe_schema_geometry_empty():
    """An empty GeoDataFrame yields geometry_type 'unknown' without raising."""
    if gpd is None:
        pytest.skip("geopandas is not installed")
    gdf = gpd.GeoDataFrame({"name": [], "geometry": []})
    schema = get_dataframe_schema(gdf)
    assert schema["items"]["properties"]["geometry"]["format"] == "geometry"


def test_get_dataframe_schema_object_column_unaffected():
    """Plain object columns still become string enums (no geometry regression)."""
    df = pd.DataFrame({"cat": ["x", "y", "x"], "n": [1, 2, 3]})
    props = get_dataframe_schema(df)["items"]["properties"]
    assert props["cat"] == {"type": "string", "enum": ["x", "y"]}
    assert props["n"]["type"] == "integer"


def test_get_dataframe_schema_does_not_import_geopandas():
    """Schema detection must not speculatively import geopandas (gh-1903 review)."""
    code = (
        "import sys, pandas as pd;"
        "from lumen.util import get_dataframe_schema;"
        "get_dataframe_schema(pd.DataFrame({'a': [1, 2], 'b': ['x', 'y']}));"
        "assert 'geopandas' not in sys.modules"
    )
    subprocess.run([sys.executable, "-c", code], check=True)
