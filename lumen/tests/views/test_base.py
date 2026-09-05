import importlib.util
import json

from pathlib import Path

import pandas as pd
import panel as pn
import pytest

from lumen.filters.base import ConstantFilter, ParamFilter
from lumen.panel import DownloadButton
from lumen.panes.mosaic import Mosaic
from lumen.pipeline import Pipeline
from lumen.sources.base import FileSource, InMemorySource
from lumen.state import state
from lumen.tests.utils import Polygon, gpd, requires_geopandas
from lumen.transforms.base import Columns
from lumen.transforms.sql import SQLLimit
from lumen.variables.base import Variables
from lumen.views.base import (
    DeckGLView, MosaicView, Panel, Table, VegaLiteView, View, hvOverlayView,
    hvPlotView,
)

# duckdb is a core dependency but the minimal test-core env omits it, and the
# geometry tests skip on their own via the DuckDBSource(...) guard below.
try:
    from lumen.sources.duckdb import DuckDBSource
except ImportError:
    DuckDBSource = None

# geoviews is checked without importing it: importing geoviews (-> matplotlib)
# at collection under xdist can deadlock the font cache on macOS. hvplot imports
# it lazily when the geometry plot is actually rendered.
requires_geoviews = pytest.mark.skipif(
    importlib.util.find_spec("geoviews") is None, reason="geoviews not installed"
)


def test_resolve_module_type():
    assert View._get_type('lumen.views.base.View') is View


def test_view_hvplot_limit(set_root):
    set_root(str(Path(__file__).parent.parent))
    source = FileSource(tables={'test': 'sources/test.csv'})
    view = {
        'type': 'hvplot',
        'table': 'test',
        'x': 'A',
        'y': 'B',
        'kind': 'scatter',
        'limit': 2
    }

    view = View.from_spec(view, source, [])
    data = view.get_data()
    assert data.shape == (2, 4)


def test_view_hvplot_basis(set_root):
    set_root(str(Path(__file__).parent.parent))
    source = FileSource(tables={'test': 'sources/test.csv'})
    view = {
        'type': 'hvplot',
        'table': 'test',
        'x': 'A',
        'y': 'B',
        'kind': 'scatter',
    }

    view = View.from_spec(view, source, [])

    assert isinstance(view, hvPlotView)
    assert view.kind == 'scatter'
    assert view.pipeline.source is source
    assert view.pipeline.table == 'test'
    assert view.pipeline.filters == []
    assert view.pipeline.transforms == []
    assert view.controls == []

    df = view.get_data()

    assert isinstance(df, pd.DataFrame)
    assert df.shape == (5, 4)

    plot = view.get_plot(df)

    assert plot.kdims == ['A']
    assert plot.vdims == ['B']


def test_view_hvplot_variable(set_root):
    set_root(str(Path(__file__).parent.parent))
    source = FileSource(tables={'test': 'sources/test.csv'})
    state._variable = vars = Variables.from_spec({'y': 'B'})
    view = {
        'type': 'hvplot',
        'table': 'test',
        'x': 'A',
        'y': '$variables.y',
        'kind': 'scatter',
    }

    view = View.from_spec(view, source, [])

    assert view.y == 'B'

    df = view.get_data()
    plot = view.get_plot(df)

    assert plot.vdims == ['B']

    vars._vars['y'].value = 'C'

    plot = view.get_plot(df)

    assert plot.vdims == ['C']


def test_view_hvplot_download(set_root):
    set_root(str(Path(__file__).parent.parent))
    source = FileSource(tables={'test': 'sources/test.csv'})
    view = {
        'type': 'hvplot',
        'table': 'test',
        'x': 'A',
        'y': 'B',
        'kind': 'scatter',
        'download': 'csv'
    }

    view = View.from_spec(view, source)

    button = view.panel[0]
    assert isinstance(button, DownloadButton)

    button._on_click()
    assert button.data.startswith('data:text/plain;charset=UTF-8;base64')

def test_view_to_spec(set_root):
    set_root(str(Path(__file__).parent.parent))
    source = FileSource(tables={'test': 'sources/test.csv'})
    view = {
        'type': 'table',
        'table': 'test',
    }

    view = View.from_spec(view, source, [])
    assert view.to_spec() == {
        'pipeline': {
            'source': {
                'tables': {'test': 'sources/test.csv'},
                'type': 'file'
            },
            'table': 'test'
        },
        'type': 'table',
    }

def test_view_to_spec_with_context(set_root):
    set_root(str(Path(__file__).parent.parent))
    source = FileSource(tables={'test': 'sources/test.csv'})
    view = {
        'type': 'table',
        'table': 'test',
    }

    view = View.from_spec(view, source, [])
    context = {}
    spec = view.to_spec(context=context)
    assert context == {
        'sources': {
            source.name: {
                'tables': {'test': 'sources/test.csv'},
                'type': 'file'
            },
        },
        'pipelines': {
            view.pipeline.name: {
                'source': source.name,
                'table': 'test'
            }
        },
    }
    assert spec == {
        'pipeline': view.pipeline.name,
        'type': 'table'
    }

def test_view_to_spec_with_context_existing_context(set_root):
    set_root(str(Path(__file__).parent.parent))
    source = FileSource(tables={'test': 'sources/test.csv'})
    view = {
        'type': 'table',
        'table': 'test',
    }

    view = View.from_spec(view, source, [])
    context = {}
    view.pipeline.to_spec(context=context)
    assert context == {
        'sources': {
            source.name: {
                'tables': {'test': 'sources/test.csv'},
                'type': 'file'
            },
        },
        'pipelines': {
            view.pipeline.name: {
                'source': source.name,
                'table': 'test'
            }
        },
    }
    spec = view.to_spec(context=context)
    assert spec == {
        'pipeline': view.pipeline.name,
        'type': 'table'
    }

def test_hvplot_view_to_spec(set_root):
    set_root(str(Path(__file__).parent.parent))
    source = FileSource(tables={'test': 'sources/test.csv'})
    view = {
        'type': 'hvplot',
        'table': 'test',
        'x': 'A',
        'y': 'B',
        'alpha': 0.3
    }
    view = View.from_spec(view, source, [])
    assert view.to_spec() == {
        'pipeline': {
            'source': {
                'tables': {'test': 'sources/test.csv'},
                'type': 'file'
            },
            'table': 'test'
        },
        'type': 'hvplot',
        'x': 'A',
        'y': 'B',
        'alpha': 0.3
    }

def test_table_layout_defaults(set_root):
    set_root(str(Path(__file__).parent.parent))
    source = FileSource(tables={'test': 'sources/test.csv'})
    view = View.from_spec({'type': 'table', 'table': 'test'}, source, [])
    panel = view.get_panel()
    assert panel.sizing_mode == 'stretch_width'
    assert panel.layout == 'fit_data_stretch'
    assert panel._configuration.get('columnDefaults', {}).get('maxInitialWidth') == 300


def test_table_user_configuration_merges(set_root):
    set_root(str(Path(__file__).parent.parent))
    source = FileSource(tables={'test': 'sources/test.csv'})
    table = Table(
        pipeline=Pipeline(source=source, table='test'),
        configuration={'columnDefaults': {'maxInitialWidth': 500, 'headerSort': False}},
    )
    panel = table.get_panel()
    config = panel._configuration
    assert config['columnDefaults']['maxInitialWidth'] == 500
    assert config['columnDefaults']['headerSort'] is False


@pytest.mark.parametrize("view_type", ("table", "hvplot"))
def test_view_title(set_root, view_type):
    set_root(str(Path(__file__).parent.parent))
    source = FileSource(tables={'test': 'sources/test.csv'})
    view = {
        'type': view_type,
        'table': 'test',
        'title': 'Test title',
    }

    view = View.from_spec(view, source)

    title = view.panel[0]
    assert isinstance(title, pn.pane.HTML)


@pytest.mark.parametrize("view_type", ("table", "hvplot"))
def test_view_title_download(set_root, view_type):
    set_root(str(Path(__file__).parent.parent))
    source = FileSource(tables={'test': 'sources/test.csv'})
    view = {
        'type': view_type,
        'table': 'test',
        'title': 'Test title',
        'download': 'csv',
    }

    view = View.from_spec(view, source)

    title = view.panel[0][0]
    assert isinstance(title, pn.pane.HTML)

    button = view.panel[0][1]
    assert isinstance(button, DownloadButton)

    button._on_click()
    assert button.data.startswith('data:text/plain;charset=UTF-8;base64')

    assert view.download.filename is None
    assert view.download.format == 'csv'


@pytest.mark.parametrize("view_type", ("table", "hvplot"))
def test_view_title_download_filename(set_root, view_type):
    set_root(str(Path(__file__).parent.parent))
    source = FileSource(tables={'test': 'sources/test.csv'})
    view = {
        'type': view_type,
        'table': 'test',
        'title': 'Test title',
        'download': 'example.csv',
    }

    view = View.from_spec(view, source)

    title = view.panel[0][0]
    assert isinstance(title, pn.pane.HTML)

    button = view.panel[0][1]
    assert isinstance(button, DownloadButton)

    button._on_click()
    assert button.data.startswith('data:text/plain;charset=UTF-8;base64')

    assert view.download.filename == 'example'
    assert view.download.format == 'csv'


def test_view_list_param_function_roundtrip(set_root):
    set_root(str(Path(__file__).parent.parent))
    original_spec = {
        "layers": [
            {
                "kind": "line",
                "operations": [
                    {
                        "rolling_window": 3,
                        "type": "holoviews.operation.timeseries.rolling",
                    },
                    {
                        "rolling_window": 3,
                        "sigma": 0.1,
                        "type": "holoviews.operation.timeseries.rolling_outlier_std",
                    },
                ],
                "pipeline": {
                    "source": {'tables': {'test': 'sources/test.csv'}, 'type': 'file'},
                    "table": "test",
                    "transforms": [{"type": "columns", "columns": ["A", "B"]}]
                },
                "x": "A",
                "y": "B",
                "type": "hvplot",
            },
        ],
        "type": "hv_overlay",
    }
    view = View.from_spec(original_spec)
    assert isinstance(view, hvOverlayView)
    assert view.to_spec() == original_spec


def test_vega_datasets(set_root):
    set_root(str(Path(__file__).parent.parent))
    source = FileSource(tables={'test': 'sources/test.csv'})
    pipeline = Pipeline(source=source, table="test")

    spec = {
        "$schema": "https://vega.github.io/schema/vega-lite/v6.json",
        "width": 700,
        "height": 400,
        "title": {"text": "US States Map"},
        "data": {"url": "https://cdn.jsdelivr.net/npm/us-atlas@3/states-10m.json", "format": {"type": "topojson", "feature": "states"}},
        "projection": {"type": "albersUsa"},
        "mark": "geoshape",
        "encoding": {"color": {"value": "#ccc"}, "tooltip": {"field": "properties.name", "type": "nominal"}},
    }
    final_spec = VegaLiteView(spec=spec, pipeline=pipeline).get_panel().object
    assert final_spec["data"]["url"] == "https://cdn.jsdelivr.net/npm/us-atlas@3/states-10m.json"
    pd.testing.assert_frame_equal(final_spec["datasets"]["test"], pipeline.data)


def _choropleth_spec(dataset_name):
    """A choropleth joining the table onto US state outlines via a lookup."""
    return {
        "data": {
            "url": "https://cdn.jsdelivr.net/npm/us-atlas@3/states-10m.json",
            "format": {"type": "topojson", "feature": "states"},
        },
        "transform": [{
            "lookup": "properties.name",
            "from": {"data": {"name": dataset_name}, "key": "A", "fields": ["B"]},
        }],
        "mark": "geoshape",
        "projection": {"type": "albersUsa"},
        "encoding": {"color": {"field": "B", "type": "quantitative"}},
    }


def test_vega_lookup_dataset_name_is_preserved_when_correct(set_root):
    set_root(str(Path(__file__).parent.parent))
    pipeline = Pipeline(source=FileSource(tables={'test': 'sources/test.csv'}), table="test")

    final_spec = VegaLiteView(spec=_choropleth_spec("test"), pipeline=pipeline).get_panel().object

    assert final_spec["transform"][0]["from"]["data"]["name"] == "test"


def test_vega_lookup_naming_another_registered_dataset_is_left_alone(set_root):
    """A spec may declare its own datasets; a lookup resolving to one of those is
    already correct and must not be dragged onto the pipeline table."""
    set_root(str(Path(__file__).parent.parent))
    pipeline = Pipeline(source=FileSource(tables={'test': 'sources/test.csv'}), table="test")
    spec = _choropleth_spec("populations")
    spec["datasets"] = {"populations": [{"A": "a", "B": 1}]}

    final_spec = VegaLiteView(spec=spec, pipeline=pipeline).get_panel().object

    assert final_spec["transform"][0]["from"]["data"]["name"] == "populations"
    assert set(final_spec["datasets"]) == {"populations", "test"}


def test_vega_lookup_dataset_name_is_repointed_when_wrong(set_root):
    """A lookup naming a dataset that was never registered renders an empty map
    and raises nothing, so the only correct table name is substituted in."""
    set_root(str(Path(__file__).parent.parent))
    pipeline = Pipeline(source=FileSource(tables={'test': 'sources/test.csv'}), table="test")
    spec = _choropleth_spec("some_table_that_was_never_registered")

    final_spec = VegaLiteView(spec=spec, pipeline=pipeline).get_panel().object

    assert final_spec["transform"][0]["from"]["data"]["name"] == "test"
    assert "test" in final_spec["datasets"]
    # the caller's spec must not be edited underneath them
    assert spec["transform"][0]["from"]["data"]["name"] == "some_table_that_was_never_registered"


def test_vega_lookup_with_its_own_data_is_left_alone(set_root):
    """A lookup carrying inline values is self-contained and must not be repointed."""
    set_root(str(Path(__file__).parent.parent))
    pipeline = Pipeline(source=FileSource(tables={'test': 'sources/test.csv'}), table="test")
    spec = _choropleth_spec("irrelevant")
    spec["transform"][0]["from"]["data"] = {"values": [{"A": 1, "B": 2}]}

    final_spec = VegaLiteView(spec=spec, pipeline=pipeline).get_panel().object

    assert final_spec["transform"][0]["from"]["data"] == {"values": [{"A": 1, "B": 2}]}


def test_vega_datasets_are_not_accumulated_across_renders(set_root):
    """self.spec is reused between renders, so registering into its own datasets
    dict would leave a DataFrame behind on the caller's spec."""
    set_root(str(Path(__file__).parent.parent))
    pipeline = Pipeline(source=FileSource(tables={'test': 'sources/test.csv'}), table="test")
    spec = _choropleth_spec("test")
    spec["datasets"] = {"populations": [{"A": "a", "B": 1}]}
    view = VegaLiteView(spec=spec, pipeline=pipeline)

    view.get_panel()
    view.get_panel()

    assert set(view.spec["datasets"]) == {"populations"}


def test_vega_layered_choropleth_registers_the_named_dataset(set_root):
    """A layered choropleth carries the boundary url on each layer, not at the top
    level, so registration has to look deeper or the lookup joins nothing."""
    set_root(str(Path(__file__).parent.parent))
    pipeline = Pipeline(source=FileSource(tables={'test': 'sources/test.csv'}), table="test")
    boundaries = {
        "url": "https://cdn.jsdelivr.net/npm/us-atlas@3/states-10m.json",
        "format": {"type": "topojson", "feature": "states"},
    }
    spec = {
        "projection": {"type": "albersUsa"},
        "layer": [
            {"data": boundaries, "mark": {"type": "geoshape", "fill": None}},
            {
                "data": boundaries,
                "transform": [{
                    "lookup": "properties.name",
                    "from": {"data": {"name": "test"}, "key": "A", "fields": ["B"]},
                }],
                "mark": "geoshape",
                "encoding": {"color": {"field": "B", "type": "quantitative"}},
            },
        ],
    }

    final_spec = VegaLiteView(spec=spec, pipeline=pipeline).get_panel().object

    assert "test" in final_spec["datasets"]
    # the table must not also replace the boundaries as primary data
    assert "data" not in final_spec


def test_vega_defaults_schema(set_root):
    set_root(str(Path(__file__).parent.parent))
    source = FileSource(tables={'test': 'sources/test.csv'})
    pipeline = Pipeline(source=source, table="test")
    spec = {
        "mark": "bar",
        "encoding": {
            "x": {"field": "A", "type": "nominal"},
            "y": {"field": "B", "type": "quantitative"},
        },
    }

    final_spec = VegaLiteView(spec=spec, pipeline=pipeline).get_panel().object

    assert final_spec["$schema"] == "https://vega.github.io/schema/vega-lite/v5.json"


@requires_geopandas
@requires_geoviews
def test_view_hvplot_geometry_auto_kind():
    """A GeoDataFrame view renders its geometry with an auto-selected kind."""
    try:
        source = DuckDBSource(
            uri=':memory:',
            initializers=["INSTALL spatial;", "LOAD spatial;"],
            tables={'geo': 'SELECT * FROM geo_tbl'},
        )
    except Exception as e:  # pragma: no cover - environment dependent
        pytest.skip(f"duckdb spatial extension unavailable: {e}")
    gdf = gpd.GeoDataFrame(
        {
            'pop': [1, 2],
            'geometry': [
                Polygon([(0, 0), (1, 0), (1, 1)]),
                Polygon([(2, 0), (3, 0), (3, 1)]),
            ],
        },
        crs='EPSG:4326',
    )
    tmp = pd.DataFrame({'pop': gdf['pop'], 'geometry': gdf['geometry'].to_wkb()})
    source._connection.register('geo_temp', tmp)
    source._connection.execute(
        'CREATE TABLE geo_tbl AS '
        'SELECT pop, ST_GeomFromWKB(geometry) AS geometry FROM geo_temp'
    )
    pipeline = Pipeline(source=source, table='geo')
    view = hvPlotView(pipeline=pipeline)  # no kind, no geo passed
    df = view.get_data()
    assert isinstance(df, gpd.GeoDataFrame)
    plot = view.get_plot(df)
    # Polygon geometry -> a Polygons element rather than an empty/failed plot
    assert type(plot).__name__ == 'Polygons'


@requires_geopandas
def test_table_view_geometry_rendered_as_wkt():
    """Table view converts geometry columns to WKT so Bokeh can serialize them."""
    try:
        source = DuckDBSource(
            uri=':memory:',
            initializers=["INSTALL spatial;", "LOAD spatial;"],
            tables={'geo': 'SELECT * FROM geo_tbl'},
        )
    except Exception as e:  # pragma: no cover - environment dependent
        pytest.skip(f"duckdb spatial extension unavailable: {e}")
    gdf = gpd.GeoDataFrame(
        {'pop': [1, 2], 'geometry': [
            Polygon([(0, 0), (1, 0), (1, 1)]), Polygon([(2, 0), (3, 0), (3, 1)])]},
        crs='EPSG:4326',
    )
    tmp = pd.DataFrame({'pop': gdf['pop'], 'geometry': gdf['geometry'].to_wkb()})
    source._connection.register('geo_temp', tmp)
    source._connection.execute(
        'CREATE TABLE geo_tbl AS SELECT pop, ST_GeomFromWKB(geometry) AS geometry FROM geo_temp'
    )
    pipeline = Pipeline(source=source, table='geo')
    view = Table(pipeline=pipeline)
    value = view._get_params()['value']
    # geometry column is now WKT text, no shapely objects
    assert not isinstance(value, gpd.GeoDataFrame)
    assert all(isinstance(v, str) and v.startswith('POLYGON') for v in value['geometry'])
    # panel builds without raising
    view.get_panel()


@requires_geopandas
def test_deckgl_view_geometry_as_geojson():
    """DeckGLView emits a GeoDataFrame as a GeoJSON FeatureCollection for layers."""
    try:
        source = DuckDBSource(
            uri=':memory:',
            initializers=["INSTALL spatial;", "LOAD spatial;"],
            tables={'geo': 'SELECT * FROM geo_tbl'},
        )
    except Exception as e:  # pragma: no cover - environment dependent
        pytest.skip(f"duckdb spatial extension unavailable: {e}")
    gdf = gpd.GeoDataFrame(
        {'pop': [1, 2], 'geometry': [
            Polygon([(0, 0), (1, 0), (1, 1)]), Polygon([(2, 0), (3, 0), (3, 1)])]},
        crs='EPSG:4326',
    )
    tmp = pd.DataFrame({'pop': gdf['pop'], 'geometry': gdf['geometry'].to_wkb()})
    source._connection.register('geo_temp', tmp)
    source._connection.execute(
        'CREATE TABLE geo_tbl AS SELECT pop, ST_GeomFromWKB(geometry) AS geometry FROM geo_temp'
    )
    pipeline = Pipeline(source=source, table='geo')
    spec = {'initialViewState': {'latitude': 0, 'longitude': 0, 'zoom': 6},
            'layers': [{'@@type': 'GeoJsonLayer'}]}
    view = DeckGLView(pipeline=pipeline, spec=spec)
    data = view._get_params()['object']['layers'][0]['data']
    assert data['type'] == 'FeatureCollection'
    assert len(data['features']) == 2
    assert data['features'][0]['geometry']['type'] == 'Polygon'
    json.dumps(data)  # must be serializable


@requires_geopandas
def test_vegalite_view_geometry_as_geoshape_data():
    """VegaLiteView emits a GeoDataFrame as a FeatureCollection with the
    format.property geoshape needs, rather than an unserializable frame."""
    gdf = gpd.GeoDataFrame(
        {'pop': [1, 2],
         'date': [pd.Timestamp('2026-06-15'), pd.Timestamp('2026-06-15')],
         'geometry': [
             Polygon([(0, 0), (1, 0), (1, 1)]), Polygon([(2, 0), (3, 0), (3, 1)])]},
        crs='EPSG:4326',
    )
    pipeline = Pipeline(source=InMemorySource(tables={'geo': gdf}), table='geo')
    spec = {'mark': 'geoshape',
            'encoding': {'color': {'field': 'properties.pop', 'type': 'quantitative'}}}
    data = VegaLiteView(pipeline=pipeline, spec=spec)._get_params()['object']['data']

    assert data['format'] == {'type': 'json', 'property': 'features'}
    assert data['values']['type'] == 'FeatureCollection'
    assert len(data['values']['features']) == 2
    assert data['values']['features'][0]['properties']['pop'] == 1
    json.dumps(data)  # must be serializable


@requires_geopandas
def test_vegalite_view_plain_frame_unchanged():
    """A non-geometry frame still goes through as data.values records."""
    df = pd.DataFrame({'x': [1, 2], 'y': [3, 4]})
    pipeline = Pipeline(source=InMemorySource(tables={'plain': df}), table='plain')
    spec = {'mark': 'point', 'encoding': {'x': {'field': 'x'}}}
    data = VegaLiteView(pipeline=pipeline, spec=spec)._get_params()['object']['data']

    assert 'format' not in data
    assert data['values'].equals(df)


@requires_geopandas
def test_deckgl_view_geometry_with_datetime_column():
    """A datetime column alongside geometry must not break serialization."""
    gdf = gpd.GeoDataFrame(
        {'date': [pd.Timestamp('2026-06-15')],
         'geometry': [Polygon([(0, 0), (1, 0), (1, 1)])]},
        crs='EPSG:4326',
    )
    data = DeckGLView._layer_data(gdf)
    assert data['features'][0]['properties']['date'].startswith('2026-06-15')
    json.dumps(data)  # must be serializable


def test_deckgl_geojson_layer_without_geometry_raises(set_root):
    """A GeoJsonLayer handed plain records draws an empty basemap and reports
    nothing, so the mismatch has to surface as an error instead."""
    set_root(str(Path(__file__).parent.parent))
    pipeline = Pipeline(source=FileSource(tables={'test': 'sources/test.csv'}), table="test")
    spec = {
        "layers": [{"@@type": "GeoJsonLayer", "id": "choropleth"}],
        "initialViewState": {"latitude": 39.5, "longitude": -98.35, "zoom": 3},
    }

    with pytest.raises(ValueError, match="needs geometry"):
        DeckGLView(pipeline=pipeline, spec=spec)._get_params()


def test_deckgl_non_geojson_layers_still_receive_records(set_root):
    """Only GeoJsonLayer needs geometry; other layers take records as before."""
    set_root(str(Path(__file__).parent.parent))
    pipeline = Pipeline(source=FileSource(tables={'test': 'sources/test.csv'}), table="test")
    spec = {
        "layers": [{"@@type": "ScatterplotLayer", "id": "points"}],
        "initialViewState": {"latitude": 39.5, "longitude": -98.35, "zoom": 3},
    }

    params = DeckGLView(pipeline=pipeline, spec=spec)._get_params()

    assert params["object"]["layers"][0]["data"] == pipeline.data.to_dict(orient="records")


def test_deckgl_geojson_layer_with_its_own_data_is_left_alone(set_root):
    """A layer that already carries data is not the injection path."""
    set_root(str(Path(__file__).parent.parent))
    pipeline = Pipeline(source=FileSource(tables={'test': 'sources/test.csv'}), table="test")
    spec = {
        "layers": [{"@@type": "GeoJsonLayer", "id": "boundaries", "data": "https://example.com/x.json"}],
        "initialViewState": {"latitude": 39.5, "longitude": -98.35, "zoom": 3},
    }

    params = DeckGLView(pipeline=pipeline, spec=spec)._get_params()

    assert params["object"]["layers"][0]["data"] == "https://example.com/x.json"


def test_mosaic_view_renders_a_native_panel_pane():
    """The view renders through Lumen's own Panel component, not an ipywidgets
    bridge, and registers the pipeline data under the pipeline's table name."""
    df = pd.DataFrame({'x': [1, 2, 3], 'y': [4.0, 5.0, 6.0]})
    pipeline = Pipeline(source=InMemorySource(tables={'points': df}), table='points')
    spec = {'plot': [{'mark': 'dot', 'data': {'from': 'points'}, 'x': 'x', 'y': 'y'}]}

    pane = MosaicView(pipeline=pipeline, spec=spec).get_panel()

    assert isinstance(pane, Mosaic)
    assert pane.connection.query('SELECT count(*) FROM points').fetchone()[0] == 3


def test_mosaic_view_rebinds_every_from_reference_to_the_pipeline_table():
    """A view is backed by exactly one pipeline table, so a spec naming any
    other table is rewritten rather than left to fail silently in the browser."""
    df = pd.DataFrame({'x': [1, 2], 'y': [3.0, 4.0]})
    pipeline = Pipeline(source=InMemorySource(tables={'points': df}), table='points')
    spec = {'plot': [
        {'mark': 'dot', 'data': {'from': 'some_other_table', 'filterBy': '$brush'},
         'x': 'x', 'y': 'y'},
    ]}

    pane = MosaicView(pipeline=pipeline, spec=spec).get_panel()

    mark = pane.spec['plot'][0]
    assert mark['data']['from'] == 'points'
    assert mark['data']['filterBy'] == '$brush'  # non-`from` keys untouched


def test_mosaic_view_does_not_mutate_the_spec_it_was_given():
    """Rebinding works on a copy; the view's own spec stays as authored so the
    editor round-trips what the user wrote."""
    df = pd.DataFrame({'x': [1], 'y': [2.0]})
    pipeline = Pipeline(source=InMemorySource(tables={'points': df}), table='points')
    spec = {'plot': [{'mark': 'dot', 'data': {'from': 'elsewhere'}, 'x': 'x', 'y': 'y'}]}

    view = MosaicView(pipeline=pipeline, spec=spec)
    view.get_panel()

    assert view.spec['plot'][0]['data']['from'] == 'elsewhere'


def test_mosaic_view_reports_an_empty_pipeline():
    """An empty result would draw a blank chart with no explanation, so it is
    raised where the retry loop can see it."""
    df = pd.DataFrame({'x': [], 'y': []})
    pipeline = Pipeline(source=InMemorySource(tables={'points': df}), table='points')
    spec = {'plot': [{'mark': 'dot', 'data': {'from': 'points'}, 'x': 'x', 'y': 'y'}]}

    with pytest.raises(ValueError, match='empty result'):
        MosaicView(pipeline=pipeline, spec=spec).get_panel()


def _mosaic_charted_rows(view, columns):
    """The rows the chart will actually query, whichever data path was taken."""
    pane = view.get_panel()
    name = pane.spec['plot'][0]['data']['from']
    cols = ', '.join(columns)
    return pane.connection.query(f'SELECT {cols} FROM "{name}" ORDER BY v').fetchall()


def _mosaic_pandas_rows(pipeline, columns):
    return list(pipeline.data[list(columns)].itertuples(index=False, name=None))


MOSAIC_DOT_SPEC = {'plot': [{'mark': 'dot', 'data': {'from': 't'}, 'x': 'cat', 'y': 'v'}]}


@pytest.mark.skipif(DuckDBSource is None, reason='duckdb not installed')
def test_mosaic_view_queries_lumen_duckdb_instead_of_copying():
    """A DuckDB-backed pipeline is served from its own connection, so the frame
    is never materialized in Python -- the point of Mosaic on large tables."""
    df = pd.DataFrame({'cat': list('aabbcc'), 'v': [1, 2, 3, 4, 5, 6]})
    source = DuckDBSource(uri=':memory:', mirrors={'t': df})
    pipeline = Pipeline(source=source, table='t')
    view = MosaicView(pipeline=pipeline, spec=MOSAIC_DOT_SPEC)

    assert view._sql_query() is not None
    assert _mosaic_charted_rows(view, ('cat', 'v')) == _mosaic_pandas_rows(pipeline, ('cat', 'v'))
    # The temp view must not disturb the table the rest of Lumen reads.
    assert source.connection.query('SELECT count(*) FROM t').fetchone()[0] == 6


@pytest.mark.skipif(DuckDBSource is None, reason='duckdb not installed')
def test_mosaic_view_pushdown_applies_filters_and_sql_transforms():
    """Pushing down must reproduce the pipeline's filters and SQL transforms.
    Charting the raw table instead would silently show unfiltered data."""
    df = pd.DataFrame({'cat': list('aabbcc'), 'v': [1, 2, 3, 4, 5, 6]})

    filtered = Pipeline(
        source=DuckDBSource(uri=':memory:', mirrors={'t': df}), table='t',
        filters=[ConstantFilter(field='cat', value='b')],
    )
    view = MosaicView(pipeline=filtered, spec=MOSAIC_DOT_SPEC)
    assert _mosaic_charted_rows(view, ('cat', 'v')) == [('b', 3), ('b', 4)]
    assert _mosaic_charted_rows(view, ('cat', 'v')) == _mosaic_pandas_rows(filtered, ('cat', 'v'))

    limited = Pipeline(
        source=DuckDBSource(uri=':memory:', mirrors={'t': df}), table='t',
        sql_transforms=[SQLLimit(limit=2)],
    )
    view = MosaicView(pipeline=limited, spec=MOSAIC_DOT_SPEC)
    assert _mosaic_charted_rows(view, ('cat', 'v')) == _mosaic_pandas_rows(limited, ('cat', 'v'))


@pytest.mark.skipif(DuckDBSource is None, reason='duckdb not installed')
def test_mosaic_view_falls_back_when_work_happens_outside_sql():
    """Anything Lumen computes in pandas after the query is absent from the
    pushdown SQL, so those pipelines must send the computed frame instead."""
    df = pd.DataFrame({'cat': list('aabbcc'), 'v': [1, 2, 3, 4, 5, 6]})

    # A Python transform runs after the SQL.
    transformed = Pipeline(
        source=DuckDBSource(uri=':memory:', mirrors={'t': df}), table='t',
        transforms=[Columns(columns=['cat', 'v'])],
    )
    assert MosaicView(pipeline=transformed, spec=MOSAIC_DOT_SPEC)._sql_query() is None

    # A ParamFilter is applied in pandas.
    param_filtered = Pipeline(
        source=DuckDBSource(uri=':memory:', mirrors={'t': df}), table='t',
        filters=[ParamFilter(field='cat')],
    )
    assert MosaicView(pipeline=param_filtered, spec=MOSAIC_DOT_SPEC)._sql_query() is None

    # A chained pipeline filters the parent's frame in pandas.
    chained = Pipeline(
        source=DuckDBSource(uri=':memory:', mirrors={'t': df}), table='t',
    ).chain(filters=[ConstantFilter(field='cat', value='b')])
    assert MosaicView(pipeline=chained, spec=MOSAIC_DOT_SPEC)._sql_query() is None
    # ...and still charts the right rows via the copy path.
    view = MosaicView(pipeline=chained, spec=MOSAIC_DOT_SPEC)
    assert _mosaic_charted_rows(view, ('cat', 'v')) == _mosaic_pandas_rows(chained, ('cat', 'v'))


@pytest.mark.skipif(DuckDBSource is None, reason='duckdb not installed')
def test_mosaic_view_reports_an_empty_pipeline_on_the_pushdown_path():
    """The empty-result error must fire on both data paths, so the agent's
    retry loop behaves the same however the pipeline is backed."""
    df = pd.DataFrame({'cat': list('aabb'), 'v': [1, 2, 3, 4]})
    pipeline = Pipeline(
        source=DuckDBSource(uri=':memory:', mirrors={'t': df}), table='t',
        filters=[ConstantFilter(field='cat', value='nope')],
    )

    with pytest.raises(ValueError, match='empty result'):
        MosaicView(pipeline=pipeline, spec=MOSAIC_DOT_SPEC).get_panel()
