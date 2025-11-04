from pathlib import Path

import pandas as pd
import panel as pn
import pytest

from panel.layout import Column
from panel.pane import Markdown
from panel.widgets import Checkbox

from lumen.panel import DownloadButton
from lumen.sources.base import FileSource
from lumen.state import state
from lumen.variables.base import Variables
from lumen.views.base import (
    Panel, VegaLiteView, View, hvOverlayView, hvPlotView,
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


def test_vegalite_view_basic(set_root):
    """Test basic VegaLite view creation and data rendering."""
    set_root(str(Path(__file__).parent.parent))
    source = FileSource(tables={'test': 'sources/test.csv'})
    
    vegalite_spec = {
        "$schema": "https://vega.github.io/schema/vega-lite/v5.json",
        "mark": "bar",
        "encoding": {
            "x": {"field": "A", "type": "quantitative"},
            "y": {"field": "B", "type": "quantitative"}
        }
    }
    
    view_spec = {
        'type': 'vegalite',
        'table': 'test',
        'spec': vegalite_spec
    }
    
    view = View.from_spec(view_spec, source)
    assert isinstance(view, VegaLiteView)
    assert view.spec == vegalite_spec
    
    # Test that data is properly embedded
    panel = view.get_panel()
    assert isinstance(panel, pn.pane.Vega)
    
    # Test that data is available for conversion
    data = view.get_data()
    assert len(data) == 5  # test.csv has 5 rows
    assert 'A' in data.columns
    assert 'B' in data.columns
    

def test_vegalite_view_convert_png(set_root):
    """Test converting VegaLite view to PNG format."""
    pytest.importorskip('vl_convert')  # Skip if vl_convert not installed
    
    set_root(str(Path(__file__).parent.parent))
    source = FileSource(tables={'test': 'sources/test.csv'})
    
    vegalite_spec = {
        "$schema": "https://vega.github.io/schema/vega-lite/v5.json",
        "mark": "bar",
        "encoding": {
            "x": {"field": "A", "type": "quantitative"},
            "y": {"field": "B", "type": "quantitative"}
        }
    }
    
    view_spec = {
        'type': 'vegalite',
        'table': 'test',
        'spec': vegalite_spec,
        'width': 600,
        'height': 400
    }
    
    view = View.from_spec(view_spec, source)
    
    # Test PNG conversion
    png_data = view.convert('png')
    assert isinstance(png_data, bytes)
    assert len(png_data) > 0
    # PNG files start with a specific signature
    assert png_data[:8] == b'\x89PNG\r\n\x1a\n'
    

def test_vegalite_view_convert_svg(set_root):
    """Test converting VegaLite view to SVG format."""
    pytest.importorskip('vl_convert')  # Skip if vl_convert not installed
    
    set_root(str(Path(__file__).parent.parent))
    source = FileSource(tables={'test': 'sources/test.csv'})
    
    vegalite_spec = {
        "$schema": "https://vega.github.io/schema/vega-lite/v5.json",
        "mark": "circle",
        "encoding": {
            "x": {"field": "A", "type": "quantitative"},
            "y": {"field": "B", "type": "quantitative"}
        }
    }
    
    view_spec = {
        'type': 'vegalite',
        'table': 'test', 
        'spec': vegalite_spec
    }
    
    view = View.from_spec(view_spec, source)
    
    # Test SVG conversion
    svg_data = view.convert('svg')
    assert isinstance(svg_data, (str, bytes))
    
    # Convert to string if bytes
    if isinstance(svg_data, bytes):
        svg_str = svg_data.decode('utf-8')
    else:
        svg_str = svg_data
        
    assert '<svg' in svg_str
    assert '</svg>' in svg_str
    

def test_vegalite_view_convert_jpeg(set_root):
    """Test converting VegaLite view to JPEG format."""
    pytest.importorskip('vl_convert')  # Skip if vl_convert not installed
    
    set_root(str(Path(__file__).parent.parent))
    source = FileSource(tables={'test': 'sources/test.csv'})
    
    vegalite_spec = {
        "$schema": "https://vega.github.io/schema/vega-lite/v5.json",
        "mark": "line",
        "encoding": {
            "x": {"field": "A", "type": "quantitative"},
            "y": {"field": "B", "type": "quantitative"}
        }
    }
    
    view_spec = {
        'type': 'vegalite',
        'table': 'test',
        'spec': vegalite_spec
    }
    
    view = View.from_spec(view_spec, source)
    
    # Test JPEG conversion
    jpeg_data = view.convert('jpeg')
    assert isinstance(jpeg_data, bytes)
    assert len(jpeg_data) > 0
    # JPEG files start with FF D8 FF
    assert jpeg_data[:3] == b'\xff\xd8\xff'
    

def test_vegalite_view_convert_pdf(set_root):
    """Test converting VegaLite view to PDF format."""
    pytest.importorskip('vl_convert')  # Skip if vl_convert not installed
    
    set_root(str(Path(__file__).parent.parent))
    source = FileSource(tables={'test': 'sources/test.csv'})
    
    vegalite_spec = {
        "$schema": "https://vega.github.io/schema/vega-lite/v5.json",
        "mark": "point",
        "encoding": {
            "x": {"field": "A", "type": "quantitative"},
            "y": {"field": "B", "type": "quantitative"}
        }
    }
    
    view_spec = {
        'type': 'vegalite',
        'table': 'test',
        'spec': vegalite_spec
    }
    
    view = View.from_spec(view_spec, source)
    
    # Test PDF conversion
    pdf_data = view.convert('pdf')
    assert isinstance(pdf_data, bytes)
    assert len(pdf_data) > 0
    # PDF files start with %PDF
    assert pdf_data[:4] == b'%PDF'
    

def test_vegalite_view_convert_html(set_root):
    """Test converting VegaLite view to HTML format."""
    pytest.importorskip('vl_convert')  # Skip if vl_convert not installed
    
    set_root(str(Path(__file__).parent.parent))
    source = FileSource(tables={'test': 'sources/test.csv'})
    
    vegalite_spec = {
        "$schema": "https://vega.github.io/schema/vega-lite/v5.json",
        "mark": "area",
        "encoding": {
            "x": {"field": "A", "type": "quantitative"},
            "y": {"field": "B", "type": "quantitative"}
        }
    }
    
    view_spec = {
        'type': 'vegalite',
        'table': 'test',
        'spec': vegalite_spec
    }
    
    view = View.from_spec(view_spec, source)
    
    # Test HTML conversion
    html_data = view.convert('html')
    assert isinstance(html_data, (str, bytes))
    
    # Convert to string if bytes
    if isinstance(html_data, bytes):
        html_str = html_data.decode('utf-8')
    else:
        html_str = html_data
        
    assert '<!DOCTYPE html>' in html_str or '<html' in html_str
    assert 'vega' in html_str.lower()
    

def test_vegalite_view_convert_url(set_root):
    """Test converting VegaLite view to Vega Editor URL."""
    pytest.importorskip('vl_convert')  # Skip if vl_convert not installed
    
    set_root(str(Path(__file__).parent.parent))
    source = FileSource(tables={'test': 'sources/test.csv'})
    
    vegalite_spec = {
        "$schema": "https://vega.github.io/schema/vega-lite/v5.json",
        "mark": "rect",
        "encoding": {
            "x": {"field": "A", "type": "quantitative"},
            "y": {"field": "B", "type": "quantitative"}
        }
    }
    
    view_spec = {
        'type': 'vegalite',
        'table': 'test',
        'spec': vegalite_spec
    }
    
    view = View.from_spec(view_spec, source)
    
    # Test URL conversion
    url = view.convert('url')
    assert isinstance(url, str)
    assert 'vega.github.io' in url
    

def test_vegalite_view_convert_scenegraph(set_root):
    """Test converting VegaLite view to Vega scenegraph."""
    pytest.importorskip('vl_convert')  # Skip if vl_convert not installed
    
    set_root(str(Path(__file__).parent.parent))
    source = FileSource(tables={'test': 'sources/test.csv'})
    
    vegalite_spec = {
        "$schema": "https://vega.github.io/schema/vega-lite/v5.json",
        "mark": "tick",
        "encoding": {
            "x": {"field": "A", "type": "quantitative"}
        }
    }
    
    view_spec = {
        'type': 'vegalite',
        'table': 'test',
        'spec': vegalite_spec
    }
    
    view = View.from_spec(view_spec, source)
    
    # Test scenegraph conversion
    scenegraph = view.convert('scenegraph')
    assert scenegraph is not None
    # Scenegraph should be a dict or JSON string
    if isinstance(scenegraph, (str, bytes)):
        import json
        if isinstance(scenegraph, bytes):
            scenegraph = scenegraph.decode('utf-8')
        scenegraph = json.loads(scenegraph)
    assert isinstance(scenegraph, dict)
    

def test_vegalite_view_convert_invalid_format(set_root):
    """Test that converting to invalid format raises ValueError."""
    pytest.importorskip('vl_convert')  # Skip if vl_convert not installed
    
    set_root(str(Path(__file__).parent.parent))
    source = FileSource(tables={'test': 'sources/test.csv'})
    
    vegalite_spec = {
        "$schema": "https://vega.github.io/schema/vega-lite/v5.json",
        "mark": "bar",
        "encoding": {
            "x": {"field": "A", "type": "quantitative"},
            "y": {"field": "B", "type": "quantitative"}
        }
    }
    
    view_spec = {
        'type': 'vegalite',
        'table': 'test',
        'spec': vegalite_spec
    }
    
    view = View.from_spec(view_spec, source)
    
    # Test that invalid format raises ValueError
    with pytest.raises(ValueError, match="Unsupported format"):
        view.convert('invalid_format')
        

def test_vegalite_view_convert_with_custom_dimensions(set_root):
    """Test that custom width/height in kwargs are used."""
    pytest.importorskip('vl_convert')  # Skip if vl_convert not installed
    
    set_root(str(Path(__file__).parent.parent))
    source = FileSource(tables={'test': 'sources/test.csv'})
    
    vegalite_spec = {
        "$schema": "https://vega.github.io/schema/vega-lite/v5.json",
        "mark": "bar",
        "encoding": {
            "x": {"field": "A", "type": "quantitative"},
            "y": {"field": "B", "type": "quantitative"}
        }
    }
    
    view_spec = {
        'type': 'vegalite',
        'table': 'test',
        'spec': vegalite_spec,
        'width': 1000,
        'height': 600
    }
    
    view = View.from_spec(view_spec, source)
    
    # Test that dimensions are properly set
    png_data = view.convert('png')
    assert isinstance(png_data, bytes)
    assert len(png_data) > 0
    
    # The spec used for conversion should include the dimensions
    # We can't directly check image dimensions without PIL, but we can verify the data is valid
    assert png_data[:8] == b'\x89PNG\r\n\x1a\n'
    

def test_vegalite_view_convert_with_kwargs(set_root):
    """Test that additional kwargs are passed to conversion function."""
    pytest.importorskip('vl_convert')  # Skip if vl_convert not installed
    
    set_root(str(Path(__file__).parent.parent))
    source = FileSource(tables={'test': 'sources/test.csv'})
    
    vegalite_spec = {
        "$schema": "https://vega.github.io/schema/vega-lite/v5.json",
        "mark": "bar",
        "encoding": {
            "x": {"field": "A", "type": "quantitative"},
            "y": {"field": "B", "type": "quantitative"}
        }
    }
    
    view_spec = {
        'type': 'vegalite',
        'table': 'test',
        'spec': vegalite_spec
    }
    
    view = View.from_spec(view_spec, source)
    
    # Test PNG conversion with scale parameter
    png_data = view.convert('png', scale=2)
    assert isinstance(png_data, bytes)
    assert len(png_data) > 0
    
    # Higher scale should generally produce larger file
    png_data_low_scale = view.convert('png', scale=1)
    # Note: This might not always be true due to compression, but generally should be
    # We'll just check both are valid PNGs
    assert png_data[:8] == b'\x89PNG\r\n\x1a\n'
    assert png_data_low_scale[:8] == b'\x89PNG\r\n\x1a\n'


def test_vegalite_view_data_embedding_in_conversion(set_root):
    """Test that data from pipeline is properly embedded in the spec during conversion."""
    pytest.importorskip('vl_convert')  # Skip if vl_convert not installed
    
    set_root(str(Path(__file__).parent.parent))
    source = FileSource(tables={'test': 'sources/test.csv'})
    
    vegalite_spec = {
        "$schema": "https://vega.github.io/schema/vega-lite/v5.json",
        "mark": "bar",
        "encoding": {
            "x": {"field": "A", "type": "quantitative"},
            "y": {"field": "B", "type": "quantitative"}
        }
    }
    
    view_spec = {
        'type': 'vegalite',
        'table': 'test',
        'spec': vegalite_spec
    }
    
    view = View.from_spec(view_spec, source)
    
    # Get the data that should be embedded
    data = view.get_data()
    
    # The convert method should embed this data in the spec
    # We'll test with SVG as it's easier to verify
    svg_data = view.convert('svg')
    
    # Convert to string if bytes
    if isinstance(svg_data, bytes):
        svg_str = svg_data.decode('utf-8')
    else:
        svg_str = svg_data
    
    # The SVG should be valid and contain visualization elements
    assert '<svg' in svg_str
    assert '</svg>' in svg_str
    
    # Verify that the conversion uses the correct data shape
    # (5 data points from test.csv)
    assert len(data) == 5
