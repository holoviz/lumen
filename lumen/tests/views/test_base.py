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
    Panel, View, hvOverlayView, hvPlotView,
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


def test_panel_view_roundtrip():
    md = Markdown("TEST", width=420)

    spec = Panel(object=md).to_spec()
    assert spec == {
        'type': 'panel',
        'object': {
            'name': f'"{md.name}"',
            'object': '"TEST"',
            'type': 'panel.pane.markup.Markdown',
            'width': '420',
        }
    }

    panel_view = Panel.from_spec(spec)
    obj = panel_view.object
    assert isinstance(obj, Markdown)
    assert obj.name == md.name
    assert obj.object == md.object
    assert obj.width == 420


def test_panel_layout_roundtrip():
    a, b = Markdown("A"), Markdown("B")
    column = Column(a, b)

    spec = Panel(object=column).to_spec()
    assert spec == {
        'type': 'panel',
        'object': {
            'name': f'"{column.name}"',
            'objects': [
                {'type': 'panel.pane.markup.Markdown', 'object': '"A"', 'name': f'"{a.name}"'},
                {'type': 'panel.pane.markup.Markdown', 'object': '"B"', 'name': f'"{b.name}"'}
            ],
            'type': 'panel.layout.base.Column'
        }
    }

def test_panel_cross_reference_param():
    a = Checkbox(name="A")
    b = Markdown("B", visible=a)
    column = Column(a, b)

    spec = Panel(object=column).to_spec()
    assert spec == {
        'type': 'panel',
        'object': {
            'name': f'"{column.name}"',
            'objects': [
                {
                    'type': 'panel.widgets.input.Checkbox',
                    'name': f'"{a.name}"'
                },
                {
                    'type': 'panel.pane.markup.Markdown',
                    'object': '"B"',
                    'name': f'"{b.name}"',
                    'visible': {
                        'name': 'value',
                        'owner': 'A',
                        'type': 'param'
                    }
                }
            ],
            'type': 'panel.layout.base.Column'
        }
    }

def test_panel_cross_reference_rx():
    a = Checkbox(name="A")
    b = Markdown("B", visible=a.rx().rx.not_())
    column = Column(a, b)
    spec = Panel(object=column).to_spec()
    assert spec == {
        'type': 'panel',
        'object': {
            'name': f'"{column.name}"',
            'objects': [
                {
                    'type': 'panel.widgets.input.Checkbox',
                    'name': f'"{a.name}"'
                },
                {
                    'type': 'panel.pane.markup.Markdown',
                    'object': '"B"',
                    'name': f'"{b.name}"',
                    'visible': {
                        'input_obj': None,
                        'kwargs': {},
                        'method': None,
                        'operation': {
                            'args': [],
                            'fn': '_operator.not_',
                            'kwargs': {},
                            'reverse': False,
                        },
                        'prev': {
                            'type': 'rx',
                            'input_obj': None,
                            'kwargs': {},
                            'method': None,
                            'operation': None,
                            'prev': {
                                'name': 'value',
                                'owner': 'A',
                                'type': 'param'
                            }
                        },
                        'type': 'rx',
                    }
                }
            ],
            'type': 'panel.layout.base.Column'
        }
    }
