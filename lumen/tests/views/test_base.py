from pathlib import Path

import pandas as pd
import panel as pn
import pytest

from lumen.panel import DownloadButton
from lumen.sources.base import FileSource
from lumen.state import state
from lumen.variables.base import Variables
from lumen.views.base import View, hvPlotView


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
