from pathlib import Path

import holoviews as hv
import numpy as np
import pytest

from bokeh.document import Document
from panel.io.server import set_curdoc
from panel.layout import Card, Column, Row
from panel.pane import HoloViews
from panel.param import Param
from panel.widgets import Tabulator

from lumen.pipeline import Pipeline
from lumen.sources import DerivedSource, FileSource
from lumen.state import state
from lumen.target import Target
from lumen.transforms import Astype
from lumen.views import View


def test_view_controls(set_root):
    set_root(str(Path(__file__).parent))
    source = FileSource(tables={'test': 'sources/test.csv'})
    views = {
        'test': {
            'type': 'hvplot', 'table': 'test', 'controls': ['x', 'y'],
            'x': 'A', 'y': 'B', 'kind': 'scatter'
        }
    }
    target = Target.from_spec({'source': source, 'views': views})

    filter_panel = target.get_filter_panel()
    param_pane = filter_panel[0][0]
    assert isinstance(param_pane, Param)
    assert param_pane.parameters == ['x', 'y']

    assert len(target._cards) == 1
    card = target._cards[0]
    hv_pane = card._card[0][0]
    isinstance(hv_pane.object, hv.Scatter)
    assert hv_pane.object.kdims == ['A']
    assert hv_pane.object.vdims == ['B']

    param_pane._widgets['x'].value = 'C'
    param_pane._widgets['y'].value = 'D'

    isinstance(hv_pane.object, hv.Scatter)
    assert hv_pane.object.kdims == ['C']
    assert hv_pane.object.vdims == ['D']


def test_transform_controls(set_root):
    set_root(str(Path(__file__).parent))
    source = FileSource(tables={'test': 'sources/test.csv'})
    views = {
        'test': {
            'type': 'hvplot',
            'table': 'test',
            'x': 'A',
            'y': 'B',
            'kind': 'scatter',
            'transforms': [{
                'type': 'sort',
                'controls': ['by']
            }]
        }
    }

    doc = Document()
    with set_curdoc(doc):
        target = Target.from_spec({'views': views, 'source': source})
        filter_panel = target.get_filter_panel()
        param_pane = filter_panel[0][0]

        assert isinstance(param_pane, Param)
        assert param_pane.parameters == ['by']

        assert len(target._cards) == 1
        card = target._cards[0]
        hv_pane = card._card[0][0]
        isinstance(hv_pane.object, hv.Scatter)
        assert hv_pane.object.kdims == ['A']
        assert hv_pane.object.vdims == ['B']

        assert np.array_equal(hv_pane.object['A'], np.array([0, 1, 2, 3, 4]))
        assert np.array_equal(hv_pane.object['B'], np.array([0, 1, 0, 1, 0]))

        param_pane.object.by = ['B']

        assert np.array_equal(hv_pane.object['A'], np.array([0., 2, 4, 1, 3]))
        assert np.array_equal(hv_pane.object['B'], np.array([0, 0, 0, 1, 1]))


def test_view_controls_facetted(set_root):
    set_root(str(Path(__file__).parent))
    source = FileSource(tables={'test': 'sources/test.csv'})
    state.sources['test'] = source
    views = {
        'test': {
            'type': 'hvplot', 'table': 'test', 'controls': ['x', 'y'],
            'x': 'A', 'y': 'B', 'kind': 'scatter'
        },
        'test2': {
            'type': 'hvplot', 'table': 'test',
            'x': 'A', 'y': 'B', 'kind': 'scatter'
        }
    }
    spec = {
        'source': 'test',
        'facet': {'by': 'C'},
        'views': views
    }
    target = Target.from_spec(spec, sources={'test': source})

    filter_panel = target.get_filter_panel()
    param_pane = filter_panel[3][0]
    assert isinstance(param_pane, Param)
    assert param_pane.parameters == ['x', 'y']

    assert len(target._cards) == 5
    for card in target._cards:
        hv_pane = card._card[0][0]
        isinstance(hv_pane.object, hv.Scatter)
        assert hv_pane.object.kdims == ['A']
        assert hv_pane.object.vdims == ['B']

    param_pane._widgets['x'].value = 'C'
    param_pane._widgets['y'].value = 'D'

    for card in target._cards:
        hv_pane = card._card[0][0]
        isinstance(hv_pane.object, hv.Scatter)
        assert hv_pane.object.kdims == ['C']
        assert hv_pane.object.vdims == ['D']



def test_transform_controls_facetted(set_root):
    set_root(str(Path(__file__).parent))
    source = FileSource(tables={'test': 'sources/test.csv'})
    derived = DerivedSource(source=source, transforms=[Astype(dtypes={'B': 'int'})])
    views = {
        'test': {
            'type': 'hvplot',
            'table': 'test',
            'x': 'D',
            'y': 'A',
            'kind': 'scatter',
            'transforms': [{
                'type': 'sort',
                'by': 'C',
                'controls': ['ascending']
            }]
        },
    }
    spec = {
        'source': 'test',
        'facet': {'by': 'B'},
        'views': views
    }

    doc = Document()
    with set_curdoc(doc):
        state.sources['test'] = derived
        target = Target.from_spec(spec, sources={'test': derived})
        filter_panel = target.get_filter_panel()
        param_pane = filter_panel[3][0]

        assert isinstance(param_pane, Param)
        assert param_pane.parameters == ['ascending']

        assert len(target._cards) == 2
        card1, card2 = target._cards
        hv_pane1 = card1._card[0][0]
        isinstance(hv_pane1.object, hv.Scatter)
        assert hv_pane1.object.kdims == ['D']
        assert hv_pane1.object.vdims == ['A']
        hv_pane2 = card2._card[0][0]
        isinstance(hv_pane2.object, hv.Scatter)
        assert hv_pane2.object.kdims == ['D']
        assert hv_pane2.object.vdims == ['A']

        assert np.array_equal(hv_pane1.object['A'], np.array([0, 2, 4]))
        assert np.array_equal(hv_pane2.object['A'], np.array([1, 3]))

        param_pane.object.ascending = False

        assert np.array_equal(hv_pane1.object['A'], np.array([4, 2, 0]))
        assert np.array_equal(hv_pane2.object['A'], np.array([3, 1]))


@pytest.mark.parametrize(
    "layout,error",
    [
        ([[0]], None),
        ([[0], [1]], IndexError),
        ([[0], [1, 2]], IndexError),
        ([['test']], None),
        ([['test1']], KeyError),
    ]
)
def test_layout_view(set_root, layout, error):
    set_root(str(Path(__file__).parent))
    source = FileSource(tables={'test': 'sources/test.csv'})
    derived = DerivedSource(source=source, transforms=[Astype(dtypes={'B': 'int'})])
    views = {
        'test': {
            'type': 'table',
            'table': 'test'
        },
    }
    spec = {
        'source': 'test',
        'views': views,
        'layout': layout
    }

    doc = Document()
    with set_curdoc(doc):
        state.sources['test'] = derived

        if error is not None:
            with pytest.raises(error):
                Target.from_spec(spec, sources={'test': derived})
        else:
            Target.from_spec(spec, sources={'test': derived})


def test_target_constructor_with_views_instance_list(set_root):
    set_root(str(Path(__file__).parent))
    source = FileSource(tables={'test': 'sources/test.csv'})
    view1 = View.from_spec({
        'type': 'hvplot', 'table': 'test',
        'x': 'A', 'y': 'B', 'kind': 'scatter'
    }, source=source)
    view2 = View.from_spec({
        'type': 'table', 'table': 'test'
    }, source=source)
    target = Target(views=[view1, view2])
    layout = target.panels

    assert isinstance(layout, Column)
    assert len(layout) == 1
    assert isinstance(layout[0], Card)
    assert len(layout[0]) == 1
    assert isinstance(layout[0][0], Column)
    assert len(layout[0][0]) == 2
    assert isinstance(layout[0][0][0], HoloViews)
    assert isinstance(layout[0][0][1], Tabulator)

def test_target_constructor_with_views_instance_dict(set_root):
    set_root(str(Path(__file__).parent))
    source = FileSource(tables={'test': 'sources/test.csv'})
    view1 = View.from_spec({
        'type': 'hvplot', 'table': 'test',
        'x': 'A', 'y': 'B', 'kind': 'scatter'
    }, source=source)
    view2 = View.from_spec({
        'type': 'table', 'table': 'test'
    }, source=source)
    target = Target(views={'hv': view1, 'table': view2}, layout=[['hv'], ['table']])
    layout = target.panels

    assert isinstance(layout, Column)
    assert len(layout) == 1
    assert isinstance(layout[0], Card)
    assert len(layout[0]) == 1
    assert isinstance(layout[0][0], Column)
    assert len(layout[0][0]) == 2
    assert isinstance(layout[0][0][0], Row)
    assert isinstance(layout[0][0][0][0], HoloViews)
    assert isinstance(layout[0][0][1], Row)
    assert isinstance(layout[0][0][1][0], Tabulator)

def test_target_to_spec_view_list(set_root):
    set_root(str(Path(__file__).parent))
    source = FileSource(tables={'test': 'sources/test.csv'})
    pipeline = Pipeline(source=source, table='test')
    view1 = View.from_spec({
        'type': 'hvplot', 'x': 'A', 'y': 'B', 'kind': 'scatter'
    }, pipeline=pipeline)
    view2 = View.from_spec({'type': 'table'}, pipeline=pipeline)
    target = Target(views=[view1, view2], title='Plots')

    context = {}
    spec = target.to_spec(context)
    assert context == {
        'sources': {
            source.name: {
                'tables': {'test': 'sources/test.csv'},
                'type': 'file'
            },
        },
        'pipelines': {
            pipeline.name: {
                'source': source.name,
                'table': 'test'
            }
        },
    }
    assert spec == {
        'title': 'Plots',
        'views': [{
            'kind': 'scatter',
            'pipeline': pipeline.name,
            'type': 'hvplot',
            'x': 'A',
            'y': 'B'
        },
        {
            'pipeline': pipeline.name,
            'type': 'table'
        }],
    }


def test_target_to_spec_with_views_instance_dict(set_root):
    set_root(str(Path(__file__).parent))
    source = FileSource(tables={'test': 'sources/test.csv'})
    pipeline = Pipeline(source=source, table='test')
    view1 = View.from_spec({'type': 'hvplot', 'x': 'A', 'y': 'B', 'kind': 'scatter'}, pipeline=pipeline)
    view2 = View.from_spec({'type': 'table'}, pipeline=pipeline)
    target = Target(
        views={'hv': view1, 'table': view2},
        layout=[['hv'], ['table']],
        title="Test"
    )
    assert view1.pipeline is view2.pipeline

    context = {}
    spec = target.to_spec(context)
    assert context == {
        'sources': {
            source.name: {
                'tables': {'test': 'sources/test.csv'},
                'type': 'file'
            },
        },
        'pipelines': {
            pipeline.name: {
                'source': source.name,
                'table': 'test'
            }
        },
    }
    assert spec == {
        'title': 'Test',
        'layout': [['hv'], ['table']],
        'views': {
            'hv': {
                'kind': 'scatter',
                'pipeline': pipeline.name,
                'type': 'hvplot',
                'x': 'A',
                'y': 'B'
            },
            'table': {
                'pipeline': pipeline.name,
                'type': 'table'
            }
        }
    }
