from pathlib import Path

import holoviews as hv

from panel.param import Param

from lumen.sources import FileSource
from lumen.target import Target


def test_view_controls(set_root):
    set_root(str(Path(__file__).parent))
    source = FileSource(tables={'test': 'sources/test.csv'})
    views = {
        'test': {
            'type': 'hvplot', 'table': 'test', 'controls': ['x', 'y'],
            'x': 'A', 'y': 'B', 'kind': 'scatter'
        }
    }
    target = Target(source=source, views=views)

    filter_panel = target.get_filter_panel()
    param_pane = filter_panel[0][0]
    assert isinstance(param_pane, Param)
    assert param_pane.parameters == ['x', 'y']

    assert len(target._cards) == 1
    card = target._cards[0]
    hv_pane = card[0][0]
    isinstance(hv_pane.object, hv.Scatter)
    assert hv_pane.object.kdims == ['A']
    assert hv_pane.object.vdims == ['B']

    param_pane._widgets['x'].value = 'C'
    param_pane._widgets['y'].value = 'D'

    isinstance(hv_pane.object, hv.Scatter)
    assert hv_pane.object.kdims == ['C']
    assert hv_pane.object.vdims == ['D']


def test_view_controls_facetted():
    source = FileSource(tables={'test': 'sources/test.csv'}, root=str(Path(__file__).parent))
    views = {
        'test': {
            'type': 'hvplot', 'table': 'test', 'controls': ['x', 'y'],
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
    param_pane = filter_panel[4][0]
    assert isinstance(param_pane, Param)
    assert param_pane.parameters == ['x', 'y']

    assert len(target._cards) == 5
    for card in target._cards:
        hv_pane = card[0][0]
        isinstance(hv_pane.object, hv.Scatter)
        assert hv_pane.object.kdims == ['A']
        assert hv_pane.object.vdims == ['B']

    param_pane._widgets['x'].value = 'C'
    param_pane._widgets['y'].value = 'D'

    for card in target._cards:
        hv_pane = card[0][0]
        isinstance(hv_pane.object, hv.Scatter)
        assert hv_pane.object.kdims == ['C']
        assert hv_pane.object.vdims == ['D']
