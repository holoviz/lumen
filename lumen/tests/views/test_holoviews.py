from pathlib import Path

import holoviews as hv
import numpy as np
import pandas as pd

from holoviews.core import Store

from lumen.pipeline import Pipeline
from lumen.sources.base import FileSource
from lumen.views.base import HoloViews


def test_serialize_holoviews_element():
    element = hv.Curve([1, 2, 3])
    view = HoloViews(object=element)

    spec = view.to_spec()
    assert spec == {
        'object': {
            'type': 'holoviews.element.chart.Curve',
            'kdims': [{'name': 'x'}],
            'vdims': [{'name': 'y'}],
            'data': {
                'x': np.array([0, 1, 2]).tolist(),
                'y': np.array([1, 2, 3]).tolist()
            },
        },
        'streaming': False,
        'type': 'holoviews'
    }

    view = HoloViews.from_spec(spec)
    assert isinstance(view.object, hv.Curve)
    assert view.pipeline is None
    assert np.array_equal(view.object.data['x'], element.data['x'])
    assert np.array_equal(view.object.data['y'], element.data['y'])

def test_serialize_holoviews_element_with_pipeline(set_root):
    set_root(str(Path(__file__).parent.parent))
    source = FileSource(tables={'test': 'sources/test.csv'})
    pipeline = Pipeline(source=source, table='test')
    element = hv.Curve(pipeline.data, 'A', 'B')
    view = HoloViews(object=element, pipeline=pipeline)

    spec = view.to_spec()
    assert spec == {
        'object': {
            'type': 'holoviews.element.chart.Curve',
            'kdims': [{'name': 'A'}],
            'vdims': [{'name': 'B'}],
            'data': '$data',
        },
        'streaming': False,
        'type': 'holoviews',
        'pipeline': {
            'source': {
                'type': 'file',
                'tables': {'test': 'sources/test.csv'}
            },
            'table': 'test',
        }
    }

    view = HoloViews.from_spec(spec)
    assert isinstance(view.object, hv.Curve)
    assert view.pipeline.data is view.object.data
    pd.testing.assert_frame_equal(view.object.data, pipeline.data)

def test_serialize_holoviews_element_with_operation():
    element = hv.Curve([1, 2, 3]).opts(width=100, height=100)
    view = HoloViews(object=element)
    spec = view.to_spec()
    assert spec == {
        'object': {
            'type': 'holoviews.element.chart.Curve',
            'kdims': [{'name': 'x'}],
            'vdims': [{'name': 'y'}],
            'data': {
                'x': np.array([0, 1, 2]).tolist(),
                'y': np.array([1, 2, 3]).tolist()
            },
            'operation': {
                'type': 'holoviews.operation.element.chain',
                 'operations': [
                     {
                         'type': 'holoviews.operation.element.factory',
                         'editor_type': {'type': 'holoviews.core.accessors.Opts', 'instance': False},
                         'kwargs': {'mode': None}
                     },
                     {
                         'type': 'holoviews.operation.element.method',
                         'input_type': {'type': 'holoviews.core.accessors.Opts', 'instance': False},
                         'kwargs': {'width': 100, 'height': 100}
                     }
                 ]
            }
        },
        'streaming': False,
        'type': 'holoviews'
    }

    view = HoloViews.from_spec(spec)
    assert isinstance(view.object, hv.Curve)
    opts = Store.lookup_options('bokeh', view.object, 'plot')
    assert opts.options['width'] == 100
    assert opts.options['height'] == 100

def test_serialize_holoviews_overlay():
    overlay = hv.Curve([1, 2, 3]) * hv.Curve([4, 5, 6])
    view = HoloViews(object=overlay)

    spec = view.to_spec()
    assert spec == {
        'object': {
            'type': 'holoviews.core.overlay.Overlay',
            'group': 'Overlay',
            'data': [
                (
                    ('Curve', 'I'),
                    {
                        'type': 'holoviews.element.chart.Curve',
                        'kdims': [{'name': 'x'}],
                        'vdims': [{'name': 'y'}],
                        'data': {
                            'x': np.array([0, 1, 2]).tolist(),
                            'y': np.array([1, 2, 3]).tolist()
                        }
                    }
                ), (
                    ('Curve', 'II'),
                    {
                        'type': 'holoviews.element.chart.Curve',
                        'kdims': [{'name': 'x'}],
                        'vdims': [{'name': 'y'}],
                        'data': {
                            'x': np.array([0, 1, 2]).tolist(),
                            'y': np.array([4, 5, 6]).tolist()
                        }
                    }
                )
            ]
        },
        'streaming': False,
        'type': 'holoviews'
    }

    view = HoloViews.from_spec(spec)
    assert isinstance(view.object, hv.Overlay)
    el1, el2 = view.object.data.values()
    assert np.array_equal(el1.data['x'], overlay.get(0).data['x'])
    assert np.array_equal(el1.data['y'], overlay.get(0).data['y'])
    assert np.array_equal(el2.data['x'], overlay.get(1).data['x'])
    assert np.array_equal(el2.data['y'], overlay.get(1).data['y'])
