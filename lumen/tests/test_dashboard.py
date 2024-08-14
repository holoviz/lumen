import pathlib

import pandas as pd
import panel as pn
import pytest

from lumen.config import config
from lumen.dashboard import Dashboard
from lumen.state import state
from lumen.validation import ValidationError

from .test_pipeline import sql_available


def test_dashboard_with_local_view(set_root):
    root = pathlib.Path(__file__).parent / 'sample_dashboard'
    set_root(str(root))
    dashboard = Dashboard(str(root / 'dashboard.yaml'))
    dashboard._render_dashboard()
    view = dashboard.layouts[0]._cards[0].views[0]
    assert isinstance(view, config._modules[str(root / 'views.py')].TestView)

def test_dashboard_with_local_view_legacy(set_root):
    root = pathlib.Path(__file__).parent / 'sample_dashboard'
    set_root(str(root))
    dashboard = Dashboard(str(root / 'dashboard_legacy.yaml'))
    dashboard._render_dashboard()
    layout = dashboard.layouts[0]
    view = layout._cards[0].views[0]
    assert isinstance(view, config._modules[str(root / 'views.py')].TestView)

def test_dashboard_from_spec():
    spec = {
        'sources': {
            'test': {'type': 'file', 'tables': ['./sources/test.csv']}
        },
        'layouts': [{
            'title': 'Test',
            'source': 'test',
            'views': [{'table': 'test', 'type': 'table'}],
        }]
    }
    dashboard = Dashboard(spec, root=str(pathlib.Path(__file__).parent))
    dashboard._render_dashboard()
    assert state.spec == spec
    layout = dashboard.layouts[0]
    card = layout._cards[0]
    assert card.views[0].view_type == 'table'

def test_dashboard_from_spec_invalid():
    with pytest.raises(ValidationError):
        Dashboard({'foo': 'bar'})

def test_dashboard_reload_layout(set_root):
    root = pathlib.Path(__file__).parent / 'sample_dashboard'
    set_root(str(root))
    dashboard = Dashboard(str(root / 'dashboard.yaml'))
    dashboard._render_dashboard()

    reload_button, ts = dashboard._sidebar[-1][-1]
    old_timestamp = ts.object
    old_objects = dashboard._layout.objects
    reload_button._button_click()
    assert ts.object is not old_timestamp
    assert dashboard._layout.objects is not old_objects

def test_dashboard_with_url_sync(set_root, document):
    root = pathlib.Path(__file__).parent / 'sample_dashboard'
    set_root(str(root))
    dashboard = Dashboard(str(root / 'sync_query.yaml'))
    dashboard._render_dashboard()
    assert pn.state.location.search == '?layout=0'
    pn.state.location.search = '?layout=1'
    assert dashboard._layout.active == 1

def test_dashboard_with_url_sync_filters(set_root, document):
    root = pathlib.Path(__file__).parent / 'sample_dashboard'
    set_root(str(root))
    dashboard = Dashboard(str(root / 'sync_query_filters.yaml'))
    dashboard._render_dashboard()
    layout = dashboard.layouts[0]
    f1, f2 = next(iter(layout._pipelines.values())).filters
    f1.value = (0.1, 0.7)
    assert pn.state.location.query_params == {'A': [0.1, 0.7], 'C': []}
    pn.state.location.search = '?A=%5B0.3%2C+0.8%5D'
    assert f1.value == (0.3, 0.8)
    f2.value = ['foo1', 'foo2']
    assert pn.state.location.query_params == {'A': [0.3, 0.8], 'C': ['foo1', 'foo2']}
    pn.state.location.search = '?A=%5B0.3%2C+0.8%5D&C=%5B%22foo1%22%2C+%22foo2%22%2C+%22foo3%22%5D'
    assert f2.value == ['foo1', 'foo2', 'foo3']

def test_dashboard_with_url_sync_filters_with_default(set_root, document):
    root = pathlib.Path(__file__).parent / 'sample_dashboard'
    set_root(str(root))
    dashboard = Dashboard(str(root / 'sync_query_filters_default.yaml'))
    dashboard._render_dashboard()
    layout = dashboard.layouts[0]
    f1, f2 = next(iter(layout._pipelines.values())).filters
    f1.value = (0.1, 0.7)
    assert pn.state.location.query_params == {'C': ['foo1'], 'A': [0.1, 0.7]}
    pn.state.location.search = '?A=%5B0.3%2C+0.8%5D'
    assert f1.value == (0.3, 0.8)
    assert f2.value == ['foo1']
    assert f1.widget.value == (0.3, 0.8)
    assert f2.widget.value == ['foo1']
    f2.value = ['foo1', 'foo2']
    assert pn.state.location.query_params == {'C': ['foo1', 'foo2'], 'A': [0.3, 0.8]}
    pn.state.location.search = '?A=%5B0.3%2C+0.8%5D&C=%5B%22foo1%22%2C+%22foo2%22%2C+%22foo3%22%5D'
    assert f2.value == ['foo1', 'foo2', 'foo3']
    assert f2.widget.value == ['foo1', 'foo2', 'foo3']

def test_dashboard_with_url_sync_filters_with_overwritten_default(set_root, document):
    root = pathlib.Path(__file__).parent / 'sample_dashboard'
    set_root(str(root))
    dashboard = Dashboard(str(root / 'sync_query_filters_default.yaml'))
    dashboard._render_dashboard()
    layout = dashboard.layouts[0]
    f1, f2 = next(iter(layout._pipelines.values())).filters
    f1.value = (0.1, 0.7)
    f2.value = []  # overwriting default with empty list
    assert pn.state.location.query_params == {'C': [], 'A': [0.1, 0.7]}
    assert f2.widget.value == []

@sql_available
def test_dashboard_with_sql_source_and_transforms(set_root, document, mixed_df_object_type):
    root = pathlib.Path(__file__).parent / 'sample_dashboard'
    set_root(str(root))
    dashboard = Dashboard(str(root / 'sql_dashboard.yaml'))
    dashboard._render_dashboard()
    layout = dashboard.layouts[0]
    layout.update()

    table = layout._cards[0]._card[0][0]
    pd.testing.assert_frame_equal(table.value, mixed_df_object_type)

    dashboard._sidebar[0][0][0]._widgets['limit'].value = 2

    pd.testing.assert_frame_equal(table.value, mixed_df_object_type.iloc[:2])

def test_dashboard_with_transform_variable(set_root, document, mixed_df):
    root = pathlib.Path(__file__).parent / 'sample_dashboard'
    set_root(str(root))
    dashboard = Dashboard(str(root / 'transform_variable.yaml'))
    dashboard._render_dashboard()
    layout = dashboard.layouts[0]
    layout.update()

    table = layout._cards[0]._card[0][0]
    pd.testing.assert_frame_equal(table.value, mixed_df)

    state.variables.length = 2

    pd.testing.assert_frame_equal(table.value, mixed_df.iloc[:2])

def test_dashboard_with_source_variable(set_root, document, mixed_df):
    root = pathlib.Path(__file__).parent / 'sample_dashboard'
    set_root(str(root))
    dashboard = Dashboard(str(root / 'source_variable.yaml'))
    dashboard._render_dashboard()
    layout = dashboard.layouts[0]
    layout.update()

    table = layout._cards[0]._card[0][0]
    pd.testing.assert_frame_equal(table.value, mixed_df)

    state.variables.tables = {'test': '../sources/test2.csv'}

    pd.testing.assert_frame_equal(table.value, mixed_df.iloc[::-1].reset_index(drop=True))

def test_dashboard_with_nested_source_variable(set_root, document, mixed_df):
    root = pathlib.Path(__file__).parent / 'sample_dashboard'
    set_root(str(root))
    dashboard = Dashboard(str(root / 'source_nested_variable.yaml'))
    dashboard._render_dashboard()
    layout = dashboard.layouts[0]
    layout.update()

    table = layout._cards[0]._card[0][0]
    pd.testing.assert_frame_equal(table.value, mixed_df)

    state.variables.ticker = '../sources/test2.csv'

    pd.testing.assert_frame_equal(table.value, mixed_df.iloc[::-1].reset_index(drop=True))

def test_dashboard_with_view_variable(set_root, document):
    root = pathlib.Path(__file__).parent / 'sample_dashboard'
    set_root(str(root))
    dashboard = Dashboard(str(root / 'view_variable.yaml'))
    dashboard._render_dashboard()
    layout = dashboard.layouts[0]
    layout.update()

    table = layout._cards[0]._card[0][0]

    assert table.page_size == 20

    state.variables.page_size = 10

    assert table.page_size == 10

def test_dashboard_with_view_and_transform_variable(set_root, document):
    root = pathlib.Path(__file__).parent / 'sample_dashboard'
    set_root(str(root))
    dashboard = Dashboard(str(root / 'view_transform_variable.yaml'))
    dashboard._render_dashboard()
    layout = dashboard.layouts[0]
    layout.update()

    plot = layout._cards[0]._card[0][0]

    assert plot.object.vdims == ['Z']

    state.variables.rename = 'Y'

    assert plot.object.vdims == ['Z']

    next(iter(layout._pipelines.values())).param.trigger('update')

    assert plot.object.vdims == ['Y']


def test_dashboard_with_template_string(set_root, document):
    root = pathlib.Path(__file__).parent / 'sample_dashboard'
    set_root(str(root))
    dashboard = Dashboard(str(root / 'template.yaml'))
    dashboard._render_dashboard()
    assert isinstance(dashboard._template, pn.template.VanillaTemplate)

    assert dashboard.to_spec()['config']['template'] == "vanilla"


def test_dashboard_with_template_params(set_root, document):
    root = pathlib.Path(__file__).parent / 'sample_dashboard'
    set_root(str(root))
    dashboard = Dashboard(str(root / 'template_params.yaml'))
    dashboard._render_dashboard()
    assert isinstance(dashboard._template, pn.template.VanillaTemplate)
    assert dashboard._template.collapsed_sidebar

    assert dashboard.to_spec()['config']['template'] == {'type': 'vanilla', 'collapsed_sidebar': True}
