import pathlib

import pandas as pd
import panel as pn

from lumen.config import config
from lumen.dashboard import Dashboard
from lumen.state import state
from lumen.views import View


def test_dashboard_with_local_view(set_root):
    root = pathlib.Path(__file__).parent / 'sample_dashboard'
    set_root(str(root))
    dashboard = Dashboard(str(root / 'dashboard.yml'))
    target = dashboard.targets[0]
    view = View.from_spec(target.views[0], target.source, [])
    assert isinstance(view, config._modules[str(root / 'views.py')].TestView)

def test_dashboard_with_url_sync(set_root, document):
    root = pathlib.Path(__file__).parent / 'sample_dashboard'
    set_root(str(root))
    dashboard = Dashboard(str(root / 'sync_query.yml'))
    dashboard._render_dashboard()
    assert pn.state.location.search == '?target=0'
    pn.state.location.search = '?target=1'
    assert dashboard._layout.active == 1

def test_dashboard_with_url_sync_filters(set_root, document):
    root = pathlib.Path(__file__).parent / 'sample_dashboard'
    set_root(str(root))
    dashboard = Dashboard(str(root / 'sync_query_filters.yml'))
    dashboard._render_dashboard()
    target = dashboard.targets[0]
    f1, f2 = target.filters
    f1.value = (0.1, 0.7)
    assert pn.state.location.search == '?A=%5B0.1%2C+0.7%5D'
    pn.state.location.search = '?A=%5B0.3%2C+0.8%5D'
    assert f1.value == (0.3, 0.8)
    f2.value = ['foo1', 'foo2']
    assert pn.state.location.search == '?A=%5B0.3%2C+0.8%5D&C=%5B%22foo1%22%2C+%22foo2%22%5D'
    pn.state.location.search = '?A=%5B0.3%2C+0.8%5D&C=%5B%22foo1%22%2C+%22foo2%22%2C+%22foo3%22%5D'
    assert f2.value == ['foo1', 'foo2', 'foo3']

def test_dashboard_with_sql_source_and_transforms(set_root, document):
    root = pathlib.Path(__file__).parent / 'sample_dashboard'
    set_root(str(root))
    dashboard = Dashboard(str(root / 'sql_dashboard.yml'))
    dashboard._render_dashboard()
    target = dashboard.targets[0]
    target.update()

    table = target._cards[0][0][0]
    expected = pd._testing.makeMixedDataFrame()
    pd.testing.assert_frame_equal(table.value, expected)

    dashboard._sidebar[0][0][0][1]._widgets['limit'].value = 2

    pd.testing.assert_frame_equal(table.value, expected.iloc[:2])

def test_dashboard_with_transform_variable(set_root, document):
    root = pathlib.Path(__file__).parent / 'sample_dashboard'
    set_root(str(root))
    dashboard = Dashboard(str(root / 'transform_variable.yml'))
    dashboard._render_dashboard()
    target = dashboard.targets[0]
    target.update()

    table = target._cards[0][0][0]
    expected = pd._testing.makeMixedDataFrame()
    pd.testing.assert_frame_equal(table.value, expected)

    state.variables.length = 2

    pd.testing.assert_frame_equal(table.value, expected.iloc[:2])

def test_dashboard_with_source_variable(set_root, document):
    root = pathlib.Path(__file__).parent / 'sample_dashboard'
    set_root(str(root))
    dashboard = Dashboard(str(root / 'source_variable.yml'))
    dashboard._render_dashboard()
    target = dashboard.targets[0]
    target.update()

    table = target._cards[0][0][0]
    expected = pd._testing.makeMixedDataFrame()
    pd.testing.assert_frame_equal(table.value, expected)

    state.variables.tables = {'test': '../sources/test2.csv'}

    pd.testing.assert_frame_equal(table.value, expected.iloc[::-1].reset_index(drop=True))

def test_dashboard_with_view_variable(set_root, document):
    root = pathlib.Path(__file__).parent / 'sample_dashboard'
    set_root(str(root))
    dashboard = Dashboard(str(root / 'view_variable.yml'))
    dashboard._render_dashboard()
    target = dashboard.targets[0]
    target.update()

    table = target._cards[0][0][0]

    assert table.page_size == 20

    state.variables.page_size = 10

    assert table.page_size == 10
