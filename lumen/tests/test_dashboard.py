import os

from lumen.dashboard import Dashboard
from lumen.views import View

def test_dashboard_with_local_view():
    root = os.path.dirname(__file__)
    dashboard = Dashboard(os.path.join(root, 'sample_dashboard', 'dashboard.yml'))
    target = dashboard.targets[0]
    view = View.from_spec(target.views[0], target.source, [])
    assert isinstance(view, dashboard._modules['views'].TestView)
