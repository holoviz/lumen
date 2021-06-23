import pathlib

from lumen.config import config
from lumen.dashboard import Dashboard
from lumen.views import View

def test_dashboard_with_local_view(set_root):
    root = pathlib.Path(__file__).parent / 'sample_dashboard'
    set_root(str(root))
    dashboard = Dashboard(str(root / 'dashboard.yml'))
    target = dashboard.targets[0]
    view = View.from_spec(target.views[0], target.source, [])
    assert isinstance(view, config._modules[str(root / 'views.py')].TestView)
