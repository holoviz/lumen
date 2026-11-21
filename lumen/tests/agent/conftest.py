from importlib.util import find_spec

if find_spec("panel_flowdash") is None:
    # lumen.agent runs on panel-flowdash, which needs Python 3.12.
    collect_ignore_glob = ["test_*.py"]
