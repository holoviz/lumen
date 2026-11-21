"""
Dashboards saved before the view nodes shared a base load and save unchanged.

The fixtures were saved by ``lumen.flowdash`` at the Lumen FlowDash stack
top before ``LumenViewNode`` existed: a YAML export (``flowdash: 1``) and the
stored model (version 4) of the same dashboard of source, SQL, table, filter
and Vega-Lite view nodes.
"""
import json

from pathlib import Path

import pytest
import yaml

pytest.importorskip("duckdb")

from panel_flowdash import FlowDash
from panel_flowdash.agent import (
    DashboardOps, dashboard_from_yaml, dashboard_to_yaml,
)

from lumen.flowdash import COMPONENTS, ViewNode, bind_source
from lumen.views.base import VegaLiteView

FIXTURES = Path(__file__).parent / "fixtures"


def _loaded(source, dashboard):
    editor = FlowDash(components=COMPONENTS)
    bind_source(editor, source)
    editor.import_dashboard(dashboard)
    return DashboardOps(editor)


def _view(ops, name):
    return ops.editor._component_instances[ops._instance(name)]


def _without_load_turn(saved, exported):
    """``exported`` without the turn loading ``saved`` adds after the turns it kept."""
    turns = exported["turns"]
    assert turns[:-1] == saved["turns"]
    assert turns[-1]["summary"] == "Loaded a dashboard file."
    return {**exported, "turns": turns[:-1]}


def _check_views(ops):
    for name in ("chart", "points"):
        node = _view(ops, name)
        assert isinstance(node, ViewNode)
        assert node._error is None
        assert isinstance(node._object, VegaLiteView)
    assert _view(ops, "chart").debounce == 250
    assert _view(ops, "points")._object.spec["width"] == 300
    assert not [issue for issue in ops.validate() if issue.severity == "error"]


def test_saved_yaml_dashboard_loads_and_saves_unchanged(source):
    text = (FIXTURES / "saved_dashboard.yaml").read_text()
    ops = _loaded(source, dashboard_from_yaml(text))

    _check_views(ops)
    saved = yaml.safe_load(text)
    assert _without_load_turn(saved, yaml.safe_load(dashboard_to_yaml(ops.editor))) == saved
    assert saved["flowdash"] == 1


def test_saved_dashboard_model_loads_and_saves_unchanged(source):
    model = json.loads((FIXTURES / "saved_dashboard.json").read_text())
    ops = _loaded(source, model)

    _check_views(ops)
    exported = json.loads(json.dumps(ops.editor.export_dashboard(), default=str))
    assert _without_load_turn(model, exported) == model
    assert model["version"] == 4
