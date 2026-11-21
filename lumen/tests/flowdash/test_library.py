import pytest

pytest.importorskip("duckdb")

from panel_flowdash import FlowDash
from panel_flowdash.agent import DashboardOps
from panel_flowdash.app import FlowDashApp
from panel_flowdash.component_library import (
    ComponentLibraryError, library_components, merge_libraries,
)
from panel_flowdash.dashboard_store import DashboardStore

from lumen.flowdash import (
    COMPONENTS, DATA_ID, FILTER_ID, SOURCE_ID, TABLE_ID, bind_source,
)


def test_lumen_flowdash_is_a_component_library():
    entries = library_components("lumen.flowdash")

    assert set(entries) == set(COMPONENTS)
    for app_id, entry in entries.items():
        assert entry.load() is COMPONENTS[app_id]
        assert entry.metadata.component and not entry.metadata.page


def test_library_nodes_build_a_pushed_down_dashboard(source, executed):
    editor = FlowDash(components=merge_libraries({}, ["lumen.flowdash"]))
    bind_source(editor, source)
    ops = DashboardOps(editor)
    ops.add_node(SOURCE_ID, {"ref": "data"}, name="data")
    ops.add_node(TABLE_ID, {"table": "sales", "limit": 5}, name="sales")
    ops.connect("data.source", "sales.source")
    ops.add_node("Widgets/MultiChoice", {"default_options": ["A", "B"]}, name="pick")
    ops.add_node(FILTER_ID, {"field": "category"}, name="by_category")
    ops.connect("sales.pipeline", "by_category.pipeline")
    ops.connect("pick.selected", "by_category.value")
    ops.add_node(DATA_ID, {}, name="rows")
    ops.connect("by_category.filtered", "rows.pipeline")

    editor._component_instances[ops._instance("pick")]._widget.value = ["B"]

    data = editor.graph.get_state(ops._instance("rows")).data
    assert data.values.tolist() == [["B", 20]]
    assert executed[-1].endswith("WHERE \"category\" IN ('B') LIMIT 5")


def test_flowdash_app_offers_the_library(tmp_path):
    app = FlowDashApp(
        project_dir=str(tmp_path), registry={}, store=DashboardStore(str(tmp_path / "dashboards.db")),
        component_libraries=["lumen.flowdash"],
    )

    assert set(COMPONENTS) <= set(app._registry)
    assert set(COMPONENTS) <= set(app._editor._component_specs)


def test_library_ids_clash_with_a_project_component():
    project = {SOURCE_ID: library_components("lumen.flowdash")[TABLE_ID]}
    with pytest.raises(ComponentLibraryError, match="Lumen/source"):
        merge_libraries(project, ["lumen.flowdash"])
