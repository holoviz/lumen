"""The Lumen table view node: a Tabulator of a pipeline's rows, configured by a spec."""
import asyncio

import pandas as pd
import panel as pn
import pytest

pytest.importorskip("duckdb")
spec_settings = pytest.importorskip("panel_flowdash.spec_settings")

from panel_flowdash import FlowDash
from panel_flowdash.agent import DashboardOps, OperationError
from panel_flowdash.ui.node_preview import hook_preview
from panel_flowdash.ui.spec_field import SpecEditor
from panel_flowdash.ui.status_bar import select_on_canvas

from lumen.data.contracts import FULL
from lumen.flowdash import (
    COMPONENTS, PIPELINE_TYPE, SOURCE_ID, TABLE_ID, TABLE_VIEW_ID,
    CheckDiagnostic, TableViewNode, bind_source,
)
from lumen.flowdash.contracts import usages
from lumen.flowdash.table_view import (
    TABLE_SETTINGS, check_table, table_fields, tabulator_options,
)
from lumen.views.base import Table

COLUMNS = ["category", "amount"]


def dashboard(source, spec=None, **editor_params):
    editor = FlowDash(components=COMPONENTS, notifications=False, **editor_params)
    bind_source(editor, source)
    ops = DashboardOps(editor)
    ops.add_node(SOURCE_ID, {"ref": "data"}, name="data")
    ops.add_node(TABLE_ID, {"table": "sales"}, name="sales")
    ops.connect("data.source", "sales.source")
    ops.add_node(TABLE_VIEW_ID, {} if spec is None else {"spec": spec}, name="rows")
    ops.connect("sales.pipeline", "rows.pipeline")
    return ops


def node(ops, name="rows"):
    return ops.editor._component_instances[ops._instance(name)]


def test_the_table_view_is_a_registered_view_that_needs_no_spec():
    meta = TableViewNode.__panel_app_metadata__
    assert COMPONENTS[TABLE_VIEW_ID] is TableViewNode
    assert meta.title == "Lumen Tabulator view" and "page_size" not in meta.description
    assert "page_size" in TableViewNode.param.spec.doc
    assert meta.requires == [{"key": "pipeline", "type": PIPELINE_TYPE, "required": True}]
    assert meta.config == ["spec"]
    assert "table" in meta.tags
    assert meta.default_size == {"height": 400} and meta.min_size == {"height": 200}
    assert TableViewNode.requires_spec is False and TableViewNode.view_type == "table"
    assert spec_settings.declared_settings(TableViewNode.param.spec) == TABLE_SETTINGS


def test_an_empty_spec_draws_every_row_without_the_index_filling_the_tile(source):
    ops = dashboard(source)
    rows = node(ops)
    assert rows._error is None and isinstance(rows._object, Table)
    table = rows._object.panel
    assert isinstance(table, pn.widgets.Tabulator)
    assert len(table.value) == 3 and list(table.value.columns) == COLUMNS
    assert table.show_index is False and table.layout == "fit_data_fill"
    assert table.sizing_mode == "stretch_both" and table.disabled
    assert table.pagination is None
    assert not [issue for issue in ops.validate() if issue.severity == "error"]


def test_spec_options_become_tabulator_options(source):
    spec = {
        "page_size": 2, "show_index": True, "layout": "fit_columns", "header_filters": True,
        "frozen_columns": ["category"], "hidden_columns": ["amount"],
        "sorters": [{"field": "amount", "dir": "desc"}],
        "formatters": {"amount": {"type": "progress", "max": 20}},
        "titles": {"category": "Category"}, "widths": {"category": 120},
    }
    table = node(dashboard(source, spec))._object.panel
    assert table.pagination == "remote" and table.page_size == 2
    assert table.show_index and table.layout == "fit_columns" and table.header_filters
    assert table.frozen_columns == ["category"] and table.hidden_columns == ["amount"]
    assert table.sorters == [{"field": "amount", "dir": "desc"}]
    assert table.formatters == {"amount": {"type": "progress", "max": 20}}
    assert table.titles == {"category": "Category"} and table.widths == {"category": 120}


@pytest.mark.parametrize("page_size", [None, 2])
def test_a_sorter_without_a_direction_sorts_ascending(source, page_size):
    spec = {"sorters": [{"field": "amount"}], **({"page_size": page_size} if page_size else {})}
    rows = node(dashboard(source, spec))
    assert rows._error is None
    table = rows._object.panel
    assert table.sorters == [{"dir": "asc", "field": "amount"}]
    assert list(table.current_view["amount"]) == [5, 10, 20]
    assert rows.spec == spec


def test_tabulator_options_do_not_change_the_spec():
    spec = {"page_size": None, "formatters": {"amount": {"type": "money"}}}
    options = tabulator_options(spec)
    assert "page_size" not in options and "pagination" not in options
    options["formatters"]["amount"]["symbol"] = "$"
    assert spec == {"page_size": None, "formatters": {"amount": {"type": "money"}}}


@pytest.mark.parametrize("spec, message", [
    ({"colour": "red"}, "Unknown table option"),
    ({"page_size": 0}, "page_size"),
    ({"page_size": True}, "page_size"),
    ({"show_index": "yes"}, "show_index"),
    ({"layout": "fit_everything"}, "layout"),
    ({"frozen_columns": "category"}, "list of column names"),
    ({"hidden_columns": ["missing"]}, "No column(s) 'missing'"),
    ({"sorters": [{"field": "amount", "dir": "down"}]}, "sorters"),
    ({"sorters": [{"field": "nope"}]}, "No column(s) 'nope'"),
    ({"formatters": {"amount": "html"}}, "formatter of 'amount'"),
    ({"formatters": {"amount": {"type": "link", "urlPrefix": "javascript:"}}}, "formatter of 'amount'"),
    ({"formatters": {"amount": {"type": "money", "symbol": "<img src=x>"}}}, "without HTML"),
    ({"formatters": {"amount": {"type": "progress", "legend": {"x": 1}}}}, "without HTML"),
    ({"titles": {"amount": "<b>Amount</b>"}}, "without HTML"),
    ({"widths": {"amount": "wide"}}, "width of 'amount'"),
])
def test_check_table_refuses_what_it_cannot_draw(spec, message):
    with pytest.raises(ValueError, match=message.replace("(", r"\(").replace(")", r"\)")):
        check_table(spec, COLUMNS)


def test_check_table_accepts_options_naming_the_columns():
    spec = {"frozen_columns": ["category"], "formatters": {"amount": "money"}, "sorters": [{"field": "amount"}]}
    assert check_table(spec, COLUMNS) == spec
    assert check_table({"hidden_columns": ["anything"]}, []) == {"hidden_columns": ["anything"]}
    assert check_table(None, COLUMNS) is None


def test_a_bad_spec_is_refused_before_the_table_changes(source):
    ops = dashboard(source, {"page_size": 2})
    drawn = node(ops)._object
    with pytest.raises(OperationError, match="No column"):
        ops.set_config("rows", {"spec": {"frozen_columns": ["nope"]}})
    assert node(ops)._object is drawn and ops.node_config("rows")["spec"] == {"page_size": 2}

    check = TableViewNode.__panel_app_metadata__.check_config
    pipeline = node(ops, "sales")._object
    found = check({"spec": {"hidden_columns": ["nope"]}}, {"pipeline": pipeline})
    assert isinstance(found, CheckDiagnostic) and found.field == "spec"
    assert check({"spec": {}}, {"pipeline": pipeline}) is None


def test_the_card_previews_the_rows_as_the_table_shows_them(source):
    ops = dashboard(source, {"hidden_columns": ["amount"], "titles": {"category": "Category"}})
    rows = node(ops)
    preview = rows.__flowdash_preview__()
    assert isinstance(preview, pd.DataFrame)
    assert list(preview.columns) == ["Category"] and len(preview) == 3
    assert hook_preview(rows) is not None
    ops.editor._open_card_popout(ops._instance("rows"))


def test_the_summary_counts_rows_and_shown_columns(source):
    rows = node(dashboard(source, {"hidden_columns": ["amount"]}))
    data = rows._object.pipeline.data
    assert rows._summary(rows.spec, data) == "Table, 3 rows, 1 column"
    assert rows._summary({}, data.head(1)) == "Table, 1 row, 2 columns"
    assert rows._summary({}, None) == "Table"


def test_without_a_pipeline_the_table_waits(source):
    ops = dashboard(source)
    ops.disconnect("sales.pipeline", "rows.pipeline")
    rows = node(ops)
    assert rows._object is None and rows.__flowdash_preview__() is None
    assert "Waiting for a pipeline" in rows._body().object


@pytest.mark.parametrize("key, value", [
    ("page_size", 25), ("layout", "fit_columns"), ("frozen_columns", "category"),
    ("show_index", True), ("header_filters", True),
])
def test_each_setting_round_trips_and_keeps_the_rest(key, value):
    settings = {setting.key: setting for setting in TABLE_SETTINGS}
    spec = {"formatters": {"amount": "money"}}
    written = settings[key].write(spec, value)
    assert settings[key].read(written) == value
    assert written["formatters"] == {"amount": "money"}
    assert spec_settings.unknown_paths(written, TABLE_SETTINGS) == [("formatters",)]


def test_a_list_of_frozen_columns_is_left_to_the_spec():
    settings = {setting.key: setting for setting in TABLE_SETTINGS}
    spec = {"frozen_columns": ["category", "amount"]}
    assert spec_settings.unrepresented_paths(spec, TABLE_SETTINGS) == [("frozen_columns",)]
    assert settings["frozen_columns"].write(spec, None) == {}


async def test_the_inspector_edits_the_spec_through_the_form(source):
    ops = dashboard(source, history_coalesce=0)
    editor = ops.editor
    select_on_canvas(editor._flow, node_id=ops._instance("rows"))
    (field,) = pn.Column(*editor._inspector._body).select(SpecEditor)
    await editor._inspector.settle()
    for _ in range(50):
        if field.choices.get("frozen_columns"):
            break
        await asyncio.sleep(0.02)
    assert field.choices["frozen_columns"].values() == COLUMNS
    field._controls["page_size"][1].value = 2
    await editor._inspector.settle()
    assert ops.node_config("rows")["spec"] == {"page_size": 2}
    assert node(ops)._object.panel.page_size == 2
    assert editor.undo()
    assert ops.node_config("rows").get("spec") in ({}, None)


def test_the_schema_contract_records_the_columns_the_spec_names():
    spec = {"frozen_columns": ["category"], "formatters": {"amount": "money"}, "titles": {"category": "C"}}
    assert table_fields(spec) == {"category", "amount"}
    assert table_fields({}) == set() and table_fields(None) == set()
    doc = {
        "flowdash": 4,
        "nodes": {
            "data": {"component": SOURCE_ID, "config": {"ref": "data"}},
            "sales": {"component": TABLE_ID, "config": {"table": "sales"}},
            "rows": {"component": TABLE_VIEW_ID, "config": {"spec": {"hidden_columns": ["amount"]}}},
            "all": {"component": TABLE_VIEW_ID, "config": {}},
        },
        "edges": [
            {"from": "data.source", "to": "sales.source"},
            {"from": "sales.pipeline", "to": "rows.pipeline"},
            {"from": "sales.pipeline", "to": "all.pipeline"},
        ],
    }
    schema = {"sales": {"category": "string", "amount": "integer"}}
    found = {u.node: (u.coverage, {t: sorted(c) for t, c in u.tables.items()}) for u in usages(doc, "data", schema, "duckdb")}
    assert found["rows"] == (FULL, {"sales": ["amount"]})
    assert found["all"] == (FULL, {})
