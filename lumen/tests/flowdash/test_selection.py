import datetime as dt

import pandas as pd
import pytest

pytest.importorskip("duckdb")
selection_module = pytest.importorskip("panel_flowdash.selection")

from panel_flowdash import FlowDash, Predicate, Selection
from panel_flowdash.agent import DashboardOps

from lumen.flowdash import (
    COMPONENTS, DATA_ID, FILTER_ID, SOURCE_ID, TABLE_ID, VIEW_ID, bind_source,
    check_view_config, pipeline_sql, selection_conditions, selection_warnings,
)
from lumen.transforms.base import Range

BRUSH = {
    "params": [{"name": "brush", "select": {"type": "interval", "encodings": ["x"]}}],
    "mark": "bar",
    "encoding": {
        "x": {"field": "category", "type": "nominal"},
        "y": {"aggregate": "sum", "field": "amount", "type": "quantitative"},
    },
}


def output(ops, name, port):
    return getattr(ops.editor.graph.get_state(ops._instance(name)), port)


def instance(ops, name):
    return ops.editor._component_instances[ops._instance(name)]


def data_queries(executed):
    return [q for q in executed if "__count" not in q and not (q.endswith(" LIMIT 1") and "subquery" not in q)]


def brushed_dashboard(source, spec=BRUSH):
    editor = FlowDash(components=COMPONENTS)
    bind_source(editor, source)
    ops = DashboardOps(editor)
    ops.add_node(SOURCE_ID, {"ref": "data"}, name="data")
    ops.add_node(TABLE_ID, {"table": "sales"}, name="sales")
    ops.connect("data.source", "sales.source")
    ops.add_node(VIEW_ID, {"spec": spec}, name="chart")
    ops.connect("sales.pipeline", "chart.pipeline")
    ops.add_node(FILTER_ID, {}, name="brushed")
    ops.connect("sales.pipeline", "brushed.pipeline")
    ops.connect("chart.selection", "brushed.selection")
    ops.add_node(DATA_ID, {}, name="rows")
    ops.connect("brushed.filtered", "rows.pipeline")
    return ops


def brush(ops, value):
    pane = instance(ops, "chart")._object._panel
    pane.selection.param.update(brush=value)


def rows(ops):
    return sorted(map(tuple, output(ops, "rows", "data").values.tolist()))


def test_brushing_the_chart_filters_in_the_source_query(source, executed):
    ops = brushed_dashboard(source)
    assert len(rows(ops)) == 3
    executed.clear()

    brush(ops, {"category": ["A"]})

    selection = output(ops, "chart", "selection")
    assert selection.predicates == (Predicate("category", "in", ("A",)),)
    assert selection.source == ops._instance("chart")
    assert rows(ops) == [("A", 5), ("A", 10)]
    statement = pipeline_sql(output(ops, "brushed", "filtered"))
    assert "WHERE \"category\" IN ('A')" in statement
    assert data_queries(executed) == [" ".join(statement.split())]


def test_the_brushed_chart_keeps_every_row(source, executed):
    ops = brushed_dashboard(source)
    brush(ops, {"category": ["A"]})
    assert len(instance(ops, "chart")._object.get_data()) == 3


def test_clearing_the_brush_restores_every_row_without_a_query(source, executed):
    ops = brushed_dashboard(source)
    brush(ops, {"category": ["A"]})
    executed.clear()

    brush(ops, {})

    assert output(ops, "chart", "selection").empty
    assert len(rows(ops)) == 3
    assert data_queries(executed) == []


def test_chart_cannot_read_the_filter_its_selection_drives(source):
    ops = brushed_dashboard(source)
    ops.add_node(VIEW_ID, {"spec": BRUSH}, name="loop")
    ops.add_node(FILTER_ID, {}, name="loop_filter")
    ops.connect("sales.pipeline", "loop_filter.pipeline")
    ops.connect("loop_filter.filtered", "loop.pipeline")
    with pytest.raises(Exception, match="cycle"):
        ops.connect("loop.selection", "loop_filter.selection")


def test_view_debounce_reaches_the_chart(source):
    ops = brushed_dashboard(source)
    pane = instance(ops, "chart")._object._panel
    assert pane.debounce == selection_module.DEBOUNCE
    ops.set_config("chart", {"spec": BRUSH, "debounce": 300})
    assert instance(ops, "chart")._object._panel.debounce == 300


def test_selection_conditions():
    selection = Selection((
        Predicate("category", "in", ("A", "B")),
        Predicate("amount", "range", (1, 5)),
        Predicate("date", "range", (dt.datetime(2025, 2, 1), dt.datetime(2025, 5, 1)), closed="left"),
    ))
    assert selection_conditions(selection) == [
        ("category", ["A", "B"]),
        ("amount", (1, 5)),
        ("date", Range(dt.datetime(2025, 2, 1), dt.datetime(2025, 5, 1), closed="left")),
    ]


def test_left_closed_ranges_are_pushed_down_half_open(source, executed):
    ops = brushed_dashboard(source)
    instance(ops, "brushed").selection = Selection((Predicate("amount", "range", (5, 20), closed="left"),))
    statement = pipeline_sql(output(ops, "brushed", "filtered"))
    assert "\"amount\" >= '5' AND \"amount\" < '20'" in statement
    assert rows(ops) == [("A", 5), ("A", 10)]


def test_a_selection_by_position_is_refused(source):
    ops = brushed_dashboard(source)
    instance(ops, "brushed").selection = Selection(index=(0,))
    failure = output(ops, "brushed", "filtered")
    assert "by their position" in str(getattr(failure, "exception", failure))


def test_a_selection_on_a_column_the_data_lacks_is_refused(source):
    ops = brushed_dashboard(source)
    instance(ops, "brushed").selection = Selection((Predicate("region", "in", ("X",)),))
    failure = output(ops, "brushed", "filtered")
    assert "no column 'region'" in str(getattr(failure, "exception", failure))


HISTOGRAM = {
    "params": [{"name": "brush", "select": "interval"}],
    "mark": "bar",
    "encoding": {
        "x": {"field": "amount", "bin": True, "type": "quantitative"},
        "y": {"aggregate": "count", "type": "quantitative"},
    },
}


@pytest.mark.parametrize("spec, warning", [
    (HISTOGRAM, "give the bin a fixed step"),
    (dict(HISTOGRAM, encoding=dict(HISTOGRAM["encoding"], x={"field": "amount", "bin": {"maxbins": 5}, "type": "quantitative"})), "fixed step"),
    ({**BRUSH, "encoding": {"x": {"field": "date", "timeUnit": "month", "type": "temporal"}}}, "repeating time unit"),
    ({**BRUSH, "params": [{"name": "pick", "select": {"type": "point"}}]}, "selects rows by position"),
])
def test_selection_warnings(spec, warning):
    (message,) = selection_warnings(spec)
    assert warning in message


@pytest.mark.parametrize("spec", [
    BRUSH,
    dict(HISTOGRAM, encoding=dict(HISTOGRAM["encoding"], x={"field": "amount", "bin": {"step": 5}, "type": "quantitative"})),
    {**HISTOGRAM, "params": []},
    {**BRUSH, "encoding": {"x": {"field": "date", "timeUnit": "yearmonth", "type": "temporal"}}},
    {**BRUSH, "params": [{"name": "pick", "select": {"type": "point", "fields": ["category"]}}]},
])
def test_no_selection_warnings(spec):
    assert selection_warnings(spec) == []


def test_view_check_warns_about_a_histogram_without_a_bin_step(source):
    ops = brushed_dashboard(source)
    pipeline = output(ops, "sales", "pipeline")
    (warning,) = check_view_config({"spec": HISTOGRAM}, {"pipeline": pipeline})
    assert type(warning).__name__ == "CheckWarning"
    assert check_view_config({"spec": BRUSH}, {"pipeline": pipeline}) is None


def test_agent_with_connect_is_told_how_to_link_selections(monkeypatch):
    pytest.importorskip("pydantic_ai")
    from lumen.agent import assistant

    given = []
    monkeypatch.setattr(
        assistant, "create_flowdash_agent",
        lambda *args, **kwargs: given.append(kwargs["instructions"]) or assistant.Agent("test"),
    )
    assistant.create_agent("test")
    assistant.create_agent("test", dashboard_tools="layout")
    assistant.create_agent("test", dashboard_tools=None)
    assert [assistant.SELECTION_INSTRUCTIONS in text for text in given] == [False, True, True]


def test_selection_frame_filter_matches_the_pushed_down_rows(source, executed):
    ops = brushed_dashboard(source)
    selection = Selection((Predicate("amount", "range", (5, 10)),))
    instance(ops, "brushed").selection = selection
    frame = pd.DataFrame(output(ops, "sales", "pipeline").data)
    assert sorted(map(tuple, selection.filter(frame).values.tolist())) == rows(ops)
