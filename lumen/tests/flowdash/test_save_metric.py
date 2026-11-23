import pytest

pytest.importorskip("duckdb")
node_actions = pytest.importorskip("panel_flowdash.node_actions")
if not hasattr(node_actions, "NodeActionContext"):
    pytest.skip("panel-flowdash without node action contexts", allow_module_level=True)

import yaml

from panel_flowdash import FlowDash
from panel_flowdash.agent import DashboardOps, dashboard_to_yaml
from panel_flowdash.node_actions import NodeActionContext
from panel_flowdash.ui import tile_layout

from lumen.flowdash import (
    AGGREGATE_ID, COMPONENTS, DATA_ID, METRIC_ID, METRICS_KEY, SQL_ID,
    TABLE_ID, bind_source,
)
from lumen.flowdash.metrics import MetricSet, ProjectMetrics
from lumen.flowdash.project import ProjectError, load_project
from lumen.flowdash.save_metric import (
    ACTION_KEY, SaveMetricDialog, agree, check_definition, derive_draft,
    describe_aggregate, describe_query, point_at_metric, preview_metric,
    save_metric, scrub, structured_form,
)
from lumen.sources.duckdb import DuckDBSource

# Each turbine row repeats its project's capacity, so summing p_cap over the rows counts a project once per turbine.
TURBINES = [
    ("Alta", "CA", 3.0, 1.0), ("Alta", "CA", 3.0, 1.0), ("Alta", "CA", 3.0, 1.0),
    ("Roscoe", "TX", 2.0, 1.0), ("Roscoe", "TX", 2.0, 1.0),
    ("Story", "IA", 1.0, 1.0),
]
INFLATED, CAPACITY = 14.0, 6.0

CAPACITY_SQL = (
    "SELECT SUM(p_cap) AS capacity FROM (SELECT p_name, MAX(p_cap) AS p_cap FROM turbines GROUP BY p_name) AS p"
)


def turbines_sql():
    rows = ", ".join(f"('{n}', '{s}', {p}, {t})" for n, s, p, t in TURBINES)
    return f"SELECT * FROM (VALUES {rows}) AS t(p_name, t_state, p_cap, t_cap)"


@pytest.fixture
def duck():
    return DuckDBSource(uri=":memory:", tables={"turbines": turbines_sql()})


@pytest.fixture
def admin():
    return {"value": True}


@pytest.fixture
def project_file(tmp_path):
    path = tmp_path / "flowdash.yaml"
    path.write_text("# The app's project file.\nmetrics: {}\n")
    return path


@pytest.fixture
def ops(duck, project_file, admin):
    editor = FlowDash(components=COMPONENTS, notifications=False, authorize_admin=lambda fresh=False: admin["value"])
    ref = bind_source(editor, duck)
    project = load_project(project_file, editor.context)
    editor.context[METRICS_KEY] = ProjectMetrics(project)
    editor.project = project
    editor.new_dashboard("Wind")
    ops = DashboardOps(editor)
    ops.ref = ref
    yield ops
    editor.dispose()


def ctx_of(ops, name=None):
    """The node action context of node ``name``, or of a source node placed for actions on no node in particular."""
    if name is None:
        name = "any_source"
        if name not in [n["name"] for n in ops.nodes()]:
            ops.add_node("Lumen/source", {"ref": ops.ref}, name=name)
    return NodeActionContext(ops.editor, ops._instance(name))


def output(ops, name, port):
    return getattr(ops.editor.graph.get_state(ops._instance(name)), port)


def ad_hoc_kpi(ops):
    """A KPI showing the sum of p_cap over all turbine rows, wired as add_kpi wires one."""
    ops.add_node(TABLE_ID, {"ref": ops.ref, "table": "turbines"}, name="turbines")
    ops.add_node(AGGREGATE_ID, {"columns": ["p_cap"], "method": "sum"}, name="total_capacity_agg")
    ops.connect("turbines.pipeline", "total_capacity_agg.pipeline")
    ops.add_node(DATA_ID, {}, name="total_capacity_value")
    ops.connect("total_capacity_agg.result", "total_capacity_value.pipeline")
    ops.add_node("Widgets/KPI", {"label": "Total capacity", "format": ",.0f"}, name="total_capacity")
    ops.connect("total_capacity_value.data", "total_capacity.value",
                [{"op": "column", "column": "p_cap"}, {"op": "aggregate", "how": "last"}])
    ops.place_tile("total_capacity", area="top")
    return ops._instance("total_capacity")


# Labels

def test_an_ad_hoc_query_says_how_it_computes_its_number():
    assert describe_query("SELECT SUM(p_cap) AS total FROM turbines") == (
        "Sum of p_cap over all rows of turbines, computed by its own SQL query; no metric was used.")
    assert describe_query("SELECT t_state, AVG(t_cap) FROM turbines GROUP BY t_state").startswith(
        "Average of t_cap by t_state of turbines")
    assert "of the rows it filters" in describe_query("SELECT COUNT(*) FROM turbines WHERE t_cap > 1")
    assert describe_query("SELECT * FROM turbines").startswith("Rows of turbines")
    assert describe_query("not sql at all (").startswith("Computed by its own SQL query")
    assert describe_aggregate("sum", ["p_cap"], []) == "Sum of p_cap over all rows of its input; no metric was used."
    assert describe_aggregate("count", [], ["t_state"]).startswith("Count of rows by t_state")


def test_views_read_ad_hoc_or_metric_labels_from_the_nodes_they_read(ops):
    kpi = ad_hoc_kpi(ops)
    provenance, by = ops.editor.node_provenance(kpi)
    assert provenance.label == "Ad hoc" and provenance.kind == "ad hoc"
    assert provenance.detail.startswith("Sum of p_cap over all rows")
    assert by == ops._instance("total_capacity_agg")
    ops.add_node(SQL_ID, {"sql": "SELECT SUM(t_cap) AS turbines_mw FROM turbines"}, name="sql_total")
    ops.add_node("Lumen/source", {"ref": ops.ref}, name="src")
    ops.connect("src.source", "sql_total.source")
    provenance, _ = ops.editor.node_provenance(ops._instance("sql_total"))
    assert provenance.detail.startswith("Sum of t_cap over all rows of turbines")


# Deriving a definition

def test_a_plain_aggregate_query_maps_to_measures_and_dimensions():
    shape, why = structured_form("SELECT t_state AS state, SUM(t_cap) AS mw, COUNT(*) AS n FROM turbines GROUP BY t_state")
    assert why is None
    assert shape == {"table": "turbines", "measures": {"mw": "SUM(t_cap)", "n": "COUNT(*)"},
                     "dimensions": {"state": {"column": "t_state"}}}


@pytest.mark.parametrize(("sql", "why"), [
    ("SELECT SUM(t_cap) AS mw FROM turbines WHERE t_state = 'TX'", "filters rows with WHERE"),
    ("SELECT SUM(t_cap) FROM turbines", "has no plain name"),
    ("SELECT SUM(p_cap) AS c FROM (SELECT DISTINCT p_name, p_cap FROM turbines) AS p", "reads a subquery"),
    ("SELECT t_state, SUM(t_cap) AS mw FROM turbines GROUP BY t_state ORDER BY mw", "sorts its rows"),
    ("SELECT a.t_state, SUM(a.t_cap) AS mw FROM turbines a JOIN turbines b ON a.p_name = b.p_name GROUP BY a.t_state",
     "joins tables"),
    ("SELECT p_name, t_cap FROM turbines", "neither a grouped column nor an aggregate"),
])
def test_other_queries_do_not_map_and_say_why(sql, why):
    shape, reason = structured_form(sql)
    assert shape is None and why in reason


def test_a_kpi_over_an_aggregate_derives_a_structured_metric(ops):
    draft = derive_draft(ctx_of(ops, ad_hoc_kpi(ops)))
    assert draft.name == "total_capacity"
    assert draft.structured and draft.notes == []
    assert draft.definition == {"source": ops.ref, "table": "turbines", "measures": {"p_cap": 'SUM("p_cap")'}}
    assert draft.measure == "p_cap"
    assert draft.node == ops._instance("total_capacity_agg")
    assert draft.current == INFLATED


def test_a_sql_node_with_a_filter_derives_a_sql_metric_and_says_what_to_adjust(ops):
    ops.add_node("Lumen/source", {"ref": ops.ref}, name="src")
    ops.add_node(SQL_ID, {"sql": "SELECT SUM(t_cap) AS mw FROM turbines WHERE t_state = 'TX'"}, name="texas")
    ops.connect("src.source", "texas.source")
    draft = derive_draft(ctx_of(ops, ops._instance("texas")))
    assert not draft.structured
    assert draft.definition == {"source": ops.ref, "sql": "SELECT SUM(t_cap) AS mw FROM turbines WHERE t_state = 'TX'"}
    assert "filters rows with WHERE" in draft.notes[0]
    assert draft.current == 2.0


def test_nothing_is_derived_from_a_node_without_a_query(ops):
    ops.add_node(TABLE_ID, {"ref": ops.ref, "table": "turbines"}, name="turbines")
    with pytest.raises(ValueError, match="do not come from a Lumen SQL or aggregate node"):
        derive_draft(ctx_of(ops, ops._instance("turbines")))


# Previewing

def test_the_preview_of_the_derived_metric_agrees_with_the_node(ops):
    draft = derive_draft(ctx_of(ops, ad_hoc_kpi(ops)))
    preview = preview_metric(draft.name, draft.definition, ops.editor.context, draft.measure)
    assert preview.ok and preview.value == INFLATED and agree(draft.current, preview.value)
    assert 'SUM("p_cap")' in preview.sql and "turbines" in preview.sql


def test_the_preview_of_a_corrected_metric_shows_the_right_number(ops):
    preview = preview_metric("capacity", {"source": ops.ref, "sql": CAPACITY_SQL}, ops.editor.context, "p_cap")
    assert preview.ok and preview.column == "capacity" and preview.value == CAPACITY
    assert agree(INFLATED, preview.value) is False


def test_the_preview_reports_problems_without_raising(ops):
    preview = preview_metric("capacity", {"source": ops.ref, "table": "turbines", "measures": {"c": "SUM(nope)"}},
                             ops.editor.context)
    assert not preview.ok and "Unknown column 'nope'" in preview.problems[0]
    preview = preview_metric("drop; table", {"source": ops.ref, "sql": "SELECT 1"}, ops.editor.context)
    assert not preview.ok and "letters, digits and underscores" in preview.problems[0]
    preview = preview_metric("capacity", {"source": ops.ref, "sql": "DROP TABLE turbines"}, ops.editor.context)
    assert not preview.ok


def test_previews_hide_the_credentials_of_a_connection():
    assert scrub("could not connect to postgresql://admin:hunter2@db/prod") == (
        "could not connect to postgresql://***@db/prod")


# Saving

def test_saving_writes_the_metric_and_every_metric_node_sees_it(ops, project_file):
    shared = ops.editor.context[METRICS_KEY]
    save_metric(ctx_of(ops), "capacity", {"description": "Capacity, one p_cap per project.", "source": ops.ref,
                                         "sql": CAPACITY_SQL})
    written = yaml.safe_load(project_file.read_text())
    assert written["metrics"]["capacity"]["sql"] == CAPACITY_SQL
    assert "capacity" in shared and shared["capacity"].description == "Capacity, one p_cap per project."
    ops.add_node(METRIC_ID, {"metric": "capacity"}, name="cap")
    assert output(ops, "cap", "data")["capacity"].tolist() == [CAPACITY]


def test_saving_refuses_non_admins_and_leaves_the_file_alone(ops, project_file, admin):
    before = project_file.read_text()
    admin["value"] = False
    with pytest.raises(PermissionError, match="Only an admin"):
        save_metric(ctx_of(ops), "capacity", {"source": ops.ref, "sql": CAPACITY_SQL})
    assert project_file.read_text() == before


def test_saving_fails_closed_on_a_bad_definition_or_a_taken_name(ops, project_file):
    before = project_file.read_text()
    with pytest.raises(ProjectError, match="Unknown column"):
        save_metric(ctx_of(ops), "capacity", {"source": ops.ref, "table": "turbines", "measures": {"c": "SUM(x)"}})
    with pytest.raises(ProjectError, match="letters, digits and underscores"):
        save_metric(ctx_of(ops), "<script>", {"source": ops.ref, "sql": CAPACITY_SQL})
    assert project_file.read_text() == before
    save_metric(ctx_of(ops), "capacity", {"source": ops.ref, "sql": CAPACITY_SQL})
    with pytest.raises(ValueError, match="named 'capacity' exists already"):
        save_metric(ctx_of(ops), "capacity", {"source": ops.ref, "sql": "SELECT COUNT(*) AS n FROM turbines"})


def test_check_definition_reads_the_metric_as_the_project_file_would(ops):
    metrics = check_definition("capacity", {"source": ops.ref, "sql": CAPACITY_SQL}, ops.editor.context)
    assert metrics["capacity"].form == "sql"
    with pytest.raises(ProjectError, match="serves no source 'elsewhere'"):
        check_definition("capacity", {"source": "elsewhere", "sql": CAPACITY_SQL}, ops.editor.context)


# The action

def test_only_admins_are_offered_the_action_and_only_on_nodes_with_a_query(ops, admin):
    editor = ops.editor
    kpi = ad_hoc_kpi(ops)
    assert ACTION_KEY in [a.key for a in editor.node_actions(kpi)]
    assert ACTION_KEY in [a.key for a in editor.node_actions(ops._instance("total_capacity_agg"))]
    assert ACTION_KEY not in [a.key for a in editor.node_actions(ops._instance("turbines"))]
    admin["value"] = False
    assert ACTION_KEY not in [a.key for a in editor.node_actions(kpi)]
    with pytest.raises(PermissionError):
        editor.run_node_action(kpi, ACTION_KEY)


def test_the_action_is_not_offered_without_a_project_file(ops):
    kpi = ad_hoc_kpi(ops)
    ops.editor.project = None
    ops.editor.context[METRICS_KEY] = MetricSet()
    assert ACTION_KEY not in [a.key for a in ops.editor.node_actions(kpi)]


# Pointing the node at the metric

def test_pointing_a_kpi_at_the_metric_keeps_its_slot_and_undo_restores_it(ops):
    editor = ops.editor
    kpi = ad_hoc_kpi(ops)
    draft = derive_draft(ctx_of(ops, kpi))
    save_metric(editor, "capacity", {"source": ops.ref, "sql": CAPACITY_SQL})
    assert editor.save()
    before = yaml.safe_load(dashboard_to_yaml(editor))
    name = point_at_metric(ctx_of(ops, draft.target), draft, "capacity", "capacity")
    assert name == "capacity_metric"
    assert output(ops, "total_capacity", "value") == CAPACITY
    names = {n["name"] for n in ops.nodes()}
    assert {"total_capacity_agg", "total_capacity_value"}.isdisjoint(names)
    assert {"turbines", "total_capacity", "capacity_metric"} <= names
    assert [i["instance_id"] for i in tile_layout.area_items(editor, tile_layout.TOP)] == [kpi]
    provenance, _ = editor.node_provenance(kpi)
    assert provenance.label == "Metric: capacity"

    assert editor.undo()
    names = {n["name"] for n in ops.nodes()}
    assert {"total_capacity_agg", "total_capacity_value"} <= names and "capacity_metric" not in names
    assert output(ops, "total_capacity", "value") == INFLATED
    assert editor.node_provenance(kpi)[0].label == "Ad hoc"
    after = yaml.safe_load(dashboard_to_yaml(editor))
    assert after["nodes"] == before["nodes"] and after.get("edges") == before.get("edges")


def test_pointing_a_sql_node_at_the_metric_gives_the_metric_node_its_tile(ops):
    editor = ops.editor
    ops.add_node("Lumen/source", {"ref": ops.ref}, name="src")
    ops.add_node(SQL_ID, {"sql": "SELECT t_state, SUM(t_cap) AS mw FROM turbines GROUP BY t_state"}, name="by_state")
    ops.connect("src.source", "by_state.source")
    ops.add_node("Lumen/view", {"spec": {"mark": "bar", "encoding": {
        "x": {"field": "t_state", "type": "nominal"}, "y": {"field": "mw", "type": "quantitative"}}}}, name="chart")
    ops.connect("by_state.pipeline", "chart.pipeline")
    ops.place_tile("by_state", visible=True, index=0, width=50)
    draft = derive_draft(ctx_of(ops, ops._instance("by_state")))
    assert draft.structured and draft.dimensions == ["t_state"]
    save_metric(editor, "state_mw", draft.definition)
    assert editor.save()
    slot_before = [i["instance_id"] for i in tile_layout.area_items(editor, tile_layout.GRID)].index(
        ops._instance("by_state"))
    name = point_at_metric(ctx_of(ops, draft.target), draft, "state_mw", draft.measure)
    grid = [i["instance_id"] for i in tile_layout.area_items(editor, tile_layout.GRID)]
    assert grid.index(ops._instance(name)) == slot_before
    assert tile_layout.tile_entry(editor, ops._instance(name)).get("width") == 50
    assert "by_state" not in {n["name"] for n in ops.nodes()}
    assert sorted(output(ops, name, "data")["mw"].tolist()) == [1.0, 2.0, 3.0]
    assert editor.node_provenance(ops._instance("chart"))[0].label == "Metric: state_mw"
    assert editor.undo()
    assert "by_state" in {n["name"] for n in ops.nodes()} and name not in {n["name"] for n in ops.nodes()}


def test_a_node_reading_the_sql_node_as_a_source_is_not_rewired(ops):
    ops.add_node("Lumen/source", {"ref": ops.ref}, name="src")
    ops.add_node(SQL_ID, {"sql": "SELECT p_name, MAX(p_cap) AS p_cap FROM turbines GROUP BY p_name"}, name="projects")
    ops.connect("src.source", "projects.source")
    ops.add_node(SQL_ID, {"sql": "SELECT SUM(p_cap) AS capacity FROM projects"}, name="later")
    ops.connect("projects.result", "later.source")
    draft = derive_draft(ctx_of(ops, ops._instance("projects")))
    save_metric(ctx_of(ops), "projects_cap", draft.definition)
    with pytest.raises(ValueError, match="reads the SQL node as a source"):
        point_at_metric(ctx_of(ops, draft.target), draft, "projects_cap")
    assert "projects" in {n["name"] for n in ops.nodes()}


# The dialog

def test_the_dialog_previews_saves_and_points_the_node(ops, project_file):
    kpi = ad_hoc_kpi(ops)
    dialog = SaveMetricDialog(ctx_of(ops, kpi), derive_draft(ctx_of(ops, kpi)))
    assert dialog.preview.value == INFLATED and dialog._verdict.object == "They agree."
    assert not dialog._save.disabled
    dialog.metric_name = "capacity"
    dialog.description = "Capacity <b>once</b> per project."
    dialog.definition = yaml.safe_dump({"source": ops.ref, "sql": CAPACITY_SQL})
    assert dialog._save.disabled
    preview = dialog.run_preview()
    assert preview.value == CAPACITY and dialog._verdict.object.startswith("They differ")
    assert dialog.save()
    assert yaml.safe_load(project_file.read_text())["metrics"]["capacity"]["description"] == (
        "Capacity <b>once</b> per project.")
    assert dialog._point.visible
    assert dialog.point() == "capacity_metric"
    assert output(ops, "total_capacity", "value") == CAPACITY


def test_the_dialog_refuses_to_save_for_someone_no_longer_admin(ops, project_file, admin):
    dialog = SaveMetricDialog(ctx_of(ops, ad_hoc_kpi(ops)), derive_draft(ctx_of(ops, "total_capacity")))
    admin["value"] = False
    before = project_file.read_text()
    assert not dialog.save()
    assert "Only an admin" in dialog._message.object
    assert project_file.read_text() == before


def test_the_dialog_shows_a_hostile_name_and_errors_as_text(ops, project_file):
    dialog = SaveMetricDialog(ctx_of(ops, ad_hoc_kpi(ops)), derive_draft(ctx_of(ops, "total_capacity")))
    before = project_file.read_text()
    dialog.metric_name = "<img src=x onerror=alert(1)>"
    assert not dialog.run_preview().ok
    assert "<img" not in dialog._message.object and "letters, digits and underscores" in dialog._message.object
    assert not dialog.save()
    assert project_file.read_text() == before


def test_saves_take_the_project_files_lock_and_keep_other_writers_metrics(ops, project_file):
    import threading

    from lumen.flowdash import metrics_admin

    class Admin:
        _identity = None

        def _still_admin(self):
            return True

    other = load_project(project_file, ops.editor.context)
    assert metrics_admin.save_metric(other, "turbine_count", {"source": ops.ref, "sql": "SELECT COUNT(*) AS n FROM turbines"},
                                     app=Admin()) == []
    save_metric(ctx_of(ops), "capacity", {"source": ops.ref, "sql": CAPACITY_SQL})
    assert set(yaml.safe_load(project_file.read_text())["metrics"]) == {"turbine_count", "capacity"}

    errors = []

    def save(i):
        try:
            save_metric(ctx_of(ops), f"count_{i}", {"source": ops.ref, "sql": f"SELECT COUNT(*) + {i} AS n FROM turbines"})
        except Exception as exc:
            errors.append(exc)

    ctx_of(ops)
    threads = [threading.Thread(target=save, args=(i,)) for i in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert errors == []
    saved = yaml.safe_load(project_file.read_text())["metrics"]
    assert {f"count_{i}" for i in range(8)} | {"turbine_count", "capacity"} == set(saved)


def test_saved_sql_is_written_as_a_block(ops, project_file):
    save_metric(ctx_of(ops), "capacity", {"source": ops.ref, "sql": "SELECT SUM(p_cap) AS capacity\nFROM turbines"})
    text = project_file.read_text()
    assert "sql: |" in text and "\\n" not in text


def test_only_single_numbers_are_labelled_ad_hoc(ops):
    ops.add_node("Lumen/source", {"ref": ops.ref}, name="src")
    ops.add_node(SQL_ID, {"sql": "SELECT t_state, SUM(t_cap) AS mw FROM turbines GROUP BY t_state"}, name="by_state")
    ops.connect("src.source", "by_state.source")
    ops.add_node("Lumen/view", {"spec": {"mark": "bar", "encoding": {
        "x": {"field": "t_state", "type": "nominal"}, "y": {"field": "mw", "type": "quantitative"}}}}, name="chart")
    ops.connect("by_state.pipeline", "chart.pipeline")
    assert ops.editor.node_provenance(ops._instance("chart")) is None
    ops.add_node(TABLE_ID, {"ref": ops.ref, "table": "turbines"}, name="rows")
    ops.add_node(AGGREGATE_ID, {"columns": ["p_cap"], "method": "sum", "by": ["t_state"]}, name="per_state")
    ops.connect("rows.pipeline", "per_state.pipeline")
    assert ops.editor.node_provenance(ops._instance("per_state")) is None


def test_the_action_is_not_offered_in_view_mode(ops):
    kpi = ad_hoc_kpi(ops)
    ops.editor.editable = False
    assert ACTION_KEY not in [a.key for a in ops.editor.node_actions(kpi)]
    with pytest.raises(PermissionError):
        ops.editor.run_node_action(kpi, ACTION_KEY)


def test_pointing_says_undo_keeps_the_metric(ops):
    kpi = ad_hoc_kpi(ops)
    ctx = ctx_of(ops, kpi)
    notes = []
    ctx.notify = lambda kind, message: notes.append(message)
    dialog = SaveMetricDialog(ctx, derive_draft(ctx))
    dialog.metric_name = "capacity"
    dialog.definition = yaml.safe_dump({"source": ops.ref, "sql": CAPACITY_SQL})
    dialog.run_preview()
    assert dialog.save() and dialog.point() == "capacity_metric"
    assert notes == ["total_capacity now reads the metric capacity. Undo points it back as before; "
                     "the metric capacity stays saved in Admin, Metrics."]
    assert ops.editor.undo()
    assert "capacity" in ops.editor.context[METRICS_KEY]
