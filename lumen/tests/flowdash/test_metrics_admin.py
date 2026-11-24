"""The admin view's Metrics section and the setup wizard's Key numbers step, without a browser."""
import pytest

pytest.importorskip("duckdb")
pytest.importorskip("panel_flowdash.admin.wizard")

from panel_flowdash import MemoryDashboardStore
from panel_flowdash.admin.form import using_app
from panel_flowdash.dashboard_store import DashboardItem

from lumen.flowdash import METRICS_KEY, SOURCES_KEY
from lumen.flowdash.app import build_routes
from lumen.flowdash.contracts import METRIC_ID
from lumen.flowdash.metrics import ProjectMetrics, read_metrics
from lumen.flowdash.metrics_admin import (
    NAME_KEY, KeyNumbersStep, MetricsSettings, check_metric, delete_metric,
    join_spec, metric_usage, preview_metric, raw_section, save_metric,
    split_spec,
)
from lumen.flowdash.project import PROJECT_FILE, load_project
from lumen.sources.duckdb import DuckDBSource

# Two projects of two and three turbines: p_cap, the project's capacity in MW, repeats on each turbine row.
ROWS = [
    ("Alta", "CA", 2010, 6.0, 3000),
    ("Alta", "CA", 2010, 6.0, 3000),
    ("Roscoe", "TX", 2012, 4.5, 1500),
    ("Roscoe", "TX", 2012, 4.5, 1500),
    ("Roscoe", "TX", 2012, 4.5, 1500),
]

CAPACITY = {
    "description": "Installed capacity, each project's p_cap counted once.",
    "source": "turbines",
    "sql": "SELECT SUM(p_cap) AS capacity_mw, COUNT(*) AS projects "
           "FROM (SELECT p_name, MAX(p_cap) AS p_cap FROM turbines GROUP BY p_name)",
}

FLEET = {
    "source": "turbines",
    "table": "turbines",
    "measures": {"turbines": "COUNT(*)", "turbine_mw": "SUM(t_cap) / 1000.0"},
    "dimensions": {"state": "t_state"},
}

YAML = """\
general:
  admins: [ana]
metrics:
  fleet:
    source: turbines
    table: turbines
    measures:
      turbines: COUNT(*)
      turbine_mw: SUM(t_cap) / 1000.0
    dimensions:
      state: t_state
"""


class Admin:
    def __init__(self, admin=True, store=None):
        self.admin = admin
        self.store = store
        self.notified = []

    def _still_admin(self):
        return self.admin

    def _notify(self, severity, message, duration=3000):
        self.notified.append((severity, message))


@pytest.fixture
def source(tmp_path):
    data = tmp_path / "turbines.csv"
    data.write_text("p_name,t_state,p_year,p_cap,t_cap\n" + "".join(f"{n},{s},{y},{p},{c}\n" for n, s, y, p, c in ROWS))
    return DuckDBSource(uri=":memory:", tables={"turbines": f"SELECT * FROM read_csv_auto('{data}')"})


@pytest.fixture
def project(tmp_path, source):
    path = tmp_path / PROJECT_FILE
    path.write_text(YAML)
    return load_project(path, {SOURCES_KEY: {"turbines": source}})


def test_a_valid_metric_has_no_problems(project):
    assert check_metric(project, "capacity", CAPACITY) == []


@pytest.mark.parametrize(("name", "spec", "keys", "message"), [
    ("bad name", CAPACITY, ("bad name", NAME_KEY), "letters, digits and underscores"),
    ("fleet", CAPACITY, ("fleet", NAME_KEY), "exists already"),
    ("m", {**FLEET, "measures": {"x": "SUM(nope)"}}, ("m", "measures", "x", "expr"), "Unknown column 'nope'"),
    ("m", {**FLEET, "measures": {"x": "t_cap"}}, ("m", "measures", "x", "expr"), "must aggregate"),
    ("m", {**FLEET, "table": "nope"}, ("m", "table"), "has no table 'nope'"),
    ("m", {**FLEET, "source": "nope"}, ("m", "source"), "serves no source 'nope'"),
    ("m", {**FLEET, "colour": "red"}, ("m", "colour"), "Unknown key 'colour'"),
    ("m", {"source": "turbines", "sql": "DELETE FROM turbines"}, ("m", "sql"), ""),
])
def test_problems_name_the_field_they_concern(project, name, spec, keys, message):
    problems = check_metric(project, name, spec)
    assert problems
    assert problems[0].keys[:len(keys)] == keys
    assert message in problems[0].message


def test_an_admin_saves_a_metric_and_keeps_the_others(project):
    assert save_metric(project, "capacity", CAPACITY, app=Admin()) == []
    section = raw_section(project)
    assert list(section) == ["fleet", "capacity"]
    assert section["capacity"] == CAPACITY
    assert load_project(project.path).get("general") == {"admins": ["ana"]}
    assert list(read_metrics(project)) == ["fleet", "capacity"]


def test_renaming_keeps_the_metric_in_its_place(project):
    assert save_metric(project, "capacity", CAPACITY, app=Admin()) == []
    base = raw_section(project)["fleet"]
    assert save_metric(project, "turbine_fleet", base, app=Admin(), original="fleet", base=base) == []
    assert list(raw_section(project)) == ["turbine_fleet", "capacity"]


@pytest.mark.parametrize("app", [None, Admin(admin=False)], ids=["no app", "no longer admin"])
def test_only_an_admin_saves_or_deletes(project, app):
    before = open(project.path).read()
    assert "Only admins" in save_metric(project, "capacity", CAPACITY, app=app)[0].message
    assert "Only admins" in delete_metric(project, "fleet", app=app)[0].message
    assert open(project.path).read() == before


def test_an_invalid_metric_is_not_written(project):
    before = open(project.path).read()
    problems = save_metric(project, "m", {**FLEET, "measures": {"x": "SUM(nope)"}}, app=Admin())
    assert problems[0].keys == ("m", "measures", "x", "expr")
    assert open(project.path).read() == before


def test_a_metric_changed_since_the_editor_opened_is_not_overwritten(project):
    base = raw_section(project)["fleet"]
    changed = {**base, "description": "Another admin's edit."}
    assert save_metric(project, "fleet", changed, app=Admin(), original="fleet", base=base) == []
    mine = {**base, "description": "Mine."}
    problems = save_metric(project, "fleet", mine, app=Admin(), original="fleet", base=base)
    assert "changed since you opened it" in problems[0].message
    assert raw_section(project)["fleet"]["description"] == "Another admin's edit."
    assert "changed since you opened it" in delete_metric(project, "fleet", app=Admin(), base=base)[0].message


def test_delete_removes_only_that_metric(project):
    save_metric(project, "capacity", CAPACITY, app=Admin())
    assert delete_metric(project, "fleet", app=Admin()) == []
    assert list(raw_section(project)) == ["capacity"]
    assert "not in the project file" in delete_metric(project, "fleet", app=Admin())[0].message


def _dashboard(store, title, *metrics):
    model = store.create_dashboard("ana", title)
    model.items = [DashboardItem(f"m{i}", METRIC_ID, config={"metric": m}) for i, m in enumerate(metrics)]
    store.save_dashboard(model)
    return model


def test_usage_lists_the_dashboards_reading_each_metric():
    store = MemoryDashboardStore()
    fleet = _dashboard(store, "Fleet", "fleet", "fleet")
    both = _dashboard(store, "Both", "fleet", "capacity")
    _dashboard(store, "None")
    usage = metric_usage(store._all_dashboards())
    assert sorted(r.title for r in usage["fleet"]) == ["Both", "Fleet"]
    assert [r.dashboard_id for r in usage["capacity"]] == [both.dashboard_id]
    assert fleet.dashboard_id not in [r.dashboard_id for r in usage["capacity"]]


def test_the_preview_runs_the_measures_alone_and_by_a_dimension(project):
    problems, (total, by_state) = preview_metric(project, "fleet", FLEET)
    assert problems == []
    assert total.columns == ["turbines", "turbine_mw"] and total.rows == [[5, 10.5]]
    assert by_state.label == "By state" and by_state.rows == [["TX", 3, 4.5], ["CA", 2, 6.0]]
    assert "LIMIT 10" in by_state.sql


def test_the_preview_counts_each_projects_capacity_once(project):
    problems, (result,) = preview_metric(project, "capacity", CAPACITY)
    assert problems == [] and result.rows == [[10.5, 2]]


def test_the_preview_is_capped_and_read_only(project):
    problems, (result,) = preview_metric(project, "rows", {"source": "turbines", "sql": "SELECT * FROM turbines"}, limit=2)
    assert len(result.rows) == 2 and "LIMIT 2" in result.sql
    problems, _ = preview_metric(project, "drop", {"source": "turbines", "sql": "DROP TABLE turbines"})
    assert problems and problems[0].keys[:2] == ("drop", "sql")


def test_a_preview_error_masks_credentials(project, monkeypatch):
    from lumen.flowdash import nodes

    def fail(*args, **kwargs):
        raise RuntimeError("could not connect to postgresql://ana:hunter2@db.example.com/fleet")

    monkeypatch.setattr(nodes, "execute", fail)
    _, (result, _) = preview_metric(project, "fleet", FLEET)
    assert "hunter2" not in result.error and "***" in result.error


def test_the_form_and_the_yaml_make_one_definition():
    form, text = split_spec(FLEET)
    assert form == {"description": None, "source": "turbines", "table": "turbines"}
    spec, problems, positions = join_spec(form, text)
    assert spec == FLEET and problems == []
    assert positions[("measures", "turbines")][0] == 2
    _, problems, _ = join_spec(form, "source: other\n")
    assert "in the form above" in problems[0].message
    _, problems, _ = join_spec(form, "measures: [\n")
    assert problems[0].keys == ("", "yaml")


def test_the_app_uses_what_an_admin_saves(project):
    metrics = ProjectMetrics(project)
    assert list(metrics) == ["fleet"]
    save_metric(project, "capacity", CAPACITY, app=Admin())
    assert list(metrics) == ["fleet", "capacity"]


def test_build_routes_gives_the_admin_view_the_project_it_reads(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "turbines.csv").write_text("p_name,t_state,p_year,p_cap,t_cap\nAlta,CA,2010,6.0,3000\n")
    (tmp_path / PROJECT_FILE).write_text(YAML)
    app = build_routes(["turbines.csv"], db_path=tmp_path / "dash.db", project=tmp_path / PROJECT_FILE)["/"]()
    project = app._settings_project()
    assert project.path == str(tmp_path / PROJECT_FILE)
    assert save_metric(project, "capacity", CAPACITY, app=Admin()) == []
    assert list(app._editor.context[METRICS_KEY]) == ["fleet", "capacity"]


def test_without_a_project_file_the_admin_view_writes_one_in_the_working_directory(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "turbines.csv").write_text("p_name,t_state,p_year,p_cap,t_cap\nAlta,CA,2010,6.0,3000\n")
    app = build_routes(["turbines.csv"], db_path=tmp_path / "dash.db")["/"]()
    project = app._settings_project()
    assert project.path == str(tmp_path / PROJECT_FILE)
    assert save_metric(project, "capacity", CAPACITY, app=Admin()) == []
    assert list(load_project(tmp_path / PROJECT_FILE).raw("metrics")) == ["capacity"]
    assert list(app._editor.context[METRICS_KEY]) == ["capacity"]


def test_the_section_escapes_what_the_file_says(project):
    hostile = {**FLEET, "description": "<img src=x onerror=alert(1)>"}
    assert save_metric(project, "fleet", hostile, app=Admin(), original="fleet") == []
    with using_app(Admin(store=MemoryDashboardStore())):
        view = MetricsSettings(project)
    html = "".join(str(getattr(c, "object", "")) for c in view._table._rows[0].select())
    assert "&lt;img src=x onerror=alert(1)&gt;" in html and "<img" not in html


def test_the_section_refuses_to_save_for_someone_no_longer_admin(project):
    app = Admin(store=MemoryDashboardStore())
    with using_app(app):
        view = MetricsSettings(project)
    view.open_metric(None)
    view._editor.load("capacity", CAPACITY)
    app.admin = False
    assert not view.save()
    assert "Only admins" in view._drawer_alert.object
    assert "capacity" not in raw_section(project)


def test_the_wizard_step_writes_nothing_when_left_blank(project):
    step = KeyNumbersStep(project)
    assert step.changes() == {}
    assert step.summary() == ["No metric defined"]
    step.editor.load("capacity", CAPACITY)
    assert step.changes() == {"metrics": {"fleet": raw_section(project)["fleet"], "capacity": CAPACITY}}
    assert step.summary() == ["capacity: a SQL query on turbines"]
    step.editor.load("bad name", CAPACITY)
    with pytest.raises(ValueError, match="letters, digits"):
        step.changes()


def test_a_demoted_admin_cannot_preview(project):
    from lumen.flowdash.metrics_admin import MetricEditor

    app = Admin()
    with using_app(app):
        editor = MetricEditor(project)
    editor.load("fleet", FLEET)
    assert editor.preview()
    app.admin = False
    assert editor.preview() == []
    assert "Only admins can preview" in editor._preview.object
    with using_app(None):
        assert MetricEditor(project).preview() == []


def test_the_yaml_field_refuses_aliases_and_long_text():
    import time

    from lumen.flowdash.metrics_admin import YAML_LIMIT

    laughs = 'a: &a ["lol","lol","lol","lol","lol","lol","lol","lol","lol"]\n' + "".join(
        f"{chr(98 + i)}: &{chr(98 + i)} [{', '.join(['*' + chr(97 + i)] * 9)}]\n" for i in range(9)
    )
    start = time.monotonic()
    _, problems, _ = join_spec({}, laughs)
    assert time.monotonic() - start < 1
    assert "Anchors and aliases" in problems[0].message
    _, problems, _ = join_spec({}, "measures:\n  n: COUNT(*)\n" + "#" * YAML_LIMIT)
    assert "longer than" in problems[0].message


def test_only_a_lumen_app_shows_the_metrics_admin_and_key_numbers(tmp_path, monkeypatch):
    from bokeh.document import Document
    from panel.io.state import set_curdoc
    from panel_flowdash.admin.page import admin_sections
    from panel_flowdash.admin.wizard import DataStep, wizard_steps
    from panel_flowdash.app import FlowDashApp

    monkeypatch.chdir(tmp_path)
    (tmp_path / "turbines.csv").write_text("p_name,t_state,p_year,p_cap,t_cap\nAlta,CA,2010,6.0,3000\n")
    routes = build_routes(["turbines.csv"], db_path=tmp_path / "dash.db")
    with set_curdoc(Document()):
        lumen_app = routes["/"]()
        assert "metrics" in [s.name for s in admin_sections()]
        steps = wizard_steps(lumen_app._settings_project(), lumen_app)
        assert isinstance(steps[1], KeyNumbersStep)
        assert not any(isinstance(step, DataStep) for step in steps)
    plain = tmp_path / "plain"
    plain.mkdir()
    with set_curdoc(Document()):
        plain_app = FlowDashApp(registry={}, project_dir=plain, store=MemoryDashboardStore(),
                                notifications=False, include_builtin_components=False)
        assert "metrics" not in [s.name for s in admin_sections()]
        steps = wizard_steps(plain_app._settings_project(), plain_app)
        assert isinstance(steps[1], DataStep)
        assert not any(isinstance(step, KeyNumbersStep) for step in steps)


@pytest.mark.parametrize("text", ["measures: " + "[" * 1000 + "]" * 1000, "measures: " + "{a: " * 1000 + "1" + "}" * 1000])
def test_deeply_nested_yaml_is_a_field_error(text):
    _, problems, _ = join_spec({"source": "turbines"}, text)
    assert problems[0].keys == ("", "yaml") and "nests deeper" in problems[0].message


SECRET_URL = "postgresql://ana:hunter2@db.example.com/fleet"


@pytest.fixture
def source_down(monkeypatch):
    """The engine refusing every metric with an error that quotes its connection URL."""
    from lumen.flowdash import metrics

    calls = []

    def fail(metric, source):
        calls.append(metric.name)
        raise RuntimeError(f"could not connect to {SECRET_URL}")

    monkeypatch.setattr(metrics, "check_metric", fail)
    return calls


def _reloaded(project, section):
    """``project`` as a reload of its file with ``section`` as its metrics would leave it, nothing loaded yet."""
    project.data = {**project.data, "metrics": section}
    project._values.pop("metrics", None)


def test_metrics_that_fail_to_load_are_logged_once_masked(project, source_down, caplog):
    _reloaded(project, raw_section(project))
    metrics = ProjectMetrics(project)
    with caplog.at_level("WARNING", logger="lumen.flowdash.metrics"):
        for _ in range(30):
            assert list(metrics) == [] and len(metrics) == 0 and "fleet" not in metrics
    assert source_down == ["fleet"]
    warnings = [r.getMessage() for r in caplog.records if r.name == "lumen.flowdash.metrics"]
    assert len(warnings) == 1
    assert "hunter2" not in warnings[0] and "***" in warnings[0]


def test_metrics_that_fail_to_reload_keep_the_last_good_set(project, monkeypatch, caplog):
    metrics = ProjectMetrics(project, retry=0)
    assert list(metrics) == ["fleet"]
    from lumen.flowdash import metrics as module

    checked = module.check_metric
    calls = []

    def fail(metric, source):
        calls.append(metric.name)
        raise RuntimeError(f"could not connect to {SECRET_URL}")

    monkeypatch.setattr(module, "check_metric", fail)
    _reloaded(project, {**raw_section(project), "capacity": CAPACITY})
    with caplog.at_level("WARNING", logger="lumen.flowdash.metrics"):
        for _ in range(5):
            assert list(metrics) == ["fleet"]
            assert metrics["fleet"].table == "turbines"
    warnings = [r.getMessage() for r in caplog.records if r.name == "lumen.flowdash.metrics"]
    assert len(warnings) == 1 and "Keeping the 1 metric read before" in warnings[0]
    assert "hunter2" not in warnings[0]
    # With the source back, the next retry reads the changed metrics.
    monkeypatch.setattr(module, "check_metric", checked)
    assert list(metrics) == ["fleet", "capacity"]


def test_a_failed_load_is_not_retried_before_its_time(project, source_down):
    _reloaded(project, raw_section(project))
    metrics = ProjectMetrics(project, retry=3600)
    for _ in range(10):
        dict(metrics.items())
    assert source_down == ["fleet"]


def test_a_metric_deleted_while_the_metrics_are_read_does_not_break_the_reading(project):
    from lumen.flowdash.nodes import host_metrics

    save_metric(project, "capacity", CAPACITY, app=Admin())
    metrics = ProjectMetrics(project)
    for read in (lambda m: m.items(), lambda m: m.values(), lambda m: host_metrics({METRICS_KEY: m}).items()):
        seen = []
        for item in read(metrics):
            seen.append(item)
            if len(seen) == 1:
                delete_metric(project, "capacity", app=Admin())
        assert len(seen) == 2
        assert list(metrics) == ["fleet"]
        save_metric(project, "capacity", CAPACITY, app=Admin())
    snapshot = metrics.snapshot()
    delete_metric(project, "capacity", app=Admin())
    assert list(snapshot) == ["fleet", "capacity"] and snapshot["capacity"].source == "turbines"
    assert list(metrics.snapshot()) == ["fleet"]


def test_check_problems_mask_credentials(project, source_down):
    problems = check_metric(project, "capacity", FLEET)
    assert problems and all("hunter2" not in p.message for p in problems)
    assert "***" in problems[0].message
    assert all("hunter2" not in p.message for p in save_metric(project, "capacity", FLEET, app=Admin()))


def _settings(project, *dashboards):
    from lumen.flowdash.metrics_admin import _set_text

    store = MemoryDashboardStore()
    for title, metric in dashboards:
        _dashboard(store, title, metric)
    with using_app(Admin(store=store)):
        view = MetricsSettings(project)
    return view, lambda name: _set_text(view._editor._name, name)


def test_renaming_a_metric_dashboards_use_asks_first(project):
    view, rename = _settings(project, ("Fleet overview", "fleet"), ("Other", "capacity"))
    view.open_metric("fleet")
    rename("fleet_v2")
    assert not view.save()
    assert view._dialog.open and view._dialog.title == "Rename metric fleet?"
    assert "1 dashboard uses fleet" in view._dialog_text.object and "Fleet overview" in view._dialog_text.object
    assert "Other" not in view._dialog_text.object and view._confirm.label == "Rename metric"
    assert list(raw_section(project)) == ["fleet"]

    view.cancel_delete()
    assert not view._dialog.open and view._drawer.open
    assert list(raw_section(project)) == ["fleet"]

    assert not view.save()
    assert view.confirm()
    assert not view._dialog.open and not view._drawer.open
    assert list(raw_section(project)) == ["fleet_v2"]


def test_renaming_a_metric_no_dashboard_uses_saves_at_once(project):
    view, rename = _settings(project, ("Other", "capacity"))
    view.open_metric("fleet")
    rename("fleet_v2")
    assert view.save()
    assert not view._dialog.open
    assert list(raw_section(project)) == ["fleet_v2"]


def test_an_invalid_rename_shows_its_problems_without_asking(project):
    view, rename = _settings(project, ("Fleet overview", "fleet"))
    view.open_metric("fleet")
    rename("fleet v2")
    assert not view.save()
    assert not view._dialog.open
    assert "letters, digits" in view._editor._name.helper_text
