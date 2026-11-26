"""Uploading data files: the handler registry, the uploads: section, the store, the dialog and the catalog."""
import hashlib
import io
import json
import logging

from types import SimpleNamespace

import pytest

pytest.importorskip("duckdb")
pytest.importorskip("panel_flowdash.admin.wizard")

from panel_flowdash import FlowDash
from panel_flowdash.admin.form import using_app
from panel_flowdash.agent import DashboardOps, FlowDashDeps
from panel_flowdash.auth import Identity

from lumen.agent.data import list_tables
from lumen.flowdash import (
    CATALOG_KEY, COMPONENTS, SOURCES_KEY, ProjectCatalog, hidden_tables,
    uploads as uploads_module,
)
from lumen.flowdash.add_data import UploadDialog
from lumen.flowdash.app import build_routes
from lumen.flowdash.catalog import raw_section
from lumen.flowdash.catalog_admin import CatalogSettings, set_assistant
from lumen.flowdash.project import PROJECT_FILE, ProjectError, load_project
from lumen.flowdash.uploads import (
    UPLOADS_KEY, UploadError, UploadFormat, Uploads, register_upload_handler,
    upload_formats, upload_settings,
)
from lumen.flowdash.uploads_admin import UploadsForm
from lumen.sources.duckdb import DuckDBSource

CSV = b"region,revenue\nNorth,120.5\nSouth,98\n"

ON = "uploads:\n  enabled: true\n"


class App:
    """The parts of a FlowDash app's session an upload uses."""

    def __init__(self, user="ana", groups=(), admin=False, editor=None):
        self._identity = Identity(user=user, groups=frozenset(groups))
        self.admin = admin
        self._editor = editor
        self.notified = []

    def _still_admin(self):
        return self.admin

    def _notify(self, severity, message, duration=3000):
        self.notified.append((severity, message))


def read_notes(context, file, alias, filename):
    """A made-up format: ``key=value`` lines, one row each."""
    import pandas as pd

    rows = [line.split("=", 1) for line in file.read().decode().splitlines() if "=" in line]
    return DuckDBSource.from_df({alias: pd.DataFrame(rows, columns=["key", "value"])})


def await_(coroutine):
    import asyncio

    return asyncio.run(coroutine)


def _excel(**sheets):
    import pandas as pd

    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer) as writer:
        for name, frame in sheets.items():
            pd.DataFrame(frame).to_excel(writer, sheet_name=name, index=False)
    return buffer.getvalue()


@pytest.fixture
def project(tmp_path):
    path = tmp_path / PROJECT_FILE
    path.write_text(ON)
    return load_project(path, {SOURCES_KEY: {}})


@pytest.fixture
def uploads(tmp_path, project):
    store = Uploads(tmp_path / "uploads", project.context[SOURCES_KEY], project=project)
    project.context[UPLOADS_KEY] = store
    return store


def test_lumen_registers_its_file_types():
    formats = upload_formats()
    assert {"csv", "tsv", "parquet", "json", "jsonl"} <= set(formats)
    assert formats["csv"].label == "CSV" and formats["xlsx"].requires == ("openpyxl",)


def test_entry_points_and_app_handlers_add_file_types(monkeypatch):
    entry = SimpleNamespace(name=".notes", value="pkg:read_notes", load=lambda: read_notes)
    broken = SimpleNamespace(name="bad", value="pkg:missing", load=lambda: (_ for _ in ()).throw(ImportError()))
    monkeypatch.setattr("importlib.metadata.entry_points", lambda group: [entry, broken] if group == "lumen.upload_handlers" else [])
    uploads_module._entry_point_formats.cache_clear()
    try:
        formats = upload_formats({"CSV": read_notes, "h5": UploadFormat("h5", "HDF5", read_notes, ("no_such_module",))})
    finally:
        uploads_module._entry_point_formats.cache_clear()
    assert formats["notes"].handler is read_notes and "bad" not in formats
    assert formats["csv"].handler is read_notes and "h5" not in formats


def test_register_upload_handler_adds_a_type_for_every_app(monkeypatch):
    monkeypatch.setattr(uploads_module, "_FORMATS", dict(uploads_module._FORMATS))
    register_upload_handler(".Notes", read_notes, label="Notes")
    assert upload_formats()["notes"].label == "Notes"


def test_uploads_are_off_by_default(tmp_path):
    path = tmp_path / PROJECT_FILE
    path.write_text("")
    assert upload_settings(load_project(path, {})) == uploads_module.UploadSettings()
    path.write_text("uploads:\n  enabled: true\n  groups: [analysts]\n  formats: [.CSV, xlsx]\n  max_mb: 5\n")
    settings = upload_settings(load_project(path, {}))
    assert (settings.enabled, settings.groups, settings.formats, settings.max_mb) == (True, ("analysts",), ("csv", "xlsx"), 5)


@pytest.mark.parametrize(("text", "message"), [
    ("uploads:\n  enabled: sure\n", "true or false"),
    ("uploads:\n  max_mb: 0\n", "at least 1"),
    ("uploads:\n  formats: [csv, 'x y']\n", "lists extensions"),
    ("uploads:\n  folder: x\n", "Unknown setting 'folder'"),
])
def test_a_wrong_setting_is_reported(tmp_path, text, message):
    path = tmp_path / PROJECT_FILE
    path.write_text(text)
    with pytest.raises(ProjectError, match=message):
        load_project(path, {})


def test_who_may_upload(tmp_path, project, uploads):
    member, other, admin = App(groups=["analysts"]), App(user="bo"), App(user="cy", admin=True)
    assert uploads.offered(member) and uploads.offered(other)
    project.set("uploads", {"enabled": True, "groups": ["analysts"]})
    assert uploads.offered(member) and not uploads.offered(other) and uploads.offered(admin)
    project.set("uploads", {})
    assert not uploads.offered(admin) and uploads.may_upload(admin) and not uploads.may_upload(member)


def test_an_upload_becomes_a_source_with_who_uploaded_it(tmp_path, uploads):
    upload = uploads.add(CSV, "Regional Sales.csv", app=App())
    assert (upload.ref, upload.tables, upload.uploaded_by, upload.size) == ("regional_sales", ["regional_sales"], "ana", len(CSV))
    source = uploads.sources["regional_sales"]
    assert source.name == "regional_sales"
    assert source.get("regional_sales")["revenue"].tolist() == [120.5, 98.0]
    assert (tmp_path / "uploads" / upload.id / "Regional Sales.csv").read_bytes() == CSV
    manifest = json.loads((tmp_path / "uploads" / "manifest.json").read_text())
    assert manifest["uploads"][0] == {
        "ref": "regional_sales", "filename": "Regional Sales.csv", "format": "csv", "tables": ["regional_sales"],
        "uploaded_by": "ana", "uploaded_at": upload.uploaded_at, "size": len(CSV), "sheet": None, "id": upload.id,
        "sha256": hashlib.sha256(CSV).hexdigest(),
    }
    assert uploads.add(CSV, "regional_sales.csv", app=App()).ref == "regional_sales_2"


def test_excel_gives_a_table_per_sheet_or_the_sheet_picked(uploads):
    pytest.importorskip("openpyxl")
    content = _excel(Q1={"region": ["North"], "revenue": [1.5]}, Q2={"region": ["South"], "revenue": [2.5]})
    assert uploads.add(content, "sales.xlsx", app=App()).tables == ["sales_q1", "sales_q2"]
    upload = uploads.add(content, "sales.xlsx", "second quarter", app=App(), sheet="Q2")
    assert (upload.ref, upload.tables, upload.sheet) == ("second_quarter", ["second_quarter"], "Q2")
    assert uploads.sources["second_quarter"].get("second_quarter")["region"].tolist() == ["South"]


def test_json_and_parquet_uploads(tmp_path, uploads):
    import pandas as pd

    assert uploads.add(b'[{"a": 1, "b": "x"}, {"a": 2, "b": "y"}]', "rows.json", app=App()).tables == ["rows"]
    pd.DataFrame({"a": [1, 2, 3]}).to_parquet(tmp_path / "f.parquet")
    upload = uploads.add((tmp_path / "f.parquet").read_bytes(), "f.parquet", app=App())
    assert len(uploads.sources[upload.ref].get(upload.ref)) == 3


@pytest.mark.parametrize(("settings", "app", "filename", "content", "message"), [
    ({}, App(), "a.csv", CSV, "turned off for you"),
    ({"enabled": True, "groups": ["analysts"]}, App(), "a.csv", CSV, "turned off for you"),
    ({"enabled": True, "formats": ["parquet"]}, App(), "a.csv", CSV, "Allowed: Parquet"),
    ({"enabled": True}, App(), "notes", CSV, "not a file type"),
    ({"enabled": True, "max_mb": 1}, App(), "a.csv", b"x" * (2**20 + 1), "larger than the 1 MB"),
    ({"enabled": True}, App(), "a.parquet", b"not parquet", "could not be read"),
])
def test_refused_uploads_leave_nothing_behind(tmp_path, project, uploads, settings, app, filename, content, message):
    project.set("uploads", settings)
    with pytest.raises(UploadError, match=message):
        uploads.add(content, filename, app=app)
    assert uploads.sources == {} and uploads.uploads() == []
    assert [p.name for p in (tmp_path / "uploads").glob("*") if p.is_dir()] == []


def test_file_contents_are_not_logged(uploads, caplog):
    secret = b"name,ssn\nAda,123-45-6789\n"
    with caplog.at_level(logging.DEBUG, logger="lumen"):
        uploads.add(secret, "people.csv", app=App())
        with pytest.raises(UploadError):
            uploads.add(b'"unclosed\n123-45-6789', "broken.json", app=App())
    assert "123-45-6789" not in caplog.text
    assert "people.csv" in caplog.text


def test_uploads_are_served_again_after_a_restart(tmp_path, uploads):
    uploads.add(CSV, "sales.csv", app=App())
    restored = Uploads(tmp_path / "uploads", {"turbines": DuckDBSource(uri=":memory:", tables={})})
    restored.restore()
    assert list(restored.sources) == ["turbines", "sales"]
    assert restored.get("sales").uploaded_by == "ana"
    taken = Uploads(tmp_path / "uploads", {"sales": DuckDBSource(uri=":memory:", tables={})})
    taken.restore()
    assert taken.get("sales") is None


def test_a_manifest_entry_from_before_uploaders_were_recorded_loads_fine(tmp_path, uploads):
    upload = uploads.add(CSV, "sales.csv", app=App(user="ana"))
    manifest_path = tmp_path / "uploads" / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    del manifest["uploads"][0]["uploaded_by"]
    manifest_path.write_text(json.dumps(manifest))
    restored = Uploads(tmp_path / "uploads", {})
    restored.restore()
    restored_upload = restored.get(upload.ref)
    assert restored_upload is not None and restored_upload.uploaded_by == ""
    assert "Only admins" in restored.delete(upload.ref, app=App(user="ana"))[0].message
    assert restored.delete(upload.ref, app=App(admin=True)) == []


def test_every_session_gets_an_upload_and_loses_a_deleted_one(uploads):
    sessions = [App(editor=FlowDash(components=COMPONENTS, context={SOURCES_KEY: {}})) for _ in range(2)]
    for app in sessions:
        uploads.attach(app)
    uploads.add(CSV, "sales.csv", app=sessions[0])
    assert all("sales" in app._editor.context[SOURCES_KEY] for app in sessions)
    assert "Only admins" in uploads.delete("sales", app=App(user="cy"))[0].message
    # sessions[0] uploaded it (as "ana", App's default user), so it, not just an admin, may delete it.
    assert uploads.delete("sales", app=sessions[0]) == []
    assert all("sales" not in app._editor.context[SOURCES_KEY] for app in sessions)
    assert uploads.uploads() == []


def test_the_uploader_may_delete_their_own_upload(uploads):
    upload = uploads.add(CSV, "sales.csv", app=App(user="ana"))
    assert uploads.delete(upload.ref, app=App(user="ana")) == []
    assert "sales" not in uploads.sources and uploads.uploads() == []


def test_a_non_admin_non_uploader_cannot_delete_someone_elses_upload(uploads):
    upload = uploads.add(CSV, "sales.csv", app=App(user="ana"))
    problems = uploads.delete(upload.ref, app=App(user="cy"))
    assert problems and "Only admins" in problems[0].message
    assert "sales" in uploads.sources


def test_an_admin_may_delete_anyones_upload(uploads):
    upload = uploads.add(CSV, "sales.csv", app=App(user="ana"))
    assert uploads.delete(upload.ref, app=App(user="cy", admin=True)) == []
    assert "sales" not in uploads.sources


def test_an_upload_with_no_recorded_uploader_is_only_an_admins_to_delete(uploads):
    import dataclasses

    upload = uploads.add(CSV, "sales.csv", app=App(user="ana"))
    # A record kept from before uploaded_by was recorded, simulated by clearing it.
    uploads._uploads[upload.ref] = dataclasses.replace(upload, uploaded_by="")
    problems = uploads.delete(upload.ref, app=App(user="ana"))
    assert problems and "Only admins" in problems[0].message
    assert uploads.delete(upload.ref, app=App(admin=True)) == []


def test_a_deleted_upload_waits_in_the_trash_and_can_be_brought_back(uploads):
    import datetime as dt

    upload = uploads.add(CSV, "sales.csv", app=App())
    admin = App(user="cy", admin=True)
    assert uploads.delete("sales", app=admin) == []
    trash = uploads.directory / ".trash" / upload.id
    note = json.loads((trash / "upload.json").read_text())
    assert (trash / "sales.csv").read_bytes() == CSV and note["deleted_by"] == "cy"
    assert not uploads.folder(upload).exists()

    uploads.add(CSV, "sales.csv", app=App())
    with pytest.raises(UploadError, match="Only admins"):
        uploads.recover(upload.id, app=App())
    recovered = uploads.recover(upload.id, app=admin)
    assert (recovered.ref, recovered.id, recovered.uploaded_by) == ("sales_2", upload.id, "ana")
    assert uploads.sources["sales_2"].get("sales_2")["revenue"].tolist() == [120.5, 98.0]

    uploads.delete("sales_2", app=admin)
    assert uploads.purge_trash(dt.datetime.now(dt.UTC) + dt.timedelta(days=29)) == []
    assert uploads.purge_trash(dt.datetime.now(dt.UTC) + dt.timedelta(days=31)) == [upload.id]
    assert not trash.exists()


def test_recover_cannot_be_made_to_read_outside_the_trash(tmp_path, uploads):
    admin = App(admin=True)
    # A real deleted upload, so the trash folder it lives beside actually exists.
    upload = uploads.add(CSV, "sales.csv", app=App())
    uploads.delete("sales", app=admin)

    outside = tmp_path / "secret"
    outside.mkdir()
    (outside / "upload.json").write_text(json.dumps({
        "ref": "evil", "id": "evil", "filename": "x", "format": "csv", "tables": [],
        "uploaded_by": "x", "uploaded_at": "2024-01-01T00:00:00", "size": 0,
    }))
    for upload_id in ("../../secret", "..", "../.."):
        with pytest.raises(UploadError, match="no deleted upload"):
            uploads.recover(upload_id, app=admin)
    assert outside.exists() and "evil" not in uploads.sources
    # The real deleted upload is still recoverable by its actual id.
    assert uploads.recover(upload.id, app=admin).ref == "sales"


def test_a_traversal_attempt_in_an_uploaded_filename_is_kept_to_its_own_folder(tmp_path, uploads):
    upload = uploads.add(CSV, "../../etc/notes.csv", app=App())
    assert upload.filename == "notes.csv"
    assert (uploads.folder(upload) / "notes.csv").read_bytes() == CSV
    assert not (tmp_path / "etc").exists()
    with pytest.raises(UploadError, match="not a file type"):
        uploads.add(CSV, "..", app=App())


def test_the_add_data_button_shows_on_dashboards_for_someone_who_may_upload(project, uploads):
    app = App(editor=FlowDash(components=COMPONENTS, context={SOURCES_KEY: {}}))
    button = uploads.attach(app)
    assert button._layout in app._editor.toolbar_extra and button._layout.visible
    app._editor.editable = False
    assert button._layout.visible and "1023.95px" in button.button.stylesheets[0]
    app._editor.editable = True
    assert button._layout.visible and "1599.95px" in button.button.stylesheets[0]
    project.set("uploads", {})
    button.sync()
    assert not button._layout.visible


def test_the_dialog_uploads_the_file_chosen(uploads):
    pytest.importorskip("openpyxl")
    app = App()
    dialog = UploadDialog(uploads, app)
    dialog.open()
    dialog.choose(_excel(Q1={"x": [1]}, Q2={"x": [2]}), "Book 1.xlsx")
    assert dialog._name.value == "book_1" and dialog._sheet.visible
    assert list(dialog._sheet.options.values()) == ["", "Q1", "Q2"] and not dialog._submit.disabled
    dialog._sheet.value = "Q2"
    upload = dialog.submit()
    assert upload.tables == ["book_1"] and not dialog._dialog.open
    assert app.notified == [("success", "Added the table book_1 as source book_1; find it in the component list.")]
    dialog.open()
    dialog.choose(CSV, "notes.txt")
    assert dialog._submit.disabled and "not a file type" in dialog._alert.object


def test_a_custom_handler_goes_through_the_whole_path(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "turbines.csv").write_text("p_name,p_cap\nAlta,6.0\n")
    (tmp_path / PROJECT_FILE).write_text(ON + "  formats: [csv, notes]\n")
    routes = build_routes(["turbines.csv"], db_path=tmp_path / "dash.db", project=tmp_path / PROJECT_FILE,
                          upload_handlers={".notes": read_notes})
    app = routes["/"]()
    context = app._editor.context
    uploads = context[UPLOADS_KEY]
    assert uploads.directory == tmp_path / "uploads"
    assert "notes" in uploads.allowed() and "json" not in uploads.allowed()

    upload = uploads.add(b"colour=blue\nsize=large\n", "settings.notes", app=App(editor=app._editor))
    assert upload.tables == ["settings"]
    assert context[SOURCES_KEY]["settings"].get("settings")["value"].tolist() == ["blue", "large"]
    assert "settings" in routes["/"]()._editor.context[SOURCES_KEY]

    project = app._settings_project()
    with using_app(App(admin=True)):
        rows = {(r["ref"], r["table"]): r for r in CatalogSettings(project).rows()}
    assert rows[("settings", "settings")]["upload"].uploaded_by == "ana"

    deps = FlowDashDeps(DashboardOps(app._editor))
    assert "settings" in await_(list_tables(SimpleNamespace(deps=deps)))
    assert set_assistant(project, "settings", "settings", False, app=App(admin=True)) == []
    assert hidden_tables(context[SOURCES_KEY]["settings"], context) == {"settings"}

    restarted = build_routes(["turbines.csv"], db_path=tmp_path / "dash.db", project=tmp_path / PROJECT_FILE,
                             upload_handlers={"notes": read_notes})["/"]()
    assert restarted._editor.context[SOURCES_KEY]["settings"].get("settings")["key"].tolist() == ["colour", "size"]


def test_the_catalog_lists_uploads_and_an_admin_deletes_one(tmp_path, project, uploads):
    uploads.sources["served"] = DuckDBSource(uri=":memory:", tables={"t": "SELECT 1 AS x"}, name="served")
    uploads.add(CSV, "sales.csv", app=App())
    set_assistant(project, "sales", "sales", False, app=App(admin=True))
    with using_app(App(admin=True)):
        view = CatalogSettings(project)
    assert [c.label for c in view._chip_row] == ["All 2", "served 1", "sales 1", "Assistant off 1"]
    view.filter = "source:sales"
    assert [r["table"] for r in view.visible_rows()] == ["sales"]
    view.open_table("sales", "sales")
    assert view._delete.visible and "Uploaded by" in view._details.object and "<dd>ana</dd>" in view._details.object
    view.ask_delete()
    assert view._confirm_dialog.open and "sales.csv" in view._confirm_text.object
    assert view.delete_upload()
    assert "sales" not in uploads.sources and "sales" not in raw_section(project)
    assert [r["table"] for r in view.rows()] == ["t"]
    view.open_table("served", "t")
    assert not view._delete.visible


def test_the_catalog_upload_button_is_for_admins(project, uploads):
    project.set("uploads", {})
    with using_app(App(admin=True)):
        view = CatalogSettings(project)
    assert view._upload.visible
    view._upload_dialog.open()
    view._upload_dialog.choose(CSV, "sales.csv")
    assert view._upload_dialog.submit() is not None
    assert [r["table"] for r in view.rows()] == ["sales"]


def test_the_uploads_form_saves_only_what_was_changed(project, uploads):
    with using_app(App(admin=True)):
        form = UploadsForm(project)
    assert form._widgets["formats"].value == list(uploads.formats) and not form.dirty
    form._widgets["formats"].value = ["csv", "xlsx"]
    form._widgets["max_mb"].value = 20
    assert form.dirty and form.save()
    assert load_project(project.path).raw("uploads") == {"enabled": True, "formats": ["csv", "xlsx"], "max_mb": 20}
    form._widgets["formats"].value = list(uploads.formats)
    form._widgets["enabled"].value = False
    assert form.save()
    assert load_project(project.path).raw("uploads") == {"max_mb": 20}


def test_the_catalog_switch_applies_to_uploaded_tables(project, uploads):
    uploads.add(CSV, "sales.csv", app=App())
    context = {SOURCES_KEY: uploads.sources, CATALOG_KEY: ProjectCatalog(project)}
    set_assistant(project, "sales", "sales", False, app=App(admin=True))
    assert hidden_tables(uploads.sources["sales"], context) == {"sales"}
