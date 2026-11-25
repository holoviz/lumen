"""
Data files people upload to a Lumen FlowDash app.

An admin turns uploads on in the admin view's Uploads section, which
writes the project file's ``uploads:`` section:

    uploads:
      enabled: true
      groups: [analysts]       # who may upload; empty or left out: everyone who can open the app
      formats: [csv, xlsx]     # left out: every registered file type
      max_mb: 100

People who may upload then get an Add data button in the dashboard bar,
and admins one in the Catalog section. An uploaded file becomes a source
of its own, named by the table name the uploader picks, that every
session's dashboards, component list and assistant can use, and that the
catalog lists with who uploaded it and when; the catalog's assistant
switch applies to its tables like to any other.

The files are kept beside the project file, in an ``uploads`` folder: each
upload in a folder named by its stable id, and ``manifest.json`` listing
them with who uploaded each, when, and its checksum. They are read again
when the app starts. Deleting an upload moves its folder to
``uploads/.trash``, where it stays for ``TRASH_DAYS`` days, so a backup
taken shortly before stays consistent and ``Uploads.recover`` can bring
it back.

A file is read by the upload handler registered for its extension. A
handler has the signature of Lumen AI's ``upload_handlers``:

    handler(context: dict, file: BinaryIO, alias: str, filename: str) -> Source | None

``alias`` is the source's ref and the table name the uploader picked,
``filename`` the uploaded file's name with its extension, and ``context``
holds the stored file's ``path``, the upload's own ``directory`` for any
files the handler derives from it, and the Excel ``sheet`` picked, if
any. CSV, TSV, Parquet, JSON, JSON Lines and Excel are registered here;
a package adds a type with ``register_upload_handler``, an entry point in
the ``lumen.upload_handlers`` group named by the extension, or an app
with ``build_routes(upload_handlers={"h5ad": handler})``.
"""
from __future__ import annotations

import asyncio
import contextlib
import datetime as dt
import functools
import hashlib
import io
import json
import logging
import os
import re
import shutil
import threading
import typing as t
import uuid
import weakref

from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
from functools import partial
from importlib.util import find_spec
from pathlib import Path

import panel as pn
import panel_material_ui as pmui
import param

from panel.viewable import Viewer

from ..sources.base import Source
from .project import (
    Problem, Project, ProjectError, ProjectSection, Section, register_section,
)

logger = logging.getLogger(__name__)

SECTION = "uploads"

# The key of the editor context the host puts its ``Uploads`` in.
UPLOADS_KEY = "lumen.uploads"

ENTRY_POINT_GROUP = "lumen.upload_handlers"

UPLOAD_DIR = "uploads"
MANIFEST = "manifest.json"
# Where deleted uploads wait, inside the uploads folder, and for how many days, so a backup
# taken shortly before a delete stays consistent and a delete by mistake can be undone.
TRASH = ".trash"
TRASH_DAYS = 30

DEFAULT_MAX_MB = 100

# Rows read from each table of an upload to check it before it is kept.
_CHECK_ROWS = 1

UploadHandler = Callable[[dict[str, t.Any], t.BinaryIO, str, str], "Source | None"]


class UploadError(ValueError):
    """Why an upload was refused or could not be read, to show the person uploading."""


# ---------------------------------------------------------------------------
# File types


@dataclass(frozen=True)
class UploadFormat:
    """A file type people may upload, by its extension, and the handler that reads it into a source."""

    extension: str
    label: str
    handler: UploadHandler
    requires: tuple[str, ...] = ()
    """Modules the handler needs; without them the type is not offered."""

    @property
    def available(self) -> bool:
        return all(find_spec(module) is not None for module in self.requires)


def extension_of(text: str) -> str:
    """``text``, a file name or an extension with or without its dot, as a lowercase extension."""
    text = str(text).strip()
    if "." in text:
        text = text.rsplit(".", 1)[1]
    return text.lower()


_FORMATS: dict[str, UploadFormat] = {}


def _format(extension: str, handler: t.Any, label: str | None = None, requires: Sequence[str] = ()) -> UploadFormat:
    if isinstance(handler, UploadFormat):
        return handler
    if not callable(handler):
        raise TypeError(f"The upload handler for .{extension} is not callable: {handler!r}.")
    return UploadFormat(extension, label or extension.upper(), handler, tuple(requires))


def register_upload_handler(
    extension: str, handler: UploadHandler, *, label: str | None = None, requires: Sequence[str] = (),
) -> UploadFormat:
    """
    Register ``handler`` for files with ``extension`` (e.g. ``"h5ad"``) in
    every app in the process, replacing the one registered before. ``label``
    names the type in the admin view; ``requires`` lists modules without
    which it is not offered.
    """
    extension = extension_of(extension)
    _FORMATS[extension] = upload_format = _format(extension, handler, label, requires)
    return upload_format


@functools.cache
def _entry_point_formats() -> dict[str, UploadFormat]:
    """The handlers installed packages register in the ``lumen.upload_handlers`` entry point group, by extension."""
    from importlib.metadata import entry_points

    found = {}
    for entry in entry_points(group=ENTRY_POINT_GROUP):
        extension = extension_of(entry.name)
        try:
            found[extension] = _format(extension, entry.load())
        except Exception as exc:
            logger.warning("Not offering .%s uploads: its handler %s failed to load: %s",
                           extension, entry.value, type(exc).__name__)
    return found


def upload_formats(handlers: Mapping[str, t.Any] | None = None) -> dict[str, UploadFormat]:
    """
    The file types an app offers, by extension: Lumen's, then those of the
    ``lumen.upload_handlers`` entry points, then ``handlers``, a mapping of
    extension to handler (or ``UploadFormat``) the app was given; a later
    one replaces an earlier one of the same extension. A type whose
    ``requires`` are not installed is left out.
    """
    formats = {**_FORMATS, **_entry_point_formats()}
    for extension, handler in (handlers or {}).items():
        extension = extension_of(extension)
        formats[extension] = _format(extension, handler)
    return {extension: f for extension, f in formats.items() if f.available}


def _slug(text: str) -> str:
    return re.sub(r"\W+", "_", str(text)).strip("_").lower()


def read_file(context: dict[str, t.Any], file: t.BinaryIO, alias: str, filename: str) -> Source:
    """CSV, TSV, Parquet and JSON Lines: a DuckDB source reading the stored file, as ``lumen serve`` reads a served one."""
    from ..sources.data import resolve_data

    (source,) = resolve_data([(alias, str(context["path"]))])
    return source


def _write_parquet(frame: t.Any, path: Path) -> None:
    import duckdb

    connection = duckdb.connect()
    try:
        connection.register("frame", frame)
        escaped = str(path).replace("'", "''")
        connection.execute(f"COPY (SELECT * FROM frame) TO '{escaped}' (FORMAT parquet)")
    finally:
        connection.close()


def read_frames(context: dict[str, t.Any], file: t.BinaryIO, alias: str, filename: str) -> Source:
    """
    JSON and Excel: parsed as Lumen AI's file upload parses them, a table
    per sheet of an Excel file unless ``context["sheet"]`` picks one, and
    stored as Parquet in the upload's directory for a DuckDB source to read.
    """
    from ..ai.controls.ingest.utils import read_file_to_dataframes
    from ..sources.data import resolve_data

    result = read_file_to_dataframes(file, extension_of(filename), alias=alias, sheet_name=context.get("sheet") or None)
    if result is None or not result.tables:
        raise UploadError(f"{filename} holds no table.")
    directory = Path(context["directory"])
    items = []
    for name, frame in result.tables.items():
        table = _slug(name) or alias
        path = directory / f"{table}.parquet"
        _write_parquet(frame, path)
        items.append((table, str(path)))
    (source,) = resolve_data(items)
    return source


for _extension, _handler, _label, _requires in (
    ("csv", read_file, "CSV", ()),
    ("tsv", read_file, "TSV", ()),
    ("parquet", read_file, "Parquet", ()),
    ("json", read_frames, "JSON", ()),
    ("jsonl", read_file, "JSON Lines", ()),
    ("xlsx", read_frames, "Excel", ("openpyxl",)),
):
    register_upload_handler(_extension, _handler, label=_label, requires=_requires)


def excel_sheets(content: bytes) -> list[str]:
    """The sheet names of an Excel file, or none if it cannot be read."""
    import pandas as pd

    try:
        return [str(name) for name in pd.ExcelFile(io.BytesIO(content)).sheet_names]
    except Exception:
        return []


# ---------------------------------------------------------------------------
# The uploads: section


@dataclass(frozen=True)
class UploadSettings:
    enabled: bool = False
    groups: tuple[str, ...] = ()
    formats: tuple[str, ...] | None = None
    max_mb: int = DEFAULT_MAX_MB


def _valid_formats(value: t.Any, section: Section) -> str | None:
    bad = [item for item in value if not re.fullmatch(r"\.?[A-Za-z0-9]+", item.strip())]
    return f"`formats` lists extensions such as csv; not {', '.join(map(repr, bad))}." if bad else None


def _settings_groups():
    from panel_flowdash.admin.form import Group, Setting

    return (
        Group(
            "Uploads",
            (
                Setting(
                    "enabled", "Let people upload data",
                    "An Add data button in the dashboard bar. Admins can upload in the Catalog either way.",
                    kind="switch", default=False,
                ),
                Setting(
                    "max_mb", "Largest file", "", kind="number", unit="MB", integer=True,
                    bounds=(1, None), default=DEFAULT_MAX_MB,
                ),
            ),
        ),
        Group(
            "Who and what",
            (
                Setting(
                    "groups", "Who can upload",
                    "Members of these groups, and admins. Without groups, everyone who can open the app.",
                    kind="names", unit="a group", default=[],
                ),
                Setting(
                    "formats", "File types", "The types of file people may upload.",
                    kind="names", unit="a file type", default=None, allow_none=True, validate=_valid_formats,
                ),
            ),
        ),
    )


def load_section(section: Section) -> dict[str, t.Any]:
    """The loader of the ``uploads:`` section: its settings, checked, as the admin view's form edits them."""
    from panel_flowdash.admin.form import settings_loader

    return settings_loader(_settings_groups())(section)


def upload_settings(project: Project | None) -> UploadSettings:
    """The ``uploads:`` section of ``project`` as it is now; uploads stay off if it cannot be read."""
    if project is None:
        return UploadSettings()
    try:
        value = project.get(SECTION) or {}
    except ProjectError as exc:
        from panel_flowdash.value_text import error_text

        logger.warning("Not using the project file's uploads settings: %s", error_text(exc))
        return UploadSettings()
    formats = value.get("formats")
    return UploadSettings(
        enabled=bool(value.get("enabled", False)),
        groups=tuple(value.get("groups") or ()),
        formats=None if formats is None else tuple(extension_of(f) for f in formats),
        max_mb=int(value.get("max_mb", DEFAULT_MAX_MB)),
    )


def _settings_view(project: Project, editable: bool) -> t.Any:
    from .uploads_admin import uploads_settings

    return uploads_settings(project, editable)


def _register() -> None:
    from .metrics import _LumenSection

    admin = {
        key: value for key, value in {"settings_view": _settings_view, "group": "Data"}.items()
        if key in getattr(ProjectSection, "__dataclass_fields__", {})
    }
    register_section((_LumenSection if admin else ProjectSection)(
        SECTION, load_section, title="Uploads", icon="upload_file",
        description="Let people add data files to the app. An upload is a source everyone who can open the app "
        "can use in dashboards and, unless turned off in the Catalog, the assistant.",
        **admin,
    ))


# ---------------------------------------------------------------------------
# Uploads


@dataclass
class Upload:
    """An uploaded file, as the manifest records it."""

    ref: str
    filename: str
    format: str
    tables: list[str]
    uploaded_by: str
    uploaded_at: str
    size: int
    sheet: str | None = None
    id: str = ""
    """Stable for the upload's life, and the name of its folder."""
    sha256: str = ""
    """The checksum of the uploaded file."""

    @property
    def when(self) -> dt.datetime | None:
        try:
            return dt.datetime.fromisoformat(self.uploaded_at)
        except ValueError:
            return None


@dataclass
class _Session:
    doc: t.Any
    loop: asyncio.AbstractEventLoop | None


def _identity_name(app: t.Any) -> str:
    return getattr(getattr(app, "_identity", None), "user", None) or "someone"


def _is_admin(app: t.Any) -> bool:
    try:
        return app is not None and bool(app._still_admin())
    except Exception:
        return False


class Uploads:
    """
    The uploads of an app: the files in ``directory``, each read by the
    handler of its type in ``formats`` into a source added to ``sources``,
    the app's sources by ref, which every new session's editor starts
    from. Sessions given to ``attach`` get each upload, and lose each
    deleted one, at once, and an Add data button in their dashboard bar.
    """

    def __init__(
        self, directory: str | os.PathLike, sources: dict[str, Source],
        formats: Mapping[str, UploadFormat] | None = None, project: Project | None = None,
        trash_days: float = TRASH_DAYS,
    ):
        self.directory = Path(directory)
        self.sources = sources
        self.formats = dict(upload_formats() if formats is None else formats)
        self.project = project
        self.trash_days = trash_days
        self._uploads: dict[str, Upload] = {}
        self._lock = threading.RLock()
        self._editors: weakref.WeakKeyDictionary[t.Any, _Session] = weakref.WeakKeyDictionary()

    # ------------------------------------------------------------------
    # What may be uploaded

    def settings(self) -> UploadSettings:
        return upload_settings(self.project)

    def allowed(self) -> dict[str, UploadFormat]:
        """The file types people may upload: the registered ones the settings allow."""
        chosen = self.settings().formats
        return {ext: f for ext, f in self.formats.items() if chosen is None or ext in chosen}

    def max_bytes(self) -> int:
        return self.settings().max_mb * 2**20

    def offered(self, app: t.Any) -> bool:
        """Whether ``app``'s session gets the Add data button: uploads are on and it is an admin or in the groups."""
        settings = self.settings()
        if not settings.enabled or app is None:
            return False
        if not settings.groups or _is_admin(app):
            return True
        identity = getattr(app, "_identity", None)
        return identity is not None and identity.in_groups(settings.groups)

    def may_upload(self, app: t.Any) -> bool:
        """Whether ``app``'s session may upload: an admin always, anyone else as ``offered`` says."""
        return _is_admin(app) or self.offered(app)

    # ------------------------------------------------------------------
    # The uploads

    def uploads(self) -> list[Upload]:
        with self._lock:
            return list(self._uploads.values())

    def get(self, ref: str) -> Upload | None:
        return self._uploads.get(ref)

    def _manifest_path(self) -> Path:
        return self.directory / MANIFEST

    def folder(self, upload: Upload) -> Path:
        """The folder holding ``upload``'s file and any files its handler derived from it."""
        return self.directory / (upload.id or upload.ref)

    def _write_manifest(self) -> None:
        self.directory.mkdir(parents=True, exist_ok=True)
        path = self._manifest_path()
        temporary = path.with_suffix(".json.tmp")
        temporary.write_text(json.dumps({"uploads": [asdict(u) for u in self._uploads.values()]}, indent=2),
                             encoding="utf-8")
        os.replace(temporary, path)

    def _read(self, upload: Upload, content: bytes | None = None) -> Source:
        """The source ``upload``'s handler reads from its stored file, named by its ref."""
        upload_format = self.formats.get(upload.format)
        if upload_format is None:
            raise UploadError(f"No upload handler reads .{upload.format} files.")
        directory = self.folder(upload)
        path = directory / upload.filename
        if content is None:
            content = path.read_bytes()
        context = {"path": path, "directory": directory, "sheet": upload.sheet, "user": upload.uploaded_by}
        source = upload_format.handler(context, io.BytesIO(content), upload.ref, upload.filename)
        if source is None:
            raise UploadError(f"{upload.filename} could not be read as {upload_format.label}.")
        with param.edit_constant(source):
            source.name = upload.ref
        return source

    def _check(self, source: Source) -> list[str]:
        """The tables of ``source``, each read once, so a file that cannot be read is refused now."""
        from .nodes import execute, listed_tables

        tables = listed_tables(source)
        if not tables:
            raise UploadError("The file holds no table.")
        if hasattr(source, "get_sql_expr"):
            from ..transforms.sql import SQLLimit

            for table in tables:
                execute(source, SQLLimit(limit=_CHECK_ROWS, write=source.dialect).apply(
                    source.get_sql_expr(source.normalize_table(table))
                ))
        return tables

    def restore(self) -> None:
        """
        Read the uploads ``manifest.json`` lists back into ``sources``; one
        that fails is logged and left out. Purges the trash.
        """
        self.purge_trash()
        path = self._manifest_path()
        if not path.is_file():
            return
        try:
            entries = json.loads(path.read_text(encoding="utf-8")).get("uploads") or []
        except (OSError, ValueError) as exc:
            logger.warning("Not reading the uploads in %s: %s", path, type(exc).__name__)
            return
        for entry in entries:
            try:
                upload = Upload(**entry)
            except TypeError:
                logger.warning("Skipping an upload %s lists that it cannot read.", path)
                continue
            if upload.ref in self.sources:
                logger.warning("Not restoring the upload %r: a served source has its name.", upload.ref)
                continue
            try:
                source = self._read(upload)
            except Exception as exc:
                # The error may quote the file, which is not logged.
                logger.warning("Could not read the upload %r (%s): %s", upload.ref, upload.filename, type(exc).__name__)
                continue
            self._uploads[upload.ref] = upload
            self.sources[upload.ref] = source

    def _ref(self, name: str) -> str:
        base = _slug(name) or "upload"
        if base[0].isdigit():
            base = f"t_{base}"
        taken = {*self.sources, *self._uploads}
        ref, n = base, 1
        while ref in taken:
            n += 1
            ref = f"{base}_{n}"
        return ref

    def refusal(self, filename: str, size: int, app: t.Any) -> str | None:
        """Why ``app``'s session may not upload ``filename`` of ``size`` bytes, or None if it may."""
        if not self.may_upload(app):
            return "Uploads are turned off for you; an admin can turn them on under Admin, Uploads."
        extension = extension_of(filename)
        allowed = self.allowed()
        if "." not in filename or extension not in allowed:
            names = ", ".join(f.label for f in allowed.values()) or "none"
            return f"{filename} is not a file type you can upload here. Allowed: {names}."
        if size > self.max_bytes():
            return f"{filename} is larger than the {self.settings().max_mb} MB uploads may have."
        return None

    def add(
        self, content: bytes, filename: str, name: str | None = None, *, app: t.Any = None,
        sheet: str | None = None,
    ) -> Upload:
        """
        Keep ``content``, uploaded as ``filename``, as a source named
        ``name`` (by default its file name), read by the handler of its
        type, and give it to every session. Raises ``UploadError`` with why
        it was refused or could not be read. Reads the file, so call it off
        the event loop.
        """
        filename = Path(str(filename).replace("\\", "/")).name
        if (refusal := self.refusal(filename, len(content), app)) is not None:
            raise UploadError(refusal)
        user = _identity_name(app)
        with self._lock:
            ref = self._ref(name or Path(filename).stem)
            upload = Upload(
                ref=ref, filename=filename, format=extension_of(filename), tables=[], uploaded_by=user,
                uploaded_at=dt.datetime.now(dt.UTC).isoformat(timespec="seconds"), size=len(content),
                sheet=sheet or None, id=uuid.uuid4().hex, sha256=hashlib.sha256(content).hexdigest(),
            )
            directory = self.folder(upload)
            directory.mkdir(parents=True)
        try:
            (directory / filename).write_bytes(content)
            source = self._read(upload, content)
            upload.tables = self._check(source)
        except Exception as exc:
            shutil.rmtree(directory, ignore_errors=True)
            if isinstance(exc, UploadError):
                raise
            from panel_flowdash.value_text import error_text, redact_secrets

            # Shown to the person who uploaded the file, but not logged, as it may quote the file.
            logger.info("Upload: %s's %s could not be read (%s).", user, filename, type(exc).__name__)
            raise UploadError(f"{filename} could not be read: {redact_secrets(error_text(exc))}") from None
        with self._lock:
            self._uploads[ref] = upload
            self._write_manifest()
            self.sources[ref] = source
            sessions = list(self._editors.items())
        logger.info("Upload: %s added %s (%d bytes) as %r with tables %s.", user, filename, len(content), ref,
                    ", ".join(upload.tables))
        for editor, session in sessions:
            self._in_session(session, partial(_bind, editor, source, ref))
        return upload

    def delete(self, ref: str, *, app: t.Any) -> list[Problem]:
        """
        Delete the upload ``ref`` for everyone: its source and its catalog
        settings. Its folder moves to the trash, with a note of who deleted
        it and when, and is removed ``trash_days`` later. Only an admin may.
        """
        if not _is_admin(app):
            return [Problem("Only admins can delete uploads, so this was not deleted.")]
        user = _identity_name(app)
        with self._lock:
            upload = self._uploads.pop(ref, None)
            if upload is None:
                return [Problem(f"There is no upload {ref!r} any more.")]
            self.sources.pop(ref, None)
            self._write_manifest()
            sessions = list(self._editors.items())
            self._to_trash(upload, user)
        self.purge_trash()
        for editor, session in sessions:
            self._in_session(session, partial(_unbind, editor, ref))
        if self.project is not None:
            self._forget_catalog(ref)
        logger.info("Upload: %s deleted the upload %r (%s).", user, ref, upload.filename)
        return []

    def _to_trash(self, upload: Upload, user: str) -> None:
        folder = self.folder(upload)
        trash = self.directory / TRASH / folder.name
        trash.parent.mkdir(parents=True, exist_ok=True)
        if folder.is_dir():
            if trash.exists():
                shutil.rmtree(trash, ignore_errors=True)
            shutil.move(str(folder), str(trash))
        else:
            trash.mkdir(parents=True, exist_ok=True)
        note = {**asdict(upload), "deleted_by": user, "deleted_at": dt.datetime.now(dt.UTC).isoformat(timespec="seconds")}
        (trash / "upload.json").write_text(json.dumps(note, indent=2), encoding="utf-8")

    def purge_trash(self, now: dt.datetime | None = None) -> list[str]:
        """Remove the deleted uploads that have been in the trash longer than ``trash_days``; returns their ids."""
        trash = self.directory / TRASH
        if not trash.is_dir():
            return []
        now = now or dt.datetime.now(dt.UTC)
        purged = []
        for folder in trash.iterdir():
            try:
                deleted = dt.datetime.fromisoformat(json.loads((folder / "upload.json").read_text())["deleted_at"])
            except (OSError, ValueError, KeyError, TypeError):
                continue
            if (now - deleted).total_seconds() > self.trash_days * 86400:
                shutil.rmtree(folder, ignore_errors=True)
                purged.append(folder.name)
        return purged

    def _trash_folder(self, upload_id: str) -> Path:
        """
        ``upload_id``'s folder directly inside the trash, as ``folder`` names one
        for a deleted upload: never a path that escapes the trash, the way
        ``add`` keeps an uploaded file's name to its last component.
        """
        name = Path(str(upload_id).replace("\\", "/")).name
        trash = self.directory / TRASH
        folder = (trash / name).resolve() if name and name not in (".", "..") else None
        if folder is None or not folder.is_relative_to(trash.resolve()):
            raise UploadError(f"There is no deleted upload {upload_id!r} in the trash.")
        return folder

    def recover(self, upload_id: str, *, app: t.Any) -> Upload:
        """Bring back the deleted upload ``upload_id`` from the trash, under a free name if its own was taken since."""
        if not _is_admin(app):
            raise UploadError("Only admins can bring back deleted uploads.")
        trash = self._trash_folder(upload_id)
        try:
            note = json.loads((trash / "upload.json").read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            raise UploadError(f"There is no deleted upload {upload_id!r} in the trash.") from exc
        fields = {key: note[key] for key in Upload.__dataclass_fields__ if key in note}
        with self._lock:
            upload = Upload(**{**fields, "ref": self._ref(fields["ref"])})
            folder = self.folder(upload)
            shutil.move(str(trash), str(folder))
            (folder / "upload.json").unlink(missing_ok=True)
            source = self._read(upload)
            self._uploads[upload.ref] = upload
            self._write_manifest()
            self.sources[upload.ref] = source
            sessions = list(self._editors.items())
        for editor, session in sessions:
            self._in_session(session, partial(_bind, editor, source, upload.ref))
        logger.info("Upload: %s brought back the upload %r (%s).", _identity_name(app), upload.ref, upload.filename)
        return upload

    def _forget_catalog(self, ref: str) -> None:
        from .catalog import SECTION as CATALOG, raw_section

        try:
            section = raw_section(self.project)
            if ref in section:
                section.pop(ref)
                self.project.set(CATALOG, section)
        except (ProjectError, OSError) as exc:
            logger.warning("Could not remove the catalog settings of the deleted upload %r: %s", ref, type(exc).__name__)

    # ------------------------------------------------------------------
    # Sessions

    @staticmethod
    def _in_session(session: _Session, callback: Callable[[], t.Any]) -> None:
        from .nodes import _schedule

        _schedule(callback, session.doc, session.loop)

    def attach(self, app: t.Any) -> UploadButton | None:
        """
        Give ``app``'s session the uploads made from now on, and an Add data
        button in its dashboard bar, shown to someone ``offered`` says may
        upload.
        """
        editor = getattr(app, "_editor", None)
        if editor is None:
            return None
        doc = pn.state.curdoc if getattr(pn.state.curdoc, "session_context", None) else None
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None
        with self._lock:
            self._editors[editor] = _Session(doc, loop)
        button = UploadButton(self, app)
        editor.toolbar_extra = [*editor.toolbar_extra, button]
        return button


def _bind(editor: t.Any, source: Source, ref: str) -> None:
    from .nodes import bind_source

    bind_source(editor, source, ref)
    _refresh_palette(editor)


def _unbind(editor: t.Any, ref: str) -> None:
    from .nodes import unbind_source

    unbind_source(editor, ref)
    _refresh_palette(editor)


def _refresh_palette(editor: t.Any) -> None:
    palette = getattr(editor, "_palette_panel", None)
    if palette is not None:
        with contextlib.suppress(Exception):
            palette.refresh()


# ---------------------------------------------------------------------------
# The dialog


def _size(size: int) -> str:
    for unit in ("bytes", "KB", "MB", "GB"):
        if size < 1024 or unit == "GB":
            return f"{size:,.0f} {unit}" if unit == "bytes" else f"{size:,.1f} {unit}"
        size /= 1024
    return str(size)


class UploadDialog(Viewer):
    """The Add data dialog: a file, the table name it gets and, for Excel, its sheet."""

    def __init__(self, uploads: Uploads, app: t.Any, on_added: Callable[[Upload], t.Any] | None = None, **params):
        super().__init__(**params)
        self.uploads = uploads
        self._app = app
        self._on_added = on_added
        self._file = pmui.FileInput(
            label="Choose a file", icon="upload_file", variant="outlined", margin=0,
            css_classes=["flowdash-upload-file"],
        )
        self._file.param.watch(self._on_file, "value")
        self._chosen = pmui.Typography("", variant="body2", margin=0, sx={"color": "text.secondary"})
        self._name = pmui.TextInput(
            label="Table name", placeholder="e.g. sales", sizing_mode="stretch_width", margin=0,
            css_classes=["flowdash-upload-name"],
        )
        self._sheet = pmui.Select(
            label="Sheet", options={"All sheets, a table each": ""}, value="", visible=False,
            sizing_mode="stretch_width", margin=0, css_classes=["flowdash-upload-sheet"],
        )
        self._note = pmui.Typography("", variant="caption", margin=0, sx={"color": "text.secondary"})
        self._alert = pmui.Alert(object="", severity="error", visible=False, sizing_mode="stretch_width", margin=0)
        self._cancel = pmui.Button(label="Cancel", variant="outlined", color="default", margin=0)
        self._cancel.on_click(lambda _event: self.close())
        self._submit = pmui.Button(
            label="Add data", icon="upload", variant="contained", margin=0, disabled=True,
            css_classes=["flowdash-upload-submit"],
        )
        self._submit.on_click(self._on_submit)
        self._dialog = pmui.Dialog(
            pn.Column(
                pn.Row(self._file, self._chosen, margin=0, styles={"align-items": "center", "gap": "12px"}),
                self._name,
                self._sheet,
                self._note,
                self._alert,
                pn.Row(pn.layout.HSpacer(), self._cancel, self._submit, margin=0, sizing_mode="stretch_width",
                       styles={"gap": "8px"}),
                margin=0, sizing_mode="stretch_width", styles={"gap": "16px", "min-width": "min(480px, 80vw)"},
                css_classes=["flowdash-upload-dialog"],
            ),
            title="Add data", open=False, show_close_button=True, close_on_click=False, width_option="sm",
            # Its content stretches, which would otherwise stretch the dashboard bar item holding it.
            sizing_mode="fixed", width=0, height=0, margin=0,
        )
        self._content: bytes | None = None
        self._filename = ""

    def _sync_limits(self):
        allowed = self.uploads.allowed()
        self._file.param.update(
            accept=",".join(f".{ext}" for ext in allowed), max_file_size=self.uploads.max_bytes(),
        )
        types = ", ".join(f.label for f in allowed.values()) or "none"
        self._note.object = (
            f"{types}, up to {self.uploads.settings().max_mb} MB. The table is shared: everyone who can open "
            "this app can use it in dashboards and, unless an admin turns it off, the assistant."
        )

    def open(self):
        """Open the dialog with no file chosen."""
        with pn.io.hold():
            self._content, self._filename = None, ""
            self._chosen.object = "No file chosen"
            self._name.value = ""
            self._sheet.param.update(options={"All sheets, a table each": ""}, value="", visible=False)
            self._alert.visible = False
            self._submit.param.update(disabled=True, loading=False)
            self._sync_limits()
            self._dialog.open = True

    def close(self):
        self._dialog.open = False
        self._content = None

    def choose(self, content: bytes, filename: str):
        """Take ``content`` as the file ``filename`` to upload, filling in its table name and, for Excel, its sheets."""
        self._content, self._filename = content, Path(filename).name
        with pn.io.hold():
            self._chosen.object = f"{self._filename}, {_size(len(content))}"
            self._name.value = _slug(Path(self._filename).stem)
            sheets = excel_sheets(content) if extension_of(self._filename) == "xlsx" else []
            self._sheet.param.update(
                options={"All sheets, a table each": "", **{s: s for s in sheets}}, value="", visible=len(sheets) > 1,
            )
            refusal = self.uploads.refusal(self._filename, len(content), self._app)
            self._alert.param.update(object=refusal or "", visible=refusal is not None)
            self._submit.disabled = refusal is not None

    def _on_file(self, event):
        if event.new is None:
            return
        self.choose(event.new, self._file.filename or "upload")
        self._file.clear()

    def submit(self) -> Upload | None:
        """Upload the chosen file; show why if it is refused. Reads the file, so prefer ``_on_submit`` on a server."""
        if self._content is None:
            return None
        try:
            upload = self.uploads.add(
                self._content, self._filename, self._name.value or None, app=self._app, sheet=self._sheet.value or None,
            )
        except UploadError as exc:
            self._alert.param.update(object=str(exc), visible=True)
            return None
        self._added(upload)
        return upload

    async def _on_submit(self, _event=None):
        if self._content is None:
            return
        self._submit.loading = True
        self._alert.visible = False
        try:
            upload = await asyncio.to_thread(
                self.uploads.add, self._content, self._filename, self._name.value or None, app=self._app,
                sheet=self._sheet.value or None,
            )
        except UploadError as exc:
            self._alert.param.update(object=str(exc), visible=True)
            return
        finally:
            self._submit.loading = False
        self._added(upload)

    def _added(self, upload: Upload):
        self.close()
        tables = ", ".join(upload.tables)
        noun = "table" if len(upload.tables) == 1 else "tables"
        if self._app is not None and hasattr(self._app, "_notify"):
            self._app._notify(
                "success", f"Added the {noun} {tables} as source {upload.ref}; find it in the component list.",
                duration=6000,
            )
        if self._on_added is not None:
            self._on_added(upload)

    def __panel__(self):
        return self._dialog


# While editing, the dashboard bar's breadcrumb needs the room of the button's label below this width.
_EDIT_LABEL_WIDTH = 1600


def _icon_only(editing: bool) -> list[str]:
    """
    The stylesheet that shows only the button's icon, with its name as a
    tooltip: on narrow screens, as the dashboard bar's Share does, and
    below ``_EDIT_LABEL_WIDTH`` while editing.
    """
    try:
        from panel_flowdash.ui.dashboard_bar import (
            ICON_ONLY_WHEN_COMPACT, icon_only_below,
        )
    except ImportError:
        return []
    return [icon_only_below(_EDIT_LABEL_WIDTH) if editing else ICON_ONLY_WHEN_COMPACT]


class UploadButton(Viewer):
    """The dashboard bar's Add data button and its dialog, for someone who may upload."""

    def __init__(self, uploads: Uploads, app: t.Any, **params):
        super().__init__(**params)
        self.uploads = uploads
        self._app = app
        self.dialog = UploadDialog(uploads, app)
        self.button = pmui.Button(
            label="Add data", icon="upload_file", variant="text", size="small", margin=(5, 2),
            description="Add data: upload a file as a table for everyone", css_classes=["flowdash-add-data"],
        )
        self.button.on_click(lambda _event: self.open())
        self._layout = pn.Row(self.button, self.dialog, margin=0, visible=False, sizing_mode="fixed")
        editor = getattr(app, "_editor", None)
        if editor is not None:
            editor.param.watch(lambda _event: self.sync(), "editable")
        self.sync()

    def sync(self):
        """Show the button if uploads are offered to this session."""
        editor = getattr(self._app, "_editor", None)
        with pn.io.hold():
            self.button.stylesheets = _icon_only(bool(editor is not None and editor.editable))
            self._layout.visible = editor is not None and self.uploads.offered(self._app)

    def open(self):
        if not self.uploads.offered(self._app):
            self.sync()
            return
        self.dialog.open()

    def __panel__(self):
        return self._layout


_register()


__all__ = [
    "ENTRY_POINT_GROUP", "SECTION", "UPLOADS_KEY", "Upload", "UploadButton", "UploadDialog", "UploadError",
    "UploadFormat", "UploadSettings", "Uploads", "excel_sheets", "extension_of", "read_file", "read_frames",
    "register_upload_handler", "upload_formats", "upload_settings",
]
