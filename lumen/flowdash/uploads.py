"""
Data files people upload to a Lumen FlowDash app.

An admin turns uploads on in the admin view's Uploads section, which
writes the project file's ``uploads:`` section:

    uploads:
      enabled: true
      groups: [analysts]       # who may upload; empty or left out: everyone who can open the app
      formats: [csv, xlsx]     # left out: every registered file type
      max_mb: 100

People who may upload then get an Add data button in the dashboard bar
and in the Data section's tables (or the admin view's Catalog, for admins,
where there is no Data section). An uploaded file becomes a source
of its own, named by the table name the uploader picks, that every
session's dashboards, component list and assistant can use, and that the
catalog lists with who uploaded it and when; the catalog's assistant
switch applies to its tables like to any other.

The files are kept beside the project file, in an ``uploads`` folder: each
upload in a folder named by its stable id, and ``manifest.json`` listing
them with who uploaded each, when, and its checksum. They are read again
when the app starts. Deleting an upload, which its uploader or an admin
may, moves its folder to ``uploads/.trash``, where it stays for
``TRASH_DAYS`` days, so a backup taken shortly before stays consistent and
``Uploads.recover`` can bring it back; only an admin can recover one.

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
import tempfile
import threading
import typing as t
import uuid
import weakref

from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from functools import partial
from importlib.util import find_spec
from pathlib import Path

import panel as pn
import param

from ..sources.base import Source
from .access import is_admin
from .project import (
    Problem, Project, ProjectError, ProjectSection, Section, register_section,
)

if t.TYPE_CHECKING:
    from .metadata import DescriptionStore

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
                    "An Add data button in the dashboard bar and with the app's tables. Admins can always add data.",
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
        "can use in dashboards and, unless its table is turned off for it, the assistant.",
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
    uploaded_at: str
    size: int
    sheet: str | None = None
    id: str = ""
    """Stable for the upload's life, and the name of its folder."""
    sha256: str = ""
    """The checksum of the uploaded file."""
    uploaded_by: str = ""
    """
    Who uploaded it, by the app's identity name; empty for a record kept
    from before this was recorded, which only an admin may then delete.
    """

    @property
    def when(self) -> dt.datetime | None:
        try:
            return dt.datetime.fromisoformat(self.uploaded_at)
        except ValueError:
            return None


@dataclass
class Connection:
    """
    A source an admin connected in the app: a database, a database file or
    a data file, by its location. The manifest never holds its credentials:
    a location with a password or token keeps only its masked form, and the
    whole location goes in ``secret``, encrypted as panel-flowdash stores a
    model provider's key (see ``panel_flowdash.secret_store``); a password
    written as a ``${VARIABLE}`` reference is kept as written and read from
    the server's environment when it connects.
    """

    ref: str
    location: str
    """What ``lumen serve`` would be given for it, a path, a URL or a connection string, credentials masked."""
    added_by: str
    added_at: str
    id: str = ""
    secret: dict[str, str] | None = None
    """The location with its credentials, encrypted, when it has any written out."""

    @property
    def when(self) -> dt.datetime | None:
        try:
            return dt.datetime.fromisoformat(self.added_at)
        except ValueError:
            return None

    @property
    def shown(self) -> str:
        """The location to show: masked, with any ``${VARIABLE}`` reference as written."""
        from panel_flowdash.value_text import redact_secrets

        return self.location if _REFERENCE.search(self.location) else redact_secrets(self.location)


# A reference to an environment variable in a location, e.g. the password of postgresql://ana:${PG_PASSWORD}@host/db.
_REFERENCE = re.compile(r"\$\{[A-Za-z_][A-Za-z0-9_]*(?::-[^}]*)?\}")


def stored_location(location: str) -> tuple[str, dict[str, str] | None]:
    """
    What the manifest keeps of ``location``: the location itself when it
    holds no credentials, or only ``${VARIABLE}`` references to them; else
    its masked form and the whole location encrypted. Raises
    ``UploadError`` when it holds credentials and nothing here can encrypt
    them.
    """
    from panel_flowdash.secret_store import (
        INSTALL_HINT, SecretError, encrypt_secret, encryption_available,
    )
    from panel_flowdash.value_text import redact_secrets

    masked = redact_secrets(location)
    if masked == location:
        return location, None
    # With each reference standing where a secret would, only a secret written out is left to mask.
    referenced = _REFERENCE.sub("***", location)
    if redact_secrets(referenced) == referenced:
        return location, None
    if not encryption_available():
        raise UploadError(
            f"{INSTALL_HINT} Or write the password as a reference to an environment variable the server "
            "has, e.g. postgresql://user:${PG_PASSWORD}@host/db."
        )
    try:
        return masked, encrypt_secret(location)
    except SecretError as exc:
        raise UploadError(f"Could not store its credentials: {exc}") from None


def resolved_location(connection: Connection) -> str:
    """
    The location ``connection`` connects to, with its credentials:
    decrypted, and its references read from the environment. Raises
    ``UploadError`` saying why they cannot be read, never what they are.
    """
    from panel_flowdash.project import expand_env
    from panel_flowdash.secret_store import SecretError, resolve_secret

    try:
        location = resolve_secret(connection.secret) if connection.secret else connection.location
        return expand_env(location)
    except (SecretError, ValueError) as exc:
        raise UploadError(f"Its credentials can't be read, so it is not connected. {exc}") from None


@dataclass
class TablePreview:
    """A table of a file or source not added yet: its columns with their types and first values."""

    table: str
    columns: list[tuple[str, str, str]] = field(default_factory=list)
    """Each column's name, type and first few values, as text."""


@dataclass
class Preview:
    """What adding a file or a source would add: the source's ref and its tables, the first ones previewed."""

    ref: str
    tables: list[str]
    previews: list[TablePreview] = field(default_factory=list)


# The rows a preview reads of a table, and the tables it previews at most.
PREVIEW_ROWS = 4
PREVIEW_TABLES = 3


def _kind(dtype: t.Any) -> str:
    """A column's pandas ``dtype`` as the catalog names types."""
    from pandas.api import types

    if types.is_bool_dtype(dtype):
        return "boolean"
    if types.is_integer_dtype(dtype):
        return "integer"
    if types.is_float_dtype(dtype) or types.is_numeric_dtype(dtype):
        return "number"
    if types.is_datetime64_any_dtype(dtype):
        return "datetime"
    return "string"


def table_preview(source: Source, table: str, rows: int = PREVIEW_ROWS) -> TablePreview:
    """The columns of ``table`` with their types and first ``rows`` values; reads them, so run it off the event loop."""
    import datetime as _dt

    import pandas as pd

    from .catalog import plain_text, sample_sql
    from .nodes import execute

    if hasattr(source, "get_sql_expr"):
        frame = execute(source, sample_sql(source, table, rows))
    else:
        frame = source.get(table).head(rows)
    preview = TablePreview(table)
    for column in frame.columns:
        values = frame[column]
        kind = _kind(values.dtype)
        items = values.tolist()
        if kind == "string" and len(values.dropna()) and values.map(lambda v: isinstance(v, _dt.date)).all():
            kind = "date"
        elif kind == "datetime" and all(pd.isna(v) or v == v.normalize() for v in items):
            # Days read from a spreadsheet arrive as midnights.
            kind, items = "date", [None if pd.isna(v) else v.date() for v in items]
        shown = ", ".join(plain_text(v) for v in items if plain_text(v))
        preview.columns.append((str(column), kind, shown))
    return preview


@dataclass
class _Session:
    doc: t.Any
    loop: asyncio.AbstractEventLoop | None


def _identity_name(app: t.Any) -> str:
    return getattr(getattr(app, "_identity", None), "user", None) or "someone"


def _has_identity(app: t.Any) -> bool:
    """Whether app's session carries a real identity, as opposed to the "someone" every identity-less session shares."""
    return bool(getattr(getattr(app, "_identity", None), "user", None))


def _is_uploader(app: t.Any, upload: Upload) -> bool:
    """Whether app's session is the one who made upload; a record kept before uploaders were recorded is no one's,
    and neither is one made or checked by a session without a real identity, since every such session would
    otherwise match under the shared "someone" placeholder."""
    return bool(upload.uploaded_by) and _has_identity(app) and _identity_name(app) == upload.uploaded_by


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
        self._connections: dict[str, Connection] = {}
        # Why each connection that is listed but not served could not connect, e.g. its credentials are unreadable.
        self._broken: dict[str, str] = {}
        self._descriptions: DescriptionStore | None = None
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
        if not settings.groups or is_admin(app):
            return True
        identity = getattr(app, "_identity", None)
        return identity is not None and identity.in_groups(settings.groups)

    def may_upload(self, app: t.Any) -> bool:
        """Whether ``app``'s session may upload: an admin always, anyone else as ``offered`` says."""
        return is_admin(app) or self.offered(app)

    def may_delete(self, upload: Upload | None, app: t.Any) -> bool:
        """Whether ``app``'s session may delete ``upload``: an admin always, or whoever uploaded it, themself."""
        return upload is not None and (is_admin(app) or _is_uploader(app, upload))

    # ------------------------------------------------------------------
    # The uploads

    def uploads(self) -> list[Upload]:
        with self._lock:
            return list(self._uploads.values())

    def get(self, ref: str) -> Upload | None:
        return self._uploads.get(ref)

    def connections(self) -> list[Connection]:
        with self._lock:
            return list(self._connections.values())

    def connection(self, ref: str) -> Connection | None:
        """The source an admin connected as ``ref``, if it is one."""
        return self._connections.get(ref)

    def connection_problem(self, ref: str) -> str | None:
        """Why the connected source ``ref`` is listed but not served, e.g. its credentials can't be read."""
        return self._broken.get(ref)

    def _manifest_path(self) -> Path:
        return self.directory / MANIFEST

    def folder(self, upload: Upload) -> Path:
        """The folder holding ``upload``'s file and any files its handler derived from it."""
        return self.directory / (upload.id or upload.ref)

    def _write_manifest(self) -> None:
        self.directory.mkdir(parents=True, exist_ok=True)
        path = self._manifest_path()
        temporary = path.with_suffix(".json.tmp")
        manifest = {
            "uploads": [asdict(u) for u in self._uploads.values()],
            "sources": [asdict(c) for c in self._connections.values()],
        }
        temporary.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        os.replace(temporary, path)

    def _read(self, upload: Upload, content: bytes | None = None, directory: Path | None = None) -> Source:
        """The source ``upload``'s handler reads from its stored file, named by its ref."""
        upload_format = self.formats.get(upload.format)
        if upload_format is None:
            raise UploadError(f"No upload handler reads .{upload.format} files.")
        directory = self.folder(upload) if directory is None else directory
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
            manifest = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            logger.warning("Not reading the uploads in %s: %s", path, type(exc).__name__)
            return
        entries = manifest.get("uploads") or []
        for entry in manifest.get("sources") or []:
            try:
                connection = Connection(**entry)
            except TypeError:
                logger.warning("Skipping a source %s lists that it cannot read.", path)
                continue
            if connection.ref in self.sources:
                logger.warning("Not restoring the source %r: a served source has its name.", connection.ref)
                continue
            # A source that cannot connect stays listed, so an admin can enter its credentials again or remove it.
            self._connections[connection.ref] = connection
            try:
                source = self._open(resolved_location(connection), connection.ref)
            except UploadError as exc:
                self._broken[connection.ref] = str(exc)
                logger.warning("Not connecting the source %r: %s", connection.ref, exc)
                continue
            except Exception as exc:
                # The error may quote the connection's credentials, which are not logged.
                self._broken[connection.ref] = f"Could not connect to it ({type(exc).__name__})."
                logger.warning("Could not connect the source %r: %s", connection.ref, type(exc).__name__)
                continue
            self.sources[connection.ref] = source
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

    def _ref(self, name: str, free: t.Collection[str] = ()) -> str:
        """A ref for ``name`` no source, upload or connection has, except those in ``free``."""
        base = _slug(name) or "upload"
        if base[0].isdigit():
            base = f"t_{base}"
        taken = {*self.sources, *self._uploads, *self._connections} - set(free)
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

    def preview(
        self, content: bytes, filename: str, name: str | None = None, *, app: t.Any = None,
        sheet: str | None = None,
    ) -> Preview:
        """
        What ``add`` would add for the same arguments: the ref, the tables
        and the first rows of the first tables, read by the handler of the
        file's type in a scratch folder that is removed again. Raises
        ``UploadError`` like ``add``. Reads the file, so call it off the
        event loop.
        """
        from .nodes import listed_tables

        filename = Path(str(filename).replace("\\", "/")).name
        if (refusal := self.refusal(filename, len(content), app)) is not None:
            raise UploadError(refusal)
        ref = self._ref(name or Path(filename).stem)
        upload = Upload(
            ref=ref, filename=filename, format=extension_of(filename), tables=[], uploaded_by=_identity_name(app),
            uploaded_at="", size=len(content), sheet=sheet or None,
        )
        with tempfile.TemporaryDirectory(prefix="lumen-preview-") as scratch:
            directory = Path(scratch)
            (directory / filename).write_bytes(content)
            try:
                source = self._read(upload, content, directory)
                tables = listed_tables(source)
                previews = [table_preview(source, table) for table in tables[:PREVIEW_TABLES]]
            except UploadError:
                raise
            except Exception as exc:
                from panel_flowdash.value_text import (
                    error_text, redact_secrets,
                )

                raise UploadError(f"{filename} could not be read: {redact_secrets(error_text(exc))}") from None
        if not tables:
            raise UploadError("The file holds no table.")
        return Preview(ref, tables, previews)

    def add(
        self, content: bytes, filename: str, name: str | None = None, *, app: t.Any = None,
        sheet: str | None = None, assistant: bool = True,
    ) -> Upload:
        """
        Keep ``content``, uploaded as ``filename``, as a source named
        ``name`` (by default its file name), read by the handler of its
        type, and give it to every session; with ``assistant`` off its
        tables are turned off for the assistant, which the person uploading
        may do for their own upload. Raises ``UploadError`` with why it was
        refused or could not be read. Reads the file, so call it off the
        event loop.
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
        self._restore_descriptions(ref, source)
        logger.info("Upload: %s added %s (%d bytes) as %r with tables %s.", user, filename, len(content), ref,
                    ", ".join(upload.tables))
        if not assistant and self.project is not None:
            self._hide(ref, upload.tables)
        for editor, session in sessions:
            self._in_session(session, partial(_bind, editor, source, ref))
        return upload

    def _hide(self, ref: str, tables: Sequence[str]) -> None:
        """Turn ``tables`` of the new source ``ref`` off for the assistant in the project file's ``catalog:``."""
        from panel_flowdash.project import _FILE_LOCK

        from .catalog import SECTION as CATALOG, raw_section

        try:
            with _FILE_LOCK:
                section = raw_section(self.project)
                section[ref] = {"tables": {table: {"assistant": False} for table in tables}}
                self.project.set(CATALOG, section)
        except (ProjectError, OSError) as exc:
            logger.warning("Could not turn the new source %r off for the assistant: %s", ref, type(exc).__name__)

    # ------------------------------------------------------------------
    # Connected sources

    def _open(self, location: str, ref: str) -> Source:
        """The one source ``location`` names, read as ``lumen serve`` reads its data, named ``ref``."""
        from .app import resolve_sources

        location = location.strip()
        if not location:
            raise UploadError("Give a file, a URL or a connection string.")
        if _REFERENCE.search(location):
            from panel_flowdash.project import expand_env

            try:
                location = expand_env(location)
            except ValueError as exc:
                raise UploadError(str(exc)) from None
        sources = resolve_sources([location])
        if len(sources) != 1:
            raise UploadError(f"That gives {len(sources)} sources; connect one at a time.")
        (source,) = sources
        with param.edit_constant(source):
            source.name = ref
        return source

    def read_location(self, location: str, name: str | None = None, *, app: t.Any) -> tuple[Preview, Source]:
        """
        The source ``location`` names, as ``connect`` would add it, and a
        preview of it; only for an admin. Raises ``UploadError`` with why it
        cannot be read. Queries the source, so call it off the event loop.
        """
        from .nodes import listed_tables

        if not is_admin(app):
            raise UploadError("Only admins can connect sources.")
        # Backslash joins the URL/path separators so a Windows local path's drive
        # letter and directories are not swept into the stem with it.
        stem = re.split(r"[/\\:?@]", location.strip().rstrip("/"))[-1].rsplit(".", 1)[0] if location else ""
        # A source listed without its credentials is replaced by connecting it again under its name.
        ref = self._ref(name or stem or "source", free=self._broken)
        try:
            source = self._open(location, ref)
            tables = listed_tables(source)
            previews = [table_preview(source, table) for table in tables[:PREVIEW_TABLES]]
        except UploadError:
            raise
        except Exception as exc:
            from panel_flowdash.value_text import error_text, redact_secrets

            raise UploadError(f"Could not read it: {redact_secrets(error_text(exc))}") from None
        if not tables:
            raise UploadError("It has no tables.")
        return Preview(ref, tables, previews), source

    def connect(
        self, location: str, name: str | None = None, *, app: t.Any, assistant: bool = True,
        source: Source | None = None,
    ) -> Connection:
        """
        Add the source ``location`` names as ``name``, for every session,
        and keep it in the manifest so it is connected again when the app
        starts; ``source`` is the one ``read_location`` read, if it did.
        Only an admin may. Raises ``UploadError`` with why it was refused or
        could not be read.
        """
        if not is_admin(app):
            raise UploadError("Only admins can connect sources.")
        location = location.strip()
        if "://" not in location and Path(location).exists():
            # A path the app was given is kept whole, so a restart elsewhere still finds it.
            location = str(Path(location).resolve())
        kept, secret = stored_location(location)
        if source is None:
            _, source = self.read_location(location, name, app=app)
        user = _identity_name(app)
        with self._lock:
            ref = self._ref(source.name, free=self._broken)
            with param.edit_constant(source):
                source.name = ref
            connection = Connection(
                ref=ref, location=kept, added_by=user,
                added_at=dt.datetime.now(dt.UTC).isoformat(timespec="seconds"), id=uuid.uuid4().hex, secret=secret,
            )
            self._broken.pop(ref, None)
            self._connections[ref] = connection
            self._write_manifest()
            self.sources[ref] = source
            sessions = list(self._editors.items())
        self._restore_descriptions(ref, source)
        logger.info("Admin: %s connected the source %r.", user, ref)
        if not assistant and self.project is not None:
            from .nodes import listed_tables

            self._hide(ref, listed_tables(source))
        for editor, session in sessions:
            self._in_session(session, partial(_bind, editor, source, ref))
        return connection

    def disconnect(self, ref: str, *, app: t.Any) -> list[Problem]:
        """Remove the source an admin connected as ``ref``, for everyone, and its catalog settings. Only an admin may."""
        if not is_admin(app):
            return [Problem("Only admins can remove sources, so this was not removed.")]
        with self._lock:
            connection = self._connections.pop(ref, None)
            if connection is None:
                return [Problem(f"There is no connected source {ref!r} any more.")]
            self._broken.pop(ref, None)
            self.sources.pop(ref, None)
            self._write_manifest()
            sessions = list(self._editors.items())
        for editor, session in sessions:
            self._in_session(session, partial(_unbind, editor, ref))
        if self.project is not None:
            self._forget_catalog(ref)
        logger.info("Admin: %s removed the source %r.", _identity_name(app), ref)
        return []

    def delete(self, ref: str, *, app: t.Any) -> list[Problem]:
        """
        Delete the upload ``ref`` for everyone: its source and its catalog
        settings. Its folder moves to the trash, with a note of who deleted
        it and when, and is removed ``trash_days`` later. Only an admin, or
        whoever uploaded it, may.
        """
        with self._lock:
            upload = self._uploads.get(ref)
        if upload is None:
            return [Problem(f"There is no upload {ref!r} any more.")]
        if not self.may_delete(upload, app):
            return [Problem("Only admins or whoever uploaded it can delete it, so this was not deleted.")]
        user = _identity_name(app)
        with self._lock:
            upload = self._uploads.pop(ref, None)
            if upload is None:
                return [Problem(f"There is no upload {ref!r} any more.")]
            self.sources.pop(ref, None)
            self._write_manifest()
            sessions = list(self._editors.items())
            self._to_trash(upload, user, self._pop_descriptions(ref))
        self.purge_trash()
        for editor, session in sessions:
            self._in_session(session, partial(_unbind, editor, ref))
        if self.project is not None:
            self._forget_catalog(ref)
        logger.info("Upload: %s deleted the upload %r (%s).", user, ref, upload.filename)
        return []

    def _to_trash(self, upload: Upload, user: str, descriptions: Mapping[str, t.Any] | None = None) -> None:
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
        if descriptions:
            note["descriptions"] = dict(descriptions)
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
        if not is_admin(app):
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
        if isinstance(descriptions := note.get("descriptions"), Mapping) and descriptions:
            self.descriptions.put(upload.ref, descriptions)
        self._restore_descriptions(upload.ref, source)
        for editor, session in sessions:
            self._in_session(session, partial(_bind, editor, source, upload.ref))
        logger.info("Upload: %s brought back the upload %r (%s).", _identity_name(app), upload.ref, upload.filename)
        return upload

    @property
    def descriptions(self) -> DescriptionStore:
        """The table and column descriptions admins saved, kept beside the manifest."""
        if self._descriptions is None:
            from .metadata import DESCRIPTIONS_FILE, DescriptionStore

            self._descriptions = DescriptionStore(self.directory / DESCRIPTIONS_FILE)
        return self._descriptions

    def _restore_descriptions(self, ref: str, source: Source) -> None:
        """Apply the saved descriptions of ``ref`` to its new source, e.g. a source connected again."""
        try:
            self.descriptions.restore({ref: source})
        except Exception as exc:
            logger.warning("Could not apply the saved descriptions of %r: %s", ref, type(exc).__name__)

    def _pop_descriptions(self, ref: str) -> dict[str, t.Any]:
        """Remove the saved descriptions of the deleted upload ``ref``, so a new upload under its name starts without them."""
        try:
            return self.descriptions.pop(ref)
        except OSError as exc:
            logger.warning("Could not remove the saved descriptions of %r: %s", ref, type(exc).__name__)
            return {}

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

    def attach(self, app: t.Any) -> t.Any:
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
        from .add_data import UploadButton

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


_register()


__all__ = [
    "ENTRY_POINT_GROUP", "SECTION", "UPLOADS_KEY", "Connection", "Preview", "TablePreview", "Upload",
    "UploadError", "UploadFormat", "UploadSettings", "Uploads", "excel_sheets", "extension_of", "read_file",
    "read_frames", "register_upload_handler", "resolved_location", "stored_location", "table_preview",
    "upload_formats", "upload_settings",
]
