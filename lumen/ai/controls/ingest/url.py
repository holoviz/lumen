from __future__ import annotations

import io
import json
import pathlib

import pandas as pd
import param

from ....sources.duckdb import DuckDBSource
from .file_row import UploadedFileRow
from .parametric import ParametricSourceControls
from .result import SourceResult
from .utils import download_file

# ─────────────────────────────────────────────────────────────────────────────
# URL SOURCE CONTROLS
# ─────────────────────────────────────────────────────────────────────────────


class URLSourceControls(ParametricSourceControls):
    """
    Parametric controls where parameter values are interpolated into a URL
    template, the result downloaded, and processed as a file.

    Example
    -------
    ::

        class PopulationControls(URLSourceControls):
            url_template = "https://api.example.com/data?region={region}&year={year}"
            region = param.Selector(default="us", objects=["us", "eu", "apac"])
            year = param.Integer(default=2024, bounds=(2000, 2030))

    The ``url_template`` string uses Python ``str.format`` substitution, so
    ``{region}`` is replaced by the current value of the ``region`` param.
    """

    url_template = param.String(default="", doc="""
        URL template with ``{param_name}`` placeholders matching query param names.""")

    label = '<span class="material-icons" style="vertical-align: middle;">link</span> URL Data Source'

    async def _fetch_data(self, action_name: str, **params) -> SourceResult:
        try:
            url = self.url_template.format(**params)
        except KeyError as e:
            return SourceResult.empty(f"URL template error — missing param: {e}")

        self.progress(f"Fetching {url[:80]}{'…' if len(url) > 80 else ''}…")
        filename, content, error = await download_file(url, progress=self.progress)

        if error:
            return SourceResult.empty(f"Download failed: {error}")

        return await self._content_to_result(filename, content)

    async def _content_to_result(self, filename: str, content: bytes) -> SourceResult:
        """
        Convert downloaded bytes to a ``SourceResult``.

        Creates a temporary ``UploadedFileRow`` for alias/extension metadata,
        then reads the content into a ``DuckDBSource``. Override for custom parsing.
        """
        suffix = pathlib.Path(filename).suffix.lstrip(".").lower()
        file_obj = io.BytesIO(content)
        card = UploadedFileRow(file_obj=file_obj, filename=filename)

        source_id = f"{self.source_name_prefix}{self._count:06d}"
        source = DuckDBSource(uri=":memory:", ephemeral=True, name=source_id, tables={})
        self._count += 1

        try:
            file_obj.seek(0)
            df = self._read_content(file_obj, suffix, card)
        except Exception as e:
            return SourceResult.empty(f"Could not parse {filename!r}: {e}")

        if df is None or df.empty:
            return SourceResult.empty(f"{filename!r} contains no data.")

        table = card.alias
        source._connection.from_df(df).to_view(table)
        source.tables[table] = f"SELECT * FROM {table}"
        source.metadata[table] = {"filename": filename}

        return SourceResult.from_source(
            source,
            table=table,
            message=f"Loaded {len(df):,} rows from '{filename}' into '{table}'",
        )

    def _read_content(self, file_obj, suffix: str, card) -> pd.DataFrame:
        """Parse file bytes into a DataFrame based on file extension."""
        if suffix == "csv":
            return pd.read_csv(file_obj, parse_dates=True, sep=None, engine="python")
        if suffix in ("parq", "parquet"):
            return pd.read_parquet(file_obj)
        if suffix == "json":
            return self._read_json_content(file_obj)
        if suffix == "xlsx":
            return pd.read_excel(file_obj, sheet_name=card.sheet)
        raise ValueError(f"Unsupported file extension: {suffix!r}")

    @staticmethod
    def _read_json_content(file_obj) -> pd.DataFrame:
        """Parse JSON bytes into a DataFrame."""
        content = file_obj.read()
        if isinstance(content, bytes):
            content = content.decode("utf-8")
        data = json.loads(content)

        if isinstance(data, list):
            return pd.json_normalize(data)
        if isinstance(data, dict):
            for key in ("data", "records", "rows", "items", "results"):
                if key in data and isinstance(data[key], list):
                    return pd.json_normalize(data[key])
            return pd.DataFrame([data])
        raise ValueError(f"Unsupported JSON root type: {type(data).__name__}")
