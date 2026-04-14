from __future__ import annotations

import io
import json
import pathlib

import pandas as pd
import param

from ....sources.duckdb import DuckDBSource
from ...translate import params_to_callable
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

    url_template = param.String(default="", precedence=-1, doc="""
        URL template with ``{param_name}`` placeholders matching query param names.""")

    label = '<span class="material-icons" style="vertical-align: middle;">link</span> URL Data Source'

    # ──────────────────────────────────────────────────────────────────────────
    # Agent integration
    # ──────────────────────────────────────────────────────────────────────────

    def as_tools(
        self, query: str | None = None, top_k: int = 5,
    ) -> list[tuple[str, callable]]:
        """Build a synthetic callable from class-level params for the agent.

        Unlike action-based controls (``CodeSourceControls``,
        ``RESTAPISourceControls``) which register explicit callables,
        ``URLSourceControls`` subclasses declare their inputs as
        class-level ``param`` attributes.  This override synthesizes a
        single callable whose signature and annotations mirror those
        params so ``FunctionTool`` can generate a schema for the LLM.

        Example::

            class PopulationControls(URLSourceControls):
                url_template = "https://api.example.com/data?region={region}&year={year}"
                region = param.Selector(default="us", objects=["us", "eu", "apac"])
                year = param.Integer(default=2024, bounds=(2000, 2030))

        ``as_tools()`` returns
        ``[("Population", <async callable(region, year)>)]`` so the
        agent sees a single tool named ``"Population"`` with typed
        ``region`` and ``year`` parameters.
        """
        if self._cached_tools is not None:
            return self._cached_tools

        query_names = self._get_query_param_names()
        if not query_names:
            self._cached_tools = []
            return self._cached_tools

        action_name = self.__class__.__name__.removesuffix("Controls")

        async def _tool_callable(**kwargs) -> SourceResult:
            return await self.load_action(action_name, **kwargs)

        query_params = {name: self.param[name] for name in query_names}
        doc = (self.__doc__ or f"Fetch data using {self.__class__.__name__}.").strip()
        params_to_callable(
            _tool_callable, query_params,
            name=action_name.lower() or "load_data",
            doc=doc,
        )

        self._cached_tools = [(action_name, _tool_callable)]
        return self._cached_tools

    # ──────────────────────────────────────────────────────────────────────────
    # Data fetching
    # ──────────────────────────────────────────────────────────────────────────

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
        # Detect API error responses before parsing
        try:
            text_preview = content[:500].decode("utf-8", errors="replace").strip()
        except Exception:
            text_preview = ""
        if text_preview.upper().startswith("ERROR"):
            return SourceResult.empty(f"API error: {text_preview[:200]}")

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

        if df is None:
            return SourceResult.empty(f"Could not parse {filename!r}.")
        if df.empty:
            return SourceResult.empty(
                "The API returned no data rows. The request may have "
                "returned headers only (e.g. today's data not yet available)."
            )

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
