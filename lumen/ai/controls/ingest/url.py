from __future__ import annotations

import io
import pathlib

import pandas as pd
import param

from ....sources.duckdb import DuckDBSource
from ...translate import params_to_callable
from .file_row import UploadedFileRow
from .parametric import ParametricSourceControls
from .result import SourceResult
from .utils import download_file, read_file_to_dataframe, read_html_tables

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
            dfs = self._read_content(file_obj, suffix, card)
        except Exception as e:
            return SourceResult.empty(f"Could not parse {filename!r}: {e}")

        if not dfs:
            return SourceResult.empty(f"Could not parse {filename!r}.")

        # dfs is a dict of {table_name: DataFrame}
        total_rows = 0
        tables_loaded = []
        for table_name, df in dfs.items():
            if df.empty:
                continue
            source._connection.from_df(df).to_view(table_name)
            source.tables[table_name] = f"SELECT * FROM {table_name}"
            source.metadata[table_name] = {"filename": filename}
            total_rows += len(df)
            tables_loaded.append(f"'{table_name}' ({len(df):,} rows)")

        if not tables_loaded:
            return SourceResult.empty(
                "The API returned no data rows. The request may have "
                "returned headers only (e.g. today's data not yet available)."
            )

        first_table = next(iter(source.tables))
        if len(tables_loaded) == 1:
            message = f"Loaded {total_rows:,} rows from '{filename}' into '{first_table}'"
        else:
            message = f"Loaded {len(tables_loaded)} tables from '{filename}': {', '.join(tables_loaded)}"

        # All tables are in source.tables; first_table is for default selection
        return SourceResult.from_source(
            source,
            table=first_table,
            message=message,
        )

    def _read_content(self, file_obj, suffix: str, card) -> dict[str, pd.DataFrame]:
        """Parse file bytes into a dict of {table_name: DataFrame}."""
        alias = card.alias
        df = read_file_to_dataframe(file_obj, suffix, sheet=card.sheet)
        if df is not None:
            return {alias: df}
        return read_html_tables(file_obj.read(), alias)
