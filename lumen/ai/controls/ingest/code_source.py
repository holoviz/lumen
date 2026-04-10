from __future__ import annotations

import asyncio

import pandas as pd
import param

from ....sources.base import Source
from .parametric import ParametricSourceControls
from .result import SourceResult

# ─────────────────────────────────────────────────────────────────────────────
# CODE SOURCE CONTROLS
# ─────────────────────────────────────────────────────────────────────────────

class CodeSourceControls(ParametricSourceControls):
    """
    Parametric controls where parameter values are passed to a Python callable
    that returns a ``DataFrame`` or a ``Source``.

    Example
    -------
    ::

        def fetch_sales(region: str, year: int) -> pd.DataFrame:
            ...

        controls = CodeSourceControls(fetch_func=fetch_sales, table_name="sales")

        # or as a subclass with declared params:
        class SalesControls(CodeSourceControls):
            fetch_func = fetch_sales
            table_name = "sales"
            region = param.Selector(default="us", objects=["us", "eu"])
            year = param.Integer(default=2024, bounds=(2000, 2030))

    The callable is run in a thread via ``asyncio.to_thread`` so it may be
    synchronous (blocking I/O, heavy computation) without stalling the event loop.
    """

    fetch_func = param.Callable(default=None, doc="""
        Callable with signature ``(**params) -> pd.DataFrame | Source``.
        Called with the current values of all query params.""")

    table_name = param.String(default="data", doc="""
        Table name used when wrapping a DataFrame result in a DuckDBSource.""")

    label = '<span class="material-icons" style="vertical-align: middle;">code</span> Custom Data Source'

    async def _fetch_data(self, **params) -> SourceResult:
        if self.fetch_func is None:
            return SourceResult.empty("No fetch_func provided.")

        self.progress("Fetching data…")

        try:
            result = await asyncio.to_thread(self.fetch_func, **params)
        except Exception as e:
            return SourceResult.empty(f"fetch_func raised an error: {e}")

        if isinstance(result, pd.DataFrame):
            return SourceResult.from_dataframe(result, self.table_name)
        elif isinstance(result, Source):
            return SourceResult.from_source(result)
        else:
            return SourceResult.empty(
                f"fetch_func must return a DataFrame or Source, got {type(result).__name__}"
            )
