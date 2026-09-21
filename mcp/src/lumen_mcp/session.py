"""In-memory session state.

The session owns a single long-lived Lumen ``DuckDBSource`` (the "workspace"). Every SQL result is
materialized into it as a real table, so results accrete in one connection and are referenced by
table name. It also keeps a registry of rendered charts.
"""

from __future__ import annotations

from typing import Any, Optional

from lumen.sources.duckdb import DuckDBSource


class Session:
    def __init__(self) -> None:
        self.source: Optional[DuckDBSource] = None
        self.charts: dict[str, dict[str, Any]] = {}
        self._result_n = 0
        self._chart_n = 0

    def require_source(self) -> DuckDBSource:
        if self.source is None:
            raise RuntimeError("No source connected. Call connect_source first.")
        return self.source

    def next_result_name(self) -> str:
        self._result_n += 1
        return f"result_{self._result_n}"

    def next_chart_id(self) -> str:
        self._chart_n += 1
        return f"chart_{self._chart_n}"


# Process-wide singleton shared by all tools.
session = Session()
