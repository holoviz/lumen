from __future__ import annotations

import pandas as pd
import param

from ....sources.duckdb import DuckDBSource


class SourceResult(param.Parameterized):
    """
    Result of a source loading operation.

    This is the return type for ``_load()`` methods in source controls.
    Use the factory methods for common patterns.
    """

    sources = param.List(default=[], doc="List of data sources loaded. Each source may contain multiple tables in source.tables.")

    table = param.String(default=None, allow_None=True, doc="Primary table name for display/default selection. All tables are in source.tables.")

    metadata = param.Dict(default={}, doc="Additional metadata about the loaded data.")

    message = param.String(default=None, allow_None=True, doc="Status message to display.")

    document_only = param.Boolean(default=False, doc="True if only a document was indexed (no tables).")

    @classmethod
    def from_dataframe(
        cls,
        df: pd.DataFrame,
        table_name: str,
        message: str | None = None,
        **metadata,
    ) -> SourceResult:
        """Create a result wrapping a DataFrame in an ephemeral DuckDBSource."""
        source = DuckDBSource.from_df(tables={table_name: df})
        source.tables[table_name] = f"SELECT * FROM {table_name}"
        if metadata:
            source.metadata = {table_name: metadata}
        return cls(
            sources=[source],
            table=table_name,
            metadata=metadata,
            message=message or f"Loaded {len(df):,} rows into '{table_name}'",
        )

    @classmethod
    def from_source(
        cls,
        source: DuckDBSource,
        table: str | None = None,
        message: str | None = None,
    ) -> SourceResult:
        """Create a result from an existing source.

        All tables in source.tables are available; `table` is the primary/default.
        """
        return cls(sources=[source], table=table, message=message)

    @classmethod
    def empty(cls, message: str = "No data loaded") -> SourceResult:
        """Return an empty result with an optional message."""
        return cls(message=message)

    @classmethod
    def from_document(cls, message: str) -> SourceResult:
        """Return a result indicating only a document was indexed (no tables)."""
        return cls(message=message, document_only=True)

    def __str__(self) -> str:
        """Return the message for LLM tool responses."""
        return self.message or "No data loaded"

    def __repr__(self) -> str:
        return self.__str__()
