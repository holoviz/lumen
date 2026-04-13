from __future__ import annotations

# ─────────────────────────────────────────────────────────────────────────────
# File type constants
# ─────────────────────────────────────────────────────────────────────────────

TABLE_EXTENSIONS = ("csv", "parquet", "parq", "json", "xlsx", "geojson", "wkt", "zip")

METADATA_EXTENSIONS = ("md", "txt", "yaml", "yml", "json", "pdf", "docx", "doc", "pptx", "ppt")
METADATA_FILENAME_PATTERNS = ("_metadata", "metadata_", "readme", "schema")

# ─────────────────────────────────────────────────────────────────────────────
# HTTP / download constants
# ─────────────────────────────────────────────────────────────────────────────

HTTP_TIMEOUT_SECONDS = 30

CONTENT_TYPE_TO_EXTENSION: dict[str, str] = {
    "text/csv": "csv",
    "application/csv": "csv",
    "application/json": "json",
    "application/geo+json": "geojson",
    "application/vnd.geo+json": "geojson",
    "application/parquet": "parquet",
    "application/vnd.apache.parquet": "parquet",
    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet": "xlsx",
    "application/vnd.ms-excel": "xlsx",
    "text/plain": "csv",
}


class DownloadConfig:
    """Configuration constants for HTTP file downloads."""

    CHUNK_SIZE = 1024 * 1024          # 1MB chunks
    TIMEOUT_SECONDS = 300             # 5 minutes
    PROGRESS_UPDATE_INTERVAL = 50     # Update every 50 chunks
    DEFAULT_HASH_MODULO = 10000
    UNKNOWN_SIZE_MAX = 1_000_000_000  # 1GB max for unknown file sizes

    # Connection settings
    CONNECTION_LIMIT = 100
    CONNECTION_LIMIT_PER_HOST = 30
    KEEPALIVE_TIMEOUT = 30
