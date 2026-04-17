from __future__ import annotations

import asyncio
import datetime
import io
import re

from urllib.parse import parse_qs, urlparse

import httpx
import pandas as pd

from .constants import (
    CONTENT_TYPE_TO_EXTENSION, METADATA_EXTENSIONS, TABLE_EXTENSIONS,
    DownloadConfig,
)

_VALID_EXTENSIONS = TABLE_EXTENSIONS + METADATA_EXTENSIONS


# ─────────────────────────────────────────────────────────────────────────────
# Formatting helpers
# ─────────────────────────────────────────────────────────────────────────────

def format_bytes(bytes_size: int) -> str:
    """Human-readable byte size string."""
    if bytes_size == 0:
        return "0 B"
    size_names = ["B", "KB", "MB", "GB", "TB"]
    i = 0
    while bytes_size >= 1024 and i < len(size_names) - 1:
        bytes_size /= 1024.0
        i += 1
    return f"{bytes_size:.1f} {size_names[i]}"


# ─────────────────────────────────────────────────────────────────────────────
# Filename extraction
# ─────────────────────────────────────────────────────────────────────────────

def extract_filename_from_url(url: str) -> str:
    """Extract a filename from a URL, falling back to a hash-based name."""
    parsed = urlparse(url)
    filename = parsed.path.split("/")[-1] if parsed.path else "downloaded_file"

    if "?" in filename:
        filename = filename.split("?")[0]

    current_ext = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""

    if current_ext not in _VALID_EXTENSIONS:
        query_params = parse_qs(parsed.query)
        format_param = query_params.get("format", [None])[0]
        if format_param and format_param.lower() in _VALID_EXTENSIONS:
            base_name = filename.rsplit(".", 1)[0] if "." in filename else filename
            filename = f"{base_name}.{format_param.lower()}"

    if not filename or "." not in filename:
        filename = f"data_{abs(hash(url)) % DownloadConfig.DEFAULT_HASH_MODULO}.json"

    return filename


def extract_filename_from_headers(response_headers: dict, default_filename: str) -> str:
    """Refine a filename using Content-Disposition or Content-Type headers."""
    if "content-disposition" in response_headers:
        cd = response_headers["content-disposition"]
        matches = re.findall('filename="?([^"]+)"?', cd)
        if matches and "." in matches[0]:
            return matches[0]

    current_ext = default_filename.rsplit(".", 1)[-1].lower() if "." in default_filename else ""

    if current_ext in _VALID_EXTENSIONS:
        return default_filename

    content_type_raw = response_headers.get("content-type", "")
    if not content_type_raw:
        return default_filename

    content_type = content_type_raw.lower().split(";")[0].strip()
    ext = CONTENT_TYPE_TO_EXTENSION.get(content_type)
    if ext is None:
        return default_filename

    base_name = default_filename.rsplit(".", 1)[0] if "." in default_filename else default_filename
    return f"{base_name}.{ext}"


# ─────────────────────────────────────────────────────────────────────────────
# Error formatting
# ─────────────────────────────────────────────────────────────────────────────

def format_download_error(error: Exception) -> str:
    """Consistent download error message."""
    if isinstance(error, httpx.HTTPStatusError):
        return f"HTTP {error.response.status_code}: {error!s}"
    if isinstance(error, httpx.RequestError):
        return f"Network error: {error!s}"
    if isinstance(error, asyncio.TimeoutError):
        return f"Download timeout ({DownloadConfig.TIMEOUT_SECONDS}s exceeded)"
    if isinstance(error, asyncio.CancelledError):
        return "Download cancelled"
    return f"Download failed: {error!s}"


# ─────────────────────────────────────────────────────────────────────────────
# Data serialization / normalization
# ─────────────────────────────────────────────────────────────────────────────

def serialize_param_value(val):
    """Serialize date/datetime values to ISO strings for API requests."""
    if isinstance(val, datetime.datetime):
        return val.isoformat()
    if isinstance(val, datetime.date):
        return val.isoformat()
    return val


def normalize_json_response(data) -> pd.DataFrame:
    """
    Flatten a JSON API response into a DataFrame.

    Handles GeoJSON feature collections, nested ``properties`` dicts
    (common in weather.gov), plain dicts, and lists.
    """
    if not isinstance(data, (dict, list)):
        return pd.DataFrame({"response": [str(data)]})

    if isinstance(data, list):
        return pd.json_normalize(data) if data else pd.DataFrame()

    # GeoJSON: extract features
    if "features" in data and isinstance(data["features"], list):
        features = data["features"]
        if not features:
            return pd.DataFrame()
        return pd.json_normalize([
            {**f.get("properties", {}), "geometry": str(f.get("geometry", ""))}
            for f in features
        ])

    # Nested properties (common in weather.gov)
    if "properties" in data and isinstance(data["properties"], dict):
        props = data["properties"]
        if "periods" in props and isinstance(props["periods"], list):
            return pd.json_normalize(props["periods"])
        return pd.json_normalize([props])

    return pd.json_normalize([data])


# ─────────────────────────────────────────────────────────────────────────────
# Progress helpers
# ─────────────────────────────────────────────────────────────────────────────

def _show_progress(progress, variant: str, message: str, value=None):
    if progress is None:
        return
    progress.description.visible = True
    progress.description.object = message
    progress.bar.variant = variant
    if value is not None:
        progress.bar.value = value
    progress.bar.visible = True


def _hide_progress(progress):
    if progress is None:
        return
    progress.bar.visible = False
    progress.description.visible = False


def _update_progress(progress, filename: str, downloaded_size: int, total_size: int):
    """Update progress bar with current download state."""
    if progress is None:
        return
    if total_size > 0:
        pct = min((downloaded_size / total_size) * 100, 100)
        progress.bar.value = pct
        progress.description.object = (
            f"Downloading {filename}: "
            f"{format_bytes(downloaded_size)}/{format_bytes(total_size)}"
        )
    else:
        progress.description.object = (
            f"Downloading {filename}: {format_bytes(downloaded_size)}"
        )


# ─────────────────────────────────────────────────────────────────────────────
# Download
# ─────────────────────────────────────────────────────────────────────────────

async def download_file(
    url: str,
    progress=None,
) -> tuple[str | None, bytes | None, str | None]:
    """
    Download a file from a URL with optional progress reporting.

    Parameters
    ----------
    url : str
        The URL to download.
    progress : Progress, optional
        A ``Progress`` instance (from ``ingest.progress``).  When provided,
        the bar and description are updated during download and hidden on
        completion.  When ``None`` the download is silent.

    Returns
    -------
    tuple[str | None, bytes | None, str | None]
        ``(filename, content_bytes, error_message)``
        On success: ``(filename, bytes, None)``
        On failure: ``(None, None, error_string)``
    """
    try:
        filename = extract_filename_from_url(url)

        async with httpx.AsyncClient(timeout=DownloadConfig.TIMEOUT_SECONDS) as client:
            async with client.stream("GET", url) as response:
                response.raise_for_status()

                filename = extract_filename_from_headers(
                    dict(response.headers), filename,
                )
                content_length = response.headers.get("content-length")

                if content_length:
                    total_size = int(content_length)
                    _show_progress(progress, "determinate", f"Downloading {filename}…", value=0)
                else:
                    total_size = 0
                    _show_progress(progress, "indeterminate", f"Downloading {filename}…")

                downloaded_data = io.BytesIO()
                downloaded_size = 0
                chunk_count = 0
                size_mismatch_detected = False

                async for chunk in response.aiter_bytes(DownloadConfig.CHUNK_SIZE):
                    if asyncio.current_task().cancelled():
                        raise asyncio.CancelledError()

                    downloaded_data.write(chunk)
                    downloaded_size += len(chunk)
                    chunk_count += 1

                    if not size_mismatch_detected and total_size > 0 and downloaded_size > total_size:
                        total_size = 0
                        size_mismatch_detected = True

                    if chunk_count % DownloadConfig.PROGRESS_UPDATE_INTERVAL == 0:
                        _update_progress(progress, filename, downloaded_size, total_size)

                    await asyncio.sleep(0)

                # Final progress update
                if progress is not None:
                    if total_size > 0:
                        progress.bar.value = min((downloaded_size / total_size) * 100, 100)
                        progress.description.object = (
                            f"Downloaded {filename}: "
                            f"{format_bytes(downloaded_size)}/{format_bytes(total_size)}"
                        )
                    else:
                        progress.bar.value = 100
                        progress.description.object = (
                            f"Downloaded {filename}: {format_bytes(downloaded_size)}"
                        )

                downloaded_data.seek(0)
                content = downloaded_data.read()
                _hide_progress(progress)
                return filename, content, None

    except asyncio.CancelledError:
        _hide_progress(progress)
        return None, None, format_download_error(asyncio.CancelledError())
    except Exception as e:
        _hide_progress(progress)
        return None, None, format_download_error(e)
