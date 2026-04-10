from __future__ import annotations

import asyncio
import io
import re

from urllib.parse import parse_qs, urlparse

import aiohttp

from .constants import METADATA_EXTENSIONS, TABLE_EXTENSIONS, DownloadConfig


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


def extract_filename_from_url(url: str) -> str:
    """Extract a filename from a URL, falling back to a hash-based name."""
    parsed = urlparse(url)
    filename = parsed.path.split('/')[-1] if parsed.path else 'downloaded_file'

    if '?' in filename:
        filename = filename.split('?')[0]

    current_ext = filename.rsplit('.', 1)[-1].lower() if '.' in filename else ''
    valid_extensions = TABLE_EXTENSIONS + METADATA_EXTENSIONS

    if current_ext not in valid_extensions:
        query_params = parse_qs(parsed.query)
        format_param = query_params.get('format', [None])[0]
        if format_param and format_param.lower() in valid_extensions:
            base_name = filename.rsplit('.', 1)[0] if '.' in filename else filename
            filename = f"{base_name}.{format_param.lower()}"

    if not filename or '.' not in filename:
        filename = f"data_{abs(hash(url)) % DownloadConfig.DEFAULT_HASH_MODULO}.json"

    return filename


def extract_filename_from_headers(response_headers: dict, default_filename: str) -> str:
    """Refine a filename using Content-Disposition or Content-Type headers."""
    if 'content-disposition' in response_headers:
        cd = response_headers['content-disposition']
        matches = re.findall('filename="?([^"]+)"?', cd)
        if matches and '.' in matches[0]:
            return matches[0]

    current_ext = default_filename.rsplit('.', 1)[-1].lower() if '.' in default_filename else ''
    valid_extensions = TABLE_EXTENSIONS + METADATA_EXTENSIONS

    if current_ext not in valid_extensions and 'content-type' in response_headers:
        content_type = response_headers['content-type'].lower().split(';')[0].strip()
        content_type_map = {
            'text/csv': 'csv',
            'application/csv': 'csv',
            'application/json': 'json',
            'application/geo+json': 'geojson',
            'application/vnd.geo+json': 'geojson',
            'application/parquet': 'parquet',
            'application/vnd.apache.parquet': 'parquet',
            'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet': 'xlsx',
            'application/vnd.ms-excel': 'xlsx',
            'text/plain': 'csv',
        }
        if content_type in content_type_map:
            ext = content_type_map[content_type]
            base_name = default_filename.rsplit('.', 1)[0] if '.' in default_filename else default_filename
            return f"{base_name}.{ext}"

    return default_filename


def format_download_error(error: Exception) -> str:
    """Consistent download error message."""
    if isinstance(error, aiohttp.ClientError):
        return f"Network error: {error!s}"
    elif isinstance(error, asyncio.TimeoutError):
        return f"Download timeout ({DownloadConfig.TIMEOUT_SECONDS}s exceeded)"
    elif isinstance(error, asyncio.CancelledError):
        return "Download cancelled"
    else:
        return f"Download failed: {error!s}"


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
    def _show_progress(variant, message, value=None):
        if progress is None:
            return
        progress.description.visible = True
        progress.description.object = message
        progress.bar.variant = variant
        if value is not None:
            progress.bar.value = value
        progress.bar.visible = True

    def _hide_progress():
        if progress is None:
            return
        progress.bar.visible = False
        progress.description.visible = False

    try:
        filename = extract_filename_from_url(url)

        timeout = aiohttp.ClientTimeout(total=DownloadConfig.TIMEOUT_SECONDS)
        connector = aiohttp.TCPConnector(
            limit=DownloadConfig.CONNECTION_LIMIT,
            limit_per_host=DownloadConfig.CONNECTION_LIMIT_PER_HOST,
            keepalive_timeout=DownloadConfig.KEEPALIVE_TIMEOUT,
            enable_cleanup_closed=True,
        )

        async with aiohttp.ClientSession(timeout=timeout, connector=connector) as session:
            async with session.get(url) as response:
                response.raise_for_status()

                filename = extract_filename_from_headers(response.headers, filename)
                content_length = response.headers.get('content-length')

                if content_length:
                    total_size = int(content_length)
                    _show_progress("determinate", f"Downloading {filename}…", value=0)
                else:
                    total_size = 0
                    _show_progress("indeterminate", f"Downloading {filename}…")

                downloaded_data = io.BytesIO()
                downloaded_size = 0
                chunk_count = 0
                size_mismatch_detected = False

                async for chunk in response.content.iter_chunked(DownloadConfig.CHUNK_SIZE):
                    if asyncio.current_task().cancelled():
                        raise asyncio.CancelledError()

                    downloaded_data.write(chunk)
                    downloaded_size += len(chunk)
                    chunk_count += 1

                    if not size_mismatch_detected and total_size > 0 and downloaded_size > total_size:
                        total_size = 0
                        size_mismatch_detected = True

                    if chunk_count % DownloadConfig.PROGRESS_UPDATE_INTERVAL == 0 and progress is not None:
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
                _hide_progress()
                return filename, content, None

    except (TimeoutError, aiohttp.ClientError, asyncio.CancelledError, Exception) as e:
        _hide_progress()
        return None, None, format_download_error(e)
