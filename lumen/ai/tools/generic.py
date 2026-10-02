"""
General-purpose LLM tools: literal Python values and HTTP GET via aiohttp.
"""

from __future__ import annotations

import ast

from .base import define_tool

try:
    import aiohttp
except ImportError:
    aiohttp = None  # type: ignore[misc, assignment]


@define_tool(purpose="Parse an expression, e.g. to perform calculations or conversions.")
def parse_literal(literal: str) -> str:
    """
    Parse *literal* as a Python literal (str, int, float, bool, None, list, dict, tuple, set).
    Only constant expressions are allowed — not arbitrary code or function calls.
    """
    s = (literal or "").strip()
    if not s:
        return "Empty input; provide a Python literal (e.g. '[1, 2]', '{\"a\": 1}')."
    try:
        value = ast.literal_eval(s)
    except (ValueError, SyntaxError) as e:
        return f"literal_eval failed: {e}"
    return repr(value)


@define_tool(purpose="Download public HTTP(S) content by URL for inspection (GET, text body, size-limited). Use when the user points at a web resource to summarize or extract from.")
async def fetch_url(
    url: str,
    timeout_seconds: float = 30.0,
    max_chars: int = 120_000,
) -> str:
    """
    HTTP GET *url* (https or http). Returns status line and response text, truncated to *max_chars*.
    """
    if aiohttp is None:
        return "aiohttp is not installed; cannot fetch URLs."
    u = (url or "").strip()
    if not u:
        return "Empty URL."
    timeout = aiohttp.ClientTimeout(total=max(1.0, min(timeout_seconds, 120.0)))
    cap = max(256, min(int(max_chars), 2_000_000))
    try:
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.get(u) as resp:
                text = await resp.text()
                if len(text) > cap:
                    text = text[:cap] + "\n... (truncated)"
                return f"HTTP {resp.status} {resp.reason}\n{text}"
    except aiohttp.ClientError as e:
        return f"Request failed: {e}"
    except TimeoutError:
        return "Request timed out."
