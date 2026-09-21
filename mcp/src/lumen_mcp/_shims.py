"""Bridges to Lumen internals.

lumen-mcp reuses Lumen's own code. A couple of pieces are not yet exposed as public API; this module
prefers the public API when the installed Lumen provides it and falls back to the validated private
path otherwise, so the server works either way.
"""

from __future__ import annotations

import asyncio
import threading
from typing import Any

import yaml

# Vega-Lite spec normalization: use the public function when the installed Lumen exposes it.
try:
    from lumen.ai.agents.vega_lite import normalize_vegalite_spec as _public_normalize

    _HAVE_PUBLIC_NORMALIZE = True
except Exception:  # pragma: no cover - depends on installed Lumen version
    _HAVE_PUBLIC_NORMALIZE = False


def _run_coro(coro: Any) -> Any:
    """Run a coroutine to completion from any context (sync tool or event loop).

    Runs in a dedicated thread with its own loop, cancels any tasks the coroutine left pending (Lumen
    spawns streaming tasks), and propagates exceptions to the caller.
    """
    result: dict[str, Any] = {}

    def runner() -> None:
        loop = asyncio.new_event_loop()
        try:
            result["value"] = loop.run_until_complete(coro)
        except BaseException as exc:  # propagate to the calling thread
            result["error"] = exc
        finally:
            pending = asyncio.all_tasks(loop)
            for task in pending:
                task.cancel()
            if pending:
                loop.run_until_complete(asyncio.gather(*pending, return_exceptions=True))
            loop.close()

    thread = threading.Thread(target=runner)
    thread.start()
    thread.join()
    if "error" in result:
        raise result["error"]
    return result["value"]


def normalize_spec(spec: dict) -> dict:
    """Normalize + validate a raw Vega-Lite spec via Lumen.

    Returns ``{"spec": <normalized dict>, "sizing_mode": ..., "min_height": ...}`` (adds ``$schema``,
    default sizing, compound-chart handling, geo interactivity). Normalization is a hard prerequisite
    for rendering: ``pn.pane.Vega`` rejects a spec dict without ``$schema``.
    """
    if _HAVE_PUBLIC_NORMALIZE:
        return _public_normalize(dict(spec))

    from lumen.ai.agents import VegaLiteAgent

    agent = VegaLiteAgent()
    return _run_coro(agent._extract_spec({}, {"yaml_spec": yaml.dump(dict(spec))}))
