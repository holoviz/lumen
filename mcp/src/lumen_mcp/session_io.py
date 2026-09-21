"""Persist and restore a session.

A session is saved as a DuckDB file holding the workspace tables plus a ``<path>.charts.json`` sidecar
holding each chart's spec and bound table. Loading restores the workspace and re-renders the charts.
"""

from __future__ import annotations

import json
import os

import duckdb

from . import sources, viz
from .session import session

_DB_SUFFIXES = (".db", ".duckdb", ".ddb")


def save_session(path: str) -> dict:
    """Save the workspace tables to a DuckDB file and chart specs to ``<path>.charts.json``.

    ``path`` gets a ``.duckdb`` suffix if it has none. Returns the saved path, tables, and chart ids.
    """
    if not path.endswith(_DB_SUFFIXES):
        path = f"{path}.duckdb"
    source = session.require_source()

    connection = duckdb.connect(path)
    try:
        for table in source.get_tables():
            frame = source.get(table)
            connection.register("_lumen_mcp_tmp", frame)
            connection.execute(f'CREATE OR REPLACE TABLE "{table}" AS SELECT * FROM _lumen_mcp_tmp')
            connection.unregister("_lumen_mcp_tmp")
    finally:
        connection.close()

    meta = {cid: {"spec": entry["spec"], "table": entry["table"]} for cid, entry in session.charts.items()}
    with open(f"{path}.charts.json", "w") as fh:
        json.dump(meta, fh)

    return {"saved": path, "tables": source.get_tables(), "charts": list(session.charts)}


def load_session(path: str) -> dict:
    """Restore a saved session: reload the workspace and re-render its charts."""
    sources.connect_source(path)
    session.charts.clear()

    meta_path = f"{path}.charts.json"
    if os.path.exists(meta_path):
        with open(meta_path) as fh:
            meta = json.load(fh)
        for cid, entry in meta.items():
            viz.render_vegalite(entry["spec"], entry["table"], chart_id=cid)

    return {
        "loaded": path,
        "tables": session.require_source().get_tables(),
        "charts": list(session.charts),
    }
