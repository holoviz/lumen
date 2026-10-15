"""Live dashboard server.

Runs a single background ``panel serve`` subprocess (the pattern from panel-live-server) that serves
``_dashboard_app.py`` against a serialized snapshot of the current session. Because the app is a real
Panel/Lumen app with the workspace data, the served dashboard is fully interactive.
"""

from __future__ import annotations

import os
import socket
import subprocess
import sys
import tempfile
import time
import urllib.request

from . import session_io

_APP = os.path.join(os.path.dirname(__file__), "_dashboard_app.py")
_ROUTE = "_dashboard_app"

# Current server: {"proc", "port", "url", "session"} or None.
_server: dict | None = None


def _free_port() -> int:
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    sock.close()
    return port


def _wait_ready(url: str, timeout: float = 45.0) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=3) as response:
                if response.status == 200:
                    return True
        except Exception:
            time.sleep(0.5)
    return False


def launch_dashboard() -> dict:
    """Serialize the session and (re)start a live Panel dashboard server. Returns the URL."""
    stop_dashboard()

    workdir = tempfile.mkdtemp(prefix="lumen-mcp-dash-")
    saved = session_io.save_session(os.path.join(workdir, "session"))
    port = _free_port()
    url = f"http://localhost:{port}/{_ROUTE}"

    env = dict(os.environ, LUMEN_MCP_DASHBOARD_SESSION=saved["saved"])
    proc = subprocess.Popen(
        [
            sys.executable, "-m", "panel", "serve", _APP,
            "--port", str(port), "--address", "127.0.0.1",
            "--allow-websocket-origin", f"localhost:{port}",
            "--allow-websocket-origin", f"127.0.0.1:{port}",
        ],
        env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )

    global _server
    _server = {"proc": proc, "port": port, "url": url, "session": saved["saved"]}
    ready = _wait_ready(url)
    return {
        "url": url,
        "port": port,
        "ready": ready,
        "charts": saved["charts"],
        "tables": saved["tables"],
    }


def stop_dashboard() -> dict:
    """Stop the running dashboard server, if any."""
    global _server
    running = _server is not None and _server["proc"].poll() is None
    if running:
        _server["proc"].terminate()
        try:
            _server["proc"].wait(timeout=5)
        except subprocess.TimeoutExpired:
            _server["proc"].kill()
    _server = None
    return {"stopped": running}
