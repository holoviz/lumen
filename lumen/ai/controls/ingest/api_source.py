from __future__ import annotations

import asyncio
import io
import json

import aiohttp
import pandas as pd
import panel as pn
import param

from panel.pane import Markdown
from panel.widgets import Tabulator
from panel_material_ui import (
    Button, Column as MuiColumn, Select, TextInput,
)

from ....sources.duckdb import DuckDBSource
from .base import BaseSourceControls
from .result import SourceResult

# ─────────────────────────────────────────────────────────────────────────────
# OpenAPI helpers
# ─────────────────────────────────────────────────────────────────────────────

def _resolve_ref(ref: str, spec: dict) -> dict:
    """Follow a JSON pointer like ``#/components/parameters/Foo``."""
    parts = ref.lstrip("#/").split("/")
    node = spec
    for p in parts:
        node = node.get(p, {})
    return node


def _collect_params(path_item: dict, operation: dict, spec: dict) -> list[dict]:
    """Merge path-level and operation-level parameters, resolving ``$ref``."""
    raw = list(path_item.get("parameters", [])) + list(operation.get("parameters", []))
    resolved = []
    seen: set[tuple] = set()
    for p in raw:
        if "$ref" in p:
            p = _resolve_ref(p["$ref"], spec)
        key = (p.get("name"), p.get("in"))
        if key not in seen:
            seen.add(key)
            resolved.append(p)
    return resolved


def _param_summary(params: list[dict], spec: dict) -> str:
    """Human-readable one-liner per parameter."""
    lines = []
    for p in params:
        schema = p.get("schema", {})
        if "$ref" in schema:
            schema = _resolve_ref(schema["$ref"], spec)
        name = p.get("name", "?")
        loc = p.get("in", "?")
        ptype = schema.get("type", "")
        required = p.get("required", False)
        desc = p.get("description", "")
        enum = schema.get("enum")
        parts = [f"{name} ({loc}, {ptype})"]
        if required:
            parts.append("[required]")
        if desc:
            parts.append(f"— {desc[:120]}")
        if enum:
            parts.append(f"values: {enum[:6]}{'…' if len(enum) > 6 else ''}")
        lines.append(" ".join(parts))
    return "\n".join(lines) if lines else "none"


def openapi_to_catalog(spec: dict) -> pd.DataFrame:
    """
    Convert an OpenAPI 3.x spec into a catalog DataFrame.

    Each row is one ``(method, path)`` endpoint.  The ``_params_raw``
    column holds the resolved parameter dicts for widget generation.
    """
    rows = []
    for path, path_item in spec.get("paths", {}).items():
        for method in ("get", "post", "put", "patch", "delete"):
            op = path_item.get(method)
            if op is None:
                continue
            params = _collect_params(path_item, op, spec)
            rows.append({
                "endpoint": f"{method.upper()} {path}",
                "operation_id": op.get("operationId", f"{method}_{path}"),
                "method": method.upper(),
                "path": path,
                "tags": ", ".join(op.get("tags", [])),
                "description": op.get("description", op.get("summary", "")),
                "parameters": _param_summary(params, spec),
                "n_params": len(params),
                "_params_raw": params,
            })
    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────────────────────
# APISourceControls
# ─────────────────────────────────────────────────────────────────────────────

class APISourceControls(BaseSourceControls):
    """
    Browse API endpoints parsed from an OpenAPI spec, then fetch data.

    Two-phase workflow:

    1. **Browse** — endpoints are shown in a Tabulator with description,
       tags, and parameter count.  Optionally embedded into a
       ``vector_store`` for agent discovery.
    2. **Fetch** — clicking a row generates a parameter form from the
       spec.  The user fills in values and clicks *Fetch Data*.
    """

    spec_url = param.String(default="", doc="URL to an OpenAPI 3.x spec (JSON or YAML).")

    spec = param.Dict(default=None, doc="Pre-loaded spec dict (takes precedence over spec_url).")

    base_url = param.String(default="", doc="Override the server base URL from the spec.")

    headers = param.Dict(default={}, doc="Extra HTTP headers for every API request.")

    catalog_df = param.DataFrame(default=None, doc="Catalog of parsed endpoints.")

    display_columns = param.Dict(default={
        "method":      {"title": "Method",      "width": "8%"},
        "path":        {"title": "Path",        "width": "30%"},
        "tags":        {"title": "Tags",        "width": "15%"},
        "description": {"title": "Description", "width": "40%"},
        "n_params":    {"title": "#",           "width": "5%"},
    })

    filter_columns = param.Dict(default={
        "path":        {"type": "input", "func": "like", "placeholder": "Search path…"},
        "tags":        {"type": "input", "func": "like", "placeholder": "Filter tags…"},
        "description": {"type": "input", "func": "like", "placeholder": "Search…"},
    })

    search_columns = param.List(default=["endpoint", "tags", "description", "parameters"])

    detail_columns = param.List(default=["operation_id", "parameters"])

    vector_store = param.Parameter(default=None, doc="""
        Optional VectorStore for agent-driven discovery.""")

    load_mode = "manual"

    label = '<span class="material-icons" style="vertical-align: middle;">api</span> API Endpoints'

    def __init__(self, **params):
        super().__init__(**params)
        self._selected_entry: pd.Series | None = None
        self._param_widgets: dict[str, TextInput | Select] = {}
        self._layout.loading = True
        pn.state.onload(self._load_spec)

    # ── Layout ────────────────────────────────────────────────────────────────

    def _render_controls(self) -> list:
        display_cols = self.display_columns
        empty_df = (
            pd.DataFrame({col: pd.Series(dtype="object") for col in display_cols})
            if display_cols else pd.DataFrame()
        )

        tabulator_kw: dict = dict(
            value=empty_df,
            page_size=5,
            pagination="local",
            sizing_mode="stretch_width",
            show_index=False,
            on_click=self._on_row_click,
        )
        if display_cols:
            tabulator_kw.update(
                titles={c: cfg.get("title", c) for c, cfg in display_cols.items()},
                widths={c: cfg.get("width", "auto") for c, cfg in display_cols.items()},
                formatters={c: cfg["formatter"] for c, cfg in display_cols.items() if "formatter" in cfg},
                editors={c: None for c in display_cols},
                header_filters=self.filter_columns,
            )
        if self.detail_columns:
            tabulator_kw["row_content"] = self._get_row_content

        self._tabulator = Tabulator(**tabulator_kw)

        self._param_form = MuiColumn(sizing_mode="stretch_width", visible=False, margin=(5, 0))
        self._fetch_button = Button(
            name="Fetch Data", icon="download",
            sizing_mode="stretch_width", height=42,
            visible=False, on_click=self._on_fetch_click,
        )
        self._selected_label = Markdown("", visible=False, margin=(5, 10))

        return [
            Markdown("*Select an endpoint, fill in parameters, and click Fetch Data.*", margin=(0, 10)),
            self._tabulator,
            self._selected_label,
            self._param_form,
            self._fetch_button,
        ]

    def _render_layout(self):
        controls = self._render_controls()
        return MuiColumn(
            *controls,
            self._error_placeholder,
            self._message_placeholder,
            self.progress.bar,
            self.progress.description,
            sizing_mode="stretch_width",
            margin=(10, 15),
        )

    # ── Spec loading ──────────────────────────────────────────────────────────

    async def _load_spec(self):
        try:
            if self.spec is None and self.spec_url:
                await self._fetch_spec()
            if self.spec is None:
                self._show_message("No OpenAPI spec provided.", error=True)
                self._layout.loading = False
                return

            self.catalog_df = openapi_to_catalog(self.spec)

            if not self.base_url:
                servers = self.spec.get("servers", [])
                if servers:
                    self.base_url = servers[0].get("url", "")

            # Resolve relative server URLs against the spec origin
            if self.base_url and self.base_url.startswith("/") and self.spec_url:
                from urllib.parse import urlparse
                parsed = urlparse(self.spec_url)
                self.base_url = f"{parsed.scheme}://{parsed.netloc}{self.base_url}"

            if self.display_columns:
                cols = [c for c in self.display_columns if c in self.catalog_df.columns]
                self._tabulator.value = self.catalog_df[cols]
            else:
                self._tabulator.value = self.catalog_df

        except Exception as e:
            self._show_message(f"Failed to load spec: {e}", error=True)
        finally:
            self._layout.loading = False

        if self.vector_store is not None and self.catalog_df is not None:
            asyncio.create_task(self._embed_catalog())  # noqa: RUF006

    async def _fetch_spec(self):
        hdrs = {"User-Agent": "lumen-api-source/1.0", **self.headers}
        async with aiohttp.ClientSession(headers=hdrs) as session:
            async with session.get(self.spec_url) as resp:
                resp.raise_for_status()
                text = await resp.text()
                if self.spec_url.endswith((".yaml", ".yml")):
                    import yaml
                    self.spec = yaml.safe_load(text)
                else:
                    self.spec = json.loads(text)

    # ── Endpoint selection ────────────────────────────────────────────────────

    async def _on_row_click(self, event):
        if self.catalog_df is None:
            return
        self._selected_entry = self.catalog_df.iloc[event.row]
        self._build_param_form(self._selected_entry)

    def _build_param_form(self, entry: pd.Series):
        """Generate widgets from the endpoint's parameter definitions."""
        self._param_form.clear()
        self._param_widgets.clear()

        params_raw = entry.get("_params_raw", [])
        self._selected_label.object = f"**Selected:** `{entry['endpoint']}`"
        self._selected_label.visible = True

        if not params_raw:
            self._param_form.append(Markdown("*This endpoint has no parameters.*"))
            self._param_form.visible = True
            self._fetch_button.visible = True
            return

        for p in params_raw:
            name = p.get("name", "?")
            desc = p.get("description", "")
            required = p.get("required", False)
            schema = p.get("schema", {})
            if "$ref" in schema and self.spec:
                schema = _resolve_ref(schema["$ref"], self.spec)

            enum = schema.get("enum")
            default = schema.get("default", "")
            label = f"{name}{'*' if required else ''}"

            if enum:
                widget = Select(
                    name=label,
                    options=[""] + [str(v) for v in enum],
                    value=str(default) if default else "",
                    description=desc[:100] if desc else "",
                    sizing_mode="stretch_width",
                )
            else:
                widget = TextInput(
                    name=label,
                    value=str(default) if default else "",
                    placeholder=desc[:80] if desc else schema.get("type", "string"),
                    sizing_mode="stretch_width",
                )

            self._param_widgets[name] = widget
            self._param_form.append(widget)

        self._param_form.visible = True
        self._fetch_button.visible = True

    # ── Data fetching ─────────────────────────────────────────────────────────

    def _on_fetch_click(self, event):
        # on_click callbacks are synchronous; schedule the async fetch
        pn.state.execute(self._do_fetch)

    async def _do_fetch(self):
        await self._run_load(self._fetch_selected())

    async def _fetch_with_params(
        self, entry: pd.Series, param_values: dict[str, str]
    ) -> SourceResult:
        """Fetch an endpoint using pre-filled parameter values.

        This is the agent-callable counterpart of ``_fetch_selected``
        which reads values from sidebar widgets. ``param_values`` maps
        parameter names to string values.
        """
        path_template = entry["path"]
        params_raw = entry.get("_params_raw", [])

        path_params: dict[str, str] = {}
        query_params: dict[str, str] = {}
        for p in params_raw:
            name = p.get("name", "")
            value = param_values.get(name, "")
            if not value:
                if p.get("required"):
                    return SourceResult.empty(f"Required parameter '{name}' is empty.")
                continue
            if p.get("in") == "path":
                path_params[name] = value
            else:
                query_params[name] = value

        try:
            path = path_template.format(**path_params) if path_params else path_template
        except KeyError as e:
            return SourceResult.empty(f"Missing path parameter: {e}")

        url = f"{self.base_url.rstrip('/')}{path}"
        if query_params:
            url += "?" + "&".join(f"{k}={v}" for k, v in query_params.items())

        self.progress(f"Fetching {url[:80]}…")

        hdrs = {"User-Agent": "lumen-api-source/1.0", **self.headers}
        try:
            async with aiohttp.ClientSession(headers=hdrs) as session:
                async with session.get(url) as resp:
                    resp.raise_for_status()
                    content_type = resp.headers.get("content-type", "")
                    raw = await resp.read()
        except Exception as e:
            return SourceResult.empty(f"Request failed: {e}")

        self.progress("Parsing response…", value=80)
        try:
            df = self._parse_response(raw, content_type)
        except Exception as e:
            return SourceResult.empty(f"Could not parse response: {e}")

        if df is None or df.empty:
            return SourceResult.empty("Response contained no tabular data.")

        table_name = (
            entry.get("operation_id", "api_data")
            .replace(".", "_").replace("-", "_").replace(" ", "_")
        )
        source = DuckDBSource.from_df(tables={table_name: df})
        return SourceResult.from_source(
            source, table=table_name,
            message=f"Loaded {len(df):,} rows from `{entry['endpoint']}`",
        )

    async def load_entry(
        self, row_idx: int, param_values: dict[str, str] | None = None
    ) -> SourceResult:
        """Load an API endpoint by index with pre-filled parameter values.

        Public API for agents — runs the full load lifecycle
        (progress, source registration, outputs trigger) and
        returns the ``SourceResult``.
        """
        if self.catalog_df is None:
            return SourceResult.empty("Catalog not loaded.")
        entry = self.catalog_df.iloc[row_idx]
        if param_values:
            return await self._run_load(self._fetch_with_params(entry, param_values))
        return await self._run_load(self._fetch_with_params(entry, {}))

    async def _fetch_selected(self) -> SourceResult:
        if self._selected_entry is None:
            return SourceResult.empty("No endpoint selected.")

        entry = self._selected_entry
        path_template = entry["path"]

        # Partition widget values into path vs query params
        path_params: dict[str, str] = {}
        query_params: dict[str, str] = {}
        for p in entry.get("_params_raw", []):
            name = p.get("name", "")
            widget = self._param_widgets.get(name)
            value = widget.value if widget else ""
            if not value:
                if p.get("required"):
                    return SourceResult.empty(f"Required parameter '{name}' is empty.")
                continue
            if p.get("in") == "path":
                path_params[name] = value
            else:
                query_params[name] = value

        # Build URL
        try:
            path = path_template.format(**path_params) if path_params else path_template
        except KeyError as e:
            return SourceResult.empty(f"Missing path parameter: {e}")

        url = f"{self.base_url.rstrip('/')}{path}"
        if query_params:
            url += "?" + "&".join(f"{k}={v}" for k, v in query_params.items())

        self.progress(f"Fetching {url[:80]}…")

        hdrs = {"User-Agent": "lumen-api-source/1.0", **self.headers}
        try:
            async with aiohttp.ClientSession(headers=hdrs) as session:
                async with session.get(url) as resp:
                    resp.raise_for_status()
                    content_type = resp.headers.get("content-type", "")
                    raw = await resp.read()
        except Exception as e:
            return SourceResult.empty(f"Request failed: {e}")

        self.progress("Parsing response…", value=80)
        try:
            df = self._parse_response(raw, content_type)
        except Exception as e:
            return SourceResult.empty(f"Could not parse response: {e}")

        if df is None or df.empty:
            return SourceResult.empty("Response contained no tabular data.")

        table_name = (
            entry.get("operation_id", "api_data")
            .replace(".", "_").replace("-", "_").replace(" ", "_")
        )
        source = DuckDBSource.from_df(tables={table_name: df})
        return SourceResult.from_source(
            source, table=table_name,
            message=f"Loaded {len(df):,} rows from `{entry['endpoint']}`",
        )

    # ── Response parsing ──────────────────────────────────────────────────────

    def _parse_response(self, raw: bytes, content_type: str) -> pd.DataFrame | None:
        """Best-effort parse of an API response into a DataFrame."""
        text = raw.decode("utf-8", errors="replace")

        if "json" in content_type or text.lstrip().startswith(("{", "[")):
            return self._json_to_df(json.loads(text))
        if "csv" in content_type:
            return pd.read_csv(io.BytesIO(raw))

        try:
            return self._json_to_df(json.loads(text))
        except json.JSONDecodeError:
            return pd.DataFrame({"raw_response": [text[:5000]]})

    @staticmethod
    def _json_to_df(data) -> pd.DataFrame | None:
        """Unwrap common JSON response patterns into a flat DataFrame."""
        if isinstance(data, list):
            return pd.json_normalize(data, max_level=2) if data else None

        if not isinstance(data, dict):
            return None

        # GeoJSON FeatureCollection
        if data.get("type") == "FeatureCollection" and "features" in data:
            rows = []
            for f in data["features"]:
                row = dict(f.get("properties", {}))
                geom = f.get("geometry") or {}
                coords = geom.get("coordinates")
                if isinstance(coords, list) and len(coords) >= 2:
                    row["longitude"], row["latitude"] = coords[0], coords[1]
                rows.append(row)
            return pd.json_normalize(rows, max_level=2) if rows else None

        # Common list-valued wrapper keys
        for key in ("features", "data", "results", "items", "records",
                     "observations", "properties", "@graph", "values"):
            if key in data and isinstance(data[key], list) and data[key]:
                return pd.json_normalize(data[key], max_level=2)

        # Flat object → single-row DataFrame
        flat = {k: v for k, v in data.items() if not isinstance(v, (dict, list))}
        return pd.DataFrame([flat]) if flat else None

    # ── Row detail ────────────────────────────────────────────────────────────

    def _get_row_content(self, row):
        lines = []
        for col in self.detail_columns:
            if col in row and pd.notna(row[col]):
                lines.append(f"  **{col}:** {row[col]}")
        return Markdown("\n\n".join(lines), sizing_mode="stretch_width")

    # ── Vector embedding ──────────────────────────────────────────────────────

    def _entry_to_text(self, entry: pd.Series) -> str:
        parts = []
        for col in self.search_columns:
            if col in entry.index and pd.notna(entry[col]):
                val = str(entry[col])
                if val:
                    parts.append(f"{col}: {val}")
        return ". ".join(parts)

    async def _embed_catalog(self):
        if self.vector_store is None or self.catalog_df is None:
            return
        items = []
        for idx, row in self.catalog_df.iterrows():
            text = self._entry_to_text(row)
            if not text:
                continue
            metadata = {
                "type": "catalog_entry",
                "_row_idx": int(idx) if not isinstance(idx, int) else idx,
                "_control_id": id(self),
            }
            # Surface required parameter names so agents know what to fill
            params_raw = row.get("_params_raw", [])
            required = [p["name"] for p in params_raw if p.get("required")]
            if required:
                metadata["_required_params"] = ",".join(required)
            # Compact JSON schema for the LLM to fill parameters
            if params_raw:
                schema_parts = []
                for p in params_raw:
                    s = p.get("schema", {})
                    desc = p.get("description", "")
                    enum = s.get("enum")
                    part = f"{p['name']}({p.get('in','?')},{s.get('type','?')})"
                    if p.get("required"):
                        part += "[req]"
                    if enum:
                        part += f"{enum[:4]}"
                    if desc:
                        part += f":{desc[:60]}"
                    schema_parts.append(part)
                metadata["_params_schema"] = "|".join(schema_parts)
            for col in ("tags", "method"):
                if col in row.index and pd.notna(row[col]):
                    metadata[col] = str(row[col])
            items.append({"text": text, "metadata": metadata})
        if items:
            await self.vector_store.upsert(items)
