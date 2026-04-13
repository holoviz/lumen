from __future__ import annotations

import asyncio
import datetime

import httpx
import panel as pn
import param

from .constants import HTTP_TIMEOUT_SECONDS
from .rest_api import RESTAPISourceControls

# ─────────────────────────────────────────────────────────────────────────────
# OpenAPI type-mapping constants
# ─────────────────────────────────────────────────────────────────────────────

# OpenAPI type string → Python type
OPENAPI_TYPE_MAP: dict[str, type] = {
    "string": str,
    "integer": int,
    "number": float,
    "boolean": bool,
}

# OpenAPI format string → Python type (overrides the base type)
OPENAPI_FORMAT_MAP: dict[str, type] = {
    "date": datetime.date,
    "date-time": datetime.datetime,
    "int32": int,
    "int64": int,
    "float": float,
    "double": float,
}

# Fallback: infer format from well-known parameter names when the spec
# doesn't include an explicit ``format`` field (common in auto-generated
# specs like the production api.weather.gov/openapi.json).
NAME_FORMAT_HINTS: dict[str, str] = {
    "date": "date",
    "start": "date-time",
    "end": "date-time",
    "effective": "date-time",
    "time": "time",
}


# ─────────────────────────────────────────────────────────────────────────────
# OPENAPI SOURCE CONTROLS
# ─────────────────────────────────────────────────────────────────────────────


class OpenAPISourceControls(RESTAPISourceControls):
    """
    Parses an OpenAPI 3.x spec and registers each endpoint as an action.

    Fetches the spec at ``spec_url`` during init, resolves ``$ref``
    references, and builds typed callables for each path/method pair.

    Parameters
    ----------
    spec_url : str
        URL to the OpenAPI JSON spec.
    base_url : str
        Override the base URL from the spec (optional).
    include_paths : list[str], optional
        Only include paths matching these prefixes (e.g. ``["/alerts", "/points"]``).
    exclude_paths : list[str], optional
        Exclude paths matching these prefixes.

    Examples
    --------
    ::

        controls = OpenAPISourceControls(
            spec_url="https://api.weather.gov/openapi.json",
            headers={"User-Agent": "MyApp/1.0"},
            include_paths=["/points", "/gridpoints", "/alerts/active"],
        )
    """

    spec_url = param.String(default="", precedence=-1, doc="""
        URL to the OpenAPI JSON spec.""")

    include_paths = param.List(default=[], precedence=-1, doc="""
        Only include paths matching these prefixes.""")

    exclude_paths = param.List(default=[], precedence=-1, doc="""
        Exclude paths matching these prefixes.""")

    label = (
        '<span class="material-icons" style="vertical-align: middle;">'
        "api"
        "</span> OpenAPI"
    )

    def __init__(self, **params):
        super().__init__(**params)
        self._layout.loading = True
        pn.state.onload(self._load_spec)

    def _setup_actions(self):
        # Actions are registered in _load_spec after the spec is fetched.
        pass

    async def _load_spec(self):
        """Fetch and parse the OpenAPI spec, then register actions."""
        try:
            async with httpx.AsyncClient(headers=self.headers, timeout=HTTP_TIMEOUT_SECONDS) as client:
                resp = await client.get(self.spec_url)
                resp.raise_for_status()
                spec = resp.json()

            endpoints = self._parse_spec(spec)
            if endpoints:
                self.endpoints = endpoints
                actions = {
                    name: self._make_endpoint_callable(ep_spec)
                    for name, ep_spec in endpoints.items()
                }
                self._register_actions(actions)
        except Exception as e:
            self._show_message(f"Failed to load OpenAPI spec: {e}", error=True)
        finally:
            self._layout.loading = False

        if self.vector_store is not None:
            asyncio.create_task(self._embed_actions())  # noqa: RUF006

    # ──────────────────────────────────────────────────────────────────────────
    # Type resolution (overrides RESTAPISourceControls default)
    # ──────────────────────────────────────────────────────────────────────────

    def _resolve_param_type(self, p: dict) -> type:
        """Map an OpenAPI parameter dict to a Python type using format/type maps."""
        fmt = p.get("format")
        if fmt and fmt in OPENAPI_FORMAT_MAP:
            return OPENAPI_FORMAT_MAP[fmt]
        return OPENAPI_TYPE_MAP.get(p.get("type", "string"), str)

    # ──────────────────────────────────────────────────────────────────────────
    # Spec parsing — staticmethods are subclass extension points
    # ──────────────────────────────────────────────────────────────────────────

    def _parse_spec(self, spec: dict) -> dict[str, dict]:
        """
        Parse an OpenAPI 3.x spec into endpoint definitions.

        Resolves ``$ref`` references for parameters and extracts
        path + query parameters with types and descriptions.
        """
        if not self.base_url:
            servers = spec.get("servers", [])
            if servers:
                self.base_url = servers[0].get("url", "")

        components = spec.get("components", {})
        paths = spec.get("paths", {})
        endpoints: dict[str, dict] = {}

        for path, path_item in paths.items():
            if not self._path_included(path):
                continue
            path_endpoints = self._parse_path_operations(path, path_item, components)
            endpoints.update(path_endpoints)

        return endpoints

    def _parse_path_operations(
        self, path: str, path_item: dict, components: dict,
    ) -> dict[str, dict]:
        """Parse all HTTP method operations for a single path."""
        path_params = [
            self._resolve_param(p, components)
            for p in path_item.get("parameters", [])
        ]

        endpoints: dict[str, dict] = {}
        for method in ("get", "post", "put", "delete"):
            if method not in path_item:
                continue

            operation = path_item[method]
            params = self._merge_params(path_params, operation, components)
            display_name = self._display_name_for_operation(method, path, operation)

            endpoints[display_name] = {
                "method": method,
                "path": path,
                "summary": operation.get("summary", ""),
                "description": operation.get("description", operation.get("summary", "")),
                "operationId": operation.get("operationId", ""),
                "parameters": params,
            }

        return endpoints

    @staticmethod
    def _merge_params(
        path_params: list[dict], operation: dict, components: dict,
    ) -> list[dict]:
        """Merge path-level and operation-level parameters, filtering to path+query only."""
        op_params = [
            OpenAPISourceControls._resolve_param(p, components)
            for p in operation.get("parameters", [])
        ]
        merged = {p["name"]: p for p in path_params}
        merged.update({p["name"]: p for p in op_params})
        return [p for p in merged.values() if p.get("in") in ("path", "query")]

    @staticmethod
    def _display_name_for_operation(method: str, path: str, operation: dict) -> str:
        """Derive a human-readable display name for an endpoint."""
        summary = operation.get("summary", "")
        if summary:
            return summary
        operation_id = operation.get("operationId", "")
        if operation_id:
            return operation_id.replace("_", " ").title()
        return f"{method.upper()} {path}"

    def _path_included(self, path: str) -> bool:
        """Check if a path passes the include/exclude filters."""
        if any(path.startswith(p) for p in self.exclude_paths):
            return False
        if not self.include_paths:
            return True
        return any(path.startswith(p) for p in self.include_paths)

    @staticmethod
    def _resolve_param(param_spec: dict, components: dict) -> dict:
        """
        Resolve a ``$ref`` parameter reference and flatten into a
        simple dict with ``name``, ``in``, ``type``, ``required``,
        ``description``, ``default``.

        .. note::
            Only handles single-level ``$ref`` resolution.  Nested
            ``$ref`` chains are not followed (a rare edge case in
            most real-world specs).
        """
        if "$ref" in param_spec:
            ref_path = param_spec["$ref"]
            parts = ref_path.lstrip("#/").split("/")
            resolved = components
            for part in parts[1:]:
                resolved = resolved.get(part, {})
            param_spec = resolved

        schema = param_spec.get("schema", {})
        enum = schema.get("enum")

        result = {
            "name": param_spec.get("name", ""),
            "in": param_spec.get("in", "query"),
            "type": schema.get("type", "string"),
            "required": param_spec.get("required", False),
            "description": param_spec.get("description", ""),
        }
        if "default" in schema:
            result["default"] = schema["default"]
        if enum:
            result["enum"] = enum

        fmt = schema.get("format")
        if not fmt:
            fmt = NAME_FORMAT_HINTS.get(result["name"].lower())
        if fmt:
            result["format"] = fmt

        return result
