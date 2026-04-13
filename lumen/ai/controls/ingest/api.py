from __future__ import annotations

import asyncio
import datetime
import inspect

from typing import Literal

import pandas as pd
import panel as pn
import param

from .parametric import ParametricSourceControls
from .result import SourceResult

# ─────────────────────────────────────────────────────────────────────────────
# REST API SOURCE CONTROLS
# ─────────────────────────────────────────────────────────────────────────────

# OpenAPI type string → Python type
_OPENAPI_TYPE_MAP: dict[str, type] = {
    "string": str,
    "integer": int,
    "number": float,
    "boolean": bool,
}

# OpenAPI format string → Python type (overrides the base type)
_OPENAPI_FORMAT_MAP: dict[str, type] = {
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
_NAME_FORMAT_HINTS: dict[str, str] = {
    "date": "date",
    "start": "date-time",
    "end": "date-time",
    "effective": "date-time",
    "time": "time",
}


class RESTAPISourceControls(ParametricSourceControls):
    """
    Base class for controls that fetch data from REST API endpoints.

    Each endpoint is described as a dict with ``method``, ``path``,
    ``summary``, ``description``, and ``parameters``.  The base class
    builds a typed Python callable for each endpoint, registers it as
    an action, and renders widgets via ``pn.Param``.

    Subclasses provide endpoint definitions — either manually or by
    parsing an API spec (see ``OpenAPISourceControls``).

    Parameters
    ----------
    base_url : str
        The root URL for all API requests (e.g. ``"https://api.weather.gov"``).
    headers : dict
        Default HTTP headers sent with every request.
    endpoints : dict, optional
        ``{display_name: endpoint_spec}`` where each spec is a dict with
        keys ``method``, ``path``, ``summary``, ``description``, ``parameters``.
    """

    base_url = param.String(default="", precedence=-1, doc="""
        Root URL for all API requests.""")

    headers = param.Dict(default={}, precedence=-1, doc="""
        Default HTTP headers sent with every request.""")

    endpoints = param.Parameter(default=None, precedence=-1, doc="""
        ``{display_name: endpoint_spec}`` — each spec has ``method``,
        ``path``, ``summary``, ``description``, ``parameters``.""")

    label = (
        '<span class="material-icons" style="vertical-align: middle;">'
        "cloud"
        "</span> REST API"
    )

    def _setup_actions(self):
        """Build callables from endpoint specs and register as actions."""
        endpoints = self.endpoints
        if not endpoints:
            return
        actions = {}
        for name, spec in endpoints.items():
            actions[name] = self._make_endpoint_callable(spec)
        if actions:
            self._register_actions(
                actions,
                param_overrides=self.param_overrides if hasattr(self, "param_overrides") else None,
                skip_params=self.skip_params if hasattr(self, "skip_params") else None,
            )
        if self.vector_store is not None:
            pn.state.execute(self._embed_actions)

    def _make_endpoint_callable(self, spec: dict) -> callable:
        """
        Build a typed callable from an endpoint specification dict.

        The resulting function has a proper ``__signature__``,
        ``__annotations__``, ``__doc__``, and ``__name__`` so that
        ``signature_to_params`` and ``function_to_model`` work on it.
        """
        method = spec.get("method", "get").lower()
        path = spec["path"]
        description = spec.get("description", spec.get("summary", ""))
        operation_id = spec.get("operationId", path.strip("/").replace("/", "_").replace("{", "").replace("}", ""))
        parameters = spec.get("parameters", [])

        # Build Python function parameters and collect descriptions
        # for a synthetic google-style docstring.
        py_params = []
        annotations = {}
        param_docs: list[str] = []  # lines for Args: section
        for p in parameters:
            pname = p["name"]
            required = p.get("required", False)
            enum = p.get("enum")
            pdesc = p.get("description", "")

            if enum:
                # Enum → Literal annotation; default to first value or None
                ptype = Literal[tuple(enum)]
                if required:
                    default = enum[0]
                else:
                    default = p.get("default", None)
            else:
                # Check format first (e.g. date, date-time) then fall back to base type
                fmt = p.get("format")
                if fmt and fmt in _OPENAPI_FORMAT_MAP:
                    ptype = _OPENAPI_FORMAT_MAP[fmt]
                else:
                    ptype = _OPENAPI_TYPE_MAP.get(p.get("type", "string"), str)
                if required:
                    default = p.get("default", inspect.Parameter.empty)
                else:
                    # Use None for optional params so Selector/widget defaults work
                    default = p.get("default", None)

            py_params.append(inspect.Parameter(
                pname,
                inspect.Parameter.KEYWORD_ONLY,
                default=default,
                annotation=ptype,
            ))
            annotations[pname] = ptype
            if pdesc:
                param_docs.append(f"    {pname}: {pdesc}")

        # Build a google-style docstring so that doc_descriptions() can
        # extract per-parameter descriptions for widget tooltips / agent use.
        docstring = description or ""
        if param_docs:
            docstring += "\n\nArgs:\n" + "\n".join(param_docs)

        base_url = self.base_url
        headers = dict(self.headers)

        async def endpoint_func(**kwargs) -> pd.DataFrame:
            import httpx

            # Separate path params from query params
            path_params = {}
            query_params = {}
            for p in parameters:
                val = kwargs.get(p["name"])
                if val is None or val == "":
                    continue
                # Serialize date/datetime to ISO strings for the API
                if isinstance(val, datetime.datetime):
                    val = val.isoformat()
                elif isinstance(val, datetime.date):
                    val = val.isoformat()
                if p.get("in") == "path":
                    path_params[p["name"]] = val
                else:
                    query_params[p["name"]] = val

            url = f"{base_url}{path}"
            for k, v in path_params.items():
                url = url.replace(f"{{{k}}}", str(v))

            async with httpx.AsyncClient(headers=headers, timeout=30) as client:
                resp = await client.request(method, url, params=query_params)
                resp.raise_for_status()

                # Handle non-JSON responses (e.g. XML from TAF endpoints)
                content_type = resp.headers.get("content-type", "")
                if "json" in content_type:
                    data = resp.json()
                else:
                    # Return raw text as a single-row DataFrame
                    return pd.DataFrame({"response": [resp.text]})

            # Flatten GeoJSON and nested responses
            if isinstance(data, dict):
                # GeoJSON: extract features
                if "features" in data and isinstance(data["features"], list):
                    features = data["features"]
                    if features:
                        return pd.json_normalize([
                            {**f.get("properties", {}), "geometry": str(f.get("geometry", ""))}
                            for f in features
                        ])
                    return pd.DataFrame()
                # Nested properties (common in weather.gov)
                if "properties" in data and isinstance(data["properties"], dict):
                    props = data["properties"]
                    # Forecast periods
                    if "periods" in props and isinstance(props["periods"], list):
                        return pd.json_normalize(props["periods"])
                    return pd.json_normalize([props])
                return pd.json_normalize([data])
            elif isinstance(data, list):
                return pd.json_normalize(data) if data else pd.DataFrame()
            return pd.DataFrame({"response": [str(data)]})

        endpoint_func.__name__ = operation_id
        endpoint_func.__qualname__ = operation_id
        endpoint_func.__doc__ = docstring
        endpoint_func.__signature__ = inspect.Signature(py_params)
        endpoint_func.__annotations__ = annotations
        return endpoint_func

    async def _fetch_data(self, action_name: str, **params) -> SourceResult:
        func = self._actions[action_name]
        self.progress(f"Calling {action_name}\u2026")

        try:
            df = await func(**params)
        except Exception as e:
            return SourceResult.empty(f"API error: {e}")

        if df is None or df.empty:
            return SourceResult.empty(f"{action_name} returned no data.")
        table = action_name.lower().replace(" ", "_")
        return SourceResult.from_dataframe(df, table)


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
        # Defer spec loading to onload so it doesn't block init
        self._layout.loading = True
        pn.state.onload(self._load_spec)

    def _setup_actions(self):
        # Don't register actions in __init__ — we need to fetch the spec first.
        # Actions are registered in _load_spec after the spec is fetched.
        pass

    async def _load_spec(self):
        """Fetch and parse the OpenAPI spec, then register actions."""
        try:
            import httpx
            async with httpx.AsyncClient(headers=self.headers, timeout=30) as client:
                resp = await client.get(self.spec_url)
                resp.raise_for_status()
                spec = resp.json()

            endpoints = self._parse_spec(spec)
            if endpoints:
                self.endpoints = endpoints
                actions = {}
                for name, ep_spec in endpoints.items():
                    actions[name] = self._make_endpoint_callable(ep_spec)
                self._register_actions(actions)
        except Exception as e:
            self._show_message(f"Failed to load OpenAPI spec: {e}", error=True)
        finally:
            self._layout.loading = False

        if self.vector_store is not None:
            asyncio.create_task(self._embed_actions())  # noqa: RUF006

    def _parse_spec(self, spec: dict) -> dict[str, dict]:
        """
        Parse an OpenAPI 3.x spec into endpoint definitions.

        Resolves ``$ref`` references for parameters and extracts
        path + query parameters with types and descriptions.
        """
        # Extract base URL from spec if not explicitly provided
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

            # Path-level parameters (shared by all methods on this path)
            path_params = [
                self._resolve_param(p, components)
                for p in path_item.get("parameters", [])
            ]

            for method in ("get", "post", "put", "delete"):
                if method not in path_item:
                    continue
                operation = path_item[method]

                # Merge path-level and operation-level parameters
                op_params = [
                    self._resolve_param(p, components)
                    for p in operation.get("parameters", [])
                ]
                # Operation params override path params by name
                merged = {p["name"]: p for p in path_params}
                merged.update({p["name"]: p for p in op_params})
                params = list(merged.values())

                # Filter out header/cookie params — only path + query
                params = [p for p in params if p.get("in") in ("path", "query")]

                summary = operation.get("summary", "")
                description = operation.get("description", summary)
                operation_id = operation.get("operationId", "")

                # Build display name
                if summary:
                    display_name = summary
                elif operation_id:
                    display_name = operation_id.replace("_", " ").title()
                else:
                    display_name = f"{method.upper()} {path}"

                endpoints[display_name] = {
                    "method": method,
                    "path": path,
                    "summary": summary,
                    "description": description,
                    "operationId": operation_id,
                    "parameters": params,
                }

        return endpoints

    def _path_included(self, path: str) -> bool:
        """Check if a path passes the include/exclude filters."""
        if self.exclude_paths:
            for prefix in self.exclude_paths:
                if path.startswith(prefix):
                    return False
        if self.include_paths:
            return any(path.startswith(prefix) for prefix in self.include_paths)
        return True

    @staticmethod
    def _resolve_param(param_spec: dict, components: dict) -> dict:
        """
        Resolve a ``$ref`` parameter reference and flatten into a
        simple dict with ``name``, ``in``, ``type``, ``required``,
        ``description``, ``default``.
        """
        if "$ref" in param_spec:
            ref_path = param_spec["$ref"]  # e.g. "#/components/parameters/AlertStatus"
            parts = ref_path.lstrip("#/").split("/")
            resolved = components
            for part in parts[1:]:  # skip "components"
                resolved = resolved.get(part, {})
            param_spec = resolved

        schema = param_spec.get("schema", {})
        ptype = schema.get("type", "string")

        # Handle enums
        enum = schema.get("enum")

        result = {
            "name": param_spec.get("name", ""),
            "in": param_spec.get("in", "query"),
            "type": ptype,
            "required": param_spec.get("required", False),
            "description": param_spec.get("description", ""),
        }
        if "default" in schema:
            result["default"] = schema["default"]
        if enum:
            result["enum"] = enum
        fmt = schema.get("format")
        if not fmt:
            # Fallback: infer format from the parameter name when the
            # spec omits it (auto-generated specs often do).
            pname = result["name"].lower()
            fmt = _NAME_FORMAT_HINTS.get(pname)
        if fmt:
            result["format"] = fmt
        return result
