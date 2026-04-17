from __future__ import annotations

import inspect

from typing import Literal

import httpx
import pandas as pd
import panel as pn
import param

from .constants import HTTP_TIMEOUT_SECONDS
from .parametric import ParametricSourceControls
from .result import SourceResult
from .utils import normalize_json_response, serialize_param_value

# ─────────────────────────────────────────────────────────────────────────────
# Pure helpers (module-level, not meant for subclass override)
# ─────────────────────────────────────────────────────────────────────────────


def _split_params(
    parameters: list[dict], kwargs: dict,
) -> tuple[dict, dict]:
    """Separate path params from query params, serializing values."""
    path_params: dict = {}
    query_params: dict = {}
    for p in parameters:
        val = kwargs.get(p["name"])
        if val is None or val == "":
            continue
        val = serialize_param_value(val)
        if p.get("in") == "path":
            path_params[p["name"]] = val
        else:
            query_params[p["name"]] = val
    return path_params, query_params


def _operation_id_from_spec(spec: dict) -> str:
    """Extract or derive a Python-safe operation id from an endpoint spec."""
    operation_id = spec.get("operationId")
    if operation_id:
        return operation_id
    path = spec.get("path", "")
    return path.strip("/").replace("/", "_").replace("{", "").replace("}", "")


def _build_docstring(description: str, param_docs: list[str]) -> str:
    """Build a google-style docstring from description + param doc lines."""
    docstring = description or ""
    if param_docs:
        docstring += "\n\nArgs:\n" + "\n".join(param_docs)
    return docstring


def _build_endpoint_closure(
    method: str,
    path: str,
    parameters: list[dict],
    base_url: str,
    headers: dict,
) -> callable:
    """Create an async closure that performs the HTTP request."""

    async def endpoint_func(**kwargs) -> pd.DataFrame:
        path_params, query_params = _split_params(parameters, kwargs)

        url = f"{base_url}{path}"
        for k, v in path_params.items():
            url = url.replace(f"{{{k}}}", str(v))

        async with httpx.AsyncClient(headers=headers, timeout=HTTP_TIMEOUT_SECONDS) as client:
            resp = await client.request(method, url, params=query_params)
            resp.raise_for_status()

            content_type = resp.headers.get("content-type", "")
            if "json" not in content_type:
                return pd.DataFrame({"response": [resp.text]})
            data = resp.json()

        return normalize_json_response(data)

    return endpoint_func


# ─────────────────────────────────────────────────────────────────────────────
# REST API SOURCE CONTROLS
# ─────────────────────────────────────────────────────────────────────────────


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
        if not self.endpoints:
            return
        actions = {
            name: self._make_endpoint_callable(spec)
            for name, spec in self.endpoints.items()
        }
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

        Subclasses can override ``_build_signature_for_param`` to
        customise how individual parameters are converted to
        ``inspect.Parameter`` objects (e.g. to add OpenAPI type mapping).
        """
        method = spec.get("method", "get").lower()
        path = spec["path"]
        description = spec.get("description", spec.get("summary", ""))
        operation_id = _operation_id_from_spec(spec)
        parameters = spec.get("parameters", [])

        py_params, annotations, param_docs = self._build_signature(parameters)
        docstring = _build_docstring(description, param_docs)
        endpoint_func = _build_endpoint_closure(
            method, path, parameters, self.base_url, dict(self.headers),
        )

        endpoint_func.__name__ = operation_id
        endpoint_func.__qualname__ = operation_id
        endpoint_func.__doc__ = docstring
        endpoint_func.__signature__ = inspect.Signature(py_params)
        endpoint_func.__annotations__ = annotations
        return endpoint_func

    def _build_signature(
        self, parameters: list[dict],
    ) -> tuple[list[inspect.Parameter], dict[str, type], list[str]]:
        """
        Convert endpoint parameter dicts into ``inspect.Parameter`` objects.

        Delegates per-parameter type resolution to
        ``_resolve_param_type`` so subclasses (e.g.
        ``OpenAPISourceControls``) can add richer type mapping.

        Returns
        -------
        tuple of (py_params, annotations, param_doc_lines)
        """
        py_params: list[inspect.Parameter] = []
        annotations: dict[str, type] = {}
        param_docs: list[str] = []

        for p in parameters:
            pname = p["name"]
            required = p.get("required", False)
            enum = p.get("enum")
            pdesc = p.get("description", "")

            if enum:
                ptype = Literal[tuple(enum)]
                default = enum[0] if required else p.get("default", None)
            else:
                ptype = self._resolve_param_type(p)
                if required:
                    default = p.get("default", inspect.Parameter.empty)
                else:
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

        return py_params, annotations, param_docs

    def _resolve_param_type(self, p: dict) -> type:
        """
        Map a parameter dict to a Python type.

        Base implementation returns ``str`` for everything.  Override
        in subclasses to add richer type mapping (e.g. OpenAPI format
        handling).
        """
        return str

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
