from __future__ import annotations

import asyncio
import collections.abc

import pandas as pd
import panel as pn
import param

from panel_material_ui import Column as MuiColumn, Select

from ....sources.base import Source
from ...translate import doc_descriptions, signature_to_params
from .base import BaseSourceControls
from .result import SourceResult

# ─────────────────────────────────────────────────────────────────────────────
# PARAMETRIC SOURCE CONTROLS
# ─────────────────────────────────────────────────────────────────────────────


class ParametricSourceControls(BaseSourceControls):
    """
    Base class for controls that expose user-facing parameters and support
    multiple named actions (functions / endpoints).

    Each action's parameters are introspected from its callable's signature
    and rendered as widgets via ``pn.Param`` on a dynamically created
    ``Parameterized`` sub-object.  When the active action changes, a fresh
    sub-object is created and the widget area is rebuilt.

    Both human (``pn.Param`` widgets) and agent (``function_to_model`` via
    :meth:`as_tools`) consume the same function signatures as their source
    of truth.

    Subclasses implement:

    - ``_fetch_data(action_name, **params) -> SourceResult``
    """

    vector_store = param.Parameter(default=None, precedence=-1, doc="""
        Optional VectorStore. When provided, actions are embedded
        for similarity-based agent discovery.""")

    _query_params = param.List(default=[], precedence=-1, doc="""
        Explicit list of param names to expose as widgets when using the
        subclass pattern (class-level params, no registered actions).""")

    load_mode = "button"
    load_button_label = "Fetch Data"
    load_button_icon = "download"

    __abstract = True

    def __init__(self, **params):
        # Set instance state BEFORE super().__init__ because the base
        # class __init__ calls _render_layout() → _render_controls()
        # which accesses these attributes.
        self._actions: dict[str, callable] = {}
        self._action_models: dict[str, param.Parameterized] = {}
        self._action_selector = Select(
            name="Action", options=[], sizing_mode="stretch_width",
            visible=False,
        )
        self._action_selector.param.watch(self._on_action_change, "value")
        self._controls_area = MuiColumn(sizing_mode="stretch_width")
        super().__init__(**params)

    # ──────────────────────────────────────────────────────────────────────────
    # Action registration
    # ──────────────────────────────────────────────────────────────────────────

    def _register_actions(
        self,
        actions: dict[str, callable],
        param_overrides: dict[str, dict[str, param.Parameter | dict]] | None = None,
    ):
        """
        Store callables and build a ``Parameterized`` sub-object per action.

        Parameters
        ----------
        actions : dict[str, callable]
            ``{display_name: callable}`` — each callable's signature is
            introspected via ``signature_to_params``.
        param_overrides : dict, optional
            ``{action_name: {param_name: override}}`` where *override* is
            either a ``param.Parameter`` instance (full replacement) or a
            ``dict`` of keyword overrides merged into the auto-detected
            parameter.
        """
        self._actions = dict(actions)
        param_overrides = param_overrides or {}

        # Build one Parameterized sub-object per action
        for name, func in self._actions.items():
            params_dict = signature_to_params(func)
            overrides = param_overrides.get(name, {})
            for pname, override in overrides.items():
                if isinstance(override, param.Parameter):
                    params_dict[pname] = override
                elif isinstance(override, dict) and pname in params_dict:
                    existing = params_dict[pname]
                    param_cls = type(existing)
                    current: dict = {}
                    if existing.default is not None:
                        current["default"] = existing.default
                    if existing.doc:
                        current["doc"] = existing.doc
                    if existing.allow_None:
                        current["allow_None"] = existing.allow_None
                    if isinstance(existing, (param.Integer, param.Number)) and existing.bounds != (None, None):
                        current["bounds"] = existing.bounds
                    current.update(override)
                    params_dict[pname] = param_cls(**current)

            cls_name = name.replace(" ", "") + "Params"
            action_cls = type(cls_name, (param.Parameterized,), params_dict)
            self._action_models[name] = action_cls()

        # Configure selector
        names = list(self._actions)
        self._action_selector.options = names
        self._action_selector.visible = len(names) > 1
        if names:
            self._action_selector.value = names[0]
            self._rebuild_controls_area()

    # ──────────────────────────────────────────────────────────────────────────
    # Action switching
    # ──────────────────────────────────────────────────────────────────────────

    def _on_action_change(self, event):
        self._rebuild_controls_area()

    def _rebuild_controls_area(self):
        """Rebuild ``_controls_area`` with widgets for the active action."""
        active = self._action_selector.value
        model = self._action_models.get(active)
        children = []
        if self._action_selector.visible:
            children.append(self._action_selector)
        if model is not None:
            children.append(
                pn.Param(model, show_name=False, sizing_mode="stretch_width")
            )
        self._controls_area[:] = children

    # ──────────────────────────────────────────────────────────────────────────
    # Subclass pattern support (class-level params, no registered actions)
    # ──────────────────────────────────────────────────────────────────────────

    def _get_query_param_names(self) -> list[str]:
        """Return class-level param names for the subclass pattern."""
        if self._query_params:
            return list(self._query_params)
        base_names = set(ParametricSourceControls.param)
        return [
            name
            for name, p in self.param.objects().items()
            if name not in base_names
            and (p.precedence is None or p.precedence >= 0)
        ]

    @property
    def _uses_actions(self) -> bool:
        """Whether this instance uses the action-based pattern."""
        return bool(self._actions)

    # ──────────────────────────────────────────────────────────────────────────
    # Widget rendering via pn.Param
    # ──────────────────────────────────────────────────────────────────────────

    def _render_controls(self) -> list:
        """Return the controls area container.

        Always included in the layout. Starts empty for the action-based
        pattern (populated by ``_rebuild_controls_area`` after init),
        or populated immediately for the subclass pattern.
        """
        if not self._uses_actions:
            query_names = self._get_query_param_names()
            if query_names:
                self._controls_area[:] = [
                    pn.Param(
                        self,
                        parameters=query_names,
                        show_name=False,
                        sizing_mode="stretch_width",
                    )
                ]
        return [self._controls_area]

    # ──────────────────────────────────────────────────────────────────────────
    # Load lifecycle
    # ──────────────────────────────────────────────────────────────────────────

    @property
    def _active_action_name(self) -> str | None:
        return self._action_selector.value if self._uses_actions else None

    async def _load(self) -> SourceResult:
        if self._uses_actions:
            active = self._active_action_name
            model = self._action_models.get(active)
            if model is None:
                return SourceResult.empty("No action selected.")
            values = model.param.values()
            param_values = {k: v for k, v in values.items() if k != "name"}
            return await self._fetch_data(active, **param_values)
        else:
            all_values = self.param.values()
            param_values = {n: all_values[n] for n in self._get_query_param_names()}
            return await self._fetch_data(self.__class__.__name__, **param_values)

    async def _fetch_data(self, action_name: str, **params) -> SourceResult:
        """
        Fetch data for the named action with the given parameter values.

        Subclasses must implement this method.
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} must implement _fetch_data(action_name, **params)"
        )

    # ──────────────────────────────────────────────────────────────────────────
    # Agent integration
    # ──────────────────────────────────────────────────────────────────────────

    async def _embed_actions(self):
        """Embed action names + docstrings into vector store."""
        if self.vector_store is None:
            return
        items = []
        for name, func in self._actions.items():
            description, _ = doc_descriptions(func)
            items.append({
                "text": f"{name}: {description}" if description else name,
                "metadata": {"type": "source_action", "action_name": name},
            })
        if items:
            await self.vector_store.upsert(items)

    def as_tools(
        self, query: str | None = None, top_k: int = 5,
    ) -> list[tuple[str, callable]]:
        """
        Return ``(display_name, callable)`` pairs for the coordinator
        to wrap in ``FunctionTool``.
        """
        return list(self._actions.items())

    async def as_tools_async(
        self, query: str | None = None, top_k: int = 5,
    ) -> list[tuple[str, callable]]:
        """Async version with vector-store filtering."""
        if query and self.vector_store and len(self._actions) > top_k:
            results = await self.vector_store.query(query, top_k=top_k)
            relevant = {r["metadata"]["action_name"] for r in results}
            return [
                (name, func)
                for name, func in self._actions.items()
                if name in relevant
            ]
        return list(self._actions.items())

    async def load_action(self, action_name: str, **params) -> SourceResult:
        """Programmatic entry point for AI agents."""
        return await self._run_load(self._fetch_data(action_name, **params))


# ─────────────────────────────────────────────────────────────────────────────
# CODE SOURCE CONTROLS
# ─────────────────────────────────────────────────────────────────────────────


class CodeSourceControls(ParametricSourceControls):
    """
    Wraps Python callables or object methods as data source controls.

    Accepts either an object instance with method names, a dict of
    callables, or a single callable.  Each callable's signature is
    introspected for both widget generation (``pn.Param``) and agent
    tool schemas (``function_to_model``).

    Parameters
    ----------
    instance : object, optional
        Object whose bound methods will be exposed as actions.
    methods : list[str], optional
        Method names on ``instance`` to expose.
    functions : callable | dict[str, callable], optional
        A single callable, or ``{display_name: callable}``.
    param_overrides : dict, optional
        ``{action_name: {param_name: override}}``.
        Override is either a ``param.Parameter`` (full replacement)
        or a ``dict`` of keyword overrides (merged into auto-detected).
    table_name : str
        Default table name when wrapping DataFrame results.

    Examples
    --------
    Zero-config for API clients::

        from massive import RESTClient
        client = RESTClient()
        controls = CodeSourceControls(
            instance=client,
            methods=["list_aggs", "get_ticker_details"],
        )

    With overrides for better defaults and widget types::

        controls = CodeSourceControls(
            instance=client,
            methods=["list_aggs", "get_ticker_details"],
            param_overrides={
                "List Aggs": {
                    "ticker": param.Selector(
                        default="AAPL",
                        objects=["AAPL", "MSFT", "GOOGL", "TSLA"],
                    ),
                    "timespan": param.Selector(
                        default="day",
                        objects=["minute", "hour", "day", "week"],
                    ),
                    "multiplier": {"default": 1, "bounds": (1, 100)},
                    "from_": {"default": "2024-01-01"},
                    "to": {"default": "2024-12-31"},
                    "limit": {"default": 5000, "bounds": (1, 50000)},
                },
            },
        )
    """

    # precedence=-1 hides from pn.Param / _get_query_param_names
    instance = param.Parameter(default=None, precedence=-1, doc="""
        Object instance whose bound methods will be exposed as actions.""")

    methods = param.List(default=[], item_type=str, precedence=-1, doc="""
        Method names on ``instance`` to expose.""")

    functions = param.Parameter(default=None, precedence=-1, doc="""
        A single callable, or a ``dict[str, callable]``.""")

    param_overrides = param.Parameter(default=None, precedence=-1, doc="""
        ``{action_name: {param_name: override}}``. Override is either
        a ``param.Parameter`` (full replacement) or a ``dict`` of
        keyword overrides merged into the auto-detected parameter.""")

    table_name = param.String(default="data", precedence=-1, doc="""
        Default table name when wrapping a bare DataFrame result.""")

    label = (
        '<span class="material-icons" style="vertical-align: middle;">'
        "code"
        "</span> Code"
    )

    def __init__(self, **params):
        # Resolve actions from params before super().__init__ builds
        # the layout — but don't register yet (that happens after).
        instance = params.get("instance")
        methods = params.get("methods", [])
        functions = params.get("functions")
        overrides = params.get("param_overrides") or {}

        actions: dict[str, callable] = {}

        # Instance + methods → bound methods
        if instance is not None:
            for method_name in methods:
                bound_method = getattr(instance, method_name)
                display = method_name.replace("_", " ").title()
                actions[display] = bound_method

        # Functions: single callable or dict
        if callable(functions):
            name = functions.__name__.replace("_", " ").title()
            actions.setdefault(name, functions)
        elif isinstance(functions, dict):
            actions.update(functions)

        self._pending_actions = actions
        self._pending_overrides = overrides

        # super().__init__ → _render_layout() → _render_controls()
        # At this point _actions is empty, _controls_area is in layout
        super().__init__(**params)

        # Now register actions and populate _controls_area in place
        if self._pending_actions:
            self._register_actions(self._pending_actions, self._pending_overrides)

        del self._pending_actions
        del self._pending_overrides

        # Background embed — fire and forget
        if self.vector_store is not None:
            pn.state.execute(self._embed_actions)

    async def _fetch_data(self, action_name: str, **params) -> SourceResult:
        func = self._actions[action_name]
        self.progress("Fetching data\u2026")

        try:
            if param.parameterized.iscoroutinefunction(func):
                result = await func(**params)
            else:
                result = await asyncio.to_thread(func, **params)
        except Exception as e:
            return SourceResult.empty(f"Error calling {action_name}: {e}")

        df = self._result_to_dataframe(result)
        if df is None:
            return SourceResult.empty(f"{action_name} returned no data.")
        return SourceResult.from_dataframe(df, self.table_name)

    @staticmethod
    def _result_to_dataframe(result) -> pd.DataFrame | None:
        """
        Normalize a raw function return value into a DataFrame.

        Handles the common shapes returned by Python API clients:
        DataFrame, iterator/generator of model objects, list[dict],
        single model object, dict.
        """
        if isinstance(result, pd.DataFrame):
            return result
        if isinstance(result, Source):
            return None

        # Materialise iterators / generators
        if isinstance(result, (collections.abc.Iterator, collections.abc.Generator)):
            try:
                result = list(result)
            except Exception:
                return None

        if isinstance(result, list):
            if not result:
                return pd.DataFrame()
            first = result[0]
            if isinstance(first, dict):
                return pd.json_normalize(result)
            if isinstance(first, (str, int, float, bool)):
                return pd.DataFrame({"value": result})
            return pd.json_normalize([
                obj.__dict__ if not isinstance(obj, dict) else obj
                for obj in result
            ])

        if isinstance(result, dict):
            try:
                return pd.json_normalize(result)
            except Exception:
                return pd.DataFrame([result])

        # Single model object
        if not isinstance(result, (str, int, float, bool, type(None))):
            try:
                return pd.json_normalize([result.__dict__])
            except Exception:
                return None

        return None
