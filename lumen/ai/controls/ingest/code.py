from __future__ import annotations

import asyncio

from collections.abc import Callable

import panel as pn
import param

from ...utils import result_to_dataframe
from .parametric import ParametricSourceControls
from .result import SourceResult

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
        A single callable, or ``{name: callable}``.
    param_overrides : dict, optional
        ``{action_key: {param_name: override}}``.
        The *action key* is the original method or function name
        (e.g. ``"list_aggs"``), not the title-cased display label
        shown in the UI.  For ``functions={name: callable}`` the
        action key is the dict key you provided.
        Override is either a ``param.Parameter`` (full replacement)
        or a ``dict`` of keyword overrides (merged into auto-detected).
    skip_params : frozenset[str], optional
        Parameter names to skip during signature introspection.
        Defaults to ``{"self", "cls", "return"}``.  Add SDK-internal
        names like ``"raw"`` or ``"params"`` here.
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
                "list_aggs": {  # keyed by the method name, not the UI label
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
            skip_params=frozenset({"self", "cls", "return", "raw"}),
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
        ``{action_key: {param_name: override}}`` where *action_key* is
        the original method or function name (e.g. ``"list_aggs"``),
        not the title-cased display label.  For ``functions={name:
        callable}`` the action key is the dict key you provided.
        Override is either a ``param.Parameter`` (full replacement)
        or a ``dict`` of keyword overrides merged into the
        auto-detected parameter.""")

    skip_params = param.Parameter(default=None, precedence=-1, doc="""
        Parameter names to skip during signature introspection.
        Passed through to ``signature_to_params``.  Defaults to
        ``{"self", "cls", "return"}``.  Add SDK-specific names
        like ``"raw"`` or ``"params"`` here.""")

    table_name = param.String(default="data", precedence=-1, doc="""
        Default table name when wrapping a bare DataFrame result.""")

    label = (
        '<span class="material-icons" style="vertical-align: middle;">'
        "code"
        "</span> Code"
    )

    def _setup_actions(self):
        """Resolve and register actions after layout is built.

        ``param_overrides`` is keyed by the original method/function
        name; UI display labels are derived by title-casing.
        """
        raw_actions: dict[str, Callable] = {}

        if self.instance is not None:
            for method_name in self.methods:
                raw_actions[method_name] = getattr(self.instance, method_name)

        if callable(self.functions):
            raw_actions.setdefault(self.functions.__name__, self.functions)
        elif isinstance(self.functions, dict):
            raw_actions.update(self.functions)

        if not raw_actions:
            return

        overrides = self.param_overrides or {}
        actions = {
            raw.replace("_", " ").title(): func
            for raw, func in raw_actions.items()
        }
        display_overrides = {
            raw.replace("_", " ").title(): overrides[raw]
            for raw in raw_actions
            if raw in overrides
        }

        self._register_actions(
            actions,
            param_overrides=display_overrides,
            skip_params=self.skip_params,
        )

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

        df = result_to_dataframe(result)
        if df is None:
            return SourceResult.empty(f"{action_name} returned no data.")
        return SourceResult.from_dataframe(df, self.table_name)
