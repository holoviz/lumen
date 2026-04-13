from __future__ import annotations

import panel as pn
import param

from panel_material_ui import Column as MuiColumn, Select

from ...translate import doc_descriptions, signature_to_params
from .base import BaseSourceControls
from .result import SourceResult

# ─────────────────────────────────────────────────────────────────────────────
# PARAMETRIC SOURCE CONTROLS
# ─────────────────────────────────────────────────────────────────────────────


class ParametricSourceControls(BaseSourceControls):
    """
    Base class for controls that expose user-facing parameters, supporting
    two patterns:

    **Action-based** (used by ``CodeSourceControls``): multiple callables
    are registered via ``_register_actions``.  Each callable's signature is
    introspected via ``signature_to_params`` and rendered as widgets via
    ``pn.Param`` on a dynamically created ``Parameterized`` sub-object.
    Switching the active action swaps the widget panel.

    **Subclass-based** (used by ``URLSourceControls``): parameters are
    declared as class-level ``param.Parameter`` attributes on the subclass.
    ``pn.Param(self, parameters=...)`` renders them directly.  No actions
    are registered.

    Both the human path (``pn.Param`` widgets) and the agent path
    (``function_to_model`` via ``as_tools``) consume the same function
    signatures as their single source of truth.

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
        self._setup_actions()

    def _setup_actions(self):
        """
        Hook for subclasses to register actions after the layout is built.

        Called at the end of ``ParametricSourceControls.__init__``, after
        ``super().__init__`` has created the layout (including the empty
        ``_controls_area``).  Override this in subclasses instead of
        manipulating ``_pending_*`` temporaries.
        """

    # ──────────────────────────────────────────────────────────────────────────
    # Action registration
    # ──────────────────────────────────────────────────────────────────────────

    def _register_actions(
        self,
        actions: dict[str, callable],
        param_overrides: dict[str, dict[str, param.Parameter | dict]] | None = None,
        skip_params: frozenset[str] | None = None,
    ):
        """
        Store callables and build a ``Parameterized`` sub-object per action.

        Parameters
        ----------
        actions : dict[str, callable]
            ``{display_name: callable}``.
        param_overrides : dict, optional
            ``{action_name: {param_name: override}}`` where *override* is
            either a ``param.Parameter`` instance (full replacement) or a
            ``dict`` of keyword overrides merged into the auto-detected
            parameter.
        skip_params : frozenset[str], optional
            Parameter names to skip during introspection (passed through
            to ``signature_to_params``).  Defaults to
            ``{"self", "cls", "return"}``.
        """
        self._actions = dict(actions)
        param_overrides = param_overrides or {}

        for name, func in self._actions.items():
            kwargs = {}
            if skip_params is not None:
                kwargs["skip"] = skip_params
            params_dict = signature_to_params(func, **kwargs)
            self._apply_overrides(params_dict, param_overrides.get(name, {}))

            cls_name = name.replace(" ", "") + "Params"
            action_cls = type(cls_name, (param.Parameterized,), params_dict)
            self._action_models[name] = action_cls()

        names = list(self._actions)
        self._action_selector.options = names
        self._action_selector.visible = len(names) > 1
        if names:
            self._action_selector.value = names[0]
            self._rebuild_controls_area()

    @staticmethod
    def _apply_overrides(
        params_dict: dict[str, param.Parameter],
        overrides: dict[str, param.Parameter | dict],
    ):
        """
        Apply user overrides to an auto-detected params dict in-place.

        Each override is either a full ``param.Parameter`` replacement or
        a ``dict`` of keyword overrides merged into the existing parameter.
        """
        for pname, override in overrides.items():
            if isinstance(override, param.Parameter):
                params_dict[pname] = override
            elif isinstance(override, dict) and pname in params_dict:
                existing = params_dict[pname]
                param_cls = type(existing)
                # Extract all non-default slot values from existing param
                current: dict = {}
                for slot in existing.__class__.__slots__:
                    if slot.startswith("_"):
                        continue
                    val = getattr(existing, slot, None)
                    if val is not None:
                        current[slot] = val
                current.update(override)
                params_dict[pname] = param_cls(**current)

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

        Always included in the layout.  Starts empty for the action-based
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
        # Action selector lives outside _controls_area to avoid
        # re-parenting the widget on every action switch.
        return [self._action_selector, self._controls_area]

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
