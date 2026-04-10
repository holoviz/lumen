from __future__ import annotations

import panel as pn
import param

from .base import BaseSourceControls, SourceResult

# ─────────────────────────────────────────────────────────────────────────────
# PARAMETRIC SOURCE CONTROLS
# ─────────────────────────────────────────────────────────────────────────────

class ParametricSourceControls(BaseSourceControls):
    """
    Base class for controls that expose user-facing parameters, then fetch
    data when the load button is clicked.

    Subclasses define their query parameters as normal ``param.Parameter``
    class attributes.  The base class auto-generates Panel widgets for every
    parameter that is *not* inherited from ``BaseSourceControls`` and whose
    ``precedence`` is ``None`` or ``>= 0`` (the Panel convention for
    "show this widget").

    Subclasses implement one method:

    - ``_fetch_data(**params) -> SourceResult``

    where ``**params`` is a dict of ``{param_name: current_value}`` for
    every auto-detected (or explicitly listed) query parameter.

    To hand-pick which params become widgets, set the class attribute::

        _query_params = ["start_date", "end_date", "region"]

    To customise the widget layout, override ``_render_controls()``.

    The ``_get_parameter_schema()`` method returns a JSON-schema-like dict
    suitable for future agent use.
    """

    load_mode = "button"
    load_button_label = "Fetch Data"
    load_button_icon = "download"

    _query_params = param.List(default=[], doc="""
        Explicit list of param names to expose as widgets.
        When empty, all params not inherited from BaseSourceControls
        and with precedence >= 0 are auto-detected.""")

    __abstract = True

    # ──────────────────────────────────────────────────────────────────────────
    # Widget auto-generation
    # ──────────────────────────────────────────────────────────────────────────

    def _get_query_param_names(self) -> list[str]:
        """Return the ordered list of param names to expose as widgets."""
        if self._query_params:
            return list(self._query_params)

        base_names = set(BaseSourceControls.param)
        base_names.add("_query_params")

        return [
            name
            for name, p in self.param.objects().items()
            if name not in base_names
            and (p.precedence is None or p.precedence >= 0)
        ]

    def _render_controls(self) -> list:
        """Auto-generate one widget per query param."""
        widgets = []
        for name in self._get_query_param_names():
            widget = pn.Param(
                self.param[name],
                widgets={name: {"sizing_mode": "stretch_width"}},
                show_name=True,
            )[0]
            widgets.append(widget)
        return widgets

    # ──────────────────────────────────────────────────────────────────────────
    # Load lifecycle
    # ──────────────────────────────────────────────────────────────────────────

    async def _load(self) -> SourceResult:
        param_values = {name: getattr(self, name) for name in self._get_query_param_names()}
        return await self._fetch_data(**param_values)

    async def _fetch_data(self, **params) -> SourceResult:
        """
        Fetch data given the current parameter values.

        Parameters
        ----------
        **params
            Current value of every query parameter, keyed by param name.

        Returns
        -------
        SourceResult
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} must implement _fetch_data(**params)"
        )

    # ──────────────────────────────────────────────────────────────────────────
    # Schema (for future agent use)
    # ──────────────────────────────────────────────────────────────────────────

    def _get_parameter_schema(self) -> dict:
        """
        Return a JSON-schema-like dict describing the query parameters.

        Useful for agent-driven parameter filling.
        """
        schema = {}
        for name in self._get_query_param_names():
            p = self.param[name]
            entry: dict = {"description": p.doc or ""}

            if isinstance(p, param.Selector):
                entry["type"] = "enum"
                entry["options"] = list(p.objects)
            elif isinstance(p, param.Integer):
                entry["type"] = "integer"
                if p.bounds:
                    entry["min"], entry["max"] = p.bounds
            elif isinstance(p, param.Number):
                entry["type"] = "number"
                if p.bounds:
                    entry["min"], entry["max"] = p.bounds
            elif isinstance(p, param.String):
                entry["type"] = "string"
            elif isinstance(p, param.Date):
                entry["type"] = "date"
            elif isinstance(p, param.Boolean):
                entry["type"] = "boolean"
            elif isinstance(p, param.List):
                entry["type"] = "list"
            else:
                entry["type"] = type(p).__name__

            schema[name] = entry
        return schema
