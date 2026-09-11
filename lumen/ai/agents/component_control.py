from textwrap import dedent
from typing import Any, NotRequired

import param

from ..config import PROMPTS_DIR
from ..context import ContextModel, TContext
from ..llm import Message
from ..tools.component_control import ComponentController
from ..utils import truncate_string
from .base import Agent


class ComponentControlOutputs(ContextModel):

    ui_state: NotRequired[str]


class ComponentControlAgent(Agent):
    """
    The ComponentControlAgent inspects and drives the components of the
    application the chat interface is embedded in, e.g. the widgets of a
    dashboard.
    """

    components = param.Parameter(default=None, doc="""
        Shorthand for ``ComponentController(components=...)``. Provide a
        widget, a list or dictionary of components or an entire layout such as
        a ``panel_material_ui.Page``. Ignored if a ``controller`` is given.""")

    conditions = param.List(
        default=[
            "Use when the user asks to change, set, select, toggle, enable, disable or reset a control of the application",
            "Use when the user asks which controls, widgets or settings the application has, or what they are currently set to",
            "Use when the user describes a desired state of the application in words, e.g. 'show only the last five years'",
            "Use when the user asks for a relative change to something the application controls, e.g. 'twice as fast', 'a bit stronger', 'half of that'",
            "Use when the user asks to reach a goal that the application controls can be tuned towards, e.g. 'find a setting that keeps the queue under ten'",
            "NOT when the user asks to query, aggregate or transform data; the application controls do not run queries",
        ]
    )

    controller = param.ClassSelector(class_=ComponentController, default=ComponentController(), doc="""
        The controller exposing the components of the application.""")

    purpose = param.String(default="""
        Inspects and controls the components of the running application, such as
        the widgets of a dashboard. Discovers which controls exist, reports their
        current values and writes new values to them.""")

    prompts = param.Dict(
        default={
            "main": {
                "template": PROMPTS_DIR / "ComponentControlAgent" / "main.jinja2",
            },
        }
    )

    user = param.String(default="UI")

    output_schema = ComponentControlOutputs

    def __init__(self, **params):
        if params.get("controller") is None and params.get("components") is not None:
            params["controller"] = ComponentController(components=params["components"])
        super().__init__(**params)
        if self.components is None:
            self.components = self.controller.components
        self._purpose = self.purpose
        # The controller resolves its components lazily, so listing it as an
        # llm_tool is enough for the tools to track a live, changing layout.
        self.llm_tools = [*self.llm_tools, self.controller]
        self.param.watch(self._sync_components, "components")

    def _sync_components(self, event):
        self.controller.components = event.new

    async def applies(self, context: TContext) -> bool:
        """
        Name the controls that currently exist on the purpose.

        A coordinator routes on the purpose alone, so spelling out what can
        actually be controlled is what lets it tell a request about the
        application apart from a request about data.
        """
        specs = self.controller.specs
        if not specs:
            return False
        controls = []
        for spec in specs:
            names = [info.name for info in spec.settable]
            controls.append(spec.key if names == ["value"] else f"{spec.key} ({', '.join(names)})")
        purpose = f"{dedent(self._purpose).strip()}\n\nThe application currently exposes: "
        self.purpose = purpose + truncate_string("; ".join(controls), max_length=1500)
        return True

    async def _gather_prompt_context(self, prompt_name: str, messages: list, context: TContext, **kwargs):
        prompt_context = await super()._gather_prompt_context(prompt_name, messages, context, **kwargs)
        prompt_context["ui_state"] = self.controller.summary()
        prompt_context["list_tool"] = self.controller._tool_name("list")
        return prompt_context

    async def respond(
        self,
        messages: list[Message],
        context: TContext,
        step_title: str | None = None,
    ) -> tuple[list[Any], dict[str, Any]]:
        result = await self._stream(messages, context)
        return [result], {"ui_state": self.controller.summary()}
