from __future__ import annotations

import param

from panel.io import state
from panel.viewable import Viewer
from panel_material_ui import (
    Card, Markdown, Popup, Row, TextInput, ToggleIcon,
)

from ...config import load_yaml
from ..utils import generate_diff


class RevisionControls(Viewer):

    active = param.Boolean(False, doc="Click to revise")

    instruction = param.String(doc="Instruction to LLM to revise output")

    interface = param.Parameter()

    layout_kwargs = param.Dict(default={})

    task = param.Parameter()

    view = param.Parameter(doc="The View to revise")

    toggle_kwargs = {}

    input_kwargs = {}

    def __init__(self, **params):
        super().__init__(**params)
        input_kwargs = dict(
            max_length=200,
            margin=10,
            size="small"
        )
        input_kwargs.update(self.input_kwargs)
        self._text_input = TextInput(**input_kwargs)
        popup = Popup(
            self._text_input,
            open=self.param.active,
            anchor_origin={"horizontal": "right", "vertical": "center"},
            styles={"z-index": '1300'}
        )
        icon = ToggleIcon.from_param(
            self.param.active,
            active_icon="cancel",
            attached=[popup],
            label="",
            margin=(5, 0),
            size="small",
            icon_size="1.1em",
            **self.toggle_kwargs
        )
        popup.param.watch(self._close, "open")
        self._text_input.param.watch(self._enter_reason, "enter_pressed")
        self._row = Row(icon, **self.layout_kwargs)

    @param.depends("active", watch=True)
    def _open(self):
        if self.active:
            state.execute(self._text_input.focus, schedule=True)

    def _close(self, event):
        self.active = event.new

    def _enter_reason(self, _):
        instruction = self._text_input.value_input
        self._text_input.value = ""
        self.param.update(active=False, instruction=instruction)

    def _report_status(self, out: str, title='➕/➖ Applied Revisions'):  # noqa: RUF001
        if not out:
            return
        md = Markdown(
            out,
            margin=0,
            styles={"padding-inline": "0", "margin-left": "0", "padding": "0"},
            stylesheets=[".codehilite { margin-top: 0; margin-bottom: 0; display: inline-block; } .codehilite pre { margin-top: -1.5em; }"],
            sizing_mode="stretch_width"
        )
        self.interface.stream(Card(md, title=title), user="Assistant")

    def __panel__(self):
        return self._row


class RetryControls(RevisionControls):

    instruction = param.String(doc="Reason for retry")

    input_kwargs = {"placeholder": "Enter feedback and press the <Enter> to retry."}

    toggle_kwargs = {"icon": "auto_awesome", "description": "Prompt LLM to retry"}

    @param.depends("instruction", watch=True)
    async def _revise(self):
        if not self.instruction:
            return
        elif not hasattr(self.task.actor, 'revise'):
            invalidation_keys = set(self.task.actor.output_schema.__annotations__)
            self.task.invalidate(invalidation_keys, start=1)
            root = self.task
            while root.parent is not None:
                root = root.parent
            with root.param.update(interface=self.interface):
                await root.execute()
            return

        old_spec = self.view.spec
        messages = list(self.task.history)
        try:
            with self.task.actor.param.update(interface=self.interface), self.view.editor.param.update(loading=True):
                new_spec = await self.task.actor.revise(
                    self.instruction, messages, self.task.out_context, self.view
                )
        except Exception as e:
            self._report_status(f"```\\n{e}\\n```", title="❌ Failed to generate revisions")
            raise

        diff = generate_diff(old_spec, new_spec, filename=self.view.language or "spec")
        diff_md = f'```diff\\n{diff}\\n```'
        try:
            self.view.spec = new_spec
        except Exception:
            self.view.spec = old_spec
            self._report_status(diff_md, title="❌ Failed to apply revisions")
            raise
        self._report_status(diff_md)


class AnnotationControls(RevisionControls):
    """Controls for adding annotations to visualizations."""

    input_kwargs = {
        "placeholder": "Describe what to annotate (e.g., 'highlight peak values', 'mark outliers')..."
    }

    toggle_kwargs = {
        "description": "Add annotations to highlight key insights",
        "icon": "chat-bubble",
    }

    @param.depends("instruction", watch=True)
    async def _annotate(self):
        if not self.instruction:
            return

        old_spec = self.view.spec
        try:
            with self.task.actor.param.update(interface=self.interface), self.view.editor.param.update(loading=True):
                new_spec = await self.task.actor.annotate(
                    self.instruction, list(self.task.history), self.task.out_context, {"spec": load_yaml(self.view.spec)}
                )
        except Exception as e:
            self._report_status(f"```\\n{e}\\n```", title="❌ Failed to generate annotations")
            raise

        diff = generate_diff(old_spec, new_spec, filename=self.view.language or "spec")
        diff_md = f'```diff\\n{diff}\\n```'
        try:
            self.view.spec = new_spec
        except Exception:
            self.view.spec = old_spec
            self._report_status(diff_md, title="❌ Failed to apply annotations")
            raise
        self._report_status(diff_md)
