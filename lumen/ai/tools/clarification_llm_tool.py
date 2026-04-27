import asyncio

import panel as pn

from panel.chat import ChatFeed
from panel_material_ui import (
    Button, Column, RadioButtonGroup, TextInput, Typography,
)

from ..context import TContext
from .base import FunctionTool


def make_clarification_llm_tool(
    interface: ChatFeed | None,
    context: TContext,
    user: str = "Clarification",
) -> FunctionTool:
    async def request_clarification(question: str, options: list[str] | None = None) -> str:
        """
        Ask the user one focused question and wait for their reply.

        Parameters
        ----------
        question:
            The exact question shown to the user. One decision per call.
        options:
            Fixed answer choices for constrained decisions — the user picks one
           via radio buttons and cannot type. Each option must be a complete
           standalone answer; never include "(please specify)", "(other)", or
           similar. Leave as None for free-form text replies, including any
           question whose answer requires the user to supply information
           (definitions, names, values, descriptions).
        """
        if interface is None:
            return "Clarification unavailable because no chat interface is active."

        cleaned_options = None
        if options:
            cleaned_options = [opt.strip() for opt in options if isinstance(opt, str) and opt.strip()]

        label = Typography(question, margin=10)
        confirm = Button(icon="check", label="Confirm", margin=(5, 0, 20, 20))
        if cleaned_options:
            widget = RadioButtonGroup(options=cleaned_options, value=None, margin=(0, 20))
        else:
            widget = TextInput(sizing_mode="stretch_width", margin=(0, 20))
            widget.param.enter_pressed.rx.pipe(lambda _: confirm.focus())
            confirm.disabled = widget.rx().strip() == ""
        out = Column(label, widget, confirm)
        interface.send(out, user=user, respond=False)

        while True:
            await asyncio.sleep(0.1)
            widget.focus()
            if not confirm.clicks:
                continue
            value = widget.value.strip()
            with pn.io.hold():
                label.object = f"{label.object}\n\n> {value}"
                out[:] = [label]
            return f"User provided following clarification: {value}"

    return FunctionTool(
        request_clarification,
        purpose=(
            "Ask the user a focused clarifying question before planning when the "
            "request is ambiguous, underspecified, or depends on a preference you "
            "cannot infer. Prefer asking over guessing. Use when: (a) multiple "
            "reasonable interpretations exist, (b) a required parameter is missing, "
            "(c) the user's goal could be satisfied by substantially different plans. "
            "Provide `question` always and `options` only for constrained choices."
        )
    )
