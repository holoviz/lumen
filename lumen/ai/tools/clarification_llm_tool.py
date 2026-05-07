import asyncio

import panel as pn

from panel.chat import ChatFeed
from panel_material_ui import (
    Button, ChatAreaInput, Column, RadioBoxGroup, TextInput, Typography,
)

from ..context import TContext
from .base import FunctionTool

OTHER_OPTION = "Other"


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
        submitted = []
        if cleaned_options:
            cleaned_options.append(OTHER_OPTION)
            confirm = Button(icon="check", label="Confirm", margin=(5, 0, 10, 20))
            widget = RadioBoxGroup(options=cleaned_options, inline=False, margin=(0, 20))
            text_input = TextInput(
                placeholder="Type your own response...",
                width=500,
                margin=(0, 20, 10, 20),
                visible=False,
            )

            def _on_confirm(_):
                if widget.value == OTHER_OPTION:
                    val = text_input.value.strip()
                else:
                    val = widget.value.strip()
                if val:
                    submitted.append(val)

            confirm.on_click(_on_confirm)
            text_input.param.watch(_on_confirm, "enter_pressed")
            widget.param.watch(
                lambda e: setattr(text_input, "visible", e.new == OTHER_OPTION), "value"
            )
            out = Column(label, widget, text_input, confirm, max_width=600)
        else:
            widget = ChatAreaInput(enable_upload=False, width=500, margin=(0, 20, 20, 20))
            widget.param.watch(lambda _: submitted.append(widget.value), "enter_pressed")
            out = Column(label, widget, max_width=600)
        interface.send(out, user=user, respond=False)

        while True:
            await asyncio.sleep(0.1)
            if not cleaned_options:
                widget.focus()
            if not submitted:
                continue
            value = submitted[-1].strip()
            if not value:
                continue

            with pn.io.hold():
                label.object = f"{label.object}\n\n> {value}"
                out[:] = [label]

            # Store clarification in context for downstream agents (e.g. validation)
            entry = f"Q: '{question}' | User Later Input: '{value}'"
            existing = context.get("clarifications", [])
            context["clarifications"] = existing + [entry]

            return f"User provided following clarification: {value}"

    return FunctionTool(
        request_clarification,
        purpose=(
            "Resolve an ambiguous or underspecified request before planning. Returns "
            "the disambiguated intent so planning can proceed with definite meanings. "
            "Use when (a) multiple reasonable interpretations would lead to "
            "substantially different plans, (b) a required parameter is missing and "
            "cannot be sensibly defaulted, or (c) a term is genuinely unrecognized. "
            "Prefer reasonable defaults and stated assumptions over resolving. "
            "All ambiguity must be routed through this tool — never ask clarifying "
            "questions in message content, as the user cannot respond to content-mode "
            "questions in this system. Provide `question` always and `options` only "
            "for constrained choices."
        )
    )
