import asyncio

from typing import Any, ClassVar

import panel as pn
import param

from ...sources.base import Source
from ..context import ContextModel, TContext
from ..controls import SourceControls
from ..llm import Message
from .base import Agent


class SourceOutputs(ContextModel):

    document_sources: list[Any]

    source: Source

    sources: list[Source]


class SourceAgent(Agent):
    """
    SourceAgent renders a form that allows a user to upload or
    provide a URI to one or more datasets.
    """

    conditions = param.List(
        default=[
            "Use ONLY when user explicitly asks to upload or connect to NEW data sources (NOT if data sources already exist)",
        ])

    purpose = param.String(default="Allows a user to upload new datasets, data, or documents.")

    source_controls: ClassVar[SourceControls] = SourceControls

    _extensions = ("filedropper",)

    output_schema = SourceOutputs

    async def respond(
        self,
        messages: list[Message],
        context: TContext,
        step_title: str | None = None,
    ) -> tuple[list[Any], SourceOutputs]:
        source_controls = self.source_controls(context=context, cancellable=True, replace_controls=True)
        output = pn.Column(source_controls)
        if "source" not in context:
            help_message = "No datasets or documents were found, **please upload at least one to continue**..."
        else:
            help_message = "**Please upload new dataset(s)/document(s) to continue**, or click cancel if this was unintended..."
        output.insert(0, help_message)
        self.interface.send(output, respond=False, user=self.__class__.__name__)
        while not source_controls._add_button.clicks > 0:
            await asyncio.sleep(0.05)
            if source_controls._cancel_button.clicks > 0:
                self.interface.undo()
                self.interface.disabled = False
                return [], {}
        return [], source_controls.outputs
