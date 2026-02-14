import traceback

from typing import Any

import panel as pn
import param

from panel.chat import ChatInterface
from panel.viewable import Viewer
from panel_material_ui.chat import ChatMessage

from ...dashboard import Config
from ...state import state
from ..actor import ContextProvider
from ..context import TContext
from ..llm import Llm, Message
from ..tools import ToolUser


class Agent(Viewer, ToolUser, ContextProvider):
    """
    Agents are actors responsible for taking a user query and
    performing a particular task, either by adding context or
    generating outputs.

    Agents have access to an LLM and are given context and can
    solve tasks by executing a series of prompts or by rendering
    contents such as forms or widgets to gather user input.
    """

    agents = param.List(doc="""
        List of agents this agent can invoke.""")

    debug = param.Boolean(default=False, doc="""
        Whether to enable verbose error reporting.""")

    llm = param.ClassSelector(class_=Llm, doc="""
        The LLM implementation to query.""")

    user = param.String(default="Agent", doc="""
        The name of the user that will be respond to the user query.""")

    # Panel extensions this agent requires to be loaded
    _extensions = ()

    # Maximum width of the output
    _max_width = 1200

    __abstract = True

    def __init__(self, **params):
        def _exception_handler(exception):
            traceback.print_exception(exception)
            if self.interface is None:
                return
            messages = self.interface.serialize()
            if messages and str(exception) in messages[-1]["content"]:
                return

            self.interface.send(
                f"Error cannot be resolved:\n\n{exception}", user="System", respond=False
            )

        super().__init__(**params)
        if not self.debug:
            pn.config.exception_handler = _exception_handler

        if state.config is None:
            state.config = Config(raise_with_notifications=True)
        else:
            state.config.raise_with_notifications = True

    async def _interface_callback(self, contents: list | str, user: str, instance: ChatInterface):
        await self.respond(contents)
        self._retries_left = 1

    @property
    def _steps_layout(self):
        """
        Stream to a dummy column to be able to suppress the steps output.
        """
        return pn.Column() if not self.steps_layout else self.steps_layout

    def __panel__(self):
        return self.interface

    async def _stream(self, messages: list[Message], context: TContext, **kwargs) -> Any:
        message = None
        output = self._stream_prompt("main", messages, context, field="output", **kwargs)
        try:
            async for output_chunk in output:
                if self.interface is None:
                    if message is None:
                        message = ChatMessage(output_chunk, user=self.user)
                    else:
                        message.object = output_chunk
                else:
                    message = self.interface.stream(output_chunk, replace=True, message=message, user=self.user, max_width=self._max_width)
        except Exception as e:
            traceback.print_exc()
            raise e
        return message

    async def _gather_prompt_context(self, prompt_name: str, messages: list, context: TContext, **kwargs):
        context = await super()._gather_prompt_context(prompt_name, messages, context, **kwargs)
        if "tool_context" not in context:
            context["tool_context"] = await self._use_tools(prompt_name, messages, context)
        return context

    # Public API

    @classmethod
    async def applies(cls, context: TContext) -> bool:
        """
        Additional checks to determine if the agent should be used.
        """
        return True

    async def respond(
        self,
        messages: list[Message],
        context: TContext,
        step_title: str | None = None,
    ) -> tuple[list[Any], TContext]:
        """
        Provides a response to the user query.

        The type of the response may be a simple string or an object.

        Arguments
        ---------
        messages: list[Message]
            The list of messages corresponding to the user query and any other
            system messages to be included.
        context: TContext
            A mapping containing context for the agent to perform its task.
        step_title: str | None
            If the Agent response is part of a longer query this describes
            the step currently being processed.
        """
        return [await self._stream(messages, context)], context
