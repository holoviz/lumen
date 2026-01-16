
from types import FunctionType
from typing import Any, TypedDict

import param

from panel.pane import HoloViews as HoloViewsPanel, panel as as_panel
from panel.viewable import Viewable

from ...views.base import HoloViews, View
from ..actor import Actor, ContextProvider
from ..config import PROMPTS_DIR
from ..context import ContextModel, TContext
from ..llm import Message
from ..translate import function_to_model

_TOOL_ANNOTATIONS_ATTR = "__lumen_tool_annotations__"


def define_tool(*, requires=None, provides=None, purpose=None, render_output=None):
    """
    Annotate a function so FunctionTool can read tool metadata.
    """
    def decorator(function):
        annotations = getattr(function, _TOOL_ANNOTATIONS_ATTR, {})
        if requires is not None:
            annotations["requires"] = list(requires)
        if provides is not None:
            annotations["provides"] = list(provides)
        if purpose is not None:
            annotations["purpose"] = purpose
        if render_output is not None:
            annotations["render_output"] = render_output
        setattr(function, _TOOL_ANNOTATIONS_ATTR, annotations)
        return function
    return decorator


class Tool(Actor, ContextProvider):
    """
    A Tool can be invoked by another Actor to provide additional
    context or respond to a question. Unlike an Agent they never
    interact with or on behalf of a user directly.
    """

    always_use = param.Boolean(default=False, doc="""
        Whether to always use this tool, even if it is not explicitly
        required by the current context.""")

    conditions = param.List(default=[
        "Always requires a supporting agent to interpret results"
    ])

    @classmethod
    async def applies(cls, context: TContext) -> bool:
        """
        Additional checks to determine if the tool should be used.
        """
        return True

    async def prepare(self, context: TContext):
        """
        Prepare the tool with the initial context.
        Called once when the tool is first initialized.
        """

    async def sync(self, context: TContext):
        """
        Allows the tool to update when the provided context changes.
        Subclasses should override this to handle context updates.
        """


class FunctionTool(Tool):
    """
    FunctionTool wraps arbitrary functions and makes them available as a tool
    for an LLM to call. It inspects the arguments of the function and generates
    a pydantic Model that the LLM will populate.

    The function may also consume information in memory, e.g. the current
    table or pipeline by declaring the requires parameter.

    The function may also return context to add to the current working memory
    by returning a dictionary and declaring the `provides` parameter. Any
    keys listed in the provides parameter will be copied into working memory.
    """

    formatter = param.Parameter(default="{function}({arguments}) returned: {output}", doc="""
        Formats the return value for inclusion in the global context.
        Accepts the 'function', 'arguments' and 'output' as formatting variables.""")

    function = param.Callable(default=None, allow_refs=False, doc="""
        The function to call.""")

    provides = param.List(default=[], readonly=False, constant=True, doc="""
        List of context values it provides to current working memory.""")

    requires = param.List(default=[], readonly=False, constant=True, doc="""
        List of context values it requires to be in memory.""")

    render_output = param.Boolean(default=False, doc="""
        Whether to render the tool output directly, even if it is not already a Lumen View or Panel Viewable.""")

    prompts = param.Dict(
        default={
            "main": {
                "template": PROMPTS_DIR / "FunctionTool" / "main.jinja2"
            },
        }
    )

    def __init__(self, function, **params):
        annotations = getattr(function, _TOOL_ANNOTATIONS_ATTR, {})
        if "requires" not in params and "requires" in annotations:
            params["requires"] = annotations["requires"]
        if "provides" not in params and "provides" in annotations:
            params["provides"] = annotations["provides"]
        if "purpose" not in params and "purpose" in annotations:
            params["purpose"] = annotations["purpose"]
        if "render_output" not in params and "render_output" in annotations:
            params["render_output"] = annotations["render_output"]
        model = function_to_model(function, skipped=self.requires)
        if "purpose" not in params:
            params["purpose"] = f"{model.__name__}: {model.__doc__}" if model.__doc__ else model.__name__
        super().__init__(
            function=function,
            name=function.__name__,
            **params
        )
        self._model = model

    @property
    def inputs(self):
        return TypedDict(f"{self.function.__name__}Inputs", {f: Any for f in self.requires})

    async def respond(
        self, messages: list[Message], context: TContext, **kwargs: dict[str, Any]
    ) -> tuple[list[Any], ContextModel]:
        prompt = await self._render_prompt("main", messages, context)
        kwargs = {}
        if any(field not in self.requires for field in self._model.model_fields):
            model_spec = self.prompts["main"].get("llm_spec", self.llm_spec_key)
            kwargs = await self.llm.invoke(
                messages,
                system=prompt,
                model_spec=model_spec,
                response_model=self._model,
                allow_partial=False,
                max_retries=3,
            )
        arguments = dict(kwargs, **{k: context[k] for k in self.requires})
        if param.parameterized.iscoroutinefunction(self.function):
            result = await self.function(**arguments)
        else:
            result = self.function(**arguments)
        if isinstance(result, (View, Viewable)):
            return [result], {}
        elif self.render_output:
            p = as_panel(result)
            if isinstance(p, HoloViewsPanel):
                p = HoloViews(object=p.object)
            return [p], {}
        out_model = {}
        if self.provides:
            if len(self.provides) == 1 and not isinstance(result, dict):
                out_model[self.provides[0]] = result
            else:
                out_model.update({result[key] for key in self.provides})
        return [self.formatter.format(
            function=self.function.__name__,
            arguments=', '.join(f'{k}={v!r}' for k, v in arguments.items()),
            output=result
        )], out_model


class ToolUser(Actor):
    """
    ToolUser is a mixin class for actors that use tools.
    """

    tools = param.List(default=[], doc="""
        List of tools to use.""")

    def __init__(self, **params):
        super().__init__(**params)
        self._tools = {}
        for prompt_name in self.prompts:
            self._tools[prompt_name] = self._initialize_tools_for_prompt(prompt_name, **params)

    def _get_tool_kwargs(self, tool, prompt_tools, **params):
        """
        Get kwargs for initializing a tool.
        Subclasses can override to provide additional kwargs.

        Parameters
        ----------
        tool : object
            The tool (class or instance) being initialized
        prompt_tools : list
            List of all tools for this prompt

        Returns
        -------
        dict
            Keyword arguments for tool initialization
        """
        return {"llm": self.llm or params.get("llm"), "interface": self.interface or params.get("interface")}

    def _initialize_tools_for_prompt(self, tools_or_key: str | list, **params) -> list:
        """
        Initialize tools for a specific prompt.

        Parameters
        ----------
        tools : str | list
            The list of tools to initialize. If None, tools are looked up from the prompt.
        **params : dict
            Additional parameters for tool initialization

        Returns
        -------
        list
            List of instantiated tools
        """
        if isinstance(tools_or_key, str):
            prompt_tools = self._lookup_prompt_key(tools_or_key, "tools")
        else:
            prompt_tools = tools_or_key
        instantiated_tools = []

        # Initialize each tool
        for tool in prompt_tools:
            if isinstance(tool, Actor):
                # For already instantiated Actors, set properties directly
                if tool.llm is None:
                    tool.llm = self.llm or params.get("llm")
                if tool.interface is None:
                    tool.interface = self.interface or params.get("interface")

                # Apply any additional configuration from subclasses
                tool_kwargs = self._get_tool_kwargs(tool, instantiated_tools or prompt_tools, **params)
                for key, value in tool_kwargs.items():
                    if key not in ('llm', 'interface') and hasattr(tool, key):
                        setattr(tool, key, value)

                instantiated_tools.append(tool)
            elif isinstance(tool, FunctionType):
                # Function tools only get basic kwargs
                tool_kwargs = self._get_tool_kwargs(tool, instantiated_tools or prompt_tools, **params)
                instantiated_tools.append(FunctionTool(tool, **tool_kwargs))
            else:
                # For classes that need to be instantiated
                tool_kwargs = self._get_tool_kwargs(tool, instantiated_tools or prompt_tools, **params)
                instantiated_tools.append(tool(**tool_kwargs))
        return instantiated_tools

    async def _use_tools(self, prompt_name: str, messages: list[Message], context: TContext) -> str:
        tools_context = ""
        # TODO: INVESTIGATE WHY or self.tools is needed
        for tool in self._tools.get(prompt_name, []) or self.tools:
            if all(requirement in context for requirement in tool.input_schema.__required_keys__):
                tool_context = await tool.respond(messages, context)
                if tool_context:
                    tools_context += f"\n{tool_context}"
        return tools_context
