from __future__ import annotations

from typing import (
    Any, Literal, TypedDict, Union,
)

import param

from panel.pane import panel as as_panel
from pydantic import BaseModel, Field, create_model

from ...views.base import View
from ..config import PROMPTS_DIR
from ..context import ContextModel, TContext
from ..llm import Message
from .base import Tool

JSON_TYPE_MAP = {
    "string": str,
    "integer": int,
    "number": float,
    "boolean": bool,
    "array": list,
    "object": dict,
    "null": type(None),
}


def _json_schema_to_field(field_spec: dict) -> tuple[type, Field]:
    """
    Convert a single JSON Schema property definition to a (type, FieldInfo) pair.
    """
    field_kwargs: dict[str, Any] = {}
    if "description" in field_spec:
        field_kwargs["description"] = field_spec["description"]
    if "default" in field_spec:
        field_kwargs["default"] = field_spec["default"]

    # Handle enum values
    if "enum" in field_spec:
        return Literal[tuple(field_spec["enum"])], Field(**field_kwargs)

    json_type = field_spec.get("type", "string")

    if isinstance(json_type, list):
        # Union types, e.g. ["string", "null"]
        types = tuple(JSON_TYPE_MAP.get(t, Any) for t in json_type)
        python_type = Union[types]  # noqa
    else:
        python_type = JSON_TYPE_MAP.get(json_type, Any)

    # Handle typed arrays
    if json_type == "array" and "items" in field_spec:
        items_spec = field_spec["items"]
        item_type = JSON_TYPE_MAP.get(items_spec.get("type", "string"), Any)
        if "enum" in items_spec:
            item_type = Literal[tuple(items_spec["enum"])]
        python_type = list[item_type]

    return python_type, Field(**field_kwargs)


def _json_schema_to_model(name: str, schema: dict) -> type[BaseModel]:
    """
    Convert a JSON Schema dict (as returned by MCP ``list_tools``) to a
    Pydantic model suitable for structured LLM output.
    """
    properties = schema.get("properties", {})
    required_fields = set(schema.get("required", []))

    fields: dict[str, tuple[type, Any]] = {}
    for field_name, field_spec in properties.items():
        python_type, field_info = _json_schema_to_field(field_spec)

        # If the field is optional and has no default, set default to None
        if field_name not in required_fields and field_info.default is None and "default" not in (field_spec or {}):
            field_info = Field(default=None, **{
                k: v for k, v in {
                    "description": field_spec.get("description"),
                }.items() if v is not None
            })
            python_type = python_type | None

        fields[field_name] = (python_type, field_info)

    description = schema.get("description", "")
    return create_model(name, __doc__=description, **fields)


class MCPTool(Tool):
    """
    MCPTool wraps a tool exposed by an MCP (Model Context Protocol) server
    and makes it available for an LLM to call. It connects to the server,
    discovers the tool's input schema, and generates a pydantic Model that
    the LLM will populate.

    The tool may also consume information in memory, e.g. the current
    table or pipeline by declaring the requires parameter.

    The tool may also return context to add to the current working memory
    by returning a dictionary and declaring the ``provides`` parameter. Any
    keys listed in the provides parameter will be copied into working memory.
    """

    formatter = param.Parameter(
        default="{tool_name}({arguments}) returned: {output}",
        doc="""
        Formats the return value for inclusion in the global context.
        Accepts 'tool_name', 'arguments' and 'output' as formatting variables.""",
    )

    provides = param.List(default=[], readonly=False, constant=True, doc="""
        List of context values it provides to current working memory.""")

    render_output = param.Boolean(default=False, doc="""
        Whether to render the tool output directly, even if it is not
        already a Lumen View or Panel Viewable.""")

    requires = param.List(default=[], readonly=False, constant=True, doc="""
        List of context values it requires to be in memory.""")

    server = param.Parameter(doc="""
        MCP server reference. Can be a URL string, a file path, a
        fastmcp.FastMCP instance, or a fastmcp.Client instance.""")

    tool_name = param.String(doc="""
        Name of the MCP tool to invoke on the server.""")

    prompts = param.Dict(
        default={
            "main": {
                "template": PROMPTS_DIR / "FunctionTool" / "main.jinja2",
            },
        }
    )

    def __init__(self, server, tool_name: str, schema: dict, description: str | None = None, **params):
        """
        Parameters
        ----------
        server :
            MCP server reference (URL, file path, FastMCP instance, or Client).
        tool_name : str
            Name of the MCP tool.
        schema : dict
            The JSON Schema ``inputSchema`` dict returned by the MCP server.
        description : str | None
            Human-readable description of the tool.
        **params :
            Additional keyword arguments forwarded to the Tool base class.
        """
        model = _json_schema_to_model(tool_name, schema)
        if "purpose" not in params:
            purpose = f"{tool_name}: {description}" if description else tool_name
            params["purpose"] = purpose
        super().__init__(
            server=server,
            tool_name=tool_name,
            name=tool_name,
            **params,
        )
        self._model = model

    @classmethod
    async def from_server(cls, server, **params) -> list[MCPTool]:
        """
        Connect to an MCP server, discover all available tools, and return
        an :class:`MCPTool` instance for each one.

        Parameters
        ----------
        server :
            MCP server reference (URL, file path, FastMCP instance, config dict, etc.).
        **params :
            Additional keyword arguments forwarded to each MCPTool constructor
            (e.g. ``llm``, ``interface``).

        Returns
        -------
        list[MCPTool]
            One MCPTool per tool advertised by the server.
        """
        from fastmcp import Client

        client = server if isinstance(server, Client) else Client(server)
        async with client:
            tools = await client.list_tools()

        return [
            cls(
                server=server,
                tool_name=tool.name,
                schema=tool.inputSchema,
                description=tool.description,
                **params,
            )
            for tool in tools
        ]

    # ------------------------------------------------------------------
    # ContextProvider interface
    # ------------------------------------------------------------------

    @property
    def inputs(self):
        return TypedDict(f"{self.tool_name}Inputs", {f: Any for f in self.requires})

    # ------------------------------------------------------------------
    # Execute / Respond
    # ------------------------------------------------------------------

    async def execute(self, **arguments: Any) -> Any:
        """
        Execute the MCP tool directly with the given arguments.

        This is the low-level entry point used by the LLM tool-calling
        infrastructure (analogous to calling ``tool.function(**args)`` on
        a :class:`FunctionTool`).

        Parameters
        ----------
        **arguments :
            Keyword arguments forwarded to the MCP tool.

        Returns
        -------
        Any
            The tool result.  Structured ``.data`` is preferred; falls
            back to concatenated text content blocks.
        """
        from fastmcp import Client

        client = self.server if isinstance(self.server, Client) else Client(self.server)
        async with client:
            result = await client.call_tool(self.tool_name, arguments)

        if result.data is not None:
            return result.data
        elif result.content:
            text_parts = [
                block.text for block in result.content if hasattr(block, "text")
            ]
            return "\n".join(text_parts) if text_parts else str(result.content)
        return ""

    async def respond(
        self,
        messages: list[Message],
        context: TContext,
        **kwargs: dict[str, Any],
    ) -> tuple[list[Any], ContextModel]:
        model_kwargs = {}
        if any(field not in self.requires for field in self._model.model_fields):
            kwargs = await self._invoke_prompt("main", messages, context, response_model=self._model, max_retries=3)
        arguments = dict(model_kwargs, **{k: context[k] for k in self.requires}, **kwargs)

        output = await self.execute(**arguments)

        # Handle renderable results
        if isinstance(output, dict) and "view_type" in output:
            output = View.from_spec(output)
        if isinstance(output, View):
            return [output], {}
        elif self.render_output:
            p = as_panel(output)
            return [p], {}

        # Populate provided context keys
        out_model: dict[str, Any] = {}
        if self.provides:
            if len(self.provides) == 1 and not isinstance(output, dict):
                out_model[self.provides[0]] = output
            else:
                out_model.update({key: output[key] for key in self.provides})

        return [
            self.formatter.format(
                tool_name=self.tool_name,
                arguments=", ".join(f"{k}={v!r}" for k, v in arguments.items()),
                output=output,
            )
        ], out_model
