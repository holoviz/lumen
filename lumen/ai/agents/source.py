from __future__ import annotations

import asyncio
import inspect

from functools import partial
from typing import Any, NotRequired

import panel as pn
import param

from panel_material_ui import Button
from pydantic import Field, create_model
from pydantic.fields import FieldInfo

from ..config import PROMPTS_DIR
from ..context import ContextModel, TContext
from ..llm import Message
from ..tools import CatalogLookupTool
from ..utils import log_debug
from .base import Agent

# ─────────────────────────────────────────────────────────────────────────────
# Dynamic response model factory (follows the SQLAgent create_model pattern)
# ─────────────────────────────────────────────────────────────────────────────

def make_fill_params_model(endpoints: list[dict]):
    """Build a pydantic model with explicit fields per endpoint.

    Instead of ``parameters: dict[str, str]`` (which LLMs often return
    as ``{}``), each endpoint gets its own sub-model with named fields
    for every required parameter.  The LLM fills typed fields reliably.

    Example output model for two endpoints::

        class FillParams(BaseModel):
            chain_of_thought: str
            endpoint_0: Endpoint0_station_id_fmt | None  # station_id, fmt
            endpoint_1: Endpoint1_area | None             # area
    """
    wrapper_fields: dict[str, Any] = {
        "chain_of_thought": (
            str,
            FieldInfo(description="Brief reasoning about which parameters could be determined"),
        ),
    }

    for i, ep in enumerate(endpoints):
        required_params = ep.get("required_params", [])
        if not required_params:
            continue

        # Parse per-param descriptions from the compact schema string
        # Format: "name(loc,type)[req]:description|name2(...):desc2"
        param_descs = {}
        schema_str = ep.get("params_schema", "")
        if schema_str:
            for part in schema_str.split("|"):
                if ":" in part:
                    pname = part.split("(")[0]
                    pdesc = part.split(":", 1)[1]
                    param_descs[pname] = pdesc

        # Build sub-model fields
        sub_fields: dict[str, Any] = {}
        for pname in required_params:
            desc = param_descs.get(pname, f"Value for {pname}")
            sub_fields[pname] = (str, FieldInfo(description=desc))

        name_suffix = "_".join(required_params)[:40]
        sub_model = create_model(f"Endpoint{i}_{name_suffix}", **sub_fields)

        label = ep.get("label", f"endpoint {i}")[:80]
        wrapper_fields[f"endpoint_{i}"] = (
            sub_model | None,
            FieldInfo(
                default=None,
                description=(
                    f"Parameters for [{i}] {label}. "
                    "Fill if ALL required params can be determined, else leave null."
                ),
            ),
        )

    return create_model("FillParamsResponse", **wrapper_fields)


# ─────────────────────────────────────────────────────────────────────────────
# Output schema
# ─────────────────────────────────────────────────────────────────────────────

class SourceOutputs(ContextModel):
    chat: NotRequired[str]


# ─────────────────────────────────────────────────────────────────────────────
# Agent
# ─────────────────────────────────────────────────────────────────────────────

class SourceAgent(Agent):
    """
    Helps users discover and load datasets through conversation.

    Uses CatalogLookupTool to search embedded catalogs, presents
    ranked results, explains relevance, asks clarifying questions
    when the query is vague, and offers **Load** buttons so the
    user can add data to their workspace directly from the chat.

    Loading follows the same pattern as file uploads from the chat
    input: the button callback calls the control's ``load_entry``
    method, which registers the source and triggers ``_sync_sources``
    in ExplorerUI. A fresh coordinator run is then kicked off via
    ``interface.send(..., respond=True)`` so the planner sees the
    newly loaded data.

    The Planner routes to this agent when the user's intent is
    data *discovery* rather than data *analysis*.
    """

    conditions = param.List(
        default=[
            "Use when the user wants to find, discover, search for, or browse datasets or data sources",
            "Use when the user describes a research question and needs to find relevant data",
            "Use when the user asks what datasets are available before any sources are loaded",
            "Use when no data sources are loaded and the user asks to get, fetch, load, or retrieve specific data (e.g. 'get station X', 'fetch alerts for PA')",
            "NOT when the user already has data loaded and wants to query, visualize, or analyze it",
            "NOT for general conversation unrelated to data discovery",
        ]
    )

    not_with = param.List(default=["ChatAgent"])

    purpose = param.String(
        default="""
        Helps users discover and load datasets by searching catalogs,
        API registries, and data portals. When no data is loaded yet,
        this is the agent that finds the right endpoint or dataset,
        fills in required parameters (e.g. station IDs, area codes),
        and offers Load buttons so the user can add data to their
        workspace directly from the chat. Use this agent whenever
        the user wants to get, fetch, or load data that isn't in
        the workspace yet."""
    )

    prompts = param.Dict(
        default={
            "main": {
                "template": PROMPTS_DIR / "SourceAgent" / "main.jinja2",
            },
            "fill_params": {
                "template": PROMPTS_DIR / "SourceAgent" / "fill_params.jinja2",
                "response_model": make_fill_params_model,
            },
        }
    )

    source_controls = param.List(default=[], doc="""
        Instantiated catalog/API source control instances.
        Used to map load-button clicks back to the correct control
        and invoke its ``load_entry`` method.""")

    output_schema = SourceOutputs

    # ──────────────────────────────────────────────────────────────────────────
    # Respond
    # ──────────────────────────────────────────────────────────────────────────

    async def respond(
        self,
        messages: list[Message],
        context: TContext,
        step_title: str | None = None,
    ) -> tuple[list[Any], dict[str, Any]]:
        # 1. Call the catalog lookup tool directly (not via _use_tools)
        #    so we get both the text summary AND structured matches.
        catalog_tool = self._get_catalog_tool()
        if catalog_tool is None:
            result = await self._stream(messages, context, tool_context="")
            return [result], {"chat": result.object}

        tool_outputs, tool_out = await catalog_tool.respond(messages, context)
        text_context = "\n".join(str(o) for o in tool_outputs)
        matches: list[dict] = tool_out.get("matches", [])

        # 2. Stream the LLM explanation to the chat
        result = await self._stream(messages, context, tool_context=text_context)

        # 3. Determine which matches are loadable (have all required params)
        loadable = await self._resolve_loadable(matches, messages, context)

        # 4. Build footer buttons for loadable entries
        if loadable:
            # Capture interface before the param.update context manager exits
            interface = self.interface
            buttons = self._build_load_buttons(loadable, messages, interface)
            if buttons:
                result.footer_objects = list(result.footer_objects or []) + buttons

        return [result], {"chat": result.object}

    # ──────────────────────────────────────────────────────────────────────────
    # Tool access
    # ──────────────────────────────────────────────────────────────────────────

    def _get_catalog_tool(self) -> CatalogLookupTool | None:
        """Find the CatalogLookupTool among the agent's tools.

        Checks both ``_tools`` (per-prompt tools) and ``self.tools``
        (tools passed via the ``tools=`` param), mirroring the
        ``or self.tools`` fallback in ``ToolUser._use_tools``.
        """
        for tools in self._tools.values():
            for tool in tools:
                if isinstance(tool, CatalogLookupTool):
                    return tool
        # Fallback: tools passed via the tools= param
        for tool in self.tools:
            if isinstance(tool, CatalogLookupTool):
                return tool
        return None

    # ──────────────────────────────────────────────────────────────────────────
    # Parameter resolution
    # ──────────────────────────────────────────────────────────────────────────

    async def _resolve_loadable(
        self,
        matches: list[dict],
        messages: list[Message],
        context: TContext,
    ) -> list[dict]:
        """Determine which matches can be loaded, filling params via LLM.

        Uses ``make_fill_params_model`` to create a dynamic pydantic
        model with explicit named fields per endpoint (no generic dicts).
        The LLM fills ``station_id: str``, ``fmt: str``, etc. directly.
        """
        if not matches:
            return []

        # Separate param-free entries (always loadable) from those needing params
        param_free = []
        param_needed = []
        for m in matches:
            if m.get("required_params"):
                param_needed.append(m)
            else:
                m["filled_params"] = {}
                param_free.append(m)

        if not param_needed:
            return param_free

        # Ask the LLM to fill parameters using a dynamic model with
        # explicit named fields per endpoint (follows SQLAgent pattern).
        try:
            fill_result = await self._invoke_prompt(
                "fill_params",
                messages,
                context,
                model_kwargs=dict(endpoints=param_needed),
                endpoints=param_needed,
            )
            # Read back the per-endpoint sub-models
            for i, match in enumerate(param_needed):
                sub = getattr(fill_result, f"endpoint_{i}", None)
                if sub is None:
                    continue
                filled = sub.model_dump()
                required = set(match.get("required_params", []))
                if required and not required.issubset(filled.keys()):
                    missing = required - set(filled.keys())
                    log_debug(
                        f"[SourceAgent] Skipping endpoint_{i}: "
                        f"LLM missed required params {missing}"
                    )
                    continue
                match["filled_params"] = filled
                param_free.append(match)
        except Exception as e:
            log_debug(f"[SourceAgent] fill_params prompt failed: {e}")

        return param_free

    # ──────────────────────────────────────────────────────────────────────────
    # Button construction
    # ──────────────────────────────────────────────────────────────────────────

    def _build_load_buttons(
        self,
        loadable: list[dict],
        messages: list[Message],
        interface,
    ) -> list[Button]:
        """Create a Load button for each loadable entry."""
        buttons = []
        for m in loadable[:5]:
            control = self._find_control(m.get("control_id"))
            if control is None:
                continue
            # Extract the endpoint path from the label
            # label format: "endpoint: GET /path. description: Some text..."
            raw_label = m.get("label", "dataset")
            endpoint = raw_label
            description = ""
            if "endpoint: " in raw_label:
                endpoint = raw_label.split("endpoint: ", 1)[1]
                if ". description: " in endpoint:
                    endpoint, description = endpoint.split(". description: ", 1)
            # Substitute filled params into the path for clarity
            # e.g. "GET /stations/{stationId}" + {stationId: KCMI} -> "GET /stations/KCMI"
            filled = m.get("filled_params", {})
            display = endpoint
            for k, v in filled.items():
                display = display.replace(f"{{{k}}}", v)
            btn = Button(
                label=display,
                description=description,
                icon="download",
                variant="outlined",
                size="small",
                on_click=partial(
                    self._on_load_click,
                    control=control,
                    row_idx=m.get("row_idx"),
                    param_values=m.get("filled_params", {}),
                    messages=messages,
                    interface=interface,
                ),
            )
            buttons.append(btn)
        return buttons

    def _find_control(self, control_id):
        """Resolve a control_id (Python ``id()``) to a control instance."""
        if control_id is None:
            return None
        for ctrl in self.source_controls:
            if id(ctrl) == control_id:
                return ctrl
        return None

    # ──────────────────────────────────────────────────────────────────────────
    # Load callback (mirrors the upload-from-chat pattern in ExplorerUI)
    # ──────────────────────────────────────────────────────────────────────────

    def _on_load_click(self, event, control, row_idx, param_values, messages, interface):
        """Sync button callback -- schedule the async load."""
        # Disable the clicked button to prevent double-loads
        btn = event.obj
        btn.disabled = True
        pn.state.execute(
            partial(
                self._do_load,
                control=control,
                row_idx=row_idx,
                param_values=param_values,
                messages=messages,
                interface=interface,
            )
        )

    async def _do_load(self, control, row_idx, param_values, messages, interface):
        """Async load -- calls the control, then sends a new message.

        This mirrors ``ExplorerUI._execute_pending_query_with``:
        after data is loaded and sources are synced via the param
        watch on ``outputs``, a new message is sent with
        ``respond=True`` to kick off a fresh coordinator run.
        """
        try:
            # Determine if load_entry accepts param_values
            sig = inspect.signature(control.load_entry)
            if "param_values" in sig.parameters and param_values:
                log_debug(f"[SourceAgent] Loading row_idx={row_idx} with params={param_values}")
                result = await control.load_entry(row_idx, param_values=param_values)
            else:
                log_debug(f"[SourceAgent] Loading row_idx={row_idx} (no params)")
                result = await control.load_entry(row_idx)
        except Exception as e:
            log_debug(f"[SourceAgent] load_entry raised: {e}")
            if interface is not None:
                interface.send(
                    f"Failed to load: {e}", user=self.user, respond=False
                )
            return

        if result is None or not result.sources:
            fail_msg = result.message if result else "Unknown error"
            log_debug(f"[SourceAgent] load_entry returned empty result: {fail_msg}")
            if interface is not None:
                interface.send(
                    f"Failed to load the dataset: {fail_msg}", user=self.user, respond=False
                )
            return

        # The control's _handle_success sets outputs["source"] synchronously
        # and triggers the param watch, but the async _sync_sources callback
        # may not have executed yet. To avoid a race where the coordinator
        # snapshots context without "source", we propagate it directly into
        # the control's shared context dict (which is the UI's global context).
        source = control.outputs.get("source")
        if source is not None:
            ctx = control.context
            if "source" not in ctx:
                ctx["source"] = source
            # Also ensure the sources list is up to date
            sources = ctx.setdefault("sources", [])
            if source not in sources:
                sources.append(source)

        table_name = result.table or "data"
        msg = result.message or f"Loaded **{table_name}**"

        # Extract the original user query to re-submit (if any)
        original_query = self._extract_user_query(messages)
        if original_query and interface is not None:
            interface.send(
                f"{msg}. Now working on: {original_query}",
                respond=True,
            )
        elif interface is not None:
            interface.send(
                f"{msg}. What would you like to do with this data?",
                respond=True,
            )

    @staticmethod
    def _extract_user_query(messages: list[Message]) -> str | None:
        """Pull the most recent user message, stripping planner boilerplate."""
        for msg in reversed(messages):
            if msg.get("role") == "user":
                content = msg["content"]
                # Strip the planner "Roadmap:" suffix if present
                if "Roadmap:" in content:
                    content = content.split("Roadmap:")[0]
                # Strip the "User: '...'" wrapper from the planner
                if content.startswith("User: "):
                    content = content[6:].strip().strip("'\"")
                return content.strip() or None
        return None
