"""
SourceAgent — fetches external data via source control actions.

Modeled on :class:`SQLAgent`: given the :class:`SourceLookup` context (which
actions are relevant), constructs :class:`FunctionTool` instances from the
callables and passes them to the LLM.  The LLM fills in parameters and calls
the function.  Results are normalized to DataFrames and registered as new
:class:`DuckDBSource` entries in context.
"""
from __future__ import annotations

import inspect
import typing as t

import param

from panel.chat import ChatStep

from ...pipeline import Pipeline
from ...sources.duckdb import DuckDBSource
from ..config import PROMPTS_DIR
from ..context import ContextModel, TContext
from ..controls import ParametricSourceControls
from ..llm import Message
from ..schemas import Metaset, get_metaset
from ..tools import FunctionTool
from ..translate import doc_descriptions
from ..utils import (
    describe_data, get_pipeline, log_debug, result_to_dataframe,
)
from .base import Agent

# ---------------------------------------------------------------------------
# Context models
# ---------------------------------------------------------------------------

class SourceInputs(ContextModel):
    source_controls: t.NotRequired[list]
    source_actions: t.NotRequired[dict[str, t.Any]]


class SourceOutputs(ContextModel):
    source: DuckDBSource
    data: t.Any
    table: str
    pipeline: Pipeline
    metaset: Metaset


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_catalog_summary(context: TContext) -> str:
    """
    Build a human-readable summary of all registered source controls
    and their actions, for inclusion in the LLM prompt.
    """
    controls = context.get("source_controls", [])
    if not controls:
        return "(no source controls configured)"

    lines = []
    for ctrl in controls:
        if not isinstance(ctrl, ParametricSourceControls):
            continue
        ctrl_label = getattr(ctrl, "label", ctrl.__class__.__name__)
        lines.append(f"### {ctrl_label}")
        tools = ctrl.as_tools()
        if not tools:
            lines.append("  (no actions registered)")
            continue
        for action_name, func in tools:
            description, param_docs = doc_descriptions(func)
            lines.append(f"- **{action_name}**: {description or '(no description)'}")
            if param_docs:
                for pname, pdesc in param_docs.items():
                    lines.append(f"    - `{pname}`: {pdesc}")
    return "\n".join(lines) if lines else "(no source controls configured)"


def _sanitize_tool_callable(
    func: t.Callable,
    action_name: str,
    skip_params: list[str],
    result_store: dict[str, t.Any] | None = None,
) -> t.Callable:
    """Hide SDK/internal params from FunctionTool schema generation."""
    if not skip_params:
        return func

    sig = inspect.signature(func)
    filtered_params = [
        parameter
        for name, parameter in sig.parameters.items()
        if name not in skip_params
    ]
    filtered_signature = inspect.Signature(filtered_params)
    kept_param_names = {parameter.name for parameter in filtered_params}

    annotations = dict(getattr(func, "__annotations__", {}))
    filtered_annotations = {
        name: annotation
        for name, annotation in annotations.items()
        if name not in skip_params
    }

    description, param_docs = doc_descriptions(func, sig)
    filtered_doc_lines = [description] if description else []
    filtered_param_docs = [
        f"    {name}: {param_docs[name]}"
        for name in sig.parameters
        if name in kept_param_names and name in param_docs
    ]
    if filtered_param_docs:
        filtered_doc_lines.extend(["", "Args:", *filtered_param_docs])
    filtered_doc = "\n".join(filtered_doc_lines) or getattr(func, "__doc__", None)

    if param.parameterized.iscoroutinefunction(func):
        async def wrapped_async(**kwargs):
            result = await func(**kwargs)
            if result_store is not None:
                result_store[action_name] = {"result": result, "kwargs": kwargs}
            return result
        wrapped_func = wrapped_async
    else:
        def wrapped_sync(**kwargs):
            result = func(**kwargs)
            if result_store is not None:
                result_store[action_name] = {"result": result, "kwargs": kwargs}
            return result
        wrapped_func = wrapped_sync

    wrapped_name = getattr(func, "__name__", action_name.replace(" ", "_"))
    wrapped_any = t.cast(t.Any, wrapped_func)
    wrapped_any.__name__ = wrapped_name
    wrapped_any.__qualname__ = wrapped_name
    wrapped_any.__doc__ = filtered_doc
    wrapped_any.__signature__ = filtered_signature
    wrapped_any.__annotations__ = filtered_annotations
    return wrapped_func


# ---------------------------------------------------------------------------
# SourceAgent
# ---------------------------------------------------------------------------

class SourceAgent(Agent):
    """
    Fetches data from external APIs, Python callables, or URL templates
    configured via source controls.

    The agent reads available actions from ``SourceLookup`` context,
    wraps the relevant callables as ``FunctionTool`` instances, and
    passes them to the LLM.  The LLM decides which action to call and
    fills in parameter values.  Results are normalized to DataFrames
    and registered as ``DuckDBSource`` entries for downstream agents
    (e.g. ``SQLAgent``, ``hvPlotAgent``) to consume.
    """

    conditions = param.List(
        default=[
            "Use when no existing data source can answer the query and external source controls are configured",
            "Use when the user asks for data from an external API or service",
            "Prefer over metadata discovery when no data sources are loaded but source controls exist",
            "This agent only fetches and registers data — it never presents results to the user; always follow with a presentation or query step",
            "NOT for listing configured external source actions or answering 'what data is available' before a concrete dataset request",
            "NOT when data is already loaded and user wants to query/visualize it",
        ]
    )

    purpose = param.String(
        default="""
        Fetches data from configured external sources (APIs, Python callables,
        URL templates).  Discovers relevant endpoints, fills parameters via
        LLM, executes the call, and registers the resulting DataFrame for
        downstream querying and visualization."""
    )

    prompts = param.Dict(
        default={
            "main": {
                "template": PROMPTS_DIR / "SourceAgent" / "main.jinja2",
            },
        }
    )

    user = param.String(default="Source")

    input_schema = SourceInputs
    output_schema = SourceOutputs

    @classmethod
    async def applies(cls, context: TContext) -> bool:
        controls = context.get("source_controls", [])
        return any(isinstance(c, ParametricSourceControls) for c in controls)

    # ------------------------------------------------------------------
    # Tool building
    # ------------------------------------------------------------------

    @staticmethod
    def _build_tools(
        context: TContext,
        source_actions: dict[str, t.Any] | None = None,
        result_store: dict[str, t.Any] | None = None,
    ) -> list[FunctionTool]:
        """
        Build FunctionTool instances from source controls.

        If ``source_actions`` is provided (from SourceLookup), only builds
        tools for those actions.  Otherwise builds tools for all actions
        in all controls.
        """
        controls = context.get("source_controls", [])
        ctrl_map: dict[int, ParametricSourceControls] = {
            id(c): c for c in controls if isinstance(c, ParametricSourceControls)
        }

        tools: list[FunctionTool] = []

        def _make_tool(
            ctrl: ParametricSourceControls,
            action_name: str,
            func: t.Callable,
        ) -> FunctionTool | None:
            """Create a FunctionTool, respecting the control's skip_params."""
            skip = list(getattr(ctrl, "skip_params", None) or [])
            tool_func = _sanitize_tool_callable(func, action_name, skip, result_store)
            description, _ = doc_descriptions(tool_func)
            try:
                return FunctionTool(
                    tool_func,
                    purpose=(
                        f"{action_name}: {description}"
                        if description
                        else action_name
                    ),
                )
            except Exception as e:
                log_debug(
                    f"[SourceAgent] Failed to build FunctionTool for "
                    f"{action_name!r}, skipping. {type(e).__name__}: {e}"
                )
                return None

        if source_actions:
            for action_name, info in source_actions.items():
                ctrl_id = info.get("control_id")
                ctrl = ctrl_map.get(ctrl_id)
                if ctrl is None:
                    continue
                func = dict(ctrl.as_tools()).get(action_name)
                if func is None:
                    continue
                tool = _make_tool(ctrl, action_name, func)
                if tool is not None:
                    tools.append(tool)
        else:
            for ctrl in ctrl_map.values():
                for action_name, func in ctrl.as_tools():
                    tool = _make_tool(ctrl, action_name, func)
                    if tool is not None:
                        tools.append(tool)

        return tools

    # ------------------------------------------------------------------
    # Respond
    # ------------------------------------------------------------------

    async def respond(
        self,
        messages: list[Message],
        context: TContext,
        step_title: str | None = None,
    ) -> tuple[list[t.Any], SourceOutputs]:
        source_actions = context.get("source_actions")
        tool_results: dict[str, t.Any] = {}
        tool_list = self._build_tools(context, source_actions, result_store=tool_results)

        if not tool_list:
            raise ValueError(
                "No source control actions available. "
                "Ensure source controls with actions are configured."
            )

        catalog_summary = _build_catalog_summary(context)

        with self._add_step(
            title=step_title or "Fetching external data…",
            steps_layout=self._steps_layout,
        ) as step:
            step.stream(
                f"Found {len(tool_list)} available action(s). "
                "Asking LLM to select and call the right one…"
            )

            result = await self._invoke_prompt(
                "main",
                messages,
                context,
                tools=tool_list,
                catalog_summary=catalog_summary,
            )

            step.stream("\n\nProcessing result…")

            payload = next(
                (r["result"] for r in reversed(tool_results.values())),
                result,
            ) if tool_results else result
            df = result_to_dataframe(payload)

            if df is None or (hasattr(df, "empty") and df.empty):
                step.stream("\n\n⚠️ The action returned no data.")
                raise ValueError("The source action returned no data.")

            # Determine a table name
            table_name = self._derive_table_name(messages, source_actions, tool_results)

            # Register as DuckDBSource
            source = DuckDBSource.from_df(tables={table_name: df})
            source.tables[table_name] = f"SELECT * FROM {table_name}"

            # Update context
            context["source"] = source

            # Create pipeline and metaset
            pipeline = await get_pipeline(source=source, table=table_name)
            metaset = await get_metaset([source], [table_name])
            summary = await describe_data(df, reduce_enums=False)
            if len(summary) >= 4000:
                summary = summary[:3997] + "..."

            step.stream(f"\n\n✅ Loaded **{len(df):,}** rows into `{table_name}`")
            step.stream(f"\n```\n{summary}\n```")

            if isinstance(step, ChatStep):
                step.param.update(
                    status="success",
                    success_title=f"Loaded {len(df):,} rows into '{table_name}'",
                )

        out_context: SourceOutputs = {
            "source": source,
            "data": summary,
            "table": table_name,
            "pipeline": pipeline,
            "metaset": metaset,
        }
        return [], out_context

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _derive_table_name(
        messages: list[Message],
        source_actions: dict[str, t.Any] | None = None,
        tool_results: dict[str, t.Any] | None = None,
    ) -> str:
        """Derive a table name from the action name + argument values.

        Examples:
            List Aggs {ticker: AAPL, timespan: day} → list_aggs_aapl_day
            Get Ticker Details {ticker: AAPL}        → get_ticker_details_aapl
        """
        from ...util import normalize_table_name

        if tool_results and len(tool_results) == 1:
            action_name, info = next(iter(tool_results.items()))
            kwargs = info.get("kwargs", {})
            slug = action_name
            for v in kwargs.values():
                s = str(v).strip()
                if s and len(s) <= 20:
                    slug += f" {s}"
            return normalize_table_name(slug)

        if source_actions and len(source_actions) == 1:
            name = next(iter(source_actions))
            return normalize_table_name(name)

        return "source_data"
