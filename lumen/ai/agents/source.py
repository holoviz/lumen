"""
SourceAgent — fetches external data via source control actions.

Modeled on :class:`SQLAgent`: given the :class:`SourceLookup` context (which
actions are relevant), constructs :class:`FunctionTool` instances from the
callables and passes them to the LLM.  The LLM fills in parameters and calls
the function.  Results are normalized to DataFrames and registered as new
:class:`DuckDBSource` entries in context.
"""
from __future__ import annotations

import asyncio
import inspect
import typing as t

import param

from panel.chat import ChatStep

from ...pipeline import Pipeline
from ...sources.duckdb import DuckDBSource
from ..config import PROMPTS_DIR
from ..context import ContextModel, TContext
from ..controls import ParametricSourceControls
from ..controls.ingest.result import SourceResult
from ..llm import Message
from ..schemas import Metaset, get_metaset
from ..tools import FunctionTool
from ..translate import doc_descriptions
from ..utils import (
    describe_data, get_pipeline, log_debug, result_to_dataframe,
    retry_llm_output,
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
    """Hide SDK/internal params from FunctionTool schema generation.

    Also wraps the callable to capture its return value in
    ``result_store`` when provided, even if no params need skipping.
    """
    needs_skip = bool(skip_params)
    needs_capture = result_store is not None

    if not needs_skip and not needs_capture:
        return func

    sig = inspect.signature(func)

    if needs_skip:
        filtered_params = [
            parameter
            for name, parameter in sig.parameters.items()
            if name not in skip_params
        ]
        filtered_signature = inspect.Signature(filtered_params)
        kept_param_names = {parameter.name for parameter in filtered_params}

        annotations = dict(getattr(func, "__annotations__", {}))
        for p in sig.parameters.values():
            if p.annotation is not inspect.Parameter.empty and p.name not in annotations:
                annotations[p.name] = p.annotation
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
    else:
        filtered_signature = sig
        filtered_annotations = dict(getattr(func, "__annotations__", {}))
        for p in sig.parameters.values():
            if p.annotation is not inspect.Parameter.empty and p.name not in filtered_annotations:
                filtered_annotations[p.name] = p.annotation
        filtered_doc = getattr(func, "__doc__", None)

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
            "Use when the loaded data does not cover the requested scope (e.g. user asks for 'whole month' but data summary only shows a week)",
            "Prefer over metadata discovery when no data sources are loaded but source controls exist",
            "This agent only fetches and registers data \u2014 it never presents results to the user; always follow with a presentation or query step",
            "NOT for listing configured external source actions or answering 'what data is available' before a concrete dataset request",
            "NOT when the loaded data already covers the requested scope and user just wants to query/visualize it",
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

    # Counter to disambiguate repeated table names
    _table_name_counts: dict[str, int] = {}

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
        ctrl_map: dict[str, ParametricSourceControls] = {}
        for c in controls:
            if isinstance(c, ParametricSourceControls):
                from ..tools.source_lookup import _control_hash
                ctrl_map[_control_hash(c)] = c

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
                ctrl_key = info.get("control_key")
                ctrl = ctrl_map.get(ctrl_key)
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
            title=step_title or "Fetching external data\u2026",
            steps_layout=self._steps_layout,
        ) as step:
            step.stream(
                f"Found {len(tool_list)} available action(s). "
                "Asking LLM to select and call the right one\u2026"
            )

            source, table_name, df, summary, pipeline, metaset = (
                await self._invoke_and_process(
                    messages, context, tool_list, tool_results,
                    source_actions, catalog_summary, step,
                )
            )

            step.stream(f"\n\n\u2705 Loaded **{len(df):,}** rows into `{table_name}`")
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

    @retry_llm_output()
    async def _invoke_and_process(
        self,
        messages: list[Message],
        context: TContext,
        tool_list: list[FunctionTool],
        tool_results: dict[str, t.Any],
        source_actions: dict[str, t.Any] | None,
        catalog_summary: str,
        step: t.Any,
        errors: list[str] | None = None,
    ) -> tuple[DuckDBSource, str, t.Any, str, Pipeline, Metaset]:
        """Call the LLM, execute the tool, and process the result.

        Decorated with ``@retry_llm_output`` so that recoverable
        errors (missing params, API errors, empty data) are fed back
        to the LLM for a retry.
        """
        # Clear previous tool results so retries start fresh
        tool_results.clear()

        result = await self._invoke_prompt(
            "main",
            messages,
            context,
            tools=tool_list,
            catalog_summary=catalog_summary,
            errors=errors,
        )

        step.stream("\n\nProcessing result\u2026")

        payload = next(
            (r["result"] for r in reversed(tool_results.values())),
            result,
        ) if tool_results else result

        # --- SourceResult path (URLSourceControls, CatalogSourceControls) ---
        if isinstance(payload, SourceResult):
            if not payload.sources:
                msg = payload.message or "The source action returned no data."
                step.stream(f"\n\n\u26a0\ufe0f {msg}")
                raise Exception(msg)
            source = payload.sources[0]
            table_name = payload.table or self._derive_table_name(
                messages, source_actions, tool_results
            )
            if table_name not in source.tables:
                table_name = next(iter(source.tables), table_name)
            context["source"] = source
            pipeline = await get_pipeline(source=source, table=table_name)
            metaset = await get_metaset([source], [table_name])
            df = await asyncio.to_thread(lambda: pipeline.data)
            summary = await describe_data(df, reduce_enums=False)
            if len(summary) >= 4000:
                summary = summary[:3997] + "..."
            return source, table_name, df, summary, pipeline, metaset

        # --- DataFrame path (CodeSourceControls, raw callables) ---
        df = result_to_dataframe(payload)

        if df is None or (hasattr(df, "empty") and df.empty):
            step.stream("\n\n\u26a0\ufe0f The action returned no data.")
            raise Exception("The source action returned no data.")

        table_name = self._derive_table_name(messages, source_actions, tool_results)

        source = DuckDBSource.from_df(tables={table_name: df})
        source.tables[table_name] = f"SELECT * FROM {table_name}"
        context["source"] = source

        pipeline = await get_pipeline(source=source, table=table_name)
        metaset = await get_metaset([source], [table_name])
        summary = await describe_data(df, reduce_enums=False)
        if len(summary) >= 4000:
            summary = summary[:3997] + "..."

        return source, table_name, df, summary, pipeline, metaset

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

        Appends a numeric suffix when a name has been used before to
        avoid silently overwriting previously-registered tables.

        Examples:
            List Aggs {ticker: AAPL, timespan: day} \u2192 list_aggs_aapl_day
            (second call with same args)             \u2192 list_aggs_aapl_day_2
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
            base = normalize_table_name(slug)
        elif source_actions and len(source_actions) == 1:
            name = next(iter(source_actions))
            base = normalize_table_name(name)
        else:
            base = "source_data"

        # Disambiguate repeated names
        counts = SourceAgent._table_name_counts
        count = counts.get(base, 0) + 1
        counts[base] = count
        return base if count == 1 else f"{base}_{count}"
