from __future__ import annotations

import asyncio
import json
import traceback

from functools import partial
from typing import Any, ClassVar, Literal

import pandas as pd
import panel as pn
import param
import yaml

from panel.chat import ChatInterface
from panel.viewable import Viewable, Viewer
from panel_material_ui import Button, Tabs
from panel_material_ui.chat import ChatMessage
from pydantic import BaseModel, create_model
from pydantic.fields import FieldInfo

from ..base import Component
from ..dashboard import Config, load_yaml
from ..pipeline import Pipeline
from ..sources.base import BaseSQLSource, Source
from ..state import state
from ..transforms.sql import SQLLimit
from ..views import (
    Panel, VegaLiteView, View, hvPlotUIView,
)
from .actor import ContextProvider
from .config import (
    PROMPTS_DIR, SOURCE_TABLE_SEPARATOR, VEGA_MAP_LAYER,
    VEGA_ZOOMABLE_MAP_ITEMS, MissingContextError, RetriesExceededError,
)
from .controls import RetryControls, SourceControls
from .llm import Llm, Message
from .memory import _Memory
from .models import (
    DbtslQueryParams, DiscoveryQueries, DiscoverySufficiency, DistinctQuery,
    PartialBaseModel, QueryCompletionValidation, RetrySpec, SampleQuery,
    SqlQuery, VegaLiteBasicSpec, VegaLiteSpecUpdate, make_sql_model,
)
from .schemas import get_metaset
from .services import DbtslMixin
from .tools import ToolUser
from .translate import param_to_pydantic
from .utils import (
    apply_changes, clean_sql, describe_data, get_data, get_pipeline,
    get_root_exception, get_schema, load_json, log_debug, report_error,
    retry_llm_output, set_nested, stream_details,
)
from .views import AnalysisOutput, LumenOutput, VegaLiteOutput


class Agent(Viewer, ToolUser, ContextProvider):
    """
    Agents are actors responsible for taking a user query and
    performing a particular task and responding by adding context to
    the current memory and creating outputs.

    Each Agent can require certain context that is needed to perform
    the task and should declare any context it itself provides.

    Agents have access to an LLM and the current memory and can
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

    async def _stream(self, messages: list[Message], system_prompt: str) -> Any:
        message = None
        model_spec = self.prompts["main"].get("llm_spec", self.llm_spec_key)
        async for output_chunk in self.llm.stream(messages, system=system_prompt, model_spec=model_spec, field="output"):
            if self.interface is None:
                if message is None:
                    message = ChatMessage(output_chunk, user=self.user)
                else:
                    message.object = output_chunk
            else:
                message = self.interface.stream(output_chunk, replace=True, message=message, user=self.user, max_width=self._max_width)
        return message

    async def _gather_prompt_context(self, prompt_name: str, messages: list, **context):
        context = await super()._gather_prompt_context(prompt_name, messages, **context)
        if "tool_context" not in context:
            context["tool_context"] = await self._use_tools(prompt_name, messages)
        return context

    # Public API

    @classmethod
    async def applies(cls, memory: _Memory) -> bool:
        """
        Additional checks to determine if the agent should be used.
        """
        return True

    async def requirements(self, messages: list[Message]) -> list[str]:
        return self.requires

    async def respond(
        self,
        messages: list[Message],
        step_title: str | None = None,
    ) -> Any:
        """
        Provides a response to the user query.

        The type of the response may be a simple string or an object.

        Arguments
        ---------
        messages: list[Message]
            The list of messages corresponding to the user query and any other
            system messages to be included.
        step_title: str | None
            If the Agent response is part of a longer query this describes
            the step currently being processed.
        """
        system_prompt = await self._render_prompt("main", messages)
        return await self._stream(messages, system_prompt)


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

    requires = param.List(default=[], readonly=True)

    provides = param.List(default=["sources", "source", "document_sources"], readonly=True)

    source_controls: ClassVar[SourceControls] = SourceControls

    _extensions = ("filedropper",)

    async def respond(
        self,
        messages: list[Message],
        step_title: str | None = None,
    ) -> Any:
        source_controls = self.source_controls(memory=self._memory, cancellable=True, replace_controls=True)

        output = pn.Column(source_controls)
        if "source" not in self._memory:
            help_message = "No datasets or documents were found, **please upload at least one to continue**..."
        else:
            help_message = "**Please upload new dataset(s)/document(s) to continue**, or click cancel if this was unintended..."
        output.insert(0, help_message)
        self.interface.send(output, respond=False, user="Assistant")
        while not source_controls._add_button.clicks > 0:
            await asyncio.sleep(0.05)
            if source_controls._cancel_button.clicks > 0:
                self.interface.undo()
                self.interface.disabled = False
                return None
        return source_controls


class ChatAgent(Agent):
    """
    ChatAgent provides general information about available data
    and other topics  to the user.
    """

    conditions = param.List(
        default=[
            "Use for high-level data information or general conversation",
            "Use for technical questions about programming, functions, methods, libraries, APIs, software tools, or 'how to' code usage",
            "NOT for data-specific questions that require querying data",
        ]
    )

    purpose = param.String(
        default="""
        Engages in conversations about high-level data topics, programming questions,
        technical documentation, and general conversation. Handles questions about
        specific functions, methods, libraries, and provides coding guidance and
        technical explanations.""")

    prompts = param.Dict(
        default={
            "main": {
                "template": PROMPTS_DIR / "ChatAgent" / "main.jinja2",
            },
        }
    )

    requires = param.List(default=[], readonly=True)

    async def respond(
        self,
        messages: list[Message],
        step_title: str | None = None,
    ) -> Any:
        context = {"tool_context": await self._use_tools("main", messages)}
        if "vector_metaset" not in self._memory and "source" in self._memory and "table" in self._memory:
            source = self._memory["source"]
            self._memory["vector_metaset"] = await get_metaset(
                [source], [f"{source.name}{SOURCE_TABLE_SEPARATOR}{self._memory['table']}"],
            )
        system_prompt = await self._render_prompt("main", messages, **context)
        return await self._stream(messages, system_prompt)


class AnalystAgent(ChatAgent):
    conditions = param.List(
        default=[
            "Use for interpreting and analyzing results from executed queries",
            "NOT for initial data queries, data exploration, or technical programming questions",
        ]
    )

    purpose = param.String(
        default="""
        Responsible for analyzing results and providing clear, concise, and actionable insights.
        Focuses on breaking down complex data findings into understandable points for
        high-level decision-making. Emphasizes detailed interpretation, trends, and
        relationships within the data, while avoiding general overviews or
        superficial descriptions.""")

    prompts = param.Dict(
        default={
            "main": {"template": PROMPTS_DIR / "AnalystAgent" / "main.jinja2", "tools": []},
        }
    )

    requires = param.List(default=["source", "pipeline"], readonly=True)

    async def respond(
        self,
        messages: list[Message],
        step_title: str | None = None,
    ) -> Any:
        messages = await super().respond(messages, step_title)
        if len(self._memory.get("data", [])) == 0 and self._memory.get("sql"):
            self._memory["sql"] = f"{self._memory['sql']}\n-- No data was returned from the query."
        return messages


class ListAgent(Agent):
    """
    Abstract base class for agents that display a list of items to the user.
    """

    purpose = param.String(default="""
        Renders a list of items to the user and lets the user pick one.""")

    requires = param.List(default=[], readonly=True)

    _extensions = ("tabulator",)

    _column_name = None

    _message_format = None

    __abstract = True

    def _get_items(self) -> dict[str, list[str]]:
        """Return dict of items grouped by source/category"""

    def _use_item(self, event):
        """Handle when a user clicks on an item in the list"""
        if event.column != "show":
            return

        tabulator = self._tabs[self._tabs.active]
        item = tabulator.value.iloc[event.row, 0]

        if self._message_format is None:
            raise ValueError("Subclass must define _message_format")

        message = self._message_format.format(item=repr(item))
        self.interface.send(message)

    async def respond(
        self,
        messages: list[Message],
        step_title: str | None = None,
    ) -> Any:
        items = self._get_items()

        # Create tabs with one tabulator per source
        tabs = []
        for source_name, source_items in items.items():
            if not source_items:
                continue

            header_filters = False
            if len(source_items) > 10:
                column_filter = {"type": "input", "placeholder": f"Filter by {self._column_name.lower()}..."}
                header_filters = {self._column_name: column_filter}

            df = pd.DataFrame({self._column_name: source_items})
            item_list = pn.widgets.Tabulator(
                df,
                buttons={"show": '<i class="fa fa-eye"></i>'},
                show_index=False,
                min_height=150,
                min_width=350,
                widths={self._column_name: "90%"},
                disabled=True,
                page_size=10,
                pagination="remote",
                header_filters=header_filters,
                sizing_mode="stretch_width",
                name=source_name
            )
            item_list.on_click(self._use_item)
            tabs.append(item_list)

        self._tabs = Tabs(*tabs, sizing_mode="stretch_width")

        self.interface.stream(
            pn.Column(
                f"The available {self._column_name.lower()}s are listed below. Click on the eye icon to show the {self._column_name.lower()} contents.",
                self._tabs
            ), user="Assistant"
        )
        return self._tabs


class TableListAgent(ListAgent):
    """
    The TableListAgent lists all available data and lets the user pick one.
    """

    conditions = param.List(default=[
        "Use when user explicitly asks to 'list data', 'show available data', or 'what data do you have'",
        "NOT for showing actual data contents, querying, or analyzing data",
    ])

    not_with = param.List(default=["DbtslAgent", "SQLAgent"])

    purpose = param.String(default="""
        Displays a list of all available data & datasets in memory. Not useful for identifying which dataset to use for analysis.""")

    requires = param.List(default=["source"], readonly=True)

    _column_name = "Data"

    _message_format = "Show the data: {item}"

    @classmethod
    async def applies(cls, memory: _Memory) -> bool:
        return len(memory.get('visible_slugs', set())) > 1

    def _get_items(self) -> dict[str, list[str]]:
        if "closest_tables" in self._memory:
            # If we have closest_tables from search, return as a single group
            return {"Search Results": self._memory["closest_tables"]}

        # Group tables by source
        visible_slugs = self._memory.get('visible_slugs', set())
        if not visible_slugs:
            return {}

        tables_by_source = {}
        for slug in visible_slugs:
            if SOURCE_TABLE_SEPARATOR in slug:
                source_name, table_name = slug.split(SOURCE_TABLE_SEPARATOR, 1)
            else:
                # Fallback if separator not found
                source_name = "Unknown"
                table_name = slug

            if source_name not in tables_by_source:
                tables_by_source[source_name] = []
            tables_by_source[source_name].append(table_name)

        # Sort tables within each source
        for source in tables_by_source:
            tables_by_source[source].sort()

        return tables_by_source


class DocumentListAgent(ListAgent):
    """
    The DocumentListAgent lists all available documents provided by the user.
    """

    conditions = param.List(
        default=[
            "Use when user asks to list or see all available documents",
            "NOT when user asks about specific document content",
        ]
    )

    purpose = param.String(default="""
        Displays a list of all available documents in memory.""")

    requires = param.List(default=["document_sources"], readonly=True)

    _column_name = "Documents"

    _message_format = "Tell me about: {item}"

    @classmethod
    async def applies(cls, memory: _Memory) -> bool:
        sources = memory.get("document_sources")
        if not sources:
            return False  # source not loaded yet; always apply
        return len(sources) > 1

    def _get_items(self) -> dict[str, list[str]]:
        # extract the filename, following this pattern `Filename: 'filename'``
        documents = [doc["metadata"].get("filename", "untitled") for doc in self._memory.get("document_sources", [])]
        # Return all documents under a single "Documents" category
        return {"Documents": documents} if documents else {}


class LumenBaseAgent(Agent):

    prompts = param.Dict(
        default={
            "retry_output": {"response_model": RetrySpec, "template": PROMPTS_DIR / "LumenBaseAgent" / "retry_output.jinja2"},
        }
    )

    user = param.String(default="Lumen")

    _max_width = None
    _output_type = LumenOutput
    _retry_target_keys = []

    def _update_spec(self, memory: _Memory, event: param.parameterized.Event):
        """
        Update the specification in memory.
        """

    @staticmethod
    def _prepare_lines_for_retry(original_output: str, retry_target_keys: list[str] | None = None) -> tuple[list[str], bool, dict | None]:
        """
        Prepare lines for retry by optionally extracting targeted sections from YAML.

        Parameters
        ----------
        original_output : str
            The original output string to process
        retry_target_keys : list[str], optional
            List of keys to target specific sections in YAML

        Returns
        -------
        tuple
            A tuple containing:

            - lines : list[str]
                List of lines to be processed
            - targeted : bool
                Whether targeting was applied
            - original_spec : dict or None
                Original parsed spec if targeting was used, None otherwise
        """
        lines = original_output.splitlines()
        targeted = False
        original_spec = None

        if retry_target_keys:
            original_spec = yaml.safe_load(original_output)
            targeted_output = original_spec.copy()
            for key in retry_target_keys:
                targeted_output = targeted_output[key]
            lines = yaml.dump(targeted_output, default_flow_style=False).splitlines()
            targeted = True

        return lines, targeted, original_spec

    @staticmethod
    def _apply_line_changes_to_output(
        lines: list[str],
        line_changes: list,
        targeted: bool,
        original_spec: dict | None = None,
        retry_target_keys: list[str] | None = None
    ) -> str:
        """
        Apply line changes to output, handling both targeted and non-targeted cases.

        Parameters
        ----------
        lines : list[str]
            The lines that were modified
        line_changes : list
            Changes to apply to the lines
        targeted : bool
            Whether this was a targeted retry
        original_spec : dict or None
            Original parsed spec for targeted retries
        retry_target_keys : list[str] or None
            Keys used for targeting

        Returns
        -------
        str
            Final output string with changes applied
        """
        if targeted:
            targeted_spec = yaml.safe_load(apply_changes(lines, line_changes))
            updated_spec = original_spec.copy()
            set_nested(updated_spec, retry_target_keys, targeted_spec)
            return yaml.dump(updated_spec)
        else:
            return apply_changes(lines, line_changes)

    async def _retry_output_by_line(
        self,
        feedback: str,
        messages: list[Message],
        memory: _Memory,
        original_output: str,
        language: str | None = None,
        **context
    ) -> str:
        """
        Retry the output by line, allowing the user to provide feedback on why the output was not satisfactory, or an error.
        """
        # Prepare lines for retry processing
        lines, targeted, original_spec = self._prepare_lines_for_retry(
            original_output, self._retry_target_keys
        )

        with self.param.update(memory=memory):
            numbered_text = "\n".join(f"{i:2d}: {line}" for i, line in enumerate(lines, 1))
            system = await self._render_prompt(
                "retry_output",
                messages=messages,
                numbered_text=numbered_text,
                language=language,
                feedback=feedback,
                **context
            )
        retry_model = self._lookup_prompt_key("retry_output", "response_model")
        invoke_kwargs = dict(
            messages=messages,
            system=system,
            response_model=retry_model,
            model_spec="edit",
        )
        result = await self.llm.invoke(**invoke_kwargs)

        # Apply line changes and return result
        return self._apply_line_changes_to_output(
            lines, result.lines_changes, targeted, original_spec, self._retry_target_keys
        )

    def _render_lumen(
        self,
        component: Component,
        messages: list | None = None,
        title: str | None = None,
        **kwargs,
    ):
        async def _retry_invoke(event: param.parameterized.Event):
            with out.param.update(loading=True):
                out.spec = await self._retry_output_by_line(event.new, messages, memory, out.spec, language=out.language)

        memory = self._memory
        retry_controls = RetryControls()
        retry_controls.param.watch(_retry_invoke, "reason")
        out = self._output_type(
            component=component,
            footer=[retry_controls],
            title=title,
            **kwargs
        )
        out.param.watch(partial(self._update_spec, self._memory), "spec")
        if "outputs" in self._memory:
            # We have to create a new list to trigger an event
            # since inplace updates will not trigger updates
            # and won't allow diffing between old and new values
            self._memory["outputs"] = self._memory["outputs"] + [out]
        if self.interface is not None:
            message_kwargs = dict(value=out, user=self.user)
            self.interface.stream(replace=True, max_width=self._max_width, **message_kwargs)


class SQLAgent(LumenBaseAgent):

    conditions = param.List(
        default=[
            "Use for displaying, examining, or querying data resulting in a data pipeline",
            "Use for calculations that require data (e.g., 'calculate average', 'sum by category')",
            "Commonly used with AnalystAgent to analyze query results",
            "NOT for non-data questions or technical programming help",
            "NOT useful if the user is using the same data for plotting",
        ]
    )

    exclusions = param.List(default=["dbtsl_metaset"])

    not_with = param.List(default=["DbtslAgent", "TableLookup", "TableListAgent"])

    exploration_enabled = param.Boolean(default=True, doc="""
        Whether to enable SQL exploration mode. When False, only attempts oneshot SQL generation.""")

    purpose = param.String(
        default="""
        Handles the display of data and the creation, modification, and execution
        of SQL queries to address user queries about the data. Executes queries in
        a single step, encompassing tasks such as table joins, filtering, aggregations,
        and calculations. If additional columns are required, SQLAgent can join the
        current table with other tables to fulfill the query requirements."""
    )

    prompts = param.Dict(
        default={
            "main": {
                "response_model": make_sql_model,
                "template": PROMPTS_DIR / "SQLAgent" / "main.jinja2",
            },
            "select_discoveries": {
                "response_model": DiscoveryQueries,
                "template": PROMPTS_DIR / "SQLAgent" / "select_discoveries.jinja2",
            },
            "check_sufficiency": {
                "response_model": DiscoverySufficiency,
                "template": PROMPTS_DIR / "SQLAgent" / "check_sufficiency.jinja2",
            },
            "retry_output": {"response_model": RetrySpec, "template": PROMPTS_DIR / "SQLAgent" / "retry_output.jinja2"},
        }
    )

    provides = param.List(default=["table", "sql", "pipeline", "data"], readonly=True)

    requires = param.List(default=["sources", "source", "sql_metaset"], readonly=True)

    user = param.String(default="SQL")

    _extensions = ("codeeditor", "tabulator")

    _output_type = LumenOutput

    def _update_spec(self, memory: _Memory, event: param.parameterized.Event):
        spec = load_yaml(event.new)
        table = spec["table"]
        source = spec["source"]
        if isinstance(source["tables"], dict):
            sql = source["tables"][table]
        else:
            sql = next((t["sql"] for t in source["tables"] if t["name"] == table), None)
        memory["sql"] = sql

    async def _generate_sql_queries(
        self, messages: list[Message], dialect: str, step_number: int,
        is_final: bool, context_entries: list[dict] | None,
        sql_plan_context: str | None = None, errors: list | None = None
    ) -> dict[str, str]:
        """Generate SQL queries using LLM."""
        # Build SQL history from context
        sql_query_history = {}
        if context_entries:
            for entry in context_entries:
                for query in entry.get("queries", []):
                    sql_query_history[query["sql"]] = query["table_status"]

        # Render prompt
        system_prompt = await self._render_prompt(
            "main",
            messages,
            dialect=dialect,
            step_number=step_number,
            is_final_step=is_final,
            current_step=messages[0]["content"] if not is_final else "",
            sql_query_history=sql_query_history,
            current_iteration=getattr(self, '_current_iteration', 1),
            sql_plan_context=sql_plan_context,
            errors=errors,
        )

        # Generate SQL
        model_spec = self.prompts["main"].get("llm_spec", self.llm_spec_key)

        sql_response_model = self._get_model("main", is_final=is_final)
        output = await self.llm.invoke(
            messages,
            system=system_prompt,
            model_spec=model_spec,
            response_model=sql_response_model,
        )
        if not output:
            raise ValueError("No output was generated.")

        sql_queries = {}
        for query_obj in ([output] if isinstance(output, SqlQuery) else output.queries):
            if query_obj.query and query_obj.expr_slug:
                sql_queries[query_obj.expr_slug.strip()] = query_obj.query.strip()

        return sql_queries

    async def _validate_sql(
        self, sql_query: str, expr_slug: str, dialect: str,
        source, messages: list[Message], step, max_retries: int = 2,
        discovery_context: str | None = None,
    ) -> str:
        """Validate and potentially fix SQL query."""
        # Clean SQL
        try:
            sql_query = clean_sql(sql_query, dialect)
        except Exception as e:
            step.stream(f"\n\n❌ SQL cleaning failed: {e}")

        # Validate with retries
        for i in range(max_retries):
            try:
                step.stream(f"\n\n`{expr_slug}`\n```sql\n{sql_query}\n```")
                source.create_sql_expr_source({expr_slug: sql_query}, materialize=True)
                step.stream("\n\n✅ SQL validation successful")
                return sql_query
            except Exception as e:
                if i == max_retries - 1:
                    step.stream(f"\n\n❌ SQL validation failed after {max_retries} attempts: {e}")
                    raise e

                # Retry with LLM fix
                step.stream(f"\n\n⚠️ SQL validation failed (attempt {i+1}/{max_retries}): {e}")
                feedback = f"{type(e).__name__}: {e!s}"
                if "KeyError" in feedback:
                    feedback += " The data does not exist; select from available data sources."

                retry_result = await self._retry_output_by_line(
                    feedback, messages, self._memory, sql_query, language=f"sql.{dialect}", discovery_context=discovery_context
                )
                sql_query = clean_sql(retry_result, dialect)
        return sql_query

    async def _execute_query(
        self, source, expr_slug: str, sql_query: str,
        is_final: bool, should_materialize: bool, step
    ) -> tuple[Pipeline, Source, str]:
        """Execute SQL query and return pipeline and summary."""
        # Create SQL source
        sql_expr_source = source.create_sql_expr_source(
            {expr_slug: sql_query}, materialize=should_materialize
        )

        if should_materialize:
            self._memory["source"] = sql_expr_source

        # Create pipeline
        if is_final:
            sql_transforms = [SQLLimit(limit=1_000_000, write=source.dialect, pretty=True, identify=False)]
            pipeline = await get_pipeline(source=sql_expr_source, table=expr_slug, sql_transforms=sql_transforms)
        else:
            pipeline = await get_pipeline(source=sql_expr_source, table=expr_slug)

        # Get data summary
        df = await get_data(pipeline)
        summary = await describe_data(df, reduce_enums=False)
        if len(summary) >= 1000:
            summary = summary[:1000-3] + "..."
        summary_formatted = f"\n```\n{summary}\n```"
        if should_materialize:
            summary_formatted += f"\n\nMaterialized data: `{sql_expr_source.name}{SOURCE_TABLE_SEPARATOR}{expr_slug}`"
        stream_details(f"{summary_formatted}", step, title=expr_slug)

        return pipeline, sql_expr_source, summary

    async def _finalize_execution(
        self,
        results: dict,
        context_entries: list[dict],
        messages: list[Message],
        step_title: str | None,
        raise_if_empty: bool = False
    ) -> None:
        """Finalize execution for final step."""
        # Get first result (typically only one for final step)
        expr_slug, result = next(iter(results.items()))

        # Update memory
        pipeline = result["pipeline"]
        df = await get_data(pipeline)

        # If dataframe is empty, raise error to fall back to exploration
        if df.empty and raise_if_empty:
            raise ValueError(f"\nQuery `{result['sql']}` returned empty results; ensure all the WHERE filter values exist in the dataset.")

        self._memory["data"] = await describe_data(df)
        self._memory["sql"] = result["sql"]
        self._memory["pipeline"] = pipeline
        self._memory["table"] = pipeline.table
        self._memory["source"] = pipeline.source
        self._memory["sql_plan_context"] = context_entries

        # Render output
        self._render_lumen(
            pipeline,
            messages=messages,
            title=step_title
        )

    async def _render_execute_query(
        self,
        messages: list[Message],
        source: Source,
        step_title: str,
        success_message: str,
        discovery_context: str | None = None,
        raise_if_empty: bool = False,
        output_title: str | None = None
    ) -> Pipeline:
        """
        Helper method that generates, validates, and executes final SQL queries.

        Parameters
        ----------
        messages : list[Message]
            List of user messages
        source : Source
            Data source to execute against
        step_title : str
            Title for the execution step
        success_message : str
            Message to show on successful completion
        discovery_context : str, optional
            Optional discovery context to include in prompt
        raise_if_empty : bool, optional
            Whether to raise error if query returns empty results
        output_title : str, optional
            Title to use for the output

        Returns
        -------
        Pipeline
            Pipeline object from successful execution
        """
        with self._add_step(title=step_title, steps_layout=self._steps_layout) as step:
            # Generate SQL using common prompt pattern
            system_prompt = await self._render_prompt(
                "main",
                messages,
                dialect=source.dialect,
                step_number=1,
                is_final_step=True,
                current_step="",
                sql_query_history={},
                current_iteration=1,
                sql_plan_context=None,
                errors=None,
                discovery_context=discovery_context,
            )

            # Generate SQL using existing model
            model_spec = self.prompts["main"].get("llm_spec", self.llm_spec_key)
            sql_response_model = self._get_model("main", is_final=True)
            output = await self.llm.invoke(
                messages,
                system=system_prompt,
                model_spec=model_spec,
                response_model=sql_response_model,
            )

            if not output:
                raise ValueError("No output was generated.")

            # Extract SQL query
            sql_query = output.query.strip()
            expr_slug = output.expr_slug.strip()

            # Validate SQL
            validated_sql = await self._validate_sql(
                sql_query, expr_slug, source.dialect, source, messages, step, discovery_context=discovery_context
            )

            # Execute and get results
            pipeline, sql_expr_source, summary = await self._execute_query(
                source, expr_slug, validated_sql, is_final=True,
                should_materialize=True, step=step
            )

            # Finalize execution
            results = {
                expr_slug: {
                    "sql": validated_sql,
                    "summary": summary,
                    "pipeline": pipeline,
                    "source": sql_expr_source
                }
            }

            await self._finalize_execution(
                results, [], messages, output_title, raise_if_empty=raise_if_empty
            )

            step.status = "success"
            step.success_title = success_message

        return pipeline

    async def _select_discoveries(
        self,
        messages: list[Message],
        error_context: str,
    ) -> DiscoveryQueries:
        """Let LLM choose which discoveries to run based on error and available tables."""
        system_prompt = await self._render_prompt(
            "select_discoveries",
            messages,
            error_context=error_context,
        )

        model_spec = self.prompts["select_discoveries"].get("llm_spec", self.llm_spec_key)
        selection = await self.llm.invoke(
            messages,
            system=system_prompt,
            model_spec=model_spec,
            response_model=DiscoveryQueries,
        )
        return selection

    async def _check_discovery_sufficiency(
        self,
        messages: list[Message],
        error_context: str,
        discovery_results: list[tuple[str, str]],
    ) -> DiscoverySufficiency:
        """Check if discovery results are sufficient to fix the error."""
        system_prompt = await self._render_prompt(
            "check_sufficiency",
            messages,
            error_context=error_context,
            discovery_results=discovery_results,
        )

        model_spec = self.prompts["check_sufficiency"].get("llm_spec", self.llm_spec_key)
        sufficiency = await self.llm.invoke(
            messages,
            system=system_prompt,
            model_spec=model_spec,
            response_model=DiscoverySufficiency,
        )
        return sufficiency

    async def _execute_discovery_query(
        self,
        query_model: SampleQuery | DistinctQuery,
        source: Source,
    ) -> tuple[str, str]:
        """Execute a single discovery query and return formatted result."""
        # Generate the actual SQL
        sql = query_model.query.format(**query_model.dict(exclude={'query'}))
        try:
            df = source.execute(sql)
            # Format results compactly for context
            if isinstance(query_model, SampleQuery):
                # Limit to first 5 rows, truncate long values
                result_summary = f"Sample from {query_model.table}:\n```\n{df.head().to_string(max_cols=10)}\n```"
            elif isinstance(query_model, DistinctQuery):
                # Show distinct values
                values = df.iloc[:, 0].tolist()[:10]  # First column, max 10 values
                result_summary = f"Distinct {query_model.column}: `{values}`"
            return query_model.__class__.__name__, result_summary
        except Exception as e:
            return query_model.__class__.__name__, f"Error: {e!s}"

    async def _run_discoveries_parallel(
        self,
        discovery_queries: list[SampleQuery | DistinctQuery],
        source: Source,
    ) -> list[tuple[str, str]]:
        """Execute all discoveries concurrently and stream results as they complete."""
        # Create all tasks
        tasks = [
            asyncio.create_task(self._execute_discovery_query(query, source))
            for query in discovery_queries
        ]

        results = []
        # Stream results as they complete (fastest first)
        for completed_task in asyncio.as_completed(tasks):
            query_type, result = await completed_task
            results.append((query_type, result))
            # Stream each result immediately when ready
            with self._add_step(title=f"Discovery: {query_type}", steps_layout=self._steps_layout) as step:
                step.stream(result)
        return results

    async def _explore_tables(
        self,
        messages: list[Message],
        source: Source,
        error_context: str,
        step_title: str | None = None,
    ) -> Any:
        """Run adaptive exploration: initial discoveries → sufficiency check → optional follow-ups → final answer."""
        # Step 1: LLM selects initial discoveries
        with self._add_step(title="Selecting initial discoveries", steps_layout=self._steps_layout) as step:
            selection = await self._select_discoveries(messages, error_context)
            step.stream(f"Strategy: {selection.reasoning}\n\nSelected {len(selection.queries)} initial discoveries")

        # Step 2: Run initial discoveries in parallel
        initial_results = await self._run_discoveries_parallel(selection.queries, source)

        # Step 3: Check if discoveries are sufficient
        with self._add_step(title="Evaluating discovery sufficiency", steps_layout=self._steps_layout) as step:
            sufficiency = await self._check_discovery_sufficiency(messages, error_context, initial_results)
            step.stream(f"Assessment: {sufficiency.reasoning}")

            if sufficiency.sufficient:
                step.stream("\n✅ Sufficient context - proceeding to final answer")
                final_results = initial_results
            else:
                step.stream(f"\n⚠️ Insufficient - running {len(sufficiency.follow_up_needed)} follow-ups")

        # Step 4: Run follow-up discoveries if needed
        if not sufficiency.sufficient and sufficiency.follow_up_needed:
            follow_up_results = await self._run_discoveries_parallel(sufficiency.follow_up_needed, source)
            final_results = initial_results + follow_up_results
        else:
            final_results = initial_results

        # Step 5: Generate final answer with all discovery context
        discovery_context = "\n".join([
            f"- **{query_type}:** {result}" for query_type, result in final_results
        ])

        return await self._render_execute_query(
            messages=messages,
            source=source,
            step_title="Generating final answer with discoveries",
            success_message="Final answer generated with discovery context",
            discovery_context=discovery_context,
            raise_if_empty=False,
            output_title=step_title
        )

    async def respond(
        self,
        messages: list[Message],
        step_title: str | None = None,
    ) -> Any:
        """Execute SQL generation with one-shot attempt first, then exploration if needed."""
        # Setup sources
        source = self._memory["source"]
        try:
            # Try one-shot approach first
            pipeline = await self._render_execute_query(
                messages=messages,
                source=source,
                step_title="Attempting one-shot SQL generation...",
                success_message="One-shot SQL generation successful",
                discovery_context=None,
                raise_if_empty=True,
                output_title=step_title
            )
        except Exception as e:
            if not self.exploration_enabled:
                # If exploration is disabled, re-raise the error instead of falling back
                raise e
            # Fall back to exploration mode if enabled
            pipeline = await self._explore_tables(messages, source, str(e), step_title)
        return pipeline


class DbtslAgent(LumenBaseAgent, DbtslMixin):
    """
    Responsible for creating and executing queries against a dbt Semantic Layer
    to answer user questions about business metrics.
    """

    conditions = param.List(
        default=[
            "Always use this when dbtsl_metaset is available",
        ]
    )

    purpose = param.String(
        default="""
        Responsible for displaying data to answer user queries about
        business metrics using dbt Semantic Layers. This agent can compile
        and execute metric queries against a dbt Semantic Layer."""
    )

    prompts = param.Dict(
        default={
            "main": {
                "response_model": DbtslQueryParams,
                "template": PROMPTS_DIR / "DbtslAgent" / "main.jinja2",
            },
            "retry_output": {"response_model": RetrySpec, "template": PROMPTS_DIR / "LumenBaseAgent" / "retry_output.jinja2"},
        }
    )

    provides = param.List(default=["table", "sql", "pipeline", "data", "dbtsl_vector_metaset", "dbtsl_sql_metaset"], readonly=True)

    requires = param.List(default=["source", "dbtsl_metaset"], readonly=True)

    source = param.ClassSelector(
        class_=BaseSQLSource,
        doc="""
        The source associated with the dbt Semantic Layer.""",
    )

    user = param.String(default="DBT")

    _extensions = ("codeeditor", "tabulator")

    _output_type = LumenOutput

    def __init__(self, source: Source, **params):
        super().__init__(source=source, **params)

    def _update_spec(self, memory: _Memory, event: param.parameterized.Event):
        """
        Update the SQL specification in memory.
        """
        memory["sql"] = event.new

    @retry_llm_output()
    async def _create_valid_query(self, messages: list[Message], title: str | None = None, errors: list | None = None):
        """
        Create a valid dbt Semantic Layer query based on user messages.
        """
        system_prompt = await self._render_prompt(
            "main",
            messages,
            errors=errors,
        )

        with self._add_step(title=title or "dbt Semantic Layer query", steps_layout=self._steps_layout) as step:
            model_spec = self.prompts["main"].get("llm_spec", self.llm_spec_key)
            response = self.llm.stream(
                messages,
                system=system_prompt,
                model_spec=model_spec,
                response_model=DbtslQueryParams,
            )

            query_params = None
            try:
                async for output in response:
                    step_message = output.chain_of_thought or ""
                    step.stream(step_message, replace=True)

                query_params = {
                    "metrics": output.metrics,
                    "group_by": output.group_by,
                    "limit": output.limit,
                    "order_by": output.order_by,
                    "where": output.where,
                }
                expr_slug = output.expr_slug

                # Ensure required fields are present
                if "metrics" not in query_params or not query_params["metrics"]:
                    raise ValueError("Query must include at least one metric")

                # Ensure limit exists with a default value if not provided
                if "limit" not in query_params or not query_params["limit"]:
                    query_params["limit"] = 10000

                formatted_params = json.dumps(query_params, indent=2)
                step.stream(f"\n\n`{expr_slug}`\n```json\n{formatted_params}\n```")

                self._memory["dbtsl_query_params"] = query_params
            except asyncio.CancelledError as e:
                step.failed_title = "Cancelled dbt Semantic Layer query generation"
                raise e

        if step.failed_title and step.failed_title.startswith("Cancelled"):
            raise asyncio.CancelledError()
        elif not query_params:
            raise ValueError("No dbt Semantic Layer query was generated.")

        try:
            # Execute the query against the dbt Semantic Layer
            client = self._instantiate_client()
            async with client.session():
                sql_query = await client.compile_sql(
                    metrics=query_params.get("metrics", []),
                    group_by=query_params.get("group_by"),
                    limit=query_params.get("limit"),
                    order_by=query_params.get("order_by"),
                    where=query_params.get("where"),
                )

                # Log the compiled SQL for debugging
                step.stream(f"\nCompiled SQL:\n```sql\n{sql_query}\n```", replace=False)

            sql_expr_source = self.source.create_sql_expr_source({expr_slug: sql_query})
            self._memory["sql"] = sql_query

            # Apply transforms
            sql_transforms = [SQLLimit(limit=1_000_000, write=self.source.dialect, pretty=True, identify=False)]
            transformed_sql_query = sql_query
            for sql_transform in sql_transforms:
                transformed_sql_query = sql_transform.apply(transformed_sql_query)
                if transformed_sql_query != sql_query:
                    stream_details(f"```sql\n{transformed_sql_query}\n```", step, title=f"{sql_transform.__class__.__name__} Applied")

            # Create pipeline and get data
            pipeline = await get_pipeline(source=sql_expr_source, table=expr_slug, sql_transforms=sql_transforms)

            df = await get_data(pipeline)
            sql_metaset = await get_metaset([sql_expr_source], [expr_slug])
            vector_metaset = sql_metaset.vector_metaset

            # Update memory
            self._memory["data"] = await describe_data(df)
            self._memory["source"] = sql_expr_source
            self._memory["pipeline"] = pipeline
            self._memory["table"] = pipeline.table
            self._memory["dbtsl_vector_metaset"] = vector_metaset
            self._memory["dbtsl_sql_metaset"] = sql_metaset
            return sql_query, pipeline
        except Exception as e:
            report_error(e, step)
            raise e

    async def respond(
        self,
        messages: list[Message],
        step_title: str | None = None,
    ) -> Any:
        """
        Responds to user messages by generating and executing a dbt Semantic Layer query.
        """
        try:
            sql_query, pipeline = await self._create_valid_query(messages, step_title)
        except RetriesExceededError as e:
            traceback.print_exception(e)
            self._memory["__error__"] = str(e)
            return None

        self._render_lumen(pipeline, messages=messages, title=step_title)
        return pipeline


class BaseViewAgent(LumenBaseAgent):

    requires = param.List(default=["pipeline", "table", "data"], readonly=True)

    provides = param.List(default=["view"], readonly=True)

    prompts = param.Dict(
        default={
            "main": {"template": PROMPTS_DIR / "BaseViewAgent" / "main.jinja2"},
        }
    )

    def __init__(self, **params):
        self._last_output = None
        super().__init__(**params)

    @retry_llm_output()
    async def _create_valid_spec(
        self,
        messages: list[Message],
        pipeline: Pipeline,
        schema: dict[str, Any],
        step_title: str | None = None,
        errors: list[str] | None = None,
    ) -> dict[str, Any]:
        errors_context = {}
        if errors:
            errors = ("\n".join(f"{i + 1}. {error}" for i, error in enumerate(errors))).strip()
            try:
                last_output = yaml.safe_load(self._last_output["yaml_spec"])
            except Exception:
                last_output = ""

            vector_metadata_map = None
            if "sql_metaset" in self._memory:
                vector_metadata_map = self._memory["sql_metaset"].vector_metaset.vector_metadata_map
            elif "dbtsl_sql_metaset" in self._memory:
                vector_metadata_map = self._memory["dbtsl_sql_metaset"].vector_metaset.vector_metadata_map

            columns_context = ""
            if vector_metadata_map is not None:
                for table_slug, vector_metadata in vector_metadata_map.items():
                    table_name = table_slug.split(SOURCE_TABLE_SEPARATOR)[-1]
                    if table_name in pipeline.table:
                        columns = [col.name for col in vector_metadata.columns]
                        columns_context += f"\nSQL: {vector_metadata.base_sql}\nColumns: {', '.join(columns)}\n\n"
            errors_context = {
                "errors": errors,
                "last_output": last_output,
                "num_errors": len(errors),
                "columns_context": columns_context,
            }

        doc = self.view_type.__doc__.split("\n\n")[0] if self.view_type.__doc__ else self.view_type.__name__
        system = await self._render_prompt(
            "main",
            messages,
            table=pipeline.table,
            doc=doc,
            **errors_context,
        )

        model_spec = self.prompts["main"].get("llm_spec", self.llm_spec_key)
        response = self.llm.stream(
            messages,
            system=system,
            model_spec=model_spec,
            response_model=self._get_model("main", schema=schema),
        )

        e = None
        error = ""
        spec = ""
        with self._add_step(title=step_title or "Generating view...", steps_layout=self._steps_layout) as step:
            async for output in response:
                chain_of_thought = output.chain_of_thought or ""
                step.stream(chain_of_thought, replace=True)

            self._last_output = spec = dict(output)

            for i in range(3):
                try:
                    spec = await self._extract_spec(spec)
                    break
                except Exception as e:
                    e = get_root_exception(e, exceptions=(MissingContextError,))
                    if isinstance(e, MissingContextError):
                        raise e

                    error = str(e)
                    traceback.print_exception(e)
                    context = f"```\n{yaml.safe_dump(yaml.safe_load(self._last_output['yaml_spec']))}\n```"
                    report_error(e, step, language="json", context=context, status="failed")
                    with self._add_step(
                        title="Re-attempted view generation",
                        steps_layout=self._steps_layout,
                    ) as retry_step:
                        view = await self._retry_output_by_line(e, messages, self._memory, yaml.safe_dump(spec), language="")
                        if "yaml_spec: " in view:
                            view = view.split("yaml_spec: ")[-1].rstrip('"').rstrip("'")
                        retry_step.stream(f"\n\n```yaml\n{view}\n```")
                    if i == 2:
                        raise

        self._last_output = spec

        if error:
            raise ValueError(error)

        log_debug(f"{self.name} settled on spec: {spec!r}.")
        return spec

    async def _extract_spec(self, spec: dict[str, Any]):
        return dict(spec)

    async def _update_spec(self, memory: _Memory, event: param.parameterized.Event):
        memory["view"] = dict(await self._extract_spec(event.new), type=self.view_type)

    async def respond(
        self,
        messages: list[Message],
        step_title: str | None = None,
    ) -> Any:
        """
        Generates a visualization based on user messages and the current data pipeline.
        """
        pipeline = self._memory.get("pipeline")
        if not pipeline:
            raise ValueError("No current pipeline found in memory.")

        schema = await get_schema(pipeline)
        if not schema:
            raise ValueError("Failed to retrieve schema for the current pipeline.")

        spec = await self._create_valid_spec(messages, pipeline, schema, step_title)
        self._memory["view"] = dict(spec, type=self.view_type)
        view = self.view_type(pipeline=pipeline, **spec)
        self._render_lumen(view, messages=messages, title=step_title)
        return view


class hvPlotAgent(BaseViewAgent):
    conditions = param.List(
        default=[
            "Use for exploratory data analysis, interactive plots, and dynamic filtering",
            "Use for quick, iterative data visualization during analysis",
        ]
    )

    purpose = param.String(default="Generates a plot of the data given a user prompt.")

    prompts = param.Dict(
        default={
            "main": {"template": PROMPTS_DIR / "hvPlotAgent" / "main.jinja2"},
        }
    )

    view_type = hvPlotUIView

    def _get_model(self, prompt_name: str, schema: dict[str, Any]) -> type[BaseModel]:
        # Find parameters
        excluded = self.view_type._internal_params + [
            "controls",
            "type",
            "source",
            "pipeline",
            "transforms",
            "download",
            "field",
            "selection_group",
        ]
        model = param_to_pydantic(
            self.view_type,
            base_model=PartialBaseModel,
            excluded=excluded,
            schema=schema,
            extra_fields={
                "chain_of_thought": (str, FieldInfo(description="Your thought process behind the plot.")),
            },
        )
        return model[self.view_type.__name__]

    async def _update_spec(self, memory: _Memory, event: param.parameterized.Event):
        spec = yaml.load(event.new, Loader=yaml.SafeLoader)
        memory["view"] = dict(await self._extract_spec(spec), type=self.view_type)

    async def _extract_spec(self, spec: dict[str, Any]):
        pipeline = self._memory["pipeline"]
        spec = {key: val for key, val in spec.items() if val is not None}
        spec["type"] = "hvplot_ui"
        self.view_type.validate(spec)
        spec.pop("type", None)

        # Add defaults
        spec["responsive"] = True
        data = await get_data(pipeline)
        if len(data) > 20000 and spec["kind"] in ("line", "scatter", "points"):
            spec["rasterize"] = True
            spec["cnorm"] = "log"
        return spec


class VegaLiteAgent(BaseViewAgent):

    conditions = param.List(
        default=[
            "Use for publication-ready visualizations or when user specifically requests Vega-Lite charts",
            "Use for polished charts intended for presentation or sharing",
        ]
    )

    purpose = param.String(default="Generates a vega-lite specification of the plot the user requested.")

    prompts = param.Dict(
        default={
            "main": {"response_model": VegaLiteBasicSpec, "template": PROMPTS_DIR / "VegaLiteAgent" / "main.jinja2"},
            "axes": {"response_model": VegaLiteSpecUpdate, "template": PROMPTS_DIR / "VegaLiteAgent" / "axes.jinja2"},
            "labels": {"response_model": VegaLiteSpecUpdate, "template": PROMPTS_DIR / "VegaLiteAgent" / "labels.jinja2"},
            "polish": {"response_model": VegaLiteSpecUpdate, "template": PROMPTS_DIR / "VegaLiteAgent" / "polish.jinja2"},
            "retry_output": {"response_model": RetrySpec, "template": PROMPTS_DIR / "VegaLiteAgent" / "retry_output.jinja2"},
        }
    )

    view_type = VegaLiteView

    _extensions = ("vega",)

    _output_type = VegaLiteOutput

    _retry_target_keys = ["spec"]

    def _deep_merge_dicts(self, base_dict: dict[str, Any], update_dict: dict[str, Any]) -> dict[str, Any]:
        """Deep merge two dictionaries, with update_dict taking precedence."""
        result = base_dict.copy()
        for key, value in update_dict.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._deep_merge_dicts(result[key], value)
            else:
                result[key] = value
        return result

    async def _update_spec_step(
        self,
        step_name: str,
        step_desc: str,
        current_spec: dict[str, Any],
        prompt_name: str,
        messages: list[Message],
        doc: str | None = None,
    ) -> tuple[str, dict[str, Any]]:
        """Update a Vega-Lite spec with incremental changes for a specific step."""
        with self.interface.param.update(callback_exception="raise"), self._add_step(title=step_desc, steps_layout=self._steps_layout) as step:
            system_prompt = await self._render_prompt(
                prompt_name,
                messages,
                current_spec=yaml.dump(current_spec["spec"], default_flow_style=False),
                doc=doc,
                table=self._memory["pipeline"].table
            )

            model_spec = self.prompts.get(prompt_name, {}).get("llm_spec", self.llm_spec_key)
            result = await self.llm.invoke(
                messages=messages,
                system=system_prompt,
                response_model=VegaLiteSpecUpdate,
                model_spec=model_spec,
            )

            step.stream(f"Reasoning: {result.chain_of_thought}")
            step.stream(f"Update:\n```yaml\n{result.yaml_update}\n```", replace=False)

            update_dict = yaml.safe_load(result.yaml_update)

            # Validate the update by attempting to merge and extract
            test_spec = self._deep_merge_dicts(current_spec, update_dict)
            await self._extract_spec({"yaml_spec": yaml.dump(test_spec)})  # Validation
            step.success_title = f"{step_name} completed"
            return step_name, update_dict

    async def _update_spec(self, memory: _Memory, event: param.parameterized.Event):
        try:
            spec = await self._extract_spec({"yaml_spec": event.new})
        except Exception as e:
            traceback.print_exception(e)
            return
        memory["view"] = dict(spec, type=self.view_type)

    def _standardize_to_layers(self, vega_spec: dict) -> None:
        """Standardize vega spec by migrating to layer format."""
        if "layer" not in vega_spec:
            vega_spec["layer"] = [{
                "encoding": vega_spec.pop("encoding", None),
                "mark": vega_spec.pop("mark", None)
            }]

    def _add_zoom_params(self, vega_spec: dict) -> None:
        """Add zoom parameters to vega spec."""
        if "params" not in vega_spec:
            vega_spec["params"] = []

        existing_param_names = {
            param.get("name") for param in vega_spec["params"]
            if isinstance(param, dict) and "name" in param
        }

        for p in VEGA_ZOOMABLE_MAP_ITEMS["params"]:
            if p.get("name") not in existing_param_names:
                vega_spec["params"].append(p)

    def _setup_projection(self, vega_spec: dict) -> None:
        """Setup map projection settings."""
        if "projection" not in vega_spec:
            vega_spec["projection"] = {"type": "mercator"}
        vega_spec["projection"].update(VEGA_ZOOMABLE_MAP_ITEMS["projection"])

    def _handle_map_compatibility(self, vega_spec: dict, vega_spec_str: str) -> None:
        """Handle map projection compatibility and add geographic outlines."""
        has_world_map = "world-110m.json" in vega_spec_str
        uses_albers_usa = vega_spec["projection"]["type"] == "albersUsa"

        if has_world_map and uses_albers_usa:
            # albersUsa incompatible with world map
            vega_spec["projection"] = "mercator"
        elif not has_world_map and not uses_albers_usa:
            # Add world map outlines if needed
            vega_spec["layer"].append(VEGA_MAP_LAYER["world"])

    def _add_geographic_items(self, vega_spec: dict, vega_spec_str: str) -> dict:
        """Add geographic visualization items to vega spec."""
        self._standardize_to_layers(vega_spec)
        self._add_zoom_params(vega_spec)
        self._setup_projection(vega_spec)
        self._handle_map_compatibility(vega_spec, vega_spec_str)
        return vega_spec

    @classmethod
    def _extract_as_keys(cls, transforms: list[dict]) -> list[str]:
        """
        Extracts all 'as' field names from a list of Vega-Lite transform definitions.

        Parameters
        ----------
        transforms : list[dict]
            A list of Vega-Lite transform objects.

        Returns
        -------
        list[str]
            A list of field names from 'as' keys (flattened, deduplicated).
        """
        as_fields = []
        for t in transforms:
            # Top-level 'as'
            if "as" in t:
                if isinstance(t["as"], list):
                    as_fields.extend(t["as"])
                elif isinstance(t["as"], str):
                    as_fields.append(t["as"])
            for key in ("aggregate", "joinaggregate", "window"):
                if key in t and isinstance(t[key], list):
                    for entry in t[key]:
                        if "as" in entry:
                            as_fields.append(entry["as"])

        return list(dict.fromkeys(as_fields))

    async def _ensure_columns_exists(self, vega_spec: dict):
        pipeline = self._memory.get("pipeline")
        if not pipeline:
            return
        schema = await get_schema(pipeline)

        fields = self._extract_as_keys(vega_spec.get('transform', [])) + list(schema)
        for layer in vega_spec.get("layer", []):
            encoding = layer.get("encoding", {})
            if not encoding:
                continue

            for enc_def in encoding.values():
                fields_to_check = []

                if isinstance(enc_def, dict) and "field" in enc_def:
                    fields_to_check.append(enc_def["field"])
                elif isinstance(enc_def, list):
                    fields_to_check.extend(item["field"] for item in enc_def if isinstance(item, dict) and "field" in item)

                for field in fields_to_check:
                    if field not in fields and field.lower() not in fields and field.upper() not in fields:
                        raise ValueError(f"Field '{field}' not found in schema.")

    async def _extract_spec(self, spec: dict[str, Any]):
        # .encode().decode('unicode_escape') fixes a JSONDecodeError in Python
        # where it's expecting property names enclosed in double quotes
        # by properly handling the escaped characters in your JSON string
        if yaml_spec := spec.get("yaml_spec"):
            vega_spec = yaml.load(yaml_spec, Loader=yaml.SafeLoader)
        elif json_spec := spec.get("json_spec"):
            vega_spec = load_json(json_spec)
        if "$schema" not in vega_spec:
            vega_spec["$schema"] = "https://vega.github.io/schema/vega-lite/v5.json"
        if "width" not in vega_spec:
            vega_spec["width"] = "container"
        if "height" not in vega_spec:
            vega_spec["height"] = "container"
        self._output_type._validate_spec(vega_spec)
        await self._ensure_columns_exists(vega_spec)

        # using string comparison because these keys could be in different nested levels
        vega_spec_str = yaml.dump(vega_spec)
        # Handle different types of interactive controls based on chart type
        if "latitude:" in vega_spec_str or "longitude:" in vega_spec_str:
            vega_spec = self._add_geographic_items(vega_spec, vega_spec_str)
        elif ("point: true" not in vega_spec_str or "params" not in vega_spec) and vega_spec_str.count("encoding:") == 1:
            # add pan/zoom controls to all plots except geographic ones and points overlaid on line plots
            # because those result in an blank plot without error
            vega_spec["params"] = [{"bind": "scales", "name": "grid", "select": "interval"}]
        return {"spec": vega_spec, "sizing_mode": "stretch_both", "min_height": 300, "max_width": 1200}

    async def respond(
        self,
        messages: list[Message],
        step_title: str | None = None,
    ) -> Any:
        """
        Generates a VegaLite visualization using progressive building approach with real-time updates.
        """
        pipeline = self._memory.get("pipeline")
        if not pipeline:
            raise ValueError("No current pipeline found in memory.")

        schema = await get_schema(pipeline)
        if not schema:
            raise ValueError("Failed to retrieve schema for the current pipeline.")

        # Step 1: Generate basic spec
        with self._add_step(title="Creating basic plot structure", steps_layout=self._steps_layout) as step:
            doc = self.view_type.__doc__.split("\n\n")[0] if self.view_type.__doc__ else self.view_type.__name__

            system_prompt = await self._render_prompt(
                "main",
                messages,
                table=pipeline.table,
                doc=doc,
            )
            model_spec = self.prompts["main"].get("llm_spec", self.llm_spec_key)

            response = self.llm.stream(
                messages,
                system=system_prompt,
                model_spec=model_spec,
                response_model=VegaLiteBasicSpec,
            )

            async for output in response:
                step.stream(output.chain_of_thought, replace=True)

            # Validate basic spec
            current_spec = await self._extract_spec({"yaml_spec": output.yaml_spec})
            step.success_title = "Basic plot structure created"

        # Step 2: Show basic plot immediately - users see it right away!
        self._memory["view"] = dict(current_spec, type=self.view_type)
        view = self.view_type(pipeline=pipeline, **current_spec)
        # Render the basic plot - this creates the 'out' object we can update
        out = self._render_lumen(view, messages=messages, title=step_title)

        # Steps 3-5: Run enhancements in parallel and update visualization in real-time
        parallel_steps = {
            "axes": "Adding axis formatting and titles",
            # "labels": "Adding colors and tooltips",
            # "polish": "Adding titles and styling",
        }

        # Create tasks for parallel execution
        tasks = []
        for step_name, step_desc in parallel_steps.items():
            task = asyncio.create_task(
                self._update_spec_step(step_name, step_desc, current_spec, step_name, messages, doc=doc)
            )
            tasks.append(task)

        for completed_task in asyncio.as_completed(tasks):
            step_name, update_dict = await completed_task
            full_spec = yaml.safe_load(out.spec)
            full_spec["spec"] = self._deep_merge_dicts(full_spec["spec"], update_dict)
            out.spec = yaml.dump(full_spec)
            log_debug(f"📊 Applied {step_name} updates and refreshed visualization")

        # Update final memory state
        final_vega_spec = await self._extract_spec({"yaml_spec": yaml.dump(current_spec)})
        self._memory["view"] = dict(final_vega_spec, type=self.view_type)
        self._memory["pipeline"] = pipeline
        return view


class AnalysisAgent(LumenBaseAgent):

    analyses = param.List([])

    conditions = param.List(
        default=[
            "Use for custom analysis, advanced analytics, or domain-specific analysis methods",
            "Use when built-in SQL/visualization agents are insufficient",
            "NOT for simple queries or basic visualizations",
        ]
    )

    purpose = param.String(default="Perform custom analyses on the data.")

    prompts = param.Dict(
        default={
            "main": {"template": PROMPTS_DIR / "AnalysisAgent" / "main.jinja2"},
        }
    )

    provides = param.List(default=["view"])

    requires = param.List(default=["pipeline"])

    _output_type = AnalysisOutput

    def _update_spec(self, memory: _Memory, event: param.parameterized.Event):
        pass

    async def respond(
        self,
        messages: list[Message],
        step_title: str | None = None
    ) -> Any:
        pipeline = self._memory["pipeline"]
        analyses = {a.name: a for a in self.analyses if await a.applies(pipeline)}
        if not analyses:
            log_debug("No analyses apply to the current data.")
            return None

        # Short cut analysis selection if there's an exact match
        if len(messages):
            analysis = messages[0].get("content").replace("Apply ", "")
            if analysis in analyses:
                analyses = {analysis: analyses[analysis]}

        if len(analyses) > 1:
            with self._add_step(title="Choosing the most relevant analysis...", steps_layout=self._steps_layout) as step:
                type_ = Literal[tuple(analyses)]
                analysis_model = create_model(
                    "Analysis", correct_name=(type_, FieldInfo(description="The name of the analysis that is most appropriate given the user query."))
                )
                system_prompt = await self._render_prompt(
                    "main",
                    messages,
                    analyses=analyses,
                    data=self._memory.get("data"),
                )
                model_spec = self.prompts["main"].get("llm_spec", self.llm_spec_key)
                analysis_name = (
                    await self.llm.invoke(
                        messages,
                        system=system_prompt,
                        model_spec=model_spec,
                        response_model=analysis_model,
                        allow_partial=False,
                    )
                ).correct_name
                step.stream(f"Selected {analysis_name}")
                step.success_title = f"Selected {analysis_name}"
        else:
            analysis_name = next(iter(analyses))

        view = None
        with self.interface.param.update(callback_exception="raise"):
            with self._add_step(title=step_title or "Creating view...", steps_layout=self._steps_layout) as step:
                await asyncio.sleep(0.1)  # necessary to give it time to render before calling sync function...
                analysis_callable = analyses[analysis_name].instance(agents=self.agents, memory=self._memory, interface=self.interface)

                data = await get_data(pipeline)
                for field in analysis_callable._field_params:
                    analysis_callable.param[field].objects = list(data.columns)
                self._memory["analysis"] = analysis_callable

                if analysis_callable.autorun:
                    if asyncio.iscoroutinefunction(analysis_callable.__call__):
                        view = await analysis_callable(pipeline)
                    else:
                        view = await asyncio.to_thread(analysis_callable, pipeline)
                    if isinstance(view, Viewable):
                        view = Panel(object=view, pipeline=self._memory.get("pipeline"))
                    spec = view.to_spec()
                    if isinstance(view, View):
                        view_type = view.view_type
                        self._memory["view"] = dict(spec, type=view_type)
                    elif isinstance(view, Pipeline):
                        self._memory["pipeline"] = view
                    # Ensure data reflects processed pipeline
                    if pipeline is not self._memory["pipeline"]:
                        pipeline = self._memory["pipeline"]
                        data = await get_data(pipeline)
                        if len(data) > 0:
                            self._memory["data"] = await describe_data(data)
                    yaml_spec = yaml.dump(spec)
                    step.stream(f"Generated view\n```yaml\n{yaml_spec}\n```")
                    step.success_title = "Generated view"
                else:
                    step.success_title = "Configure the analysis"

        analysis = self._memory["analysis"]
        pipeline = self._memory["pipeline"]
        if view is None and analysis.autorun:
            self.interface.stream("Failed to find an analysis that applies to this data")
        else:
            self._render_lumen(view, analysis=analysis, pipeline=pipeline, title=step_title)
            self.interface.stream(
                analysis.message or f"Successfully created view with {analysis_name} analysis.", user="Assistant"
            )
        return view


class ValidationAgent(Agent):
    """
    ValidationAgent focuses solely on validating whether the executed plan
    fully answered the user's original query. It identifies missing elements
    and suggests next steps when validation fails.
    """

    conditions = param.List(
        default=[
            "Use to validate whether executed plans fully answered user queries",
            "Use to identify missing elements from the original user request",
            "NOT for data analysis, pattern identification, or technical programming questions",
        ]
    )

    purpose = param.String(
        default="""
        Validates whether executed plans fully answered the user's original query.
        Identifies missing elements, assesses completeness, and suggests next steps
        when validation fails. Acts as a quality gate for plan execution."""
    )

    prompts = param.Dict(
        default={
            "main": {"template": PROMPTS_DIR / "ValidationAgent" / "main.jinja2", "response_model": QueryCompletionValidation, "tools": []},
        }
    )

    requires = param.List(default=[], readonly=True)

    provides = param.List(default=["validation_result"], readonly=True)

    async def respond(
        self,
        messages: list[Message],
        render_output: bool = False,
        step_title: str | None = None,
    ) -> Any:
        def on_click(event):
            if messages:
                user_messages = [msg for msg in reversed(messages) if msg.get("role") == "user"]
                original_query = user_messages[0].get("content", "").split("-- For context...")[0]
            suggestions_list = '\n- '.join(result.suggestions)
            self.interface.send(f"Follow these suggestions to fulfill the original intent {original_query}\n\n{suggestions_list}")

        executed_steps = None
        if "plan" in self._memory and hasattr(self._memory["plan"], "steps"):
            executed_steps = [f"{step.actor}: {step.instruction}" for step in self._memory["plan"].steps]

        system_prompt = await self._render_prompt("main", messages, executed_steps=executed_steps)
        model_spec = self.prompts["main"].get("llm_spec", self.llm_spec_key)

        result = await self.llm.invoke(
            messages=messages,
            system=system_prompt,
            model_spec=model_spec,
            response_model=QueryCompletionValidation,
        )

        self._memory["validation_result"] = result

        response_parts = []
        if result.correct:
            return result

        response_parts.append(f"**Query Validation: ✗ Incomplete** - {result.chain_of_thought}")
        if result.missing_elements:
            response_parts.append(f"**Missing Elements:** {', '.join(result.missing_elements)}")
        if result.suggestions:
            response_parts.append("**Suggested Next Steps:**")
            for i, suggestion in enumerate(result.suggestions, 1):
                response_parts.append(f"{i}. {suggestion}")

        button = Button(name="Rerun", on_click=on_click)
        footer_objects = [button]
        formatted_response = "\n\n".join(response_parts)
        self.interface.stream(formatted_response, user=self.user, max_width=self._max_width, footer_objects=footer_objects)
        return result
