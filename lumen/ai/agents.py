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
from panel_material_ui import Button
from pydantic import BaseModel, create_model
from pydantic.fields import FieldInfo

from ..base import Component
from ..dashboard import Config
from ..pipeline import Pipeline
from ..sources.base import BaseSQLSource, Source
from ..sources.duckdb import DuckDBSource
from ..state import state
from ..transforms.sql import SQLLimit
from ..views import (
    Panel, VegaLiteView, View, hvPlotUIView,
)
from .actor import ContextProvider
from .config import (
    PROMPTS_DIR, SOURCE_TABLE_SEPARATOR, VEGA_MAP_LAYER,
    VEGA_ZOOMABLE_MAP_ITEMS, RetriesExceededError,
)
from .controls import RetryControls, SourceControls
from .llm import Llm, Message
from .memory import _Memory
from .models import (
    DbtslQueryParams, NextStep, PartialBaseModel, QueryCompletionValidation,
    RetrySpec, Sql, SQLRoadmap, VegaLiteSpec,
)
from .schemas import get_metaset
from .services import DbtslMixin
from .tools import ToolUser
from .translate import param_to_pydantic
from .utils import (
    apply_changes, clean_sql, describe_data, get_data, get_pipeline,
    get_schema, load_json, log_debug, mutate_user_message, parse_table_slug,
    report_error, retry_llm_output, stream_details,
)
from .views import (
    AnalysisOutput, LumenOutput, SQLOutput, VegaLiteOutput,
)


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
            messages = self.interface.serialize()
            if messages and str(exception) in messages[-1]["content"]:
                return

            import traceback

            traceback.print_exception(exception)
            self.interface.send(f"Error cannot be resolved:\n\n{exception}", user="System", respond=False)

        if "interface" not in params:
            params["interface"] = ChatInterface(callback=self._interface_callback)
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
                {source.name: source}, [f"{source.name}{SOURCE_TABLE_SEPARATOR}{self._memory['table']}"],
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

        self._tabs = pn.Tabs(*tabs, sizing_mode="stretch_width")

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
    user = param.String(default="Lumen")

    prompts = param.Dict(
        default={
            "retry_output": {"response_model": RetrySpec, "template": PROMPTS_DIR / "LumenBaseAgent" / "retry_output.jinja2"},
        }
    )

    _output_type = LumenOutput

    _max_width = None

    def _update_spec(self, memory: _Memory, event: param.parameterized.Event):
        """
        Update the specification in memory.
        """

    async def _retry_output_by_line(
        self,
        feedback: str,
        messages: list[Message],
        memory: _Memory,
        original_output: str,
        language: str | None = None,
    ) -> str:
        """
        Retry the output by line, allowing the user to provide feedback on why the output was not satisfactory, or an error.
        """
        original_lines = original_output.splitlines()
        with self.param.update(memory=memory):
            # TODO: only input the inner spec to retry
            numbered_text = "\n".join(f"{i:2d}: {line}" for i, line in enumerate(original_lines, 1))
            system = await self._render_prompt(
                "retry_output",
                messages=messages,
                numbered_text=numbered_text,
                language=language,
                feedback=feedback,
            )
        retry_model = self._lookup_prompt_key("retry_output", "response_model")
        invoke_kwargs = dict(
            messages=messages,
            system=system,
            response_model=retry_model,
            model_spec="edit",
        )
        result = await self.llm.invoke(**invoke_kwargs)
        return apply_changes(original_lines, result.lines_changes)

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
        message_kwargs = dict(value=out, user=self.user)
        self.interface.stream(replace=True, max_width=self._max_width, **message_kwargs)


class SQLAgent(LumenBaseAgent):
    conditions = param.List(
        default=[
            "Use for displaying, examining, or querying data",
            "Use for calculations that require data (e.g., 'calculate average', 'sum by category')",
            "Commonly used with AnalystAgent to analyze query results",
            "NOT for non-data questions or technical programming help",
            "NOT useful if the user is using the same data for plotting",
            "If sql_metaset is not in memory, use with IterativeTableLookup",
        ]
    )

    exclusions = param.List(default=["dbtsl_metaset"])

    max_discovery_iterations = param.Integer(default=5, doc="""
        Maximum number of discovery iterations before requiring a final answer.""")

    not_with = param.List(default=["DbtslAgent", "TableLookup", "TableListAgent"])

    planning_enabled = param.Boolean(default=True, doc="""
        Whether to enable SQL planning mode. When False, only attempts oneshot SQL generation.""")

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
                "response_model": Sql,
                "template": PROMPTS_DIR / "SQLAgent" / "main.jinja2",
            },
            "plan_next_step": {
                "response_model": NextStep,
                "template": PROMPTS_DIR / "SQLAgent" / "plan_next_step.jinja2",
            },
            "generate_roadmap": {
                "response_model": SQLRoadmap,
                "template": PROMPTS_DIR / "SQLAgent" / "generate_roadmap.jinja2",
            },
            "retry_output": {"response_model": RetrySpec, "template": PROMPTS_DIR / "SQLAgent" / "retry_output.jinja2"},
        }
    )

    provides = param.List(default=["table", "sql", "pipeline", "data"], readonly=True)

    requires = param.List(default=["sources", "source", "sql_metaset"], readonly=True)

    max_steps = param.Integer(default=20, doc="""
        Maximum number of steps before requiring a final answer (safety limit).""")

    user = param.String(default="SQL")

    _extensions = ("codeeditor", "tabulator")

    _output_type = SQLOutput

    def __init__(self, **params):
        super().__init__(**params)
        # SQLAgent watches visible_slugs and filters its sql_metaset accordingly
        self._memory.on_change('visible_slugs', self._filter_sql_metaset_by_visibility)

    def _filter_sql_metaset_by_visibility(self, key, old_slugs, new_slugs):
        """
        Filter sql_metaset when visible_slugs changes.
        This ensures SQL operations only work with visible tables.
        """
        if new_slugs is None:
            new_slugs = set()

        sql_metaset = self._memory.get('sql_metaset')
        if not sql_metaset:
            return  # No sql_metaset to filter

        # Filter vector metadata to only include visible tables
        if sql_metaset.vector_metaset and sql_metaset.vector_metaset.vector_metadata_map:
            sql_metaset.vector_metaset.vector_metadata_map = {
                slug: metadata
                for slug, metadata in sql_metaset.vector_metaset.vector_metadata_map.items()
                if slug in new_slugs
            }

        # Filter SQL metadata to only include visible tables
        if sql_metaset.sql_metadata_map:
            sql_metaset.sql_metadata_map = {
                slug: metadata
                for slug, metadata in sql_metaset.sql_metadata_map.items()
                if slug in new_slugs
            }

        # Trigger memory update to notify other components that depend on sql_metaset
        self._memory.trigger('sql_metaset')

    def _update_spec(self, memory: _Memory, event: param.parameterized.Event):
        memory["sql"] = event.new

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
            separator=SOURCE_TABLE_SEPARATOR,
            step_number=step_number,
            is_final_step=is_final,
            current_step=messages[0]["content"],
            sql_query_history=sql_query_history,
            current_iteration=getattr(self, '_current_iteration', 1),
            sql_plan_context=sql_plan_context,
            errors=errors,
        )

        # Generate SQL
        model_spec = self.prompts["main"].get("llm_spec", self.llm_spec_key)
        output = await self.llm.invoke(
            messages,
            system=system_prompt,
            model_spec=model_spec,
            response_model=self._get_model("main"),
        )

        sql_queries = {}
        for query_obj in (output.queries if output else []):
            if query_obj.query and query_obj.expr_slug:
                sql_queries[query_obj.expr_slug.strip()] = query_obj.query.strip()

        if not sql_queries:
            raise ValueError("No SQL queries were generated.")

        return sql_queries

    async def _validate_sql(
        self, sql_query: str, expr_slug: str, dialect: str,
        source, messages: list[Message], step, max_retries: int = 2
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
                    feedback, messages, self._memory, sql_query, language=f"sql.{dialect}"
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
            if isinstance(self._memory["sources"], dict):
                self._memory["sources"][expr_slug] = sql_expr_source
            else:
                self._memory["sources"].append(sql_expr_source)
            self._memory.trigger("sources")

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
        self, results: dict, context_entries: list[dict],
        messages: list[Message], step_title: str | None,
        raise_if_empty: bool = False
    ) -> None:
        """Finalize execution for final step."""
        # Get first result (typically only one for final step)
        expr_slug, result = next(iter(results.items()))

        # Update memory
        pipeline = result["pipeline"]
        df = await get_data(pipeline)

        # If dataframe is empty, raise error to fall back to planning
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
            spec=result["sql"],
            messages=messages,
            title=step_title
        )

    @retry_llm_output(retries=3)
    async def _execute_sql_step(
        self,
        action_description: str,
        source: Source,
        step_number: int,
        is_final: bool = False,
        should_materialize: bool = True,
        context_entries: list[dict] | None = None,
        step_title: str | None = None,
        step=None,
        sql_plan_context: str | None = None,
        errors: list | None = None,
    ) -> dict[str, Any]:
        """Execute a single SQL generation step."""
        with self.interface.param.update(callback_exception="raise"):
            # Generate SQL queries
            messages = [{"role": "user", "content": action_description}]
            sql_queries = await self._generate_sql_queries(
                messages, source.dialect, step_number, is_final, context_entries, sql_plan_context, errors
            )

            # Validate and execute queries
            results = {}
            for expr_slug, sql_query in sql_queries.items():
                # Validate SQL
                validated_sql = await self._validate_sql(
                    sql_query, expr_slug, source.dialect, source, messages, step
                )

                # Execute and get results
                pipeline, sql_expr_source, summary = await self._execute_query(
                    source, expr_slug, validated_sql, is_final, should_materialize, step
                )

                results[expr_slug] = {
                    "sql": validated_sql,
                    "summary": summary,
                    "pipeline": pipeline,
                    "source": sql_expr_source
                }

            # Handle final step
            if is_final and context_entries is not None:
                await self._finalize_execution(
                    results, context_entries, messages, step_title
                )
                step.status = "success"
                return next(iter(results.values()))["pipeline"]

            step.status = "success"
            return results

    def _build_context_entry(
        self, step_number: int, next_step: NextStep, step_result: dict[str, Any]
    ) -> dict:
        """Build context entry from step results."""
        entry = {
            "step_number": step_number,
            "step_type": next_step.step_type,
            "action_description": next_step.action_description,
            "should_materialize": next_step.should_materialize,
            "queries": [],
            "materialized_tables": []
        }

        for expr_slug, result in step_result.items():
            if next_step.should_materialize:
                table_status = f"{result['source']}{SOURCE_TABLE_SEPARATOR}{expr_slug}"
            else:
                table_status = "<unmaterialized>"
            query_info = {
                "sql": result["sql"],
                "expr_slug": expr_slug,
                "table_status": table_status,
                "summary": result["summary"],
            }
            entry["queries"].append(query_info)

            if next_step.should_materialize:
                entry["materialized_tables"].append(expr_slug)

        return entry

    def _setup_source(self, sources: dict) -> tuple[dict, Any]:
        """Setup the main source and handle multiple sources if needed."""
        vector_metaset = self._memory["sql_metaset"].vector_metaset
        selected_slugs = list(vector_metaset.vector_metadata_map) or self._memory["tables_metadata"].keys()

        # Filter selected_slugs to only include visible tables
        visible_slugs = self._memory.get('visible_slugs', set())
        if visible_slugs:
            selected_slugs = [slug for slug in selected_slugs if slug in visible_slugs]

        tables_to_source = {parse_table_slug(table_slug, sources)[1]: parse_table_slug(table_slug, sources)[0] for table_slug in selected_slugs}
        source = next(iter(tables_to_source.values()))

        # Handle multiple sources if needed
        if len(set(tables_to_source.values())) > 1:
            mirrors = {}
            for a_table, a_source in tables_to_source.items():
                if not any(a_table.rstrip(")").rstrip("'").rstrip('"').endswith(ext)
                           for ext in [".csv", ".parquet", ".parq", ".json", ".xlsx"]):
                    renamed_table = a_table.replace(".", "_")
                else:
                    renamed_table = a_table
                if SOURCE_TABLE_SEPARATOR in renamed_table:
                    _, renamed_table = renamed_table.split(SOURCE_TABLE_SEPARATOR, maxsplit=1)
                mirrors[renamed_table] = (a_source, renamed_table)
            source = DuckDBSource(uri=":memory:", mirrors=mirrors)

        return tables_to_source, source

    def _update_context_for_next_iteration(self, sql_plan_context: str, iteration: int) -> str:
        """Update context with results from current iteration."""
        if hasattr(self, '_iteration_results'):
            results_summary = "\n".join(
                f"{i+1}. {r['step']}\nResult: {self._truncate_summary(r['summary'])}"
                for i, r in enumerate(self._iteration_results)
            )
            sql_plan_context += f"\nIteration {iteration} Results:\n{results_summary}"
        return sql_plan_context[-10000:]

    def _format_context(self, context_entries: list[dict]) -> str:
        """Format the enhanced context entries into a readable string."""
        if not context_entries:
            return ""

        # Build materialized tables section
        materialized_tables = []
        for entry in context_entries:
            materialized_tables.extend(entry.get("materialized_tables", []))

        # Format materialized tables
        base_context = ""
        if materialized_tables:
            recent_tables = materialized_tables[-5:]  # Keep only recent 5 tables
            base_context = "**Available Materialized Data (for reuse):**\n"
            for table in recent_tables:
                base_context += f"- `{table}`\n"
            if len(materialized_tables) > 5:
                base_context += f"- ... and {len(materialized_tables) - 5} more\n"
            base_context += "\n"

        # Build recent steps section
        steps_context = "**Recent Steps:**\n"
        for entry in context_entries:
            steps_context += f"**Step {entry['step_number']}:** {entry['action_description']}\n"

            # Add query information
            if entry['queries']:
                for query in entry['queries']:
                    steps_context += f"\n`{query['expr_slug']}:`\n```sql\n{query['sql']}\n```\n"
                    steps_context += f"Result: {query['summary']}\n"
                    if query['table_status'] != "<unmaterialized>":
                        steps_context += f"`Can be referenced as {query['table_status']}`\n"
            steps_context += "\n"

        return base_context + steps_context

    async def _plan_next_step(self, messages: list[Message], sql_plan_context: str, total_steps: int) -> NextStep:
        """Plan the next step based on current context."""
        with self._add_step(title=f"Planning step {total_steps}", steps_layout=self._steps_layout) as step:
            system_prompt = await self._render_prompt(
                "plan_next_step",
                messages,
                sql_plan_context=sql_plan_context,
                total_steps=total_steps,
            )

            model_spec = self.prompts.get("plan_next_step", {}).get("llm_spec", self.llm_spec_key)
            response = self.llm.stream(
                messages,
                system=system_prompt,
                model_spec=model_spec,
                response_model=NextStep,
            )

            output = None
            async for output in response:
                message = f"**Step Validation:**\n{output.pre_step_validation}\n\n"
                message += f"**Reasoning:** {output.reasoning}\n\n"
                message += f"**Next Action ({output.step_type}):** {output.action_description}"
                if output.is_final_answer:
                    message += "\n\n**Ready to provide final answer!**"
                step.stream(message, replace=True)

            return output

    async def _generate_roadmap(self, messages: list[Message]) -> SQLRoadmap:
        """Generate initial execution roadmap."""
        with self._add_step(title="Generating execution roadmap", steps_layout=self._steps_layout) as step:
            # Analyze query for characteristics
            query_content = messages[0].get("content", "") if messages else ""
            query_type = "complex" if any(word in query_content.lower() for word in ["compare", "vs", "between", "join"]) else "simple"
            requires_joins = "join" in query_content.lower() or "vs" in query_content.lower()
            has_temporal = any(word in query_content.lower() for word in ["date", "time", "year", "month", "season", "period"])

            system_prompt = await self._render_prompt(
                "generate_roadmap",
                messages,
                query_type=query_type,
                requires_joins=requires_joins,
                has_temporal=has_temporal,
            )

            model_spec = self.prompts.get("generate_roadmap", {}).get("llm_spec", self.llm_spec_key)
            response = self.llm.stream(
                messages,
                system=system_prompt,
                model_spec=model_spec,
                response_model=self._get_model("generate_roadmap"),
            )

            roadmap = None
            async for output in response:
                # Build summary progressively
                summary = f"**Execution Plan ({output.estimated_steps} steps)**\n\n"

                if output.discovery_steps:
                    summary += "**Discovery Phase:**\n"
                    for i, step_desc in enumerate(output.discovery_steps, 1):
                        summary += f"{i}. {step_desc}\n"

                if output.validation_checks:
                    summary += "\n**Validation Checks:**\n"
                    for check in output.validation_checks:
                        summary += f"- {check}\n"

                if output.potential_issues:
                    summary += "\n**Potential Issues:**\n"
                    for issue in output.potential_issues:
                        summary += f"⚠️ {issue}\n"

                step.stream(summary, replace=True)
                roadmap = output

            return roadmap

    def _format_roadmap(self, roadmap: SQLRoadmap) -> str:
        """Format roadmap for context."""
        context = "**Execution Roadmap:**\n"
        context += f"Estimated Steps: {roadmap.estimated_steps}\n\n"

        if roadmap.discovery_steps:
            context += "Discovery Steps:\n"
            for step in roadmap.discovery_steps:
                context += f"- {step}\n"

        if roadmap.validation_checks:
            context += "\nValidation Checks:\n"
            for check in roadmap.validation_checks:
                context += f"- {check}\n"

        if roadmap.join_strategy:
            context += f"\nJoin Strategy: {roadmap.join_strategy}\n"

        return context

    async def _attempt_oneshot_sql(
        self,
        messages: list[Message],
        source: Source,
        step_title: str | None = None,
    ) -> Pipeline | None:
        """
        Attempt one-shot SQL generation without planning context.
        Returns Pipeline if successful, None if fallback to planning is needed.
        """
        with self._add_step(
            title="Attempting one-shot SQL generation...",
            steps_layout=self._steps_layout,
            status="running"
        ) as step:
            # Generate SQL without planning context
            sql_queries = await self._generate_sql_queries(
                messages, source.dialect, step_number=1,
                is_final=True, context_entries=None
            )

            # Validate and execute the first query
            expr_slug, sql_query = next(iter(sql_queries.items()))
            validated_sql = await self._validate_sql(
                sql_query, expr_slug, source.dialect, source, messages, step
            )

            # Execute and get results
            pipeline, sql_expr_source, summary = await self._execute_query(
                source, expr_slug, validated_sql, is_final=True,
                should_materialize=True, step=step
            )

            # Use existing finalization logic
            results = {expr_slug: {
                "sql": validated_sql,
                "summary": summary,
                "pipeline": pipeline,
                "source": sql_expr_source
            }}

            await self._finalize_execution(
                results, [], messages, step_title, raise_if_empty=True
            )

            step.status = "success"
            step.success_title = "One-shot SQL generation successful"
            return pipeline

    async def _execute_planning_mode(
        self,
        messages: list[Message],
        source: Source,
        step_title: str | None = None,
    ) -> Pipeline:
        """
        Execute the planning mode with roadmap generation and iterative refinement.
        """
        # Generate initial roadmap for planning mode
        roadmap = await self._generate_roadmap(messages)

        # Initialize context tracking
        context_entries = []
        total_steps = 0
        roadmap_context = self._format_roadmap(roadmap)

        # Main execution loop (existing planning logic)
        while total_steps < self.max_steps:
            # Build context for planning
            sql_plan_context = self._format_context(context_entries)
            # Include roadmap context
            if roadmap_context and total_steps == 0:
                sql_plan_context = roadmap_context + "\n\n" + sql_plan_context
            total_steps += 1

            # Plan next step
            next_step = await self._plan_next_step(messages, sql_plan_context, total_steps)

            with self._add_step(
                title=step_title or f"Step {total_steps}: Generating SQL",
                steps_layout=self._steps_layout,
                status="running"
            ) as step:
                result = await self._execute_sql_step(
                    next_step.action_description,
                    source,
                    total_steps,
                    is_final=next_step.is_final_answer,
                    context_entries=context_entries,
                    sql_plan_context=sql_plan_context,
                    should_materialize=next_step.is_final_answer and next_step.should_materialize,
                    step_title=step_title,
                    step=step
                )
            if next_step.is_final_answer:
                return result

            # Update context
            context_entry = self._build_context_entry(
                total_steps, next_step, result
            )
            context_entries.append(context_entry)

            # Keep context window manageable
            if len(context_entries) > 10:
                context_entries = context_entries[-10:]

        raise ValueError(f"Exceeded maximum steps ({self.max_steps}) without reaching a final answer.")

    async def respond(
        self,
        messages: list[Message],
        step_title: str | None = None,
    ) -> Any:
        """Execute SQL generation with one-shot attempt first, then iterative refinement if needed."""
        # Setup sources
        sources = self._memory["sources"] if isinstance(self._memory["sources"], dict) else {source.name: source for source in self._memory["sources"]}
        tables_to_source, source = self._setup_source(sources)

        try:
            # Try one-shot approach first
            pipeline = await self._attempt_oneshot_sql(messages, source, step_title)
        except Exception as e:
            if not self.planning_enabled:
                # If planning is disabled, re-raise the error instead of falling back
                raise e
            # Fall back to planning mode if enabled
            messages = mutate_user_message(str(e), messages)
            pipeline = await self._execute_planning_mode(messages, source, step_title)
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

    _output_type = SQLOutput

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
            client = self._get_dbtsl_client()
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
            sql_metaset = await get_metaset({sql_expr_source.name: sql_expr_source}, [expr_slug])
            vector_metaset = sql_metaset.vector_metaset

            # Update memory
            self._memory["data"] = await describe_data(df)
            self._memory["sources"].append(sql_expr_source)
            self._memory["source"] = sql_expr_source
            self._memory["pipeline"] = pipeline
            self._memory["table"] = pipeline.table
            self._memory["dbtsl_vector_metaset"] = vector_metaset
            self._memory["dbtsl_sql_metaset"] = sql_metaset
            self._memory.trigger("sources")
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

        self._render_lumen(pipeline, spec=sql_query, messages=messages, title=step_title)
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
                last_output = load_json(self._last_output["json_spec"])
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
                    error = str(e)
                    traceback.print_exception(e)
                    context = f"```\n{yaml.safe_dump(load_json(self._last_output['json_spec']))}\n```"
                    report_error(e, step, language="json", context=context, status="failed")
                    with self._add_step(
                        title="Re-attempted view generation",
                        steps_layout=self._steps_layout,
                    ) as retry_step:
                        view = await self._retry_output_by_line(e, messages, self._memory, yaml.safe_dump(spec), language="")
                        if "json_spec: " in view:
                            view = view.split("json_spec: ")[-1].rstrip('"').rstrip("'")
                        spec = json.loads(view)
                        retry_step.stream(f"\n\n```json\n{spec}\n```")
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
            "main": {"response_model": VegaLiteSpec, "template": PROMPTS_DIR / "VegaLiteAgent" / "main.jinja2"},
            "retry_output": {"response_model": RetrySpec, "template": PROMPTS_DIR / "VegaLiteAgent" / "retry_output.jinja2"},
        }
    )

    view_type = VegaLiteView

    _extensions = ("vega",)

    _output_type = VegaLiteOutput

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
        schema = await get_schema(self._memory["pipeline"])

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
                analysis_callable = analyses[analysis_name].instance(agents=self.agents, memory=self._memory)

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
