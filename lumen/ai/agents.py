from __future__ import annotations

import asyncio
import json
import traceback

from copy import deepcopy
from functools import partial
from typing import Any, Literal

import pandas as pd
import panel as pn
import param
import yaml

from panel.chat import ChatInterface
from panel.layout import Column
from panel.viewable import Viewable, Viewer
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
    CheckContext, DbtslQueryParams, PartialBaseModel, QueryCompletionValidation, RetrySpec,
    Sql, VegaLiteSpec,
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

    debug = param.Boolean(default=False, doc="""
        Whether to enable verbose error reporting.""")

    llm = param.ClassSelector(class_=Llm, doc="""
        The LLM implementation to query.""")

    steps_layout = param.ClassSelector(default=None, class_=Column, allow_None=True, doc="""
        The layout progress updates will be streamed to.""")

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
        render_output: bool = False,
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
        render_output: bool
            Whether to render the output to the chat interface.
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
            "Use ONLY when user explicitly asks to upload or connect to NEW data sources",
            "Do NOT use if data sources already exist UNLESS user explicitly requests upload",
        ])

    table_upload_callbacks = param.Dict(default={}, doc="""
        Dictionary mapping from file extensions to callback function,
        e.g. {"hdf5": ...}. The callback function should accept the file bytes and
        table alias, add or modify the `source` in memory, and return a bool
        (True if the table was successfully uploaded).""")

    purpose = param.String(default="The SourceAgent allows a user to upload new datasets, tables, or documents.")

    requires = param.List(default=[], readonly=True)

    provides = param.List(default=["sources", "source", "document_sources"], readonly=True)

    _extensions = ("filedropper",)

    async def respond(
        self,
        messages: list[Message],
        render_output: bool = False,
        step_title: str | None = None,
    ) -> Any:
        source_controls = SourceControls(multiple=True, replace_controls=False, memory=self.memory, table_upload_callbacks=self.table_upload_callbacks)

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
            "Best for high-level information about data or general conversation",
            "Can be used to describe available tables",
            "Use for technical questions about programming, functions, methods, or libraries",
            "Use for 'how to' questions about specific functions or code usage",
            "Use for questions about software tools, APIs, or programming concepts",
            "Not useful for answering data specific questions that require querying tables",
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
        render_output: bool = False,
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
            "Use to explain trends, patterns, or relationships in query results",
            "NOT for initial data queries or table exploration",
            "NOT for technical programming questions",
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
        render_output: bool = False,
        step_title: str | None = None,
    ) -> Any:
        messages = await super().respond(messages, render_output, step_title)
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

    def _get_items(self) -> list[str]:
        """Return the list of items to display"""

    def _use_item(self, event):
        """Handle when a user clicks on an item in the list"""
        if event.column != "show":
            return
        item = self._df.iloc[event.row, 0]

        if self._message_format is None:
            raise ValueError("Subclass must define _message_format")

        message = self._message_format.format(item=repr(item))
        self.interface.send(message)

    async def respond(
        self,
        messages: list[Message],
        render_output: bool = False,
        step_title: str | None = None,
    ) -> Any:
        items = self._get_items()
        header_filters = False
        if len(items) > 10:
            column_filter = {"type": "input", "placeholder": f"Filter by {self._column_name.lower()}..."}
            header_filters = {self._column_name: column_filter}

        self._df = pd.DataFrame({self._column_name: items})
        item_list = pn.widgets.Tabulator(
            self._df,
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
        )

        item_list.on_click(self._use_item)
        self.interface.stream(
            pn.Column(
                "The available tables are listed below. Click on the eye icon to show the table contents.",
                item_list
            ), user="Assistant"
        )
        return item_list


class TableListAgent(ListAgent):
    """
    The TableListAgent lists all available tables and lets the user pick one.
    """

    conditions = param.List(default=[
        "Use when user explicitly asks to 'list tables', 'show available tables', or 'what tables do you have'",
        "Use for listing available data tables & datasets in source to the user, but not for planning",
        "NOT for showing actual table contents or data within tables",
        "NOT when user wants to query or analyze table data",
    ])

    not_with = param.List(default=["DbtslAgent", "SQLAgent"])

    purpose = param.String(default="""
        Displays a list of all available tables & datasets in memory.""")

    requires = param.List(default=["source"], readonly=True)

    _column_name = "Table"

    _message_format = "Show the table: {item}"

    @classmethod
    async def applies(cls, memory: _Memory) -> bool:
        # Check all sources' tables in memory and see if they have greater than 1 table
        tables_count = 0
        for source in memory.get("sources", []):
            tables_count += len(source.get_tables())
            if tables_count > 1:
                return True
        return False

    def _get_items(self) -> list[str]:
        tables = []
        if "closest_tables" in self._memory:
            tables = self._memory["closest_tables"]

        for source in self._memory["sources"]:
            tables += source.get_tables()

        # remove duplicates, keeps order in which they were added
        return pd.unique(pd.Series(tables)).tolist()


class DocumentListAgent(ListAgent):
    """
    The DocumentListAgent lists all available documents provided by the user.
    """

    conditions = param.List(
        default=[
            "Use when user explicitly asks to 'list documents', 'show available documents', or 'what documents do you have'",
            "Use when user wants to see all uploaded documents",
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

    def _get_items(self) -> list[str]:
        # extract the filename, following this pattern `Filename: 'filename'``
        return [doc["metadata"].get("filename", "untitled") for doc in self._memory.get("document_sources", [])]


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
        modified_messages = mutate_user_message(
            f"New feedback: {feedback!r}.\n\nThese were the previous instructions to use as reference:",
            deepcopy(messages), wrap='\n"""\n', suffix=False
        )
        original_lines = original_output.splitlines()
        with self.param.update(memory=memory):
            # TODO: only input the inner spec to retry
            numbered_text = "\n".join(f"{i:2d}: {line}" for i, line in enumerate(original_lines, 1))
            system = await self._render_prompt(
                "retry_output",
                messages=modified_messages,
                numbered_text=numbered_text,
                language=language,
            )
        retry_model = self._lookup_prompt_key("retry_output", "response_model")
        invoke_kwargs = dict(
            messages=modified_messages,
            system=system,
            response_model=retry_model,
            model_spec="reasoning",
        )
        result = await self.llm.invoke(**invoke_kwargs)
        return apply_changes(original_lines, result.lines_changes)

    def _render_lumen(
        self,
        component: Component,
        messages: list | None = None,
        render_output: bool = False,
        title: str | None = None,
        **kwargs,
    ):
        async def _retry_invoke(event: param.parameterized.Event):
            with out.param.update(loading=True):
                out.spec = await self._retry_output_by_line(event.new, messages, memory, out.spec, language=out.language)

        memory = self._memory
        retry_controls = RetryControls()
        out = self._output_type(component=component, footer=[retry_controls], render_output=render_output, title=title, **kwargs)
        retry_controls.param.watch(_retry_invoke, "reason")
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
            "Use for queries that need to access, filter, join, or aggregate existing data tables",
            "Use when user asks about data contained in tables (e.g., 'show me sales data', 'filter by date')",
            "Use for calculations that require data from tables (e.g., 'calculate average', 'sum by category')",
            "Use when user wants to display or examine table contents",
            "If sql_metaset is not in memory, use with IterativeTableLookup",
            "Commonly used with AnalystAgent to analyze query results",
            "For existing tables, only use if additional calculations are needed",
            "When reusing tables, reference by name rather than regenerating queries",
            "NOT for technical questions about programming, functions, or libraries",
            "NOT for questions that don't require data table access",
            "NOT useful if the user is using the same data for plotting",
        ]
    )

    exclusions = param.List(default=["dbtsl_metaset"])

    not_with = param.List(default=["DbtslAgent", "TableLookup", "TableListAgent"])

    purpose = param.String(
        default="""
        Handles the display of tables and the creation, modification, and execution
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
            "check_context": {
                "response_model": CheckContext,
                "template": PROMPTS_DIR / "SQLAgent" / "check_context.jinja2",
            }
        }
    )

    provides = param.List(default=["table", "sql", "pipeline", "data"], readonly=True)

    requires = param.List(default=["sources", "source", "sql_metaset"], readonly=True)

    max_discovery_iterations = param.Integer(default=5, doc="""
        Maximum number of discovery iterations before requiring a final answer.""")

    _extensions = (
        "codeeditor",
        "tabulator",
    )

    _output_type = SQLOutput

    def _update_spec(self, memory: _Memory, event: param.parameterized.Event):
        memory["sql"] = event.new

    async def _check_information_completeness(self, messages: list[Message], sql_plan_context: str) -> list[str]:
        """
        Check if we have complete information to answer the query or need discovery.
        Returns discovery_steps.
        """
        with self._add_step(title="Checking information completeness", steps_layout=self._steps_layout) as step:
            system_prompt = await self._render_prompt("check_context", messages, previous_sql_plan_results=sql_plan_context)
            model_spec = self.prompts["check_context"].get("llm_spec", self.llm_spec_key)

            response = self.llm.stream(
                messages,
                system=system_prompt,
                model_spec=model_spec,
                response_model=self._get_model("check_context"),
            )

            async for output in response:
                if output.information_completeness:
                    step.stream(f"**Information Assessment:** {output.information_completeness}", replace=True)

            if output.discovery_steps:
                instructions_text = "\n".join(f"- {instruction}" for instruction in output.discovery_steps)
                step.stream(f"\n\n**SQL Plan:**\n{instructions_text}")
            return output.discovery_steps

    async def _validate_sql_with_retry(self, sql_query: str, expr_slug: str, source, messages: list[Message], step, max_retries: int = 3) -> tuple[str, Any]:
        """
        Validate SQL query with retry logic. Streams directly to the provided step.

        Returns:
            tuple: (validated_sql_query, sql_expr_source)
        """
        for i in range(max_retries):
            try:
                sql_expr_source = source.create_sql_expr_source({expr_slug: sql_query})
                if i > 0:
                    step.stream("\n\n✅ SQL validation successful")
                return sql_query, sql_expr_source
            except Exception as e:
                if i == max_retries - 1:
                    step.stream(f"\n\n❌ SQL validation failed after {max_retries} attempts: {e}")
                    raise e

                step.stream(f"\n\n⚠️ SQL validation failed (attempt {i+1}/{max_retries}): {e}")
                sql_query = clean_sql(
                    await self._retry_output_by_line(str(e), messages, self._memory, sql_query, language="sql"),
                    source.dialect,
                )
                step.stream(f"\n\n**Retry {i+1}:** `{expr_slug}`\n```sql\n{sql_query}\n```")

    @retry_llm_output()
    async def _generate_and_validate_sql(
        self,
        messages: list[Message],
        dialect: str,
        source,
        step_number: int,
        current_step: str,
        sql_plan_context: str,
        step_title: str,
        errors: list | None = None,
    ) -> tuple[str, str, Any]:
        """
        Generate and validate SQL for any message set. Returns (sql_query, expr_slug, sql_expr_source).
        """
        with self._add_step(title=step_title, steps_layout=self._steps_layout) as step:
            system_prompt = await self._render_prompt(
                "main",
                messages,
                dialect=dialect,
                separator=SOURCE_TABLE_SEPARATOR,
                step_number=step_number,
                current_step=current_step,
                previous_sql_plan_results=sql_plan_context,
                errors=errors,
            )

            model_spec = self.prompts["main"].get("llm_spec", self.llm_spec_key)
            response = self.llm.stream(
                messages,
                system=system_prompt,
                model_spec=model_spec,
                response_model=self._get_model("main"),
            )

            output = None
            async for out in response:
                output = out
                if output.chain_of_thought:
                    step.stream(output.chain_of_thought, replace=True)

            if not output or not output.query:
                raise ValueError("No SQL query was generated.")

            sql_query = clean_sql(output.query, dialect)
            step.stream(f"\n\n`{output.expr_slug}`\n```sql\n{sql_query}\n```")

            # Validate SQL
            validated_sql, sql_expr_source = await self._validate_sql_with_retry(
                sql_query, output.expr_slug, source, messages, step
            )
            return validated_sql, output.expr_slug, sql_expr_source

    async def _execute_single_step(self, plan_step: str, sql_query: str, expr_slug: str, sql_expr_source, source, iteration: int, step_num: int, is_final_step: bool) -> dict:
        """
        Execute a single SQL plan step and return the result.
        """
        with self._add_step(title=f"Executing step {iteration}.{step_num}", steps_layout=self._steps_layout) as step:
            if is_final_step:
                # Final step: create full pipeline with transforms
                sql_transforms = [SQLLimit(limit=1_000_000, write=source.dialect, pretty=True, identify=False)]
                pipeline = await get_pipeline(source=sql_expr_source, table=expr_slug, sql_transforms=sql_transforms)
                df = await get_data(pipeline)

                result = {
                    "step": plan_step,
                    "sql_query": sql_query,
                    "expr_slug": expr_slug,
                    "sql_expr_source": sql_expr_source,
                    "pipeline": pipeline,
                    "data": df,
                    "has_data": len(df) > 0,
                    "summary": f"Final query returned {len(df)} rows",
                    "is_final": True
                }
            else:
                temp_source = source.create_sql_expr_source({expr_slug: sql_query}, materialize=False)
                df = await get_data(await get_pipeline(source=temp_source, table=expr_slug))

                summary = str(df.to_dict()) if len(df) < 100 else f"Found {len(df)} rows: {df.head(100).to_dict()}"
                result = {
                    "step": plan_step,
                    "sql_query": sql_query,
                    "summary": summary,
                    "has_data": len(df) > 0,
                    "is_final": False
                }
            step.stream(f"**Result:** {result['summary']}")
            return result

    async def _execute_sql_plan_steps(self, discovery_steps: list[str], iteration: int, dialect: str, source, sql_plan_context: str) -> list[dict]:
        """
        Execute all SQL plan steps sequentially, with the last step being the final answer.
        """
        results = []
        current_context = sql_plan_context

        for i, plan_step in enumerate(discovery_steps, 1):
            is_final_step = (i == len(discovery_steps))
            step_title = f"SQL Plan {iteration}.{i}"

            # Generate and validate SQL
            step_messages = [{"role": "user", "content": plan_step}]
            sql_query, expr_slug, sql_expr_source = await self._generate_and_validate_sql(
                step_messages, dialect, source, i, plan_step, current_context, step_title
            )

            # Execute and collect result
            try:
                result = await self._execute_single_step(
                    plan_step, sql_query, expr_slug, sql_expr_source, source, iteration, i, is_final_step
                )
                results.append(result)
                current_context += f"\n{plan_step!r} Result: {result['summary']}"

                # Stop if final step failed
                if is_final_step and not result['has_data'] and 'Failed' in result['summary']:
                    raise Exception(result['summary'])
            except Exception as e:
                if is_final_step:
                    raise e

        return results

    async def _execute_iteration(self, iteration: int, messages: list[Message], source, sql_plan_context: str) -> dict | None:
        """
        Execute a single iteration of SQL planning. Returns final result if successful, None otherwise.
        """
        # Get SQL plan
        discovery_steps = await self._check_information_completeness(
            messages, sql_plan_context
        )

        if not discovery_steps:
            return None

        # Execute all SQL plan steps
        dialect = self._memory["source"].dialect
        results = await self._execute_sql_plan_steps(
            discovery_steps, iteration, dialect, source, sql_plan_context
        )

        # The LAST result is always the final query/pipeline
        final_result = results[-1]
        if final_result["has_data"]:
            return final_result

        # Store results for context in next iteration
        self._iteration_results = results
        return None

    def _update_context_for_next_iteration(self, sql_plan_context: str, iteration: int) -> str:
        """Update context with results from current iteration."""
        if hasattr(self, '_iteration_results'):
            results_summary = "\n".join(
                f"{i+1}. {r['step']}\nResult: {r['summary']}"
                for i, r in enumerate(self._iteration_results)
            )
            sql_plan_context += f"\nIteration {iteration} Results:\n{results_summary}"
        return sql_plan_context

    async def _create_pipeline_from_result(self, result: dict, messages: list[Message], render_output: bool, step_title: str) -> Any:
        """Create the final pipeline from the result of the last SQL plan step."""
        # Update memory with final result
        self._memory["data"] = await describe_data(result["data"])
        self._memory["sql"] = result["sql_query"]
        self._memory["pipeline"] = result["pipeline"]
        self._memory["table"] = result["pipeline"].table

        # Update sources
        if isinstance(self._memory["sources"], dict):
            self._memory["sources"][result["expr_slug"]] = result["sql_expr_source"]
        else:
            self._memory["sources"].append(result["sql_expr_source"])
        self._memory["source"] = result["sql_expr_source"]

        # Render output
        self._render_lumen(
            result["pipeline"],
            spec=result["sql_query"],
            messages=messages,
            render_output=render_output,
            title=step_title
        )
        return result["pipeline"]

    async def respond(
        self,
        messages: list[Message],
        render_output: bool = False,
        step_title: str | None = None,
    ) -> Any:
        """
        Simplified SQL generation where the last SQL plan step is the final query.
        """
        # Setup sources and metadata
        sources = self._memory["sources"] if isinstance(self._memory["sources"], dict) else {source.name: source for source in self._memory["sources"]}
        vector_metaset = self._memory["sql_metaset"].vector_metaset
        selected_slugs = list(vector_metaset.selected_columns or vector_metaset.vector_metadata_map)
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

        # Execute SQL plan iterations
        sql_plan_context = ""
        for iteration in range(1, self.max_discovery_iterations + 1):
            final_result = await self._execute_iteration(
                iteration, messages, source, sql_plan_context
            )

            if final_result:
                return await self._create_pipeline_from_result(
                    final_result, messages, render_output, step_title
                )

            # Update context for next iteration if no final result
            sql_plan_context = self._update_context_for_next_iteration(
                sql_plan_context, iteration
            )
        return self._memory["pipeline"]


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
        Responsible for displaying tables to answer user queries about
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

    _extensions = (
        "codeeditor",
        "tabulator",
    )

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
            return sql_query, pipeline
        except Exception as e:
            report_error(e, step)
            raise e

    async def respond(
        self,
        messages: list[Message],
        render_output: bool = False,
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

        self._render_lumen(pipeline, spec=sql_query, messages=messages, render_output=render_output, title=step_title)
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
                        retry_step.stream(f"\n\n```json\n{view}\n```")
                    spec = yaml.safe_load(view)
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
        render_output: bool = False,
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
        self._render_lumen(view, messages=messages, render_output=render_output, title=step_title)
        return view


class hvPlotAgent(BaseViewAgent):
    conditions = param.List(
        default=[
            "Use for exploratory data analysis and interactive plots",
            "Use for quick, iterative data visualization during analysis",
            "Use when user requests plots or charts for data exploration",
            "Use for interactive widgets and dynamic filtering",
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
            "Use for explanatory, publication-ready visualizations",
            "Use when user specifically requests Vega-Lite charts",
            "Use for polished charts intended for presentation or sharing",
            "Use for final charts in reports, dashboards, or publications",
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

    def _add_geographic_items(self, vega_spec: dict, vega_spec_str: str):
        # standardize the vega spec, by migrating to layer
        if "layer" not in vega_spec:
            vega_spec["layer"] = [{"encoding": vega_spec.pop("encoding", None), "mark": vega_spec.pop("mark", None)}]

        # make it zoomable
        if "params" not in vega_spec:
            vega_spec["params"] = []

        # Get existing param names
        existing_param_names = {param.get("name") for param in vega_spec["params"] if isinstance(param, dict) and "name" in param}
        for p in VEGA_ZOOMABLE_MAP_ITEMS["params"]:
            if p.get("name") not in existing_param_names:
                vega_spec["params"].append(p)

        if "projection" not in vega_spec:
            vega_spec["projection"] = {"type": "mercator"}
        vega_spec["projection"].update(VEGA_ZOOMABLE_MAP_ITEMS["projection"])

        # Handle map projections and add geographic outlines
        # - albersUsa projection is incompatible with world map data
        # - Each map type needs appropriate boundary outlines
        has_world_map = "world-110m.json" in vega_spec_str
        uses_albers_usa = vega_spec["projection"]["type"] == "albersUsa"

        # If trying to use albersUsa with world map, switch to mercator projection
        if has_world_map and uses_albers_usa:
            vega_spec["projection"] = "mercator"  # cannot use albersUsa with world-110m
        # Add world map outlines if needed
        elif not has_world_map and not uses_albers_usa:
            vega_spec["layer"].append(VEGA_MAP_LAYER["world"])
        return vega_spec

    async def _ensure_columns_exists(self, vega_spec: dict):
        schema = await get_schema(self._memory["pipeline"])

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
                    if field not in schema and field.lower() not in schema and field.upper() not in schema:
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
            "Use when user requests custom analysis or advanced analytics",
            "Use when built-in SQL/visualization agents are insufficient",
            "Use when user wants to apply domain-specific analysis methods",
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
        render_output: bool = False,
        step_title: str | None = None,
        agents: list[Agent] | None = None,
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
                analysis_callable = analyses[analysis_name].instance(agents=agents)

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
            self._render_lumen(view, analysis=analysis, pipeline=pipeline, render_output=render_output, title=step_title)
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
            "Use when plan execution is complete but validation is needed",
            "Use to identify missing elements from the original user request",
            "NOT for data analysis or pattern identification",
            "NOT for technical programming questions",
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
            suggestions_list = '\n- '.join(result.suggestions)
            self.interface.send(f"Follow these suggestions: {suggestions_list}")

        if messages:
            user_messages = [msg for msg in reversed(messages) if msg.get("role") == "user"]
            original_query = user_messages[0].get("content", "")

        executed_steps = None
        if "plan" in self._memory and hasattr(self._memory["plan"], "steps"):
            executed_steps = [f"{step.actor}: {step.instruction}" for step in self._memory["plan"].steps]

        system_prompt = await self._render_prompt("main", messages, original_query=original_query, executed_steps=executed_steps)
        model_spec = self.prompts["main"].get("llm_spec", self.llm_spec_key)

        result = await self.llm.invoke(
            messages=messages,
            system=system_prompt,
            model_spec=model_spec,
            response_model=QueryCompletionValidation,
        )

        self._memory["validation_result"] = result

        response_parts = []
        if result.yes:
            return result


        response_parts.append(f"**Query Validation: ✗ Incomplete** - {result.chain_of_thought}")
        if result.missing_elements:
            response_parts.append(f"**Missing Elements:** {', '.join(result.missing_elements)}")
        if result.suggestions:
            response_parts.append("**Suggested Next Steps:**")
            for i, suggestion in enumerate(result.suggestions, 1):
                response_parts.append(f"{i}. {suggestion}")

        button = pn.widgets.Button(name="Rerun", on_click=on_click)
        footer_objects = [button]
        formatted_response = "\n\n".join(response_parts)
        self.interface.stream(formatted_response, user=self.user, max_width=self._max_width, footer_objects=footer_objects)
        return result
