from __future__ import annotations

import asyncio
import json
import re

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
from ..sources.base import BaseSQLSource
from ..sources.duckdb import DuckDBSource
from ..state import state
from ..transforms.sql import SQLLimit
from ..views import (
    Panel, VegaLiteView, View, hvPlotUIView,
)
from .actor import Actor, ContextProvider
from .config import PROMPTS_DIR, SOURCE_TABLE_SEPARATOR
from .controls import RetryControls, SourceControls
from .llm import Llm, Message
from .memory import _Memory
from .models import (
    JoinRequired, RetrySpec, Sql, TableJoins, VegaLiteSpec, make_table_model,
)
from .tools import DocumentLookup, TableLookup
from .translate import param_to_pydantic
from .utils import (
    clean_sql, describe_data, gather_table_sources, get_data, get_pipeline,
    get_schema, log_debug, mutate_user_message, report_error, retry_llm_output,
)
from .views import AnalysisOutput, LumenOutput, SQLOutput


class Agent(Viewer, Actor, ContextProvider):
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

    interface = param.ClassSelector(class_=ChatInterface, doc="""
        The ChatInterface to report progress to.""")

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
            traceback.print_exc()
            self.interface.send(
                f"Error cannot be resolved:\n\n{exception}",
                user="System",
                respond=False
            )

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
        model_spec = self.prompts["main"].get("llm_spec", "default")
        async for output_chunk in self.llm.stream(
            messages, system=system_prompt, model_spec=model_spec, field="output"
        ):
            message = self.interface.stream(
                output_chunk, replace=True, message=message, user=self.user, max_width=self._max_width
            )
        return message

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

    purpose = param.String(default="""
        The SourceAgent allows a user to upload unavailable, new datasets, tables, or documents.

        Only use this if the user is requesting to add a completely new table.
        Not useful for answering what's available or loading existing datasets.
        """)

    requires = param.List(default=[], readonly=True)

    provides = param.List(default=["source"], readonly=True)

    on_init = param.Boolean(default=True)

    _extensions = ('filedropper',)

    async def respond(
        self,
        messages: list[Message],
        render_output: bool = False,
        step_title: str | None = None,
    ) -> Any:
        source_controls = SourceControls(
            multiple=True, replace_controls=True, select_existing=False, memory=self.memory
        )
        self.interface.send(source_controls, respond=False, user="SourceAgent")
        while not source_controls._add_button.clicks > 0:
            await asyncio.sleep(0.05)
            if source_controls._cancel_button.clicks > 0:
                self.interface.undo()
                return None
        return source_controls


class ChatAgent(Agent):
    """
    ChatAgent provides general information about available data
    and other topics  to the user.
    """

    purpose = param.String(default="""
        Chats and provides info about high level data related topics,
        e.g. the columns of the data or statistics about the data,
        and continuing the conversation.

        Is capable of providing suggestions to get started or comment on interesting tidbits.
        It can talk about the data, if available.

        Usually not used concurrently with SQLAgent, unlike AnalystAgent.
        Can be used concurrently with TableListAgent to describe available tables
        and potential ideas for analysis.""")

    prompts = param.Dict(
        default={
            "main": {
                "template": PROMPTS_DIR / "ChatAgent" / "main.jinja2",
                "tools": [TableLookup, DocumentLookup]
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
        table = self._memory.get("table")
        if table:
            schema = await get_schema(self._memory["source"], table, include_count=True, limit=1000)
            context["table"] = table
            context["schema"] = schema
        elif "closest_tables" in self._memory:
            schemas = [
                await get_schema(self._memory["source"], table, include_count=True, limit=1000)
                for table in self._memory["closest_tables"]
            ]
            context["tables_schemas"] = list(zip(self._memory["closest_tables"], schemas))
        system_prompt = await self._render_prompt("main", messages, **context)
        return await self._stream(messages, system_prompt)


class AnalystAgent(ChatAgent):

    purpose = param.String(default="""
        Responsible for analyzing results from SQLAgent and
        providing clear, concise, and actionable insights.
        Focuses on breaking down complex data findings into understandable points for
        high-level decision-making. Emphasizes detailed interpretation, trends, and
        relationships within the data, while avoiding general overviews or
        superficial descriptions.""")

    requires = param.List(default=["source", "pipeline"], readonly=True)

    prompts = param.Dict(
        default={
            "main": {"template": PROMPTS_DIR / "AnalystAgent" / "main.jinja2"},
        }
    )


class TableListAgent(Agent):

    purpose = param.String(default="""
        Renders a list of all availables tables to the user and lets the user
        pick one, do not use if user has already picked a table.
        Not useful for gathering information about the tables.
        """)

    prompts = param.Dict(
        default={
            "main": {"template": PROMPTS_DIR / "TableListAgent" / "main.jinja2"},
        }
    )

    requires = param.List(default=["source"], readonly=True)

    _extensions = ('tabulator',)

    @classmethod
    async def applies(cls, memory: _Memory) -> bool:
        source = memory.get("source")
        if not source:
            return True  # source not loaded yet; always apply
        return len(source.get_tables()) > 1

    def _use_table(self, event):
        if event.column != "show":
            return
        table = self._df.iloc[event.row, 0]
        self.interface.send(f"Show the table: {table!r}")

    async def respond(
        self,
        messages: list[Message],
        render_output: bool = False,
        step_title: str | None = None,
    ) -> Any:
        tables = []
        for source in self._memory["sources"]:
            tables += source.get_tables()
        self._df = pd.DataFrame({"Table": tables})
        table_list = pn.widgets.Tabulator(
            self._df,
            buttons={'show': '<i class="fa fa-eye"></i>'},
            show_index=False,
            min_height=150,
            min_width=350,
            widths={'Table': '90%'},
            disabled=True,
            page_size=10,
            header_filters=True
        )
        table_list.on_click(self._use_table)
        self.interface.stream(table_list, user="Lumen")
        self._memory["closest_tables"] = tables[:5]
        return table_list


class LumenBaseAgent(Agent):

    user = param.String(default="Lumen")

    prompts = param.Dict(
        default={
            "retry_output": {
                "response_model": RetrySpec,
                "template": PROMPTS_DIR / "LumenBaseAgent" / "retry_output.jinja2"
            },
        }
    )

    _output_type = LumenOutput

    _max_width = None

    def _update_spec(self, memory: _Memory, event: param.parameterized.Event):
        """
        Update the specification in memory.
        """

    def _render_lumen(
        self,
        component: Component,
        message: pn.chat.ChatMessage = None,
        messages: list | None = None,
        render_output: bool = False,
        title: str | None = None,
        **kwargs
    ):
        memory = self._memory

        async def retry_invoke(event: param.parameterized.Event):
            reason = event.new
            modified_messages = mutate_user_message(
                f"New feedback: {reason!r}.\n\nThese were the previous instructions to use as reference:",
                deepcopy(messages), wrap='\n"""\n', suffix=False
            )
            with self.param.update(memory=memory):
                system = await self._render_prompt(
                    "retry_output", messages=modified_messages, spec=out.spec,
                    language=out.language
                )
            retry_model = self._lookup_prompt_key("retry_output", "response_model")
            with out.param.update(loading=True):
                result = await self.llm.invoke(
                    modified_messages, system=system, response_model=retry_model, model_spec="reasoning",
                )
                if "```" in result:
                    spec = re.search(r'(?s)```(\w+)?\s*(.*?)```', result.corrected_spec, re.DOTALL)
                else:
                    spec = result.corrected_spec
                out.spec = spec

        retry_controls = RetryControls()
        retry_controls.param.watch(retry_invoke, "reason")
        out = self._output_type(
            component=component,
            footer=[retry_controls],
            render_output=render_output,
            title=title,
            **kwargs
        )
        out.param.watch(partial(self._update_spec, self._memory), 'spec')
        if 'outputs' in self._memory:
            # We have to create a new list to trigger an event
            # since inplace updates will not trigger updates
            # and won't allow diffing between old and new values
            self._memory['outputs'] = self._memory['outputs']+[out]
        message_kwargs = dict(value=out, user=self.user)
        self.interface.stream(
            message=message, **message_kwargs, replace=True, max_width=self._max_width
        )


class SQLAgent(LumenBaseAgent):

    purpose = param.String(default="""
        Responsible for generating, modifying and executing SQL queries to
        answer user queries about the data, such querying subsets of the
        data, aggregating the data and calculating results. If the current
        table does not contain all the available data the SQL agent is
        also capable of joining it with other tables. Will generate and
        execute a query in a single step.""")

    prompts = param.Dict(
        default={
            "main": {
                "response_model": Sql,
                "template": PROMPTS_DIR / "SQLAgent" / "main.jinja2",
            },
            "select_table": {
                "response_model": make_table_model,
                "template": PROMPTS_DIR / "SQLAgent" / "select_table.jinja2",
                "tools": [TableLookup]
            },
            "require_joins": {
                "response_model": JoinRequired,
                "template": PROMPTS_DIR / "SQLAgent" / "require_joins.jinja2"
            },
            "find_joins": {
                "response_model": TableJoins,
                "template": PROMPTS_DIR / "SQLAgent" / "find_joins.jinja2"
            },
        }
    )

    provides = param.List(default=["table", "sql", "pipeline", "data", "tables_sql_schemas"], readonly=True)

    requires = param.List(default=["source"], readonly=True)

    _extensions = ('codeeditor', 'tabulator',)

    _output_type = SQLOutput

    async def _select_relevant_table(self, messages: list[Message]) -> tuple[str, BaseSQLSource]:
        """Select the most relevant table based on the user query."""
        sources = self._memory["sources"]
        tables_to_source, tables_schema_str = await gather_table_sources(sources)
        tables = tuple(tables_to_source)
        if messages and messages[-1]["content"].startswith("Show the table: '"):
            # Handle the case where explicitly requested a table
            table = re.search(r"Show the table: '([^']+)'", messages[-1]["content"]).group(1)
        elif len(tables) == 1:
            table = tables[0]
        else:
            with self.interface.add_step(title="Choosing the most relevant table...", steps_layout=self._steps_layout) as step:
                if len(tables) > 1:
                    system_prompt = await self._render_prompt(
                        "select_table", messages, tables_schema_str=tables_schema_str
                    )
                    if "closest_tables" in self._memory:
                        tables = self._memory["closest_tables"]
                    model_spec = self.prompts["select_table"].get("llm_spec", "default")
                    table_model = self._get_model("select_table", tables=tables)
                    result = await self.llm.invoke(
                        messages,
                        system=system_prompt,
                        model_spec=model_spec,
                        response_model=table_model,
                        allow_partial=False,
                        max_retries=3,
                    )
                    table = result.relevant_table
                    step.stream(f"{result.chain_of_thought}\n\nSelected table: {table}")
                else:
                    table = tables[0]
                    step.stream(f"Selected table: {table}")

        if table in tables_to_source:
            source = tables_to_source[table]
        else:
            sources = [src for src in sources if table in src]
            source = sources[0] if sources else self._memory["source"]

        return table, source

    @retry_llm_output()
    async def _create_valid_sql(
        self,
        messages: list[Message],
        system: str,
        tables_to_source,
        title: str,
        errors=None
    ):
        if errors:
            last_query = self.interface.serialize()[-1]["content"]
            sql_code_match = re.search(r'(?s)```sql\s*(.*?)```', last_query, re.DOTALL)
            if sql_code_match:
                last_query = sql_code_match.group(1)
            errors = '\n'.join(errors)
            system += (
                "Your last query did not work as intended, expertly revise these errors:\n"
                f"```python\n{errors}\n```\n\n"
                "If the error is `syntax error at or near \")\"`, double check you used "
                "table names verbatim, i.e. `read_parquet('table_name.parq')` instead of `table_name`."
            )

        with self.interface.add_step(title=title or "SQL query", steps_layout=self._steps_layout) as step:
            model_spec = self.prompts["main"].get("llm_spec", "default")
            response = self.llm.stream(messages, system=system, model_spec=model_spec, response_model=self._get_model("main"))
            sql_query = None
            try:
                async for output in response:
                    step_message = output.chain_of_thought
                    if output.query:
                        sql_query = clean_sql(output.query)
                        step_message += f"\n```sql\n{sql_query}\n```"
                    step.stream(step_message, replace=True)
            except asyncio.CancelledError as e:
                step.failed_title = "Cancelled SQL query generation"
                raise e

        if step.failed_title and step.failed_title.startswith("Cancelled"):
            raise asyncio.CancelledError()
        elif not sql_query:
            raise ValueError("No SQL query was generated.")

        sources = set(tables_to_source.values())
        if len(sources) > 1:
            mirrors = {}
            for a_table, a_source in tables_to_source.items():
                if not any(a_table.rstrip(")").rstrip("'").rstrip('"').endswith(ext)
                           for ext in [".csv", ".parquet", ".parq", ".json", ".xlsx"]):
                    renamed_table = a_table.replace(".", "_")
                    sql_query = sql_query.replace(a_table, renamed_table)
                else:
                    renamed_table = a_table
                if SOURCE_TABLE_SEPARATOR in renamed_table:
                    # Remove source prefixes from table names
                    renamed_table = re.sub(rf".*?{SOURCE_TABLE_SEPARATOR}", "", renamed_table)
                sql_query = sql_query.replace(a_table, renamed_table)
                mirrors[renamed_table] = (a_source, a_table)
            source = DuckDBSource(uri=":memory:", mirrors=mirrors)
        else:
            source = next(iter(sources))

        # check whether the SQL query is valid
        expr_slug = output.expr_slug
        try:
            sql_expr_source = source.create_sql_expr_source({expr_slug: sql_query})
            # Get validated query
            sql_query = sql_expr_source.tables[expr_slug]
            sql_transforms = [SQLLimit(limit=1_000_000)]
            pipeline = await get_pipeline(
                source=sql_expr_source, table=expr_slug, sql_transforms=sql_transforms
            )
        except Exception as e:
            report_error(e, step)
            raise e

        try:
            df = await get_data(pipeline)
        except Exception as e:
            source._connection.execute(f'DROP TABLE IF EXISTS "{expr_slug}"')
            report_error(e, step)
            raise e

        self._memory["data"] = await describe_data(df)
        self._memory["sources"].append(sql_expr_source)
        self._memory["source"] = sql_expr_source
        self._memory["pipeline"] = pipeline
        self._memory["table"] = pipeline.table
        self._memory["sql"] = sql_query
        return sql_query

    async def _check_requires_joins(
        self,
        messages: list[Message],
        schema,
        table: str
    ):
        requires_joins = None
        with self.interface.add_step(title="Checking if join is required", steps_layout=self._steps_layout) as step:
            join_prompt = await self._render_prompt(
                "require_joins",
                messages,
                schema=yaml.dump(schema),
                table=table
            )
            model_spec = self.prompts["require_joins"].get("llm_spec", "default")
            response = self.llm.stream(
                messages[-1:],
                system=join_prompt,
                model_spec=model_spec,
                response_model=self._get_model("require_joins"),
            )
            try:
                async for output in response:
                    step.stream(output.chain_of_thought, replace=True)
            except asyncio.CancelledError as e:
                step.failed_title = "SQLAgent cancelled while determining required joins"
                raise e
            requires_joins = output.requires_joins
            step.success_title = 'Query requires join' if requires_joins else 'No join required'
        return requires_joins

    async def find_join_tables(self, messages: list[Message]):
        multi_source = len(self._memory['sources']) > 1
        if multi_source:
            tables = [
                f"{SOURCE_TABLE_SEPARATOR}{a_source}{SOURCE_TABLE_SEPARATOR}{a_table}" for a_source in self._memory["sources"]
                for a_table in a_source.get_tables()
            ]
        else:
            tables = self._memory['source'].get_tables()

        find_joins_prompt = await self._render_prompt("find_joins", messages, tables=tables)
        model_spec = self.prompts["find_joins"].get("llm_spec", "default")
        with self.interface.add_step(title="Determining tables required for join", steps_layout=self._steps_layout) as step:
            output = await self.llm.invoke(
                messages,
                system=find_joins_prompt,
                model_spec=model_spec,
                response_model=self._get_model("find_joins"),
            )
            tables_to_join = output.tables_to_join
            step.stream(
                f'{output.chain_of_thought}\nJoin requires following tables: {tables_to_join}',
                replace=True
            )
            step.success_title = 'Found tables required for join'

        tables_to_source = {}
        for source_table in tables_to_join:
            sources = self._memory["sources"]
            if multi_source:
                try:
                    _, a_source_name, a_table = source_table.split(SOURCE_TABLE_SEPARATOR, maxsplit=2)
                except ValueError:
                    a_source_name, a_table = source_table.split(SOURCE_TABLE_SEPARATOR, maxsplit=1)
                for source in sources:
                    if source.name == a_source_name:
                        a_source = source
                        break
                if a_table in tables_to_source:
                    a_table = f"{SOURCE_TABLE_SEPARATOR}{a_source_name}{SOURCE_TABLE_SEPARATOR}{a_table}"
            else:
                a_source = next(iter(sources))
                a_table = source_table

            tables_to_source[a_table] = a_source
        return tables_to_source

    def _update_spec(self, memory: _Memory, event: param.parameterized.Event):
        memory['sql'] = event.new

    async def respond(
        self,
        messages: list[Message],
        render_output: bool = False,
        step_title: str | None = None,
    ) -> Any:
        """
        Steps:
        1. Retrieve the current source and table from memory.
        2. If the source lacks a `get_sql_expr` method, return `None`.
        3. Fetch the schema for the current table using `get_schema` without min/max values.
        4. Determine if a join is required by calling `_check_requires_joins`.
        5. If required, find additional tables via `find_join_tables`; otherwise, use the current source and table.
        6. For each source and table, get the schema and SQL expression, storing them in `table_schemas`.
        7. Render the SQL prompt using the table schemas, dialect, join status, and table.
        8. If a join is required, remove source/table prefixes from the last message.
        9. Construct the SQL query with `_create_valid_sql`.
        """
        table, source = await self._select_relevant_table(messages)
        if not hasattr(source, "get_sql_expr"):
            return None

        # include min max for more context for data cleaning
        schema = await get_schema(source, table, include_min_max=True)
        join_required = await self._check_requires_joins(messages, schema, table)
        if join_required is None:
            return None
        if join_required:
            tables_to_source = await self.find_join_tables(messages)
        else:
            tables_to_source = {table: source}

        tables_sql_schemas = {}
        for source_table, source in tables_to_source.items():
            if source_table == table:
                table_schema = schema
            else:
                table_schema = await get_schema(source, source_table, include_min_max=True)

            # Look up underlying table name
            table_name = source_table
            if (
                'tables' in source.param and
                isinstance(source.tables, dict) and
                'select ' not in source.tables[table_name].lower()
            ):
                table_name = source.tables[table_name]

            tables_sql_schemas[table_name] = {
                "schema": yaml.dump(table_schema),
                "sql": source.get_sql_expr(source_table),
            }
        self._memory["tables_sql_schemas"] = tables_sql_schemas

        dialect = source.dialect
        system_prompt = await self._render_prompt(
            "main",
            messages,
            tables_sql_schemas=tables_sql_schemas,
            dialect=dialect,
            join_required=join_required,
            table=table,
        )
        if join_required:
            # Remove source prefixes message, e.g. SOURCE_TABLE_SEPARATOR<source>SOURCE_TABLE_SEPARATOR<table>
            messages[-1]["content"] = re.sub(rf".*?{SOURCE_TABLE_SEPARATOR}", "", messages[-1]["content"])
        sql_query = await self._create_valid_sql(messages, system_prompt, tables_to_source, step_title)
        pipeline = self._memory['pipeline']

        self._render_lumen(pipeline, spec=sql_query, messages=messages, render_output=render_output, title=step_title)
        return pipeline


class BaseViewAgent(LumenBaseAgent):

    requires = param.List(default=["pipeline"], readonly=True)

    provides = param.List(default=["view"], readonly=True)

    prompts = param.Dict(
        default={
            "main": {"template": PROMPTS_DIR / "BaseViewAgent" / "main.jinja2"},
        }
    )

    async def _extract_spec(self, spec: dict[str, Any]):
        return dict(spec)

    async def _update_spec(self, memory: _Memory, event: param.parameterized.Event):
        memory['view'] = dict(await self._extract_spec(event.new), type=self.view_type)

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

        schema = await get_schema(pipeline, include_min_max=False)
        if not schema:
            raise ValueError("Failed to retrieve schema for the current pipeline.")

        doc = self.view_type.__doc__.split("\n\n")[0] if self.view_type.__doc__ else self.view_type.__name__
        system_prompt = await self._render_prompt(
            "main",
            messages,
            schema=yaml.dump(schema),
            table=pipeline.table,
            view=self._memory.get('view'),
            doc=doc,
        )
        model_spec = self.prompts["main"].get("llm_spec", "default")
        output = await self.llm.invoke(
            messages,
            system=system_prompt,
            model_spec=model_spec,
            response_model=self._get_model("main", schema=schema),
        )
        spec = await self._extract_spec(dict(output))
        chain_of_thought = spec.pop("chain_of_thought", None)
        if chain_of_thought:
            with self.interface.add_step(
                title=step_title or "Generating view...",
                steps_layout=self._steps_layout
            ) as step:
                step.stream(chain_of_thought)
        log_debug(f"{self.name} settled on spec: {spec!r}.")
        self._memory["view"] = dict(spec, type=self.view_type)
        view = self.view_type(pipeline=pipeline, **spec)

        self._render_lumen(view, messages=messages, render_output=render_output, title=step_title)
        return view


class hvPlotAgent(BaseViewAgent):

    purpose = param.String(default="""
        Generates a plot of the data given a user prompt.
        If the user asks to plot, visualize or render the data this is your best best.""")

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
        model = param_to_pydantic(self.view_type, excluded=excluded, schema=schema, extra_fields={
            "chain_of_thought": (str, FieldInfo(description="Your thought process behind the plot.")),
        })
        return model[self.view_type.__name__]

    async def _update_spec(self, memory: _Memory, event: param.parameterized.Event):
        spec = yaml.load(event.new, Loader=yaml.SafeLoader)
        memory['view'] = dict(await self._extract_spec(spec), type=self.view_type)

    async def _extract_spec(self, spec: dict[str, Any]):
        pipeline = self._memory["pipeline"]
        spec = {
            key: val for key, val in spec.items()
            if val is not None
        }
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

    purpose = param.String(default="""
        Generates a vega-lite specification of the plot the user requested.
        If the user asks to plot, visualize or render the data this is your best best.""")

    prompts = param.Dict(
        default={
            "main": {
                "response_model": VegaLiteSpec,
                "template": PROMPTS_DIR / "VegaLiteAgent" / "main.jinja2"
            },
        }
    )

    view_type = VegaLiteView

    _extensions = ('vega',)

    async def _update_spec(self, memory: _Memory, event: param.parameterized.Event):
        spec = yaml.load(event.new, Loader=yaml.SafeLoader)
        memory['view'] = dict(await self._extract_spec({"json_spec": json.dumps(spec)}), type=self.view_type)

    async def _extract_spec(self, spec: dict[str, Any]):
        vega_spec = json.loads(spec['json_spec'])
        if "$schema" not in vega_spec:
            vega_spec["$schema"] = "https://vega.github.io/schema/vega-lite/v5.json"
        if "width" not in vega_spec:
            vega_spec["width"] = "container"
        if "height" not in vega_spec:
            vega_spec["height"] = "container"
        return {'spec': vega_spec, "sizing_mode": "stretch_both", "min_height": 500}


class AnalysisAgent(LumenBaseAgent):

    analyses = param.List([])

    purpose = param.String(default="""
        Perform custom analyses on the data.""")

    prompts = param.Dict(
        default={
            "main": {"template": PROMPTS_DIR / "AnalysisAgent" / "main.jinja2"},
        }
    )

    provides = param.List(default=['view'])

    requires = param.List(default=['pipeline'])

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
        pipeline = self._memory['pipeline']
        analyses = {a.name: a for a in self.analyses if await a.applies(pipeline)}
        if not analyses:
            log_debug("No analyses apply to the current data.")
            return None

        # Short cut analysis selection if there's an exact match
        if len(messages):
            analysis = messages[0].get('content').replace('Apply ', '')
            if analysis in analyses:
                analyses = {analysis: analyses[analysis]}

        if len(analyses) > 1:
            with self.interface.add_step(title="Choosing the most relevant analysis...", steps_layout=self._steps_layout) as step:
                type_ = Literal[tuple(analyses)]
                analysis_model = create_model(
                    "Analysis",
                    correct_name=(type_, FieldInfo(description="The name of the analysis that is most appropriate given the user query."))
                )
                system_prompt = await self._render_prompt(
                    "main",
                    messages,
                    analyses=analyses,
                    data=self._memory.get("data"),
                )
                model_spec = self.prompts["main"].get("llm_spec", "default")
                analysis_name = (await self.llm.invoke(
                    messages,
                    system=system_prompt,
                    model_spec=model_spec,
                    response_model=analysis_model,
                    allow_partial=False,
                )).correct_name
                step.stream(f"Selected {analysis_name}")
                step.success_title = f"Selected {analysis_name}"
        else:
            analysis_name = next(iter(analyses))

        with self.interface.add_step(title=step_title or "Creating view...", steps_layout=self._steps_layout) as step:
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
                    view = Panel(object=view, pipeline=self._memory.get('pipeline'))
                spec = view.to_spec()
                if isinstance(view, View):
                    view_type = view.view_type
                    self._memory["view"] = dict(spec, type=view_type)
                elif isinstance(view, Pipeline):
                    self._memory["pipeline"] = view
                # Ensure data reflects processed pipeline
                if pipeline is not self._memory['pipeline']:
                    pipeline = self._memory['pipeline']
                    if len(data) > 0:
                        self._memory["data"] = await describe_data(data)
                yaml_spec = yaml.dump(spec)
                step.stream(f"Generated view\n```yaml\n{yaml_spec}\n```")
                step.success_title = "Generated view"
            else:
                step.success_title = "Configure the analysis"
                view = None

        analysis = self._memory["analysis"]
        pipeline = self._memory['pipeline']
        if view is None and analysis.autorun:
            self.interface.stream('Failed to find an analysis that applies to this data')
        else:
            self._render_lumen(
                view,
                analysis=analysis,
                pipeline=pipeline,
                render_output=render_output,
                title=step_title
            )
        return view
