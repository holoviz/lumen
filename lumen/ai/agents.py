from __future__ import annotations

import asyncio
import re
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
from ..sources.base import BaseSQLSource
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
    PartialBaseModel, RetrySpec, Sql, VegaLiteSpec, make_find_tables_model,
)
from .tools import ToolUser
from .translate import param_to_pydantic
from .utils import (
    clean_sql, describe_data, get_data, get_pipeline, get_schema, load_json,
    log_debug, mutate_user_message, parse_table_slug, report_error,
    retry_llm_output, stream_details,
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
        model_spec = self.prompts["main"].get("llm_spec", self.llm_spec_key)
        async for output_chunk in self.llm.stream(
            messages, system=system_prompt, model_spec=model_spec, field="output"
        ):
            message = self.interface.stream(
                output_chunk, replace=True, message=message, user=self.user, max_width=self._max_width
            )
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

    purpose = param.String(default="""
        The SourceAgent allows a user to upload unavailable, new datasets, tables, or documents.

        Only use this if the user is requesting to add a completely new table.
        Not useful for answering what is available or loading existing datasets
        so use other agents for these.
        """)

    requires = param.List(default=[], readonly=True)

    provides = param.List(default=["source", "document_sources"], readonly=True)

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

        output = pn.Column(source_controls)
        if "source" not in self._memory:
            help_message = "No datasets or documents were found, **please upload at least one to continue**..."
        else:
            help_message = "**Please upload new dataset(s)/document(s) to continue**, or click cancel if this was unintended..."
        output.insert(0, help_message)
        self.interface.send(output, respond=False, user="SourceAgent")
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

    purpose = param.String(default="""
        Chats and provides info about high level data related topics,
        e.g. the columns of the data or statistics about the data,
        and continuing the conversation.

        Is capable of providing suggestions to get started or comment on interesting tidbits.
        It can talk about the data, if available. Or, it can also solely talk about documents.

        Usually not used concurrently with SQLAgent, unlike AnalystAgent.
        Can be used to describe available tables. Often used together
        with TableLookup, DocumentLookup, or other tools.""")

    prompts = param.Dict(
        default={
            "main": {
                "template": PROMPTS_DIR / "ChatAgent" / "main.jinja2",
            },
        }
    )

    # technically not required if appended manually with tool in coordinator
    requires = param.List(default=["table_vector_metaset"], readonly=True)

    async def respond(
        self,
        messages: list[Message],
        render_output: bool = False,
        step_title: str | None = None,
    ) -> Any:
        context = {"tool_context": await self._use_tools("main", messages)}
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

    prompts = param.Dict(
        default={
            "main": {
                "template": PROMPTS_DIR / "AnalystAgent" / "main.jinja2",
                "tools": []
            },
        }
    )

    requires = param.List(default=["source", "pipeline"], readonly=True)


class ListAgent(Agent):
    """
    Abstract base class for agents that display a list of items to the user.
    """

    purpose = param.String(default="""
        Renders a list of items to the user and lets the user pick one.""")

    requires = param.List(default=[], readonly=True)

    _extensions = ('tabulator',)

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
            buttons={'show': '<i class="fa fa-eye"></i>'},
            show_index=False,
            min_height=150,
            min_width=350,
            widths={self._column_name: '90%'},
            disabled=True,
            page_size=10,
            pagination="remote",
            header_filters=header_filters,
            sizing_mode="stretch_width",
        )

        item_list.on_click(self._use_item)
        self.interface.stream(item_list, user="Lumen")
        return item_list


class TableListAgent(ListAgent):
    """
    The TableListAgent lists all available tables and lets the user pick one.
    """

    purpose = param.String(default="""
        Renders a list of all availables tables to the user and lets the user
        pick one, do not use if user has already picked a table.
        Not useful for gathering information about the tables.
        """)

    requires = param.List(default=["source"], readonly=True)

    _column_name = "Table"

    _message_format = "Show the table: {item}"

    @classmethod
    async def applies(cls, memory: _Memory) -> bool:
        source = memory.get("source")
        if not source:
            return True  # source not loaded yet; always apply
        return len(source.get_tables()) > 1

    def _get_items(self) -> list[str]:
        tables = []
        if "closest_tables" in self._memory:
            tables = self._memory["closest_tables"]

        for source in self._memory["sources"]:
            tables += source.get_tables()

        # remove duplicates, keeps order in which they were added
        return pd.unique(tables).tolist()


class DocumentListAgent(ListAgent):
    """
    The DocumentListAgent lists all available documents provided by the user.
    """

    purpose = param.String(default="""
        Renders a list of all availables documents to the user and lets the user
        pick one. Not useful for gathering details about the documents;
        use ChatAgent for that instead.
        """)

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
        return [doc["metadata"].get("filename", "untitled") for doc in self._memory["document_sources"]]


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
                # TODO: only input the inner spec to retry
                system = await self._render_prompt(
                    "retry_output", messages=modified_messages, spec=out.spec,
                    language=out.language,
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
        Responsible for displaying tables, generating, modifying and
        executing SQL queries to answer user queries about the data,
        such querying subsets of the data, aggregating the data and
        calculating results. If the current table does not contain all
        the available data the SQL agent is also capable of joining it
        with other tables. Will generate and execute a query in a single
        step. Not useful if the user is using the same data for plotting.""")

    prompts = param.Dict(
        default={
            "main": {
                "response_model": Sql,
                "template": PROMPTS_DIR / "SQLAgent" / "main.jinja2",
            },
            "find_tables": {
                "response_model": make_find_tables_model,
                "template": PROMPTS_DIR / "SQLAgent" / "find_tables.jinja2"
            },
        }
    )

    provides = param.List(default=["table", "sql", "pipeline", "data"], readonly=True)

    requires = param.List(default=["source", "sql_metaset"], readonly=True)

    _extensions = ('codeeditor', 'tabulator',)

    _output_type = SQLOutput

    @retry_llm_output()
    async def _create_valid_sql(
        self,
        messages: list[Message],
        dialect: str,
        comments: str,
        title: str,
        tables_to_source: dict[str, BaseSQLSource],
        errors=None
    ):
        if errors:
            # get the head of the tables
            columns_context = ""
            vector_metadata_map = self._memory["sql_metaset"].vector_metaset.vector_metadata_map
            for table_slug, vector_metadata in vector_metadata_map.items():
                table_name = table_slug.split(SOURCE_TABLE_SEPARATOR)[-1]
                if table_name in tables_to_source:
                    columns = [col.name for col in vector_metadata.columns]
                    columns_context += f"\nSQL: {vector_metadata.base_sql}\nColumns: {', '.join(columns)}\n\n"
            last_query = self._memory["sql"]
            num_errors = len(errors)
            errors = ('\n'.join(f"{i+1}. {error}" for i, error in enumerate(errors))).strip()
            content = (
                f"\n\nYou are a world-class SQL user. Identify why this query failed:\n```sql\n{last_query}\n```\n\n"
                f"Your goal is to try a different query to address the question while avoiding these issues:\n```\n{errors}\n```\n\n"
                f"Note a penalty of $100 will be incurred for every time the issue occurs, and thus far you have been penalized ${num_errors * 100}! "
                f"Use your best judgement to address them. If the error is `syntax error at or near \")\"`, double check you used "
                f"table names verbatim, i.e. `read_parquet('table_name.parq')` instead of `table_name`. Ensure no inline comments are present. "
                f"For extra context, here are the tables and columns available:\n{columns_context}\n"
            )
            messages = mutate_user_message(content, messages)

        join_required = len(tables_to_source) > 1
        comments = comments if join_required else ""  # comments are about joins
        system_prompt = await self._render_prompt(
            "main",
            messages,
            join_required=join_required,
            dialect=dialect,
            comments=comments,
            has_errors=bool(errors),
        )
        with self._add_step(title=title or "SQL query", steps_layout=self._steps_layout) as step:
            model_spec = self.prompts["main"].get("llm_spec", self.llm_spec_key)
            response = self.llm.stream(messages, system=system_prompt, model_spec=model_spec, response_model=self._get_model("main"))
            sql_query = None
            try:
                async for output in response:
                    step_message = output.chain_of_thought or ""
                    step.stream(step_message, replace=True)
                if output.query:
                    sql_query = clean_sql(output.query, dialect)
                if sql_query and output.expr_slug:
                    step.stream(f"\n`{output.expr_slug}`\n```sql\n{sql_query}\n```", replace=True)
                self._memory["sql"] = sql_query
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
                # Remove source prefixes from table names
                if SOURCE_TABLE_SEPARATOR in renamed_table:
                    _, renamed_table = renamed_table.split(SOURCE_TABLE_SEPARATOR, maxsplit=1)
                sql_query = sql_query.replace(a_table, renamed_table)
                mirrors[renamed_table] = (a_source, renamed_table)
            source = DuckDBSource(uri=":memory:", mirrors=mirrors)
        else:
            source = next(iter(sources))

        # check whether the SQL query is valid
        expr_slug = output.expr_slug
        try:
            # TODO: if original sql expr matches, don't recreate a new one!
            sql_expr_source = source.create_sql_expr_source({expr_slug: sql_query})
            sql_query = sql_expr_source.tables[expr_slug]
            sql_transforms = [SQLLimit(limit=1_000_000, write=source.dialect, pretty=True, identify=False)]
            transformed_sql_query = sql_query
            for sql_transform in sql_transforms:
                transformed_sql_query = sql_transform.apply(transformed_sql_query)  # not to be used elsewhere; just for transparency
                if transformed_sql_query != sql_query:
                    stream_details(f'```sql\n{transformed_sql_query}\n```', step, title=f"{sql_transform.__class__.__name__} Applied")
            pipeline = await get_pipeline(
                source=sql_expr_source, table=expr_slug, sql_transforms=sql_transforms
            )
        except Exception as e:
            report_error(e, step)
            raise e

        try:
            df = await get_data(pipeline)
        except Exception as e:
            if isinstance(source, DuckDBSource):
                source._connection.execute(f'DROP TABLE IF EXISTS "{expr_slug}"')
            report_error(e, step)
            raise e

        self._memory["data"] = await describe_data(df)
        self._memory["sources"].append(sql_expr_source)
        self._memory["source"] = sql_expr_source
        self._memory["pipeline"] = pipeline
        self._memory["table"] = pipeline.table
        return sql_query

    def _update_spec(self, memory: _Memory, event: param.parameterized.Event):
        memory['sql'] = event.new

    @retry_llm_output()
    async def _find_tables(self, messages: list[Message], errors: list | None = None) -> tuple[dict[str, BaseSQLSource], str]:
        if errors:
            content = (
                "Your goal is to try to address the question while avoiding these issues:\n"
                f"```\n{errors}\n```\n\n"
            )
            messages = mutate_user_message(content, messages)

        sources = {source.name: source for source in self._memory["sources"]}
        vector_metaset = self._memory["sql_metaset"].vector_metaset
        selected_slugs = list(
            #  if no tables/cols are subset
            vector_metaset.selected_columns or
            vector_metaset.vector_metadata_map
        )
        if len(selected_slugs) == 0:
            raise ValueError("No tables found in memory.")

        chain_of_thought = ""
        with self._add_step(title="Determining tables to use", steps_layout=self._steps_layout) as step:
            if len(selected_slugs) > 1:
                system = await self._render_prompt(
                    "find_tables", messages, separator=SOURCE_TABLE_SEPARATOR
                )
                find_tables_model = self._get_model("find_tables", tables=selected_slugs)
                model_spec = self.prompts["find_tables"].get("llm_spec", self.llm_spec_key)
                response = self.llm.stream(
                    messages,
                    system=system,
                    model_spec=model_spec,
                    response_model=find_tables_model,
                )
                async for output in response:
                    chain_of_thought = output.chain_of_thought or ""
                    if output.potential_join_issues is not None:
                        chain_of_thought += output.potential_join_issues
                    step.stream(chain_of_thought, replace=True)

                if selected_slugs := output.selected_tables:
                    stream_details('\n'.join(selected_slugs), step, title="Relevant tables", auto=False)
                    step.success_title = f'Found {len(selected_slugs)} relevant table(s)'

            tables_to_source = {}
            step.stream("Planning to use:")
            for table_slug in selected_slugs:
                a_source_obj, a_table = parse_table_slug(table_slug, sources)
                tables_to_source[a_table] = a_source_obj
                step.stream(f"\n{a_table!r}")
        return tables_to_source, chain_of_thought

    async def respond(
        self,
        messages: list[Message],
        render_output: bool = False,
        step_title: str | None = None,
    ) -> Any:
        tables_to_source, comments = await self._find_tables(messages)
        dialect = self._memory["source"].dialect
        try:
            sql_query = await self._create_valid_sql(messages, dialect, comments, step_title, tables_to_source)
            pipeline = self._memory['pipeline']
        except RetriesExceededError as e:
            traceback.print_exception(e)
            self._memory["__error__"] = str(e)
            return None
        self._render_lumen(pipeline, spec=sql_query, messages=messages, render_output=render_output, title=step_title)
        return pipeline


class BaseViewAgent(LumenBaseAgent):

    requires = param.List(default=["pipeline", "sql_metaset"], readonly=True)

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
        if errors:
           errors = '\n'.join(errors)
           if self._last_output:
            try:
                json_spec = load_json(self._last_output["json_spec"])
            except Exception:
                json_spec = ""

            vector_metadata_map = self._memory["sql_metaset"].vector_metaset.vector_metadata_map
            columns_context = ""
            for table_slug, vector_metadata in vector_metadata_map.items():
                table_name = table_slug.split(SOURCE_TABLE_SEPARATOR)[-1]
                if table_name in pipeline.table:
                    columns = [col.name for col in vector_metadata.columns]
                    columns_context += f"\nSQL: {vector_metadata.base_sql}\nColumns: {', '.join(columns)}\n\n"
            messages = mutate_user_message(
                f"\nNote, your last specification did not work as intended:\n```json\n{json_spec}\n```\n\n"
                f"Your task is to expertly address these errors so they do not occur again:\n```\n{errors}\n```\n"
                f"For extra context, here are the tables and columns available:\n{columns_context}\n",
                messages
            )

        doc = self.view_type.__doc__.split("\n\n")[0] if self.view_type.__doc__ else self.view_type.__name__
        system = await self._render_prompt(
            "main",
            messages,
            table=pipeline.table,
            has_errors=bool(errors),
            doc=doc,
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
        with self._add_step(
            title=step_title or "Generating view...",
            steps_layout=self._steps_layout
        ) as step:
            async for output in response:
                chain_of_thought = output.chain_of_thought or ""
                step.stream(chain_of_thought, replace=True)

            self._last_output = dict(output)
            try:
                spec = await self._extract_spec(self._last_output)
            except Exception as e:
                error = str(e)
                traceback.print_exception(e)
                context = f'```\n{yaml.safe_dump(load_json(self._last_output["json_spec"]))}\n```'
                report_error(e, step, language="json", context=context)

        if error:
            raise ValueError(error)

        log_debug(f"{self.name} settled on spec: {spec!r}.")
        return spec

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

        schema = await get_schema(pipeline)
        if not schema:
            raise ValueError("Failed to retrieve schema for the current pipeline.")

        spec = await self._create_valid_spec(messages, pipeline, schema, step_title)
        self._memory["view"] = dict(spec, type=self.view_type)
        view = self.view_type(pipeline=pipeline, **spec)
        self._render_lumen(view, messages=messages, render_output=render_output, title=step_title)
        return view


class hvPlotAgent(BaseViewAgent):

    purpose = param.String(default="""
        Generates a plot of the data given a user prompt.
        If the user asks to plot, visualize or render the data this is your best bet.""")

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
        model = param_to_pydantic(self.view_type, base_model=PartialBaseModel, excluded=excluded, schema=schema, extra_fields={
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
        If the user asks to plot, visualize or render the data this is your best bet.""")

    prompts = param.Dict(
        default={
            "main": {
                "response_model": VegaLiteSpec,
                "template": PROMPTS_DIR / "VegaLiteAgent" / "main.jinja2"
            },
            "retry_output": {
                "response_model": RetrySpec,
                "template": PROMPTS_DIR / "VegaLiteAgent" / "retry_output.jinja2"
            },
        }
    )

    view_type = VegaLiteView

    _extensions = ('vega',)

    _output_type = VegaLiteOutput

    async def _update_spec(self, memory: _Memory, event: param.parameterized.Event):
        try:
            spec = await self._extract_spec({"yaml_spec": event.new})
        except Exception as e:
            traceback.print_exception(e)
            return
        memory['view'] = dict(spec, type=self.view_type)

    def _add_geographic_items(self, vega_spec: dict, vega_spec_str: str):
        # standardize the vega spec, by migrating to layer
        if "layer" not in vega_spec:
            vega_spec["layer"] = [{"encoding": vega_spec.pop("encoding", None), "mark": vega_spec.pop("mark", None)}]

        # make it zoomable
        if "params" not in vega_spec:
            vega_spec["params"] = []

        # Get existing param names
        existing_param_names = {
            param.get('name') for param in vega_spec["params"]
            if isinstance(param, dict) and 'name' in param
        }
        for p in VEGA_ZOOMABLE_MAP_ITEMS["params"]:
            if p.get('name') not in existing_param_names:
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
                    fields_to_check.extend(
                        item["field"] for item in enc_def
                        if isinstance(item, dict) and "field" in item
                    )

                for field in fields_to_check:
                    if field not in schema:
                        raise ValueError(f"Field '{field}' not found in schema.")

    async def _extract_spec(self, spec: dict[str, Any]):
        # .encode().decode('unicode_escape') fixes a JSONDecodeError in Python
        # where it's expecting property names enclosed in double quotes
        # by properly handling the escaped characters in your JSON string
        if yaml_spec:= spec.get('yaml_spec'):
            vega_spec = yaml.load(yaml_spec, Loader=yaml.SafeLoader)
        elif json_spec:= spec.get('json_spec'):
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
        elif "point: true" not in vega_spec_str or "params" not in vega_spec:
            # add pan/zoom controls to all plots except geographic ones and points overlaid on line plots
            # because those result in an blank plot without error
            vega_spec["params"] = [{"bind": "scales", "name": "grid", "select": "interval"}]
        return {'spec': vega_spec, "sizing_mode": "stretch_both", "min_height": 300, "max_width": 1200}


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
            with self._add_step(title="Choosing the most relevant analysis...", steps_layout=self._steps_layout) as step:
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
                model_spec = self.prompts["main"].get("llm_spec", self.llm_spec_key)
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
