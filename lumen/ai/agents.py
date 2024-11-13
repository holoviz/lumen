import asyncio
import difflib
import json
import re
import textwrap

from typing import Any, Literal

import pandas as pd
import panel as pn
import param
import yaml

from instructor.retry import InstructorRetryException
from panel.chat import ChatInterface
from panel.layout import Column
from panel.viewable import Viewer
from pydantic import BaseModel, create_model
from pydantic.fields import FieldInfo

from ..base import Component
from ..dashboard import Config
from ..pipeline import Pipeline
from ..sources.base import BaseSQLSource
from ..sources.duckdb import DuckDBSource
from ..state import state
from ..transforms.sql import SQLLimit
from ..views import VegaLiteView, View, hvPlotUIView
from .analysis import Analysis
from .config import FUZZY_TABLE_LENGTH
from .controls import SourceControls
from .embeddings import Embeddings
from .llm import Llm, Message
from .memory import _Memory, memory
from .models import (
    DataRequired, FuzzyTable, JoinRequired, Sql, TableJoins, Topic,
    VegaLiteSpec, make_table_model,
)
from .translate import param_to_pydantic
from .utils import (
    clean_sql, describe_data, gather_table_sources, get_data, get_pipeline,
    get_schema, render_template, report_error, retry_llm_output,
)
from .views import AnalysisOutput, LumenOutput, SQLOutput


class Agent(Viewer):
    """
    An Agent in Panel is responsible for handling a specific type of
    query. Each agent consists of an LLM and optionally a set of
    embeddings.
    """

    debug = param.Boolean(default=False, doc="""
        Whether to enable verbose error reporting.""")

    embeddings = param.ClassSelector(class_=Embeddings, doc="""
        Embeddings object which is queried to provide additional context
        before asking the LLM to respond.""")

    interface = param.ClassSelector(class_=ChatInterface, doc="""
        The ChatInterface to report progress to.""")

    llm = param.ClassSelector(class_=Llm, doc="""
        The LLM implementation to query.""")

    memory = param.ClassSelector(class_=_Memory, default=None, doc="""
        Local memory which will be used to provide the agent context.
        If None the global memory will be used.""")

    steps_layout = param.ClassSelector(default=None, class_=Column, allow_None=True, doc="""
        The layout progress updates will be streamed to.""")

    provides = param.List(default=[], readonly=True, doc="""
        List of context values this Agent provides to current working memory.""")

    requires = param.List(default=[], readonly=True, doc="""
        List of context that this Agent requires to be in memory.""")

    response_model = param.ClassSelector(class_=BaseModel, is_instance=False, doc="""
        A Pydantic model determining the schema of the response.""")

    system_prompt = param.String(doc="The system prompt.")

    user = param.String(default="Agent", doc="""
        The name of the user that will be respond to the user query.""")

    # Panel extensions this agent requires to be loaded
    _extensions = ()

    # Maximum width of the output
    _max_width = 1200

    __abstract = True

    def __init__(self, **params):
        def _exception_handler(exception):
            if str(exception) in self.interface.serialize()[-1]["content"]:
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

    @property
    def _memory(self):
        return memory if self.memory is None else self.memory

    def __panel__(self):
        return self.interface

    async def _system_prompt_with_context(
        self, messages: list, context: str = ""
    ) -> str:
        system_prompt = self.system_prompt
        if self.embeddings:
            context = self.embeddings.query(messages)
        if context:
            system_prompt += f"\n### CONTEXT: {context}"
        return system_prompt

    async def _get_closest_tables(self, messages: list[Message], tables: list[str], n: int = 3) -> list[str]:
        system = (
            f"You are great at extracting keywords based on the user query to find the correct table. "
            f"The current table selected: `{self._memory.get('current_table', 'N/A')}`. "
        )
        tables = tuple(table.replace('"', "") for table in tables)

        fuzzy_table = (await self.llm.invoke(
            messages,
            system=system,
            response_model=FuzzyTable,
            allow_partial=False
        ))
        if not fuzzy_table.required:
            return [self._memory.get("current_table") or tables[0]]

        # make case insensitive, but keep original case to transform back
        if not fuzzy_table.keywords:
            return tables[:n]

        table_keywords = [keyword.lower() for keyword in fuzzy_table.keywords]
        tables_lower = {table.lower(): table for table in tables}

        # get closest matches
        closest_tables = set()
        for keyword in table_keywords:
            closest_tables |= set(difflib.get_close_matches(keyword, list(tables_lower), n=5, cutoff=0.3))
        closest_tables = [tables_lower[table] for table in closest_tables]
        if len(closest_tables) == 0:
            # if no tables are found, ask the user to select ones and load it
            tables = await self._select_table(tables)
        self._memory["closest_tables"] = tables
        return tuple(tables)

    async def _select_table(self, tables):
        input_widget = pn.widgets.AutocompleteInput(
            placeholder="Start typing parts of the table name",
            case_sensitive=False,
            options=list(tables),
            search_strategy="includes",
            min_characters=1,
        )
        self.interface.send(
            pn.chat.ChatMessage(
                "No tables were found based on the user query. Please select the table you are looking for.",
                footer_objects=[input_widget],
                user="Assistant",
            ),
            respond=False
        )
        while not input_widget.value:
            await asyncio.sleep(0.05)
        tables = [input_widget.value]
        self.interface.pop(-1)
        return tables

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
        step_title: str | None = None
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
        system_prompt = await self._system_prompt_with_context(messages)
        message = None
        async for output_chunk in self.llm.stream(
            messages, system=system_prompt, response_model=self.response_model, field="output"
        ):
            message = self.interface.stream(
                output_chunk, replace=True, message=message, user=self.user, max_width=self._max_width
            )
        return message


class SourceAgent(Agent):
    """
    The SourceAgent allows a user to upload new datasets.

    Only use this if the user is requesting to add a dataset or you think
    additional information is required to solve the user query.
    """

    requires = param.List(default=[], readonly=True)

    provides = param.List(default=["current_source"], readonly=True)

    on_init = param.Boolean(default=True)

    _extensions = ('filedropper',)

    async def respond(
        self,
        messages: list[Message],
        render_output: bool = False,
        step_title: str | None = None
    ) -> Any:
        source_controls = SourceControls(
            multiple=True, replace_controls=True, select_existing=False, memory=self.memory
        )
        self.interface.send(source_controls, respond=False, user="SourceAgent")
        while not source_controls._add_button.clicks > 0:
            await asyncio.sleep(0.05)
        return source_controls


class ChatAgent(Agent):
    """
    Chats and provides info about high level data related topics,
    e.g. what datasets are available, the columns of the data or
    statistics about the data, and continuing the conversation.

    Is capable of providing suggestions to get started or comment on interesting tidbits.
    If data is available, it can also talk about the data itself.
    """

    system_prompt = param.String(
        default=(
            "Be a helpful chatbot to talk about high-level data exploration "
            "like what datasets are available and what each column represents. "
            "Provide suggestions to get started if necessary. Do not write analysis "
            "code unless explicitly asked."
        )
    )

    response_model = param.ClassSelector(class_=BaseModel, is_instance=False)

    requires = param.List(default=["current_source"], readonly=True)

    @retry_llm_output()
    async def requirements(self, messages: list[Message], errors=None):
        if 'current_data' in self._memory:
            return self.requires

        available_sources = self._memory["available_sources"]
        _, tables_schema_str = await gather_table_sources(available_sources)
        with self.interface.add_step(title="Checking if data is required", steps_layout=self._steps_layout) as step:
            response = self.llm.stream(
                messages,
                system=(
                    "Assess if the user's prompt requires loading data. "
                    "If the inquiry is just about available tables, no data access required. "
                    "However, if relevant tables apply to the query, load the data for a "
                    "more accurate and up-to-date response. "
                    f"Here are the available tables:\n{tables_schema_str}"
                ),
                response_model=DataRequired,
            )
            async for output in response:
                step.stream(output.chain_of_thought, replace=True)
            if output.data_required:
                return self.requires + ['current_table']
        return self.requires

    async def _system_prompt_with_context(
        self, messages: list[Message], context: str = ""
    ) -> str:
        source = self._memory.get("current_source")
        if not source:
            raise ValueError("No source found in memory.")
        tables = source.get_tables()
        if len(tables) > 1:
            if len(tables) > FUZZY_TABLE_LENGTH and "closest_tables" not in self._memory:
                closest_tables = await self._get_closest_tables(messages, tables, n=5)
            else:
                closest_tables = self._memory.get("closest_tables", tables)
            context = f"Available tables: {', '.join(closest_tables)}"
        else:
            self._memory["current_table"] = table = self._memory.get("current_table", tables[0])
            schema = await get_schema(self._memory["current_source"], table)
            if schema:
                context = f"{table} with schema: {schema}"

        if "current_data" in self._memory:
            context += (
                f"\nHere's a summary of the dataset the user just asked about:\n```\n{self._memory['current_data']}\n```"
            )

        system_prompt = self.system_prompt
        if context:
            system_prompt += f"\n### CONTEXT: {context}"
        return system_prompt


class ChatDetailsAgent(ChatAgent):
    """
    Responsible for chatting and providing info about details about the table values,
    e.g. what should be noted about the data values, valuable, applicable insights about the data,
    and continuing the conversation. Does not provide overviews;
    only details, meaning, relationships, and trends of the data.
    """

    requires = param.List(default=["current_source", "current_table"], readonly=True)

    system_prompt = param.String(
        default=(
            "You are a world-class, subject scientist on the topic. Help provide guidance and meaning "
            "about the data values, highlight valuable and applicable insights. Be very precise "
            "on your subject matter expertise and do not be afraid to use specialized terminology or "
            "industry-specific jargon to describe the data values and trends. Do not provide overviews; "
            "instead, focus on the details and meaning of the data. Highlight relationships "
            "between the columns and what could be interesting to dive deeper into. Do not write analysis "
            "code unless explicitly asked"
        )
    )

    async def _system_prompt_with_context(
        self, messages: list[Message], context: str = ""
    ) -> str:
        system_prompt = self.system_prompt
        topic = (await self.llm.invoke(
            messages,
            system="What is the topic of the table?",
            response_model=Topic,
            allow_partial=False,
        )).result
        context += f"Topic you are a world-class expert on: {topic}"

        current_data = self._memory["current_data"]
        if isinstance(current_data, dict):
            columns = list(current_data["stats"].keys())
        else:
            columns = list(current_data.columns)
        context += f"\nHere are the columns of the table: {columns}"

        if context:
            system_prompt += f"\n### CONTEXT: {context}"
        return system_prompt


class LumenBaseAgent(Agent):

    user = param.String(default="Lumen")

    _output_type = LumenOutput

    _max_width = None

    def _render_lumen(
        self,
        component: Component,
        message: pn.chat.ChatMessage = None,
        render_output: bool = False,
        **kwargs
    ):
        out = self._output_type(
            component=component, render_output=render_output, **kwargs
        )
        if 'outputs' in self._memory:
            self._memory['outputs'].append(out)
        message_kwargs = dict(value=out, user=self.user)
        self.interface.stream(message=message, **message_kwargs, replace=True, max_width=self._max_width)


class TableListAgent(LumenBaseAgent):
    """
    Renders a list of all availables tables to the user.

    Not useful for gathering information about the tables.
    """

    system_prompt = param.String(
        default="You are an agent responsible for listing the available tables in the current source."
    )

    requires = param.List(default=["current_source"], readonly=True)

    _extensions = ('tabulator',)

    @classmethod
    async def applies(cls, memory: _Memory) -> bool:
        source = memory.get("current_source")
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
        step_title: str | None = None
    ) -> Any:
        tables = []
        for source in self._memory['available_sources']:
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
        return table_list


class SQLAgent(LumenBaseAgent):
    """
    Responsible for generating, modifying and executing SQL queries to
    answer user queries about the data, such querying subsets of the
    data, aggregating the data and calculating results. If the current
    table does not contain all the available data the SQL agent is
    also capable of joining it with other tables.
    """

    system_prompt = param.String(
        default=textwrap.dedent(
            """
            You are an agent responsible for writing a SQL query that will
            perform the data transformations the user requested.
            """
        )
    )

    requires = param.List(default=["current_source"], readonly=True)

    provides = param.List(default=["current_table", "current_sql", "current_pipeline"], readonly=True)

    _extensions = ('codeeditor', 'tabulator',)

    _output_type = SQLOutput

    async def _select_relevant_table(self, messages: list[Message]) -> tuple[str, BaseSQLSource]:
        """Select the most relevant table based on the user query."""
        available_sources = self._memory["available_sources"]

        tables_to_source, tables_schema_str = await gather_table_sources(available_sources)
        tables = tuple(tables_to_source)
        if messages and messages[-1]["content"].startswith("Show the table: '"):
            # Handle the case where explicitly requested a table
            table = messages[-1]["content"].replace("Show the table: '", "")[:-1]
        elif len(tables) == 1:
            table = tables[0]
        else:
            with self.interface.add_step(title="Choosing the most relevant table...", steps_layout=self._steps_layout) as step:
                closest_tables = self._memory.pop("closest_tables", [])
                if closest_tables:
                    tables = closest_tables
                elif len(tables) > FUZZY_TABLE_LENGTH:
                    tables = await self._get_closest_tables(messages, tables)
                system_prompt = await self._system_prompt_with_context(messages, context=tables_schema_str)
                if len(tables) > 1:
                    table_model = make_table_model(tables)
                    result = await self.llm.invoke(
                        messages,
                        system=system_prompt,
                        response_model=table_model,
                        allow_partial=False,
                    )
                    table = result.relevant_table
                    step.stream(f"{result.chain_of_thought}\n\nSelected table: {table}")
                else:
                    table = tables[0]
                    step.stream(f"Selected table: {table}")

        if table in tables_to_source:
            source = tables_to_source[table]
        else:
            sources = [src for src in available_sources if table in src]
            source = sources[0] if sources else self._memory["current_source"]

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
            last_query = self.interface.serialize()[-1]["content"].replace("```sql", "").rstrip("```").strip()
            errors = '\n'.join(errors)
            messages += [
                {
                    "role": "user",
                    "content": (
                        f"Your last query `{last_query}` did not work as intended, "
                        f"expertly revise, and please do not repeat these issues:\n{errors}\n\n"
                        f"If the error is `syntax error at or near \")\"`, double check you used "
                        "table names verbatim, i.e. `read_parquet('table_name.parq')` instead of `table_name`."
                    )
                }
            ]

        with self.interface.add_step(title=title or "SQL query", steps_layout=self._steps_layout) as step:
            response = self.llm.stream(messages, system=system, response_model=Sql)
            sql_query = None
            async for output in response:
                step_message = output.chain_of_thought
                if output.query:
                    sql_query = clean_sql(output.query)
                    step_message += f"\n```sql\n{sql_query}\n```"
                step.stream(step_message, replace=True)

        if not sql_query:
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
                if "//" in renamed_table:
                    # Remove source prefixes from table names
                    renamed_table = re.sub(r"//[^/]+//", "", renamed_table)
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
        except InstructorRetryException as e:
            error_msg = str(e)
            step.stream(f'\n```python\n{error_msg}\n```')
            if len(error_msg) > 50:
                error_msg = error_msg[:50] + "..."
            step.failed_title = error_msg
            step.status = "failed"
            if e.n_attempts > 1:
                # Show the last error message
                step.stream(f'\n```python\n{e.messages[-1]["content"]}\n```')
            elif e.n_attempts > 2:
                raise e
        except Exception as e:
            report_error(e, step)
            raise e

        try:
            df = await get_data(pipeline)
        except Exception as e:
            source._connection.execute(f'DROP TABLE IF EXISTS "{expr_slug}"')
            report_error(e, step)
            raise e

        if len(df) > 0:
            self._memory["current_data"] = await describe_data(df)

        self._memory["available_sources"].append(sql_expr_source)
        self._memory["current_source"] = sql_expr_source
        self._memory["current_pipeline"] = pipeline
        self._memory["current_table"] = pipeline.table
        self._memory["current_sql"] = sql_query
        return sql_query

    async def _check_join_required(
        self,
        messages: list[Message],
        schema,
        table: str
    ):
        with self.interface.add_step(title="Checking if join is required", steps_layout=self._steps_layout) as step:
            join_prompt = render_template(
                "join_required.jinja2",
                schema=yaml.dump(schema),
                table=table
            )
            response = self.llm.stream(
                messages[-1:],
                system=join_prompt,
                response_model=JoinRequired,
            )
            async for output in response:
                step.stream(output.chain_of_thought, replace=True)
            join_required = output.join_required
            step.success_title = 'Query requires join' if join_required else 'No join required'
        return join_required

    async def find_join_tables(self, messages: list):
        multi_source = len(self._memory['available_sources']) > 1
        if multi_source:
            available_tables = [
                f"//{a_source}//{a_table}" for a_source in self._memory["available_sources"]
                for a_table in a_source.get_tables()
            ]
        else:
            available_tables = self._memory['current_source'].get_tables()

        find_joins_prompt = render_template(
            "find_joins.jinja2",
            available_tables=available_tables
        )
        with self.interface.add_step(title="Determining tables required for join", steps_layout=self._steps_layout) as step:
            output = await self.llm.invoke(
                messages,
                system=find_joins_prompt,
                response_model=TableJoins,
            )
            join_tables = output.tables
            step.stream(
                f'{output.chain_of_thought}\nJoin requires following tables: {join_tables}',
                replace=True
            )
            step.success_title = 'Found tables required for join'

        tables_to_source = {}
        for source_table in join_tables:
            available_sources = self._memory["available_sources"]
            if multi_source:
                try:
                    _, a_source_name, a_table = source_table.split("//", maxsplit=2)
                except ValueError:
                    a_source_name, a_table = source_table.split("//", maxsplit=1)
                for source in available_sources:
                    if source.name == a_source_name:
                        a_source = source
                        break
                if a_table in tables_to_source:
                    a_table = f"//{a_source_name}//{a_table}"
            else:
                a_source = next(iter(available_sources))
                a_table = source_table

            tables_to_source[a_table] = a_source
        return tables_to_source

    async def respond(
        self,
        messages: list[Message],
        render_output: bool = False,
        step_title: str | None = None
    ) -> Any:
        """
        Steps:
        1. Retrieve the current source and table from memory.
        2. If the source lacks a `get_sql_expr` method, return `None`.
        3. Fetch the schema for the current table using `get_schema` without min/max values.
        4. Determine if a join is required by calling `_check_join_required`.
        5. If required, find additional tables via `find_join_tables`; otherwise, use the current source and table.
        6. For each source and table, get the schema and SQL expression, storing them in `table_schemas`.
        7. Render the SQL prompt using the table schemas, dialect, join status, and table.
        8. If a join is required, remove source/table prefixes from the last message.
        9. Construct the SQL query with `_create_valid_sql`.
        """
        table, source = await self._select_relevant_table(messages)
        if not hasattr(source, "get_sql_expr"):
            return None

        schema = await get_schema(source, table, include_min_max=False)
        join_required = await self._check_join_required(messages, schema, table)
        if join_required:
            tables_to_source = await self.find_join_tables(messages)
        else:
            tables_to_source = {table: source}

        table_schemas = {}
        for source_table, source in tables_to_source.items():
            if source_table == table:
                table_schema = schema
            else:
                table_schema = await get_schema(source, source_table, include_min_max=False)

            # Look up underlying table name
            table_name = source_table
            if (
                'tables' in source.param and
                isinstance(source.tables, dict) and
                'select ' not in source.tables[table_name].lower()
            ):
                table_name = source.tables[table_name]

            table_schemas[table_name] = {
                "schema": yaml.dump(table_schema),
                "sql": source.get_sql_expr(source_table)
            }

        dialect = source.dialect
        sql_prompt = render_template(
            "sql_query.jinja2",
            tables_sql_schemas=table_schemas,
            dialect=dialect,
            join_required=join_required,
            table=table
        )
        system = await self._system_prompt_with_context(messages[-1:], context=sql_prompt)
        if join_required:
            # Remove source prefixes message, e.g. //<source>//<table>
            messages[-1]["content"] = re.sub(r"//[^/]+//", "", messages[-1]["content"])
        sql_query = await self._create_valid_sql(messages, system, tables_to_source, step_title)
        pipeline = self._memory['current_pipeline']
        self._render_lumen(pipeline, spec=sql_query, render_output=render_output)
        return pipeline


class BaseViewAgent(LumenBaseAgent):

    requires = param.List(default=["current_pipeline"], readonly=True)

    provides = param.List(default=["current_plot"], readonly=True)

    async def _extract_spec(self, model: BaseModel):
        return dict(model)

    async def respond(
        self,
        messages: list[Message],
        render_output: bool = False,
        step_title: str | None = None
    ) -> Any:
        pipeline = self._memory["current_pipeline"]

        # Write prompts
        system_prompt = await self._system_prompt_with_context(messages)
        schema = await get_schema(pipeline, include_min_max=False)
        view_prompt = render_template(
            "plot_agent.jinja2",
            schema=yaml.dump(schema),
            table=pipeline.table,
            current_view=self._memory.get('current_view'),
            doc=self.view_type.__doc__.split("\n\n")[0]
        )
        print(f"{self.name} is being instructed that {view_prompt}.")

        # Query
        output = await self.llm.invoke(
            messages,
            system=system_prompt + view_prompt,
            response_model=self._get_model(schema),
        )
        spec = await self._extract_spec(output)
        chain_of_thought = spec.pop("chain_of_thought", None)
        if chain_of_thought:
            with self.interface.add_step(
                title=step_title or "Generating view...",
                steps_layout=self._steps_layout
            ) as step:
                step.stream(chain_of_thought)
        print(f"{self.name} settled on {spec=!r}.")
        self._memory["current_view"] = dict(spec, type=self.view_type)
        view = self.view_type(pipeline=pipeline, **spec)
        self._render_lumen(view, render_output=render_output)
        return view


class hvPlotAgent(BaseViewAgent):
    """
    Generates a plot of the data given a user prompt.

    If the user asks to plot, visualize or render the data this is your best best.
    """

    system_prompt = param.String(
        default="""
        Generate the plot the user requested.
        Note that `x`, `y`, `by` and `groupby` fields MUST ALL be unique columns;
        no repeated columns are allowed.
        Do not arbitrarily set `groupby` and `by` fields unless explicitly requested.
        If a histogram is requested, use `y` instead of `x`.
        If x is categorical or strings, prefer barh over bar, and use `y` for the values.
        """
    )

    view_type = hvPlotUIView

    @classmethod
    def _get_model(cls, schema):
        # Find parameters
        excluded = cls.view_type._internal_params + [
            "controls",
            "type",
            "source",
            "pipeline",
            "transforms",
            "download",
            "field",
            "selection_group",
        ]
        model = param_to_pydantic(cls.view_type, excluded=excluded, schema=schema, extra_fields={
            "chain_of_thought": (str, FieldInfo(description="Your thought process behind the plot.")),
        })
        return model[cls.view_type.__name__]

    async def _extract_spec(self, model):
        pipeline = self._memory["current_pipeline"]
        spec = {
            key: val for key, val in dict(model).items()
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
    """
    Generates a vega-lite specification of the plot the user requested.

    If the user asks to plot, visualize or render the data this is your best best.
    """

    system_prompt = param.String(default="""
        Generate the plot the user requested as a vega-lite specification.""")

    view_type = VegaLiteView

    _extensions = ('vega',)

    @classmethod
    def _get_model(cls, schema):
        return VegaLiteSpec

    async def _extract_spec(self, model: VegaLiteSpec):
        vega_spec = json.loads(model.json_spec)
        if "$schema" not in vega_spec:
            vega_spec["$schema"] = "https://vega.github.io/schema/vega-lite/v5.json"
        if "width" not in vega_spec:
            vega_spec["width"] = "container"
        if "height" not in vega_spec:
            vega_spec["height"] = "container"
        return {'spec': vega_spec, "sizing_mode": "stretch_both", "min_height": 500}


class AnalysisAgent(LumenBaseAgent):
    """
    Perform custom analyses on the data.
    """

    analyses = param.List([])

    requires = param.List(default=['current_pipeline'])

    provides = param.List(default=['current_view'])

    system_prompt = param.String(
        default=(
            "You are in charge of picking the appropriate analysis to perform on the data "
            "based on the user query and the available data.\n\nAvailable analyses include:\n\n"
        )
    )

    _output_type = AnalysisOutput

    async def _system_prompt_with_context(
        self,
        messages: list[Message],
        context: str = "",
        analyses: list[Analysis] = []
    ) -> str:
        system_prompt = self.system_prompt
        for name, analysis in analyses.items():
            if analysis.__doc__ is None:
                doc = analysis.__name__
            else:
                doc = analysis.__doc__.replace("\n", " ")
            system_prompt = f'- {name!r} - {doc}\n'
        current_data = self._memory["current_data"]
        if isinstance(current_data, dict):
            columns = list(current_data["stats"].keys())
        else:
            columns = list(current_data.columns)
        system_prompt += f"\nHere are the columns of the table: {columns}"
        if context:
            system_prompt += f"\n### CONTEXT: {context}".strip()
        return system_prompt

    async def respond(
        self,
        messages: list[Message],
        render_output: bool = False,
        step_title: str | None = None,
        agents: list[Agent] | None = None
    ) -> Any:
        pipeline = self._memory['current_pipeline']
        analyses = {a.name: a for a in self.analyses if await a.applies(pipeline)}
        if not analyses:
            print("NONE found...")
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
                system_prompt = await self._system_prompt_with_context(messages, analyses=analyses)
                analysis_name = (await self.llm.invoke(
                    messages,
                    system=system_prompt,
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
            self._memory["current_analysis"] = analysis_callable

            if analysis_callable.autorun:
                if asyncio.iscoroutinefunction(analysis_callable.__call__):
                    view = await analysis_callable(pipeline)
                else:
                    view = await asyncio.to_thread(analysis_callable, pipeline)
                spec = view.to_spec()
                if isinstance(view, View):
                    view_type = view.view_type
                    self._memory["current_view"] = dict(spec, type=view_type)
                elif isinstance(view, Pipeline):
                    self._memory["current_pipeline"] = view
                # Ensure current_data reflects processed pipeline
                if pipeline is not self._memory['current_pipeline']:
                    pipeline = self._memory['current_pipeline']
                    if len(data) > 0:
                        self._memory["current_data"] = await describe_data(data)
                yaml_spec = yaml.dump(spec)
                step.stream(f"Generated view\n```yaml\n{yaml_spec}\n```")
                step.success_title = "Generated view"
            else:
                step.success_title = "Configure the analysis"
                view = None

        analysis = self._memory["current_analysis"]
        pipeline = self._memory['current_pipeline']
        if view is None and analysis.autorun:
            self.interface.stream('Failed to find an analysis that applies to this data')
        else:
            self._render_lumen(
                view,
                analysis=analysis,
                pipeline=pipeline,
                render_output=render_output
            )
        return view
