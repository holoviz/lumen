import asyncio
import difflib
import io
import json
import textwrap

from typing import Literal, Optional, Type

import pandas as pd
import panel as pn
import param

from panel.chat import ChatInterface
from panel.pane import HTML
from panel.viewable import Viewer
from pydantic import BaseModel, create_model
from pydantic.fields import FieldInfo

from lumen.sources.duckdb import DuckDBSource

from ..base import Component
from ..dashboard import Config
from ..pipeline import Pipeline
from ..sources.intake_sql import IntakeBaseSQLSource
from ..state import state
from ..transforms.sql import SQLOverride, SQLTransform, Transform
from ..views import VegaLiteView, hvPlotUIView
from .embeddings import Embeddings
from .llm import Llm
from .memory import memory
from .models import (
    DataRequired, FuzzyTable, JoinRequired, Sql, TableJoins, Topic,
    VegaLiteSpec,
)
from .translate import param_to_pydantic
from .utils import (
    describe_data, get_schema, render_template, retry_llm_output,
)
from .views import LumenOutput, SQLOutput

FUZZY_TABLE_LENGTH = 10


class Agent(Viewer):
    """
    An Agent in Panel is responsible for handling a specific type of
    query. Each agent consists of an LLM and optionally a set of
    embeddings.
    """

    debug = param.Boolean(default=False)

    embeddings = param.ClassSelector(class_=Embeddings)

    interface = param.ClassSelector(class_=ChatInterface)

    llm = param.ClassSelector(class_=Llm)

    system_prompt = param.String()

    response_model = param.ClassSelector(class_=BaseModel, is_instance=False)

    user = param.String(default="Agent")

    requires = param.List(default=[], readonly=True)

    provides = param.List(default=[], readonly=True)

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
            params["interface"] = ChatInterface(callback=self._chat_invoke)
        super().__init__(**params)
        if not self.debug:
            pn.config.exception_handler = _exception_handler

        if state.config is None:
            state.config = Config(raise_with_notifications=True)
        else:
            state.config.raise_with_notifications = True

    async def _chat_invoke(self, contents: list | str, user: str, instance: ChatInterface):
        await self.invoke(contents)
        self._retries_left = 1

    def __panel__(self):
        return self.interface

    async def _system_prompt_with_context(
        self, messages: list | str, context: str = ""
    ) -> str:
        system_prompt = self.system_prompt
        if self.embeddings:
            context = self.embeddings.query(messages)
        if context:
            system_prompt += f"{system_prompt}\n### CONTEXT: {context}".strip()
        return system_prompt

    async def _get_closest_tables(self, messages: list | str, tables: list[str], n: int = 3) -> list[str]:
        system = (
            f"You are great at extracting keywords based on the user query to find the correct table. "
            f"The current table selected: `{memory.get('current_table', 'N/A')}`. "
        )
        tables = tuple(table.replace('"', "") for table in tables)

        fuzzy_table = (await self.llm.invoke(
            messages,
            system=system,
            response_model=FuzzyTable,
            allow_partial=False
        ))
        if not fuzzy_table.required:
            return [memory.get("current_table") or tables[0]]

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
        memory["closest_tables"] = tables
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

    async def requirements(self, messages: list | str):
        return self.requires

    async def invoke(self, messages: list | str):
        system_prompt = await self._system_prompt_with_context(messages)

        message = None
        async for output in self.llm.stream(
            messages, system=system_prompt, response_model=self.response_model, field="output"
        ):
            message = self.interface.stream(
                output, replace=True, message=message, user=self.user
            )


class SourceAgent(Agent):
    """
    The SourceAgent allows a user to provide an input source.

    Should only be used if the user explicitly requests adding a source
    or no source is in memory.
    """

    requires = param.List(default=[], readonly=True)

    provides = param.List(default=["current_source"], readonly=True)

    on_init = param.Boolean(default=True)

    async def answer(self, messages: list | str):
        def add_table(event):
            if input_tabs.active == 0 and upload.value:
                if upload.filename.endswith("csv"):
                    df = pd.read_csv(io.BytesIO(upload.value), parse_dates=True)
                elif upload.filename.endswith((".parq", ".parquet")):
                    df = pd.read_parquet(io.BytesIO(upload.value))
                duckdb_source._connection.from_df(df).to_table(name.value)
                duckdb_source.tables = [name.value]
                memory["current_source"] = duckdb_source
            # TODO: add URL + xlsx support
            name.value = ""
            upload.value = None
            upload.filename = ""
            src = memory["current_source"]
            tables[:] = [
                (t, pn.widgets.Tabulator(src.get(t), sizing_mode="stretch_width")) for t in src.get_tables()
            ]
            input_column.visible = False

        def add_name(event):
            if "/" in event.new:
                name.value = event.new.split("/")[-1].split(".")[0].replace("-", "_")
            elif not name.value and event.new:
                name.value = event.new.split(".")[0].replace("-", "_")
                name.visible = True

        def enable_add(event):
            add.visible = not bool(name.value)

        name = pn.widgets.TextInput(name="Name your table", visible=False)
        upload = pn.widgets.FileInput(align="end", accept=".csv,.parquet,.parq")
        add = pn.widgets.Button(
            name="Add table",
            icon="table-plus",
            visible=False,
            button_type="success",
        )
        input_tabs = pn.Tabs(("Upload", upload), sizing_mode="stretch_width")
        tables = pn.Tabs(sizing_mode="stretch_width")
        duckdb_source = DuckDBSource(uri=":memory:")
        input_column = pn.Column(
            input_tabs,
            name,
            add,
            sizing_mode="stretch_width",
        )
        menu = pn.Column(
            input_column,
            tables,
            sizing_mode="stretch_width",
        )
        upload.param.watch(add_name, "filename")
        upload.param.watch(enable_add, "value")
        add.on_click(add_table)

        self.interface.send(menu, respond=False, user="SourceAgent")
        while not add.clicks:
            await asyncio.sleep(0.05)

    async def invoke(self, messages: list[str] | str):
        await self.answer(messages)


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
            "Provide suggestions to get started if necessary."
        )
    )

    response_model = param.ClassSelector(class_=BaseModel, is_instance=False)

    requires = param.List(default=["current_source"], readonly=True)

    @retry_llm_output()
    async def requirements(self, messages: list | str, errors=None):
        if 'current_data' in memory:
            return self.requires

        with self.interface.add_step(title="Checking if data is required") as step:
            response = self.llm.stream(
                messages,
                system=(
                    "The user may or may not want to chat about a particular dataset. "
                    "Determine whether the provided user prompt requires access to "
                    "actual data. If they're only searching for one, it's not required."
                ),
                response_model=DataRequired,
            )
            async for output in response:
                step.stream(output.chain_of_thought, replace=True)
            if output.data_required:
                return self.requires + ['current_table']
        return self.requires

    async def _system_prompt_with_context(
        self, messages: list | str, context: str = ""
    ) -> str:
        source = memory.get("current_source")
        if not source:
            raise ValueError("No source found in memory.")
        tables = source.get_tables()
        if len(tables) > 1:
            if len(tables) > FUZZY_TABLE_LENGTH and "closest_tables" not in memory:
                closest_tables = await self._get_closest_tables(messages, tables, n=5)
            else:
                closest_tables = memory.get("closest_tables", tables)
            context = f"Available tables: {', '.join(closest_tables)}"
        else:
            memory["current_table"] = table = memory.get("current_table", tables[0])
            schema = get_schema(memory["current_source"], table)
            if schema:
                context = f"{table} with schema: {schema}"

        if "current_data" in memory:
            context += (
                f"\nHere's a summary of the dataset the user just asked about:\n```\n{memory['current_data']}\n```"
            )

        system_prompt = self.system_prompt
        if context:
            system_prompt += f"{system_prompt}\n### CONTEXT: {context}".strip()
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
            "between the columns and what could be interesting to dive deeper into."
        )
    )

    async def _system_prompt_with_context(
        self, messages: list | str, context: str = ""
    ) -> str:
        system_prompt = self.system_prompt
        topic = (await self.llm.invoke(
            messages,
            system="What is the topic of the table?",
            response_model=Topic,
            allow_partial=False,
        )).result
        context += f"Topic you are a world-class expert on: {topic}"

        current_data = memory["current_data"]
        if isinstance(current_data, dict):
            columns = list(current_data["stats"].keys())
        else:
            columns = list(current_data.columns)
        context += f"\nHere are the columns of the table: {columns}"

        if context:
            system_prompt += f"{system_prompt}\n### CONTEXT: {context}".strip()
        return system_prompt


class LumenBaseAgent(Agent):

    user = param.String(default="Lumen")

    def _render_lumen(self, component: Component, message: pn.chat.ChatMessage = None):
        out = LumenOutput(component=component)
        message_kwargs = dict(value=out, user=self.user)
        self.interface.stream(message=message, **message_kwargs, replace=True)


class TableAgent(LumenBaseAgent):
    """
    Displays a single table / dataset. Does not discuss.
    """

    system_prompt = param.String(
        default="You are an agent responsible for finding the correct table based on the user prompt."
    )

    requires = param.List(default=["current_source"], readonly=True)

    provides = param.List(default=["current_table", "current_pipeline"], readonly=True)

    @staticmethod
    def _create_table_model(tables):
        table_model = create_model("Table", table=(Literal[tables], FieldInfo(
            description="The most relevant table based on the user query; if none are relevant, select the first."
        )))
        return table_model

    async def answer(self, messages: list | str):
        tables = tuple(memory["current_source"].get_tables())
        if len(tables) == 1:
            table = tables[0]
        else:
            with self.interface.add_step(title="Choosing the most relevant table...") as step:
                closest_tables = memory.pop("closest_tables", [])
                if closest_tables:
                    tables = closest_tables
                elif len(tables) > FUZZY_TABLE_LENGTH:
                    tables = await self._get_closest_tables(messages, tables)
                system_prompt = await self._system_prompt_with_context(messages)
                if self.debug:
                    print(f"{self.name} is being instructed that it should {system_prompt}")
                if len(tables) > 1:
                    table_model = self._create_table_model(tables)
                    result = await self.llm.invoke(
                        messages,
                        system=system_prompt,
                        response_model=table_model,
                        allow_partial=False,
                    )
                    table = result.table
                else:
                    table = tables[0]
                step.stream(f"Selected table: {table}")
        memory["current_table"] = table
        memory["current_pipeline"] = pipeline = Pipeline(
            source=memory["current_source"], table=table
        )
        df = pipeline.__panel__()[-1].value
        if len(df) > 0:
            memory["current_data"] = describe_data(df)
        if self.debug:
            print(f"{self.name} thinks that the user is talking about {table=!r}.")
        return pipeline

    async def invoke(self, messages: list | str):
        pipeline = await self.answer(messages)
        self._render_lumen(pipeline)


class TableListAgent(LumenBaseAgent):
    """
    List all of the available tables or datasets inventory. Not useful
    if the user requests a specific table.
    """

    system_prompt = param.String(
        default="You are an agent responsible for listing the available tables in the current source."
    )

    requires = param.List(default=["current_source"], readonly=True)

    async def answer(self, messages: list | str):
        source = memory["current_source"]
        tables = memory["current_source"].get_tables()
        if not tables:
            return
        if isinstance(source, IntakeBaseSQLSource) and hasattr(
            source.cat, "_repr_html_"
        ):
            table_listing = HTML(
                textwrap.dedent(source.cat._repr_html_()),
                margin=10,
                styles={
                    "overflow": "auto",
                    "background-color": "white",
                    "padding": "10px",
                },
                tags=['catalog']
            )
        else:
            tables = tuple(table.replace('"', "") for table in tables)
            table_bullets = "\n".join(f"- {table}" for table in tables)
            table_listing = f"Available tables:\n{table_bullets}"
        self.interface.add_step(table_listing, success_title="Table List", status="success")
        return tables

    async def invoke(self, messages: list | str):
        await self.answer(messages)


class SQLAgent(LumenBaseAgent):
    """
    Responsible for generating and modifying SQL queries to answer user queries about the data,
    such querying subsets of the data, aggregating the data and calculating results.
    """

    system_prompt = param.String(
        default=textwrap.dedent("""
        You are an agent responsible for writing a SQL query that will
        perform the data transformations the user requested.
        """)
    )

    requires = param.List(default=["current_table", "current_source"], readonly=True)

    provides = param.List(default=["current_sql", "current_pipeline"], readonly=True)

    def _render_sql(self, query):
        pipeline = memory['current_pipeline']
        out = SQLOutput(component=pipeline, spec=query)
        self.interface.stream(out, user="SQL", replace=True)
        return out

    @retry_llm_output()
    async def _create_valid_sql(self, messages, system, source, tables, errors=None):
        if errors:
            last_query = self.interface.serialize()[-1]["content"].replace("```sql", "").rstrip("```").strip()
            errors = '\n'.join(errors)
            messages += [
                {
                    "role": "user",
                    "content": (
                        f"Your last query `{last_query}` did not work as intended, "
                        f"expertly revise, and please do not repeat these issues:\n{errors}")
                }
            ]

        with self.interface.add_step(title="Creating SQL query...", success_title="SQL Query") as step:
            response = self.llm.stream(messages, system=system, response_model=Sql)
            sql_query = None
            async for output in response:
                step_message = output.chain_of_thought
                if output.query:
                    sql_query = output.query.replace("```sql", "").replace("```", "").strip()
                    step_message += f"\n```sql\n{sql_query}\n```"
                step.stream(step_message, replace=True)

        if not sql_query:
            raise ValueError("No SQL query was generated.")

        # check whether the SQL query is valid
        expr_name = output.expr_name
        sql_expr_source = source.create_sql_expr_source({expr_name: sql_query})
        pipeline = Pipeline(source=sql_expr_source, table=expr_name)
        df = pipeline.data
        if len(df) > 0:
            memory["current_data"] = describe_data(df)
        memory["current_pipeline"] = pipeline
        return sql_query

    async def answer(self, messages: list | str):
        source = memory["current_source"]
        table = memory["current_table"]

        if not hasattr(source, "get_sql_expr"):
            return None

        with self.interface.add_step(title="Checking if join is required") as step:
            response = self.llm.stream(
                messages,
                system="Determine whether a table join is required to answer the user's query.",
                response_model=JoinRequired,
            )
            async for output in response:
                step.stream(output.chain_of_thought, replace=True)
            join_required = output.join_required
            step.success_title = 'Query requires join' if join_required else 'No join required'

        if join_required:
            available_tables = " ".join(str(table) for table in source.get_tables())
            with self.interface.add_step(title="Determining tables required for join") as step:
                output = await self.llm.invoke(
                    messages,
                    system=f"List the tables that need to be joined: {available_tables}.",
                    response_model=TableJoins,
                )
                tables = output.tables
                step.stream(f'\nJoin requires following tables: {tables}', replace=True)
                step.success_title = 'Found tables required for join'
        else:
            tables = [table]

        tables_sql_schemas = {
            table: {
                "schema": get_schema(source, table, include_min_max=False),
                "sql": source.get_sql_expr(table)
            } for table in tables
        }
        dialect = source.dialect
        sql_prompt = render_template(
            "sql_query.jinja2",
            tables_sql_schemas=tables_sql_schemas,
            dialect=dialect,
        )
        system = await self._system_prompt_with_context(messages) + sql_prompt
        sql_query = await self._create_valid_sql(messages, system, source, tables)
        memory["current_sql"] = sql_query
        return sql_query

    async def invoke(self, messages: list | str):
        sql_query = await self.answer(messages)
        self._render_sql(sql_query)


class PipelineAgent(LumenBaseAgent):
    """
    Generates a data pipeline by applying SQL to the data.

    If the user asks to calculate, aggregate, select or perform some other operation on the data this is your best best.
    """

    requires = param.List(default=["current_table", "current_sql"], readonly=True)

    provides = param.List(default=["current_pipeline"], readonly=True)

    async def answer(self, messages: list | str) -> Transform:
        if "current_pipeline" in memory:
            pipeline = memory["current_pipeline"]
        else:
            pipeline = Pipeline(
                source=memory["current_source"],
                table=memory["current_table"],
            )
        memory["current_pipeline"] = pipeline
        pipeline._update_data(force=True)
        memory["current_data"] = describe_data(pipeline.data)
        return pipeline

    async def invoke(self, messages: list | str):
        pipeline = await self.answer(messages)
        self._render_lumen(pipeline)


class TransformPipelineAgent(LumenBaseAgent):
    """
    Generates a data pipeline by applying transformations to the data.

    If the user asks to do Root Cause Analysis (RCA) or outlier detection, this is your best bet.
    """

    requires = param.List(default=["current_table"], readonly=True)

    system_prompt = param.String(
        default=(
            "Choose the appropriate data transformation based on the query. "
            "For `contains duplicate entries, cannot reshape`, "
            "be sure to perform an `aggregate` transform first, ensuring `with_index=False` is set."
        )
    )

    requires = param.List(default=["current_table"], readonly=True)

    provides = param.List(default=["current_pipeline"], readonly=True)

    @property
    def _available_transforms(self) -> dict[str, Type[Transform]]:
        transforms = param.concrete_descendents(Transform)
        return {
            name: transform
            for name, transform in transforms.items()
            if not issubclass(transform, SQLTransform)
        }

    def _transform_prompt(
        self, model: BaseModel, transform: Transform, table: str, schema: dict
    ) -> str:
        prompt = f"{transform.__doc__}"
        if not schema:
            raise ValueError(f"No schema found for table {table!r}")
        prompt += f"\n\nThe data columns follows the following JSON schema:\n\n```json\n{schema}\n```"
        if "current_transform" in memory:
            prompt += f"The previous transform specification was: {memory['current_transform']}"
        return prompt

    @staticmethod
    def _create_transform_model(transforms):
        transform_model = create_model(
            "Transform",
            summary_of_query=(
                str,
                FieldInfo(
                    description="A summary of the query and whether you think a transform is needed."
                ),
            ),
            transform_required=(
                bool,
                FieldInfo(
                    description="Whether transforms are required for the query; sometimes the user may not need a transform.",
                    default=True,
                ),
            ),
            transforms=(Optional[list[Literal[tuple(transforms)]]], None),
        )
        return transform_model

    async def _find_transform(
        self, messages: list | str, system_prompt: str
    ) -> Type[Transform] | None:
        picker_prompt = render_template(
            "pick_transform.jinja2", transforms=self._available_transforms.values()
        )
        transforms = self._available_transforms
        transform_model = self._create_transform_model(transforms)
        transform = await self.llm.invoke(
            messages,
            system=f"{system_prompt}\n{picker_prompt}",
            response_model=transform_model,
            allow_partial=False
        )

        if transform and transform.transform_required:
            transform = await self.llm.invoke(
                f"are the transforms, {transform.transforms!r} partially relevant to the query {messages!r}",
                system=(
                    f"You are a world class validation model. "
                    f"Capable to determine if the following value is valid for the statement, "
                    f"if it is not, explain why and suggest a new value from {picker_prompt}"
                ),
                response_model=transform_model,
                allow_partial=False
            )

        if transform is None or not transform.transform_required:
            print(f"No transform found for {messages}")
            return None

        transform_names = transform.transforms
        if transform_names:
            return [transforms[transform_name.strip("'")] for transform_name in transform_names]

    async def _construct_transform(
        self, messages: list | str, transform: Type[Transform], system_prompt: str
    ) -> Transform:
        excluded = transform._internal_params + ["controls", "type"]
        schema = get_schema(memory["current_pipeline"])
        table = memory["current_table"]
        model = param_to_pydantic(transform, excluded=excluded, schema=schema)[
            transform.__name__
        ]
        transform_prompt = self._transform_prompt(model, transform, table, schema)
        if self.debug:
            print(f"{self.name} recalls that {transform_prompt}.")
        kwargs = await self.llm.invoke(
            messages,
            system=system_prompt + transform_prompt,
            response_model=model,
            allow_partial=False,
        )
        return transform(**dict(kwargs))

    async def answer(self, messages: list | str) -> Transform:
        system_prompt = await self._system_prompt_with_context(messages)
        transform_types = await self._find_transform(messages, system_prompt)

        if "current_pipeline" in memory:
            pipeline = memory["current_pipeline"]
        else:
            pipeline = Pipeline(
                source=memory["current_source"],
                table=memory["current_table"],
            )
        pipeline.sql_transforms = [SQLOverride(override=memory["current_sql"])]
        memory["current_pipeline"] = pipeline

        if not transform_types and pipeline:
            return pipeline

        for transform_type in transform_types:
            transform = await self._construct_transform(
                messages, transform_type, system_prompt
            )
            memory["current_transform"] = spec = transform.to_spec()
            if self.debug:
                print(f"{self.name} settled on {spec=!r}.")
            if pipeline.transforms and type(pipeline.transforms[-1]) is type(transform):
                pipeline.transforms[-1] = transform
            else:
                pipeline.add_transform(transform)

        pipeline._update_data(force=True)
        memory["current_data"] = describe_data(pipeline.data)
        return pipeline

    async def invoke(self, messages: list | str):
        pipeline = await self.answer(messages)
        self._render_lumen(pipeline)


class BaseViewAgent(LumenBaseAgent):

    requires = param.List(default=["current_pipeline"], readonly=True)

    provides = param.List(default=["current_plot"], readonly=True)

    def _extract_spec(self, model: BaseModel):
        return dict(model)

    async def answer(self, messages: list | str) -> hvPlotUIView:
        pipeline = memory["current_pipeline"]

        # Write prompts
        system_prompt = await self._system_prompt_with_context(messages)
        schema = get_schema(pipeline, include_min_max=False)
        view_prompt = render_template(
            "plot_agent.jinja2",
            schema=schema,
            table=pipeline.table,
            current_view=memory.get('current_view'),
            doc=self.view_type.__doc__.split("\n\n")[0]
        )
        print(f"{self.name} is being instructed that {view_prompt}.")

        # Query
        output = await self.llm.invoke(
            messages,
            system=system_prompt + view_prompt,
            response_model=self._get_model(schema),
        )
        spec = self._extract_spec(output)
        chain_of_thought = spec.pop("chain_of_thought")
        with self.interface.add_step(title="Generating view...") as step:
            step.stream(chain_of_thought)
        print(f"{self.name} settled on {spec=!r}.")
        memory["current_view"] = dict(spec, type=self.view_type)
        return self.view_type(pipeline=pipeline, **spec)

    async def invoke(self, messages: list | str):
        view = await self.answer(messages)
        self._render_lumen(view)


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

    def _extract_spec(self, model):
        pipeline = memory["current_pipeline"]
        spec = {
            key: val for key, val in dict(model).items()
            if val is not None
        }
        spec["type"] = "hvplot_ui"
        self.view_type.validate(spec)
        spec.pop("type", None)

        # Add defaults
        spec["responsive"] = True
        if len(pipeline.data) > 20000 and spec["kind"] in ("line", "scatter", "points"):
            spec["rasterize"] = True
            spec["cnorm"] = "log"
        return spec


class VegaLiteAgent(BaseViewAgent):
    """
    Generates a vega-lite specification of the plot the user requested.

    If the user asks to plot, visualize or render the data this is your best best.
    """

    system_prompt = param.String(
        default="""
        Generate the plot the user requested as a vega-lite specification.
        """
    )
    view_type = VegaLiteView

    @classmethod
    def _get_model(cls, schema):
        return VegaLiteSpec

    def _extract_spec(self, model):
        vega_spec = json.loads(model.json_spec)
        if "$schema" not in vega_spec:
            vega_spec["$schema"] = "https://vega.github.io/schema/vega-lite/v5.json"
        if "width" not in vega_spec:
            vega_spec["width"] = "container"
        if "height" not in vega_spec:
            vega_spec["height"] = "container"
        return {'spec': vega_spec, "sizing_mode": "stretch_both", "min_height": 500}


class Analysis(param.ParameterizedFunction):
    """
    The purpose of an Analysis is to perform custom actions tailored
    to a particular use case and/or domain. It is a ParameterizedFunction
    which implements an applies method to confirm whether the analysis
    applies to a particular dataset and a __call__ method which is given
    a pipeline and must return another valid component.
    """

    columns = param.List(default=[], doc="The columns required for the analysis.", readonly=True)

    @classmethod
    def applies(cls, pipeline) -> bool:
        return all(col in pipeline.data.columns for col in cls.columns)

    def __call__(self, pipeline) -> Component:
        return pipeline


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

    async def _system_prompt_with_context(
        self, messages: list | str, analyses: list[Analysis] = []
    ) -> str:
        system_prompt = self.system_prompt
        for name, analysis in analyses.items():
            if analysis.__doc__ is None:
                doc = analysis.__name__
            else:
                doc = analysis.__doc__.replace("\n", " ")
            system_prompt = f'- {name!r} - {doc}\n'
        current_data = memory["current_data"]
        if isinstance(current_data, dict):
            columns = list(current_data["stats"].keys())
        else:
            columns = list(current_data.columns)
        system_prompt += f"\nHere are the columns of the table: {columns}"
        return system_prompt

    async def answer(self, messages: list | str):
        pipeline = memory['current_pipeline']
        analyses = {a.name: a for a in self.analyses if a.applies(pipeline)}
        if not analyses:
            print("NONE found...")
            return None

        with self.interface.add_step(title="Choosing the most relevant analysis...") as step:
            if len(analyses) > 1:
                type_ = Literal[tuple(analyses)]
                analysis_model = create_model(
                    "Analysis",
                    correct_name=(type_, FieldInfo(description="The name of the analysis that is most appropriate given the user query."))
                )
                system_prompt = await self._system_prompt_with_context(messages, analyses)
                analysis_name = (await self.llm.invoke(
                    messages,
                    system=system_prompt,
                    response_model=analysis_model,
                    allow_partial=False,
                )).correct_name
            else:
                analysis_name = list(analyses)[0]
            step.stream(f"Selected {analysis_name}")
            step.success_title = f"Selected {analysis_name}"

        with self.interface.add_step(title="Creating view...") as step:
            print(f"Creating view for {analysis_name}")
            await asyncio.sleep(0.1)  # necessary to give it time to render before calling sync function...
            analysis_callable = analyses[analysis_name]
            if asyncio.iscoroutinefunction(analysis_callable):
                view = await analysis_callable(pipeline)
            else:
                view = await asyncio.to_thread(analysis_callable, pipeline)
            spec = view.to_spec()
            step.stream(f"Generated view\n```json\n{spec}\n```")
            step.success_title = "Generated view"
        view_type = view.view_type if hasattr(view, "view_type") else  type(view)
        memory["current_view"] = dict(spec, type=view_type)
        return view

    async def invoke(self, messages: list | str):
        view = await self.answer(messages)
        if view is None:
            self.interface.stream('Failed to find an analysis that applies to this data')
        else:
            self._render_lumen(view)
