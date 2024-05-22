import asyncio
import io
import textwrap

from typing import Literal, Optional, Type

import pandas as pd
import panel as pn
import param
import yaml

from panel.chat import ChatInterface
from panel.pane import HTML
from panel.viewable import Viewer
from pydantic import BaseModel, create_model
from pydantic.fields import FieldInfo

from ..base import Component
from ..dashboard import Config, load_yaml
from ..pipeline import Pipeline
from ..sources import FileSource, InMemorySource, Source
from ..sources.intake_sql import IntakeBaseSQLSource
from ..state import state
from ..transforms.sql import SQLOverride, SQLTransform, Transform
from ..views import hvPlotUIView
from .embeddings import Embeddings
from .llm import Llm
from .memory import memory
from .models import DataRequired, Sql
from .translate import param_to_pydantic

FUZZY_TABLE_LENGTH = 1

def format_schema(schema):
    formatted = {}
    for field, spec in schema.items():
        if "enum" in spec and len(spec["enum"]) > 5:
            spec["enum"] = spec["enum"][:5] + ["..."]
        formatted[field] = spec
    return formatted


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

    user = param.String(default="Assistant")

    requires = param.List(default=[], readonly=True)

    provides = param.List(default=[], readonly=True)

    __abstract = True

    def __init__(self, **params):
        self._retries_left = 1
        def _exception_handler(exception):
            import traceback
            tb = traceback.format_exc()
            print(tb)
            if self._retries_left > 0:
                self._retries_left -= 1
                self.interface.send(
                    f"Attempting to fix: {exception}.",
                    user="Exception",
                )
            else:
                self.interface.send(
                    f"Error cannot be resolved: {exception}.",
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

    def _link_code_editor(self, value, callback, language):
        code_editor = pn.widgets.CodeEditor(
            value=value, language=language, sizing_mode="stretch_both", min_height=300
        )
        copy_icon = pn.widgets.ButtonIcon(
            icon="copy", active_icon="check", toggle_duration=1000
        )
        copy_icon.js_on_click(
            args={"code_editor": code_editor},
            code="navigator.clipboard.writeText(code_editor.code);",
        )
        download_icon = pn.widgets.ButtonIcon(
            icon="download", active_icon="check", toggle_duration=1000
        )
        download_icon.js_on_click(
            args={"code_editor": code_editor},
            code="""
            var text = code_editor.code;
            var blob = new Blob([text], {type: 'text/plain'});
            var url = window.URL.createObjectURL(blob);
            var a = document.createElement('a');
            a.href = url;
            a.download = 'lumen_spec.yaml';
            document.body.appendChild(a); // we need to append the element to the dom -> otherwise it will not work in firefox
            a.click();
            a.parentNode.removeChild(a);  //afterwards we remove the element again
            """,
        )
        icons = pn.Row(copy_icon, download_icon)
        code_col = pn.Column(code_editor, icons, sizing_mode="stretch_both")
        placeholder = pn.Column(sizing_mode="stretch_both")
        tabs = pn.Tabs(
            ("Code", code_col),
            ("Output", placeholder),
            styles={'min-width': "100%"}
        )
        placeholder.objects = [
            pn.bind(callback, code_editor.param.value, tabs.param.active)
        ]
        return tabs

    async def _chat_invoke(self, contents: list | str, user: str, instance: ChatInterface):
        await self.invoke(contents)
        self._retries_left = 1

    def __panel__(self):
        return self.interface

    async def _system_prompt_with_context(
        self, messages: list | str, context: str = ""
    ) -> str:
        system_prompt = self.system_prompt
        if context:
            system_prompt += f"{system_prompt}\n### CONTEXT: {context}".strip()
        return system_prompt

    def _get_schema(self, source: Source | Pipeline, table: str = None):
        if isinstance(source, Pipeline):
            schema = source.get_schema()
        else:
            schema = source.get_schema(table, limit=100)
        schema = dict(schema)
        for field, spec in schema.items():
            if "inclusiveMinimum" in spec:
                spec["min"] = spec.pop("inclusiveMinimum")
            if "inclusiveMaximum" in spec:
                spec["max"] = spec.pop("inclusiveMaximum")
        return schema


    async def _get_closest_tables(self, messages: list | str, source: Source, tables: list[str], n: int = 3) -> list[str]:
        if isinstance(messages, list):
            query_text = messages[-1]["content"]
        else:
            query_text = messages
        tables = [str(table) for table in tables]
        self.embeddings.client.get_or_create_collection(source.name)
        self.embeddings.add_documents(tables, ids=tables, upsert=True)
        closest_tables = self.embeddings.query(query_text)
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
        async for chunk in self.llm.stream(
            messages, system=system_prompt, response_model=self.response_model
        ):
            message = self.interface.stream(
                chunk, replace=True, message=message, user=self.user
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

    async def invoke(self, messages: list[str] | str):
        name = pn.widgets.TextInput(name="Name your table", align="center")
        upload = pn.widgets.FileInput(align="end")
        url = pn.widgets.TextInput(name="Add a file from a URL")
        add = pn.widgets.Button(
            name="Add table",
            icon="table-plus",
            disabled=True,
            align="center",
            button_type="success",
        )
        tables = pn.Tabs(sizing_mode="stretch_width")
        mem_source = InMemorySource()
        file_source = FileSource()

        def add_table(event):
            if upload.value:
                if upload.filename.endswith("csv"):
                    df = pd.read_csv(
                        io.StringIO(upload.value.decode("utf-8")), parse_dates=True
                    )
                    for col in ("date", "Date"):
                        if col in df.columns:
                            df[col] = pd.to_datetime(df[col])
                elif upload.filename.endswith((".parq", ".parquet")):
                    df = pd.read_parquet(io.BytesIO(upload.value.decode("utf-8")))
                mem_source.add_table(name.value, df)
                memory["current_source"] = mem_source
            elif url.value:
                file_source.tables[name.value] = url.value
                memory["current_source"] = file_source
            name.value = ""
            url.value = ""
            upload.value = None
            src = memory["current_source"]
            tables[:] = [
                (t, pn.widgets.Tabulator(src.get(t))) for t in src.get_tables()
            ]

        def add_name(event):
            if not name.value and event.new:
                name.value = event.new.split(".")[0]

        upload.param.watch(add_name, "filename")

        def enable_add(event):
            add.disabled = not bool(name.value and (upload.value or url.value))

        name.param.watch(enable_add, "value")
        upload.param.watch(enable_add, "value")
        url.param.watch(enable_add, "value")
        add.on_click(add_table)
        menu = pn.Column(
            pn.Row(
                name,
                pn.Tabs(("Upload", upload), ("URL", url)),
                add,
                sizing_mode="stretch_width",
            ),
            tables,
        )
        self.interface.send(menu, respond=False, user="SourceAgent")


class ChatAgent(Agent):
    """
    Responsible for chatting and providing info about high level data related topics,
    e.g. what datasets are available, the columns of the data or
    statistics about the data, and continuing the conversation, or finding
    the closest tables based on the user query.

    Is capable of providing suggestions to get started.
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

    async def requirements(self, messages: list | str):
        if 'current_data' in memory:
            return self.requires

        for i in range(3):
            result = await self.llm.invoke(
                messages,
                system=(
                    "The user may or may not want to chat about a particular dataset. "
                    "Determine whether the provided user prompt requires access to "
                    "actual data. If they're only searching for one, it's not required."
                ),
                response_model=DataRequired,
                allow_partial=False,
            )
            if result is None:
                continue
            elif result.data_required:
                return self.requires + ['current_table']
            else:
                break
        return self.requires

    async def _system_prompt_with_context(
        self, messages: list | str, context: str = ""
    ) -> str:
        source = memory.get("current_source")
        tables = source.get_tables() if source else []
        if len(tables) > 1:
            if len(tables) > FUZZY_TABLE_LENGTH and "closest_tables" not in memory:
                closest_tables = await self._get_closest_tables(messages, source, tables, n=5)
            else:
                closest_tables = memory.get("closest_tables", tables)
            context = f"Available tables: {', '.join(closest_tables)}"
        else:
            memory["current_table"] = table = memory.get("current_table", tables[0])
            schema = self._get_schema(memory["current_source"], table)
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
    and continuing the conversation. Does not provide overviews; only details and meaning of the data.
    """

    requires = param.List(default=["current_source", "current_table"], readonly=True)

    system_prompt = param.String(
        default=(
            "You are a world-class, subject expert on the topic. Help provide guidance and meaning "
            "about the data values, highlight valuable and applicable insights. Be very precise "
            "on your subject matter expertise and do not be afraid to use specialized terminology or "
            "industry-specific jargon to describe the data values and trends. Do not provide overviews; "
            "instead, focus on the details and meaning of the data. If it's unclear what the user is asking, "
            "ask clarifying questions to get more information."
        )
    )


class LumenBaseAgent(Agent):

    user = param.String(default="Lumen")

    def _describe_data(self, df: pd.DataFrame) -> str:
        def format_float(num):
            if pd.isna(num):
                return num
            # if is integer, round to 0 decimals
            if num == int(num):
                return f"{int(num)}"
            elif 0.01 <= abs(num) < 100:
                return f"{num:.1f}"  # Regular floating-point notation with two decimals
            else:
                return f"{num:.1e}"  # Exponential notation with two decimals

        size = df.size
        shape = df.shape
        if size < 250:
            out = io.StringIO()
            df.to_csv(out)
            out.seek(0)
            return out.read()

        is_summarized = False
        if shape[0] > 5000:
            is_summarized = True
            df = df.sample(5000)

        df = df.sort_index()

        for col in df.columns:
            if isinstance(df[col].iloc[0], pd.Timestamp):
                df[col] = pd.to_datetime(df[col])

        df_describe_dict = df.describe(percentiles=[]).to_dict()

        for col in df.select_dtypes(include=["object"]).columns:
            if col not in df_describe_dict:
                df_describe_dict[col] = {}
            df_describe_dict[col]["nunique"] = df[col].nunique()
            try:
                df_describe_dict[col]["lengths"] = {
                    "max": df[col].str.len().max(),
                    "min": df[col].str.len().min(),
                    "mean": float(df[col].str.len().mean()),
                }
            except AttributeError:
                pass

        for col in df.columns:
            if col not in df_describe_dict:
                df_describe_dict[col] = {}
            df_describe_dict[col]["nulls"] = int(df[col].isnull().sum())

        # select datetime64 columns
        for col in df.select_dtypes(include=["datetime64"]).columns:
            for key in df_describe_dict[col]:
                df_describe_dict[col][key] = str(df_describe_dict[col][key])
            df[col] = df[col].astype(str)  # shorten output

        # select all numeric columns and round
        for col in df.select_dtypes(include=["int64", "float64"]).columns:
            for key in df_describe_dict[col]:
                df_describe_dict[col][key] = format_float(df_describe_dict[col][key])

        for col in df.select_dtypes(include=["float64"]).columns:
            df[col] = df[col].apply(format_float)

        df_head_dict = {}
        for col in df.columns:
            df_head_dict[col] = df[col].head(10)
            # if all nan or none, replace with None
            if df_head_dict[col].isnull().all():
                df_head_dict[col] = ["all null"]
            else:
                df_head_dict[col] = df_head_dict[col].tolist()

        data = {
            "summary": {
                "total_table_cells": size,
                "total_shape": shape,
                "is_summarized": is_summarized,
            },
            "stats": df_describe_dict,
            "head": df_head_dict
        }

        return data

    def _render_lumen(self, component: Component, message: pn.chat.ChatMessage = None):
        async def _render_component(spec, active):
            yield pn.indicators.LoadingSpinner(
                value=True, name="Rendering component...", height=50, width=50
            )

            if active != 1:
                return

            # store the spec in the cache instead of memory to save tokens
            memory["current_spec"] = spec
            try:
                output = type(component).from_spec(load_yaml(spec)).__panel__()
                yield output
            except Exception as e:
                import traceback
                traceback.print_exc()
                yield pn.pane.Alert(
                    f"```\n{e}\n```\nPlease press undo, edit the YAML, or continue chatting.",
                    alert_type="danger",
                )
                # maybe offer undo

        # layout widgets
        component_spec = component.to_spec()
        spec = yaml.safe_dump(component_spec)
        spec = f"# Here's the generated Lumen spec; modify if needed\n{spec}"
        tabs = self._link_code_editor(spec, _render_component, "yaml")
        message_kwargs = dict(value=tabs, user=self.user)
        self.interface.stream(message=message, **message_kwargs, replace=True)
        tabs.active = 1


class TableAgent(LumenBaseAgent):
    """
    Responsible for only displaying a set of tables / datasets, no discussion.
    """

    system_prompt = param.String(
        default="You are an agent responsible for finding the correct table based on the user prompt."
    )

    requires = param.List(default=["current_source"], readonly=True)

    provides = param.List(default=["current_table", "current_pipeline"], readonly=True)

    async def answer(self, messages: list | str):
        source = memory["current_source"]
        tables = tuple(source.get_tables())
        if len(tables) == 1:
            table = tables[0]
        else:
            closest_tables = memory.pop("closest_tables", [])
            if closest_tables:
                tables = closest_tables
            elif len(tables) > FUZZY_TABLE_LENGTH:
                tables = await self._get_closest_tables(messages, source, tables)
            system_prompt = await self._system_prompt_with_context(messages)
            if self.debug:
                print(f"{self.name} is being instructed that it should {system_prompt}")
            if len(tables) > 1:
                table_model = create_model("Table", table=(Literal[tables], FieldInfo(
                    description="The most relevant table based on the user query; if none are relevant, select the first."
                )))
                result = await self.llm.invoke(
                    messages,
                    system=system_prompt,
                    response_model=table_model,
                    allow_partial=False,
                )
                table = result.table
            else:
                table = tables[0]
        memory["current_table"] = table
        memory["current_pipeline"] = pipeline = Pipeline(
            source=memory["current_source"], table=table
        )
        df = pipeline.__panel__()[-1].value
        if len(df) > 0:
            memory["current_data"] = self._describe_data(df)
        if self.debug:
            print(f"{self.name} thinks that the user is talking about {table=!r}.")
        return pipeline

    async def invoke(self, messages: list | str):
        pipeline = await self.answer(messages)
        self._render_lumen(pipeline)


class TableListAgent(LumenBaseAgent):
    """
    Responsible for showing and listing ALL the available tables or dataset.
    Do not use this if the user want to know about a specific table or dataset, or related keywords.
    """

    system_prompt = param.String(
        default="You are an agent responsible for listing all the available tables in the current source."
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
            self.interface.send(table_listing, user="TableLister", respond=False)
        return tables

    async def invoke(self, messages: list | str):
        await self.answer(messages)


class SQLAgent(LumenBaseAgent):
    """
    Responsible for generating and modifying SQL queries to answer user queries about the data,
    such querying subsets of the data, aggregating the data and calculating results.
    """

    system_prompt = param.String(
        default="""
        You are an agent responsible for writing a SQL query that will
        perform the data transformations the user requested.
        """
    )

    requires = param.List(default=["current_table", "current_source"], readonly=True)

    provides = param.List(default=["current_sql", "current_pipeline"], readonly=True)

    def _render_sql(self, query):
        source = memory["current_source"]

        async def _render_sql_result(query, active):
            if active == 0:
                yield pn.indicators.LoadingSpinner(
                    value=True, name="Executing SQL query...", height=50, width=50
                )

            table = memory["current_table"]

            transforms = [SQLOverride(override=query)]
            try:
                memory["current_pipeline"] = pipeline = Pipeline(
                    source=source, table=table, sql_transforms=transforms
                )
                df = pipeline.data
                if len(df) > 0:
                    memory["current_data"] = self._describe_data(df)
                yield memory["current_pipeline"].__panel__()
            except Exception as e:
                import traceback
                traceback.print_exc()
                yield pn.pane.Alert(
                    f"```\n{e}\n```\nPlease press undo, edit the YAML, or continue chatting.",
                    alert_type="danger",
                )

        tabs = self._link_code_editor(query, _render_sql_result, "sql")
        self.interface.stream(tabs, user="SQL", replace=True)
        tabs.active = 1

    def _sql_prompt(self, sql: str, table: str, schema: dict) -> str:
        prompt = (
            f"The SQL expression for the table {table!r} is {sql!r}.\n\n"
            f"The data for table {table!r} follows the following JSON schema:\n\n```{format_schema(schema)}```"
        )
        print(prompt)
        return prompt

    async def answer(self, messages: list | str):
        message = None
        source = memory["current_source"]
        table = memory["current_table"]
        if not hasattr(source, "get_sql_expr"):
            return None
        system_prompt = await self._system_prompt_with_context(messages)
        schema = self._get_schema(source, table)
        sql_expr = source.get_sql_expr(table)
        sql_prompt = self._sql_prompt(sql_expr, table, schema)
        async for chunk in self.llm.stream(
            messages,
            system=system_prompt + sql_prompt,
            response_model=Sql,
            field="query",
        ):
            if chunk is None:
                continue
            message = self.interface.stream(
                f"```sql\n{chunk}\n```",
                user="SQL",
                message=message,
                replace=True,
            )
        if message.object is None:
            return
        sql_out = message.object.replace("```sql", "").replace("```", "").strip()

        # TODO: find a better way to replace for all tables
        for table in memory.get("closest_tables", [table]):
            sql_expr = source.get_sql_expr(table)
            if f"FROM {table}" in sql_out:
                sql_in = sql_expr.replace("SELECT * FROM ", "").replace(
                    "SELECT * from ", ""
                )
                sql_out = sql_out.replace(f"FROM {table}", f"FROM {sql_in}")
            memory["current_sql"] = sql_out
        return sql_out

    async def invoke(self, messages: list | str):
        sql = await self.answer(messages)
        self._render_sql(sql)


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
        pipeline.sql_transforms = [SQLOverride(override=memory["current_sql"])]
        memory["current_pipeline"] = pipeline
        pipeline._update_data(force=True)
        memory["current_data"] = self._describe_data(pipeline.data)
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

    def _transform_picker_prompt(self) -> str:
        prompt = "This is a description of all available transforms:\n"
        for transform in self._available_transforms.values():
            if doc := (transform.__doc__ or "").strip():
                doc = doc.split("\n\n")[0].strip().replace("\n", "")
                prompt += f"- {doc}\n"
        prompt += "- `None` No transform needed\n"
        return prompt

    def _transform_prompt(
        self, model: BaseModel, transform: Transform, table: str, schema: dict
    ) -> str:
        prompt = f"{transform.__doc__}"
        if not schema:
            raise ValueError(f"No schema found for table {table!r}")
        prompt += f"\n\nThe data columns follows the following JSON schema:\n\n```json\n{format_schema(schema)}\n```"
        if "current_transform" in memory:
            prompt += f"The previous transform specification was: {memory['current_transform']}"
        return prompt

    async def _find_transform(
        self, messages: list | str, system_prompt: str
    ) -> Type[Transform] | None:
        picker_prompt = self._transform_picker_prompt()
        transforms = self._available_transforms
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
        schema = self._get_schema(memory["current_pipeline"])
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
        memory["current_data"] = self._describe_data(pipeline.data)
        return pipeline

    async def invoke(self, messages: list | str):
        pipeline = await self.answer(messages)
        self._render_lumen(pipeline)


class hvPlotAgent(LumenBaseAgent):
    """
    Generates a plot of the data given a user prompt.

    If the user asks to plot, visualize or render the data this is your best best.
    """

    system_prompt = param.String(
        default="""
        Generate the plot the user requested.
        Note that `x`, `y`, `by` and `groupby` fields MUST ALL be unique columns.
        Do not arbitrarily set `groupby` and `by` fields unless explicitly requested.
        If a histogram is requested, use `y` instead of `x`.
        """
    )

    requires = param.List(default=["current_pipeline"], readonly=True)

    provides = param.List(default=["current_plot"], readonly=True)

    def _view_prompt(
        self, model: BaseModel, view: hvPlotUIView, table: str, schema: dict
    ) -> str:
        doc = view.__doc__.split("\n\n")[0]
        prompt = f"{doc}"
        # prompt += f"\n\nThe data follows the following JSON schema:\n\n```json\n{format_schema(schema)}\n```"
        if "current_view" in memory:
            prompt += f"The previous view specification was: {memory['current_view']}"
        return prompt

    async def answer(self, messages: list | str) -> Transform:
        pipeline = memory["current_pipeline"]
        table = memory["current_table"]
        system_prompt = await self._system_prompt_with_context(messages)

        # Find parameters
        view = hvPlotUIView
        schema = self._get_schema(pipeline)
        excluded = view._internal_params + [
            "controls",
            "type",
            "source",
            "pipeline",
            "transforms",
            "download",
            "field",
            "selection_group",
        ]
        model = param_to_pydantic(view, excluded=excluded, schema=schema)[view.__name__]
        view_prompt = self._view_prompt(model, view, table, schema)
        if self.debug:
            print(f"{self.name} is being instructed that {view_prompt}.")
        kwargs = await self.llm.invoke(
            messages,
            system=system_prompt + view_prompt,
            response_model=model,
            allow_partial=False,
        )

        # Instantiate
        spec = {key: val for key, val in dict(kwargs).items() if val is not None}
        spec["responsive"] = True

        spec["type"] = "hvplot_ui"
        view.validate(spec)

        spec.pop("type", None)
        if len(pipeline.data) > 20000 and spec["kind"] in ("line", "scatter", "points"):
            spec["rasterize"] = True
            spec["cnorm"] = "log"
        memory["current_view"] = dict(spec, type=view.view_type)
        if self.debug:
            print(f"{self.name} settled on {spec=!r}.")
        return view(pipeline=pipeline, **spec)

    async def invoke(self, messages: list | str):
        view = await self.answer(messages)
        self._render_lumen(view)
