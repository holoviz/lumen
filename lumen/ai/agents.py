import io

from typing import Literal, Type, Optional

import pandas as pd
import panel as pn
import param
import yaml

from panel.chat import ChatInterface
from panel.viewable import Viewer
from pydantic import BaseModel, create_model
from instructor import Maybe

from ..base import Component
from ..dashboard import load_yaml
from ..pipeline import Pipeline
from ..sources import FileSource, InMemorySource, Source
from ..transforms.sql import SQLTransform, Transform
from ..views import hvPlotUIView
from .embeddings import Embeddings
from .llm import Llm
from .memory import memory
from .models import (
    Sql,
    String,
    Table,
)
from .translate import param_to_pydantic
from pydantic import BaseModel, BeforeValidator
from typing_extensions import Annotated
from instructor import llm_validator


class Agent(Viewer):
    """
    An Agent in Panel is responsible for handling a specific type of
    query. Each agent consists of an LLM and optionally a set of
    embeddings.
    """

    debug = param.Boolean(default=True)

    embeddings = param.ClassSelector(class_=Embeddings)

    interface = param.ClassSelector(class_=ChatInterface)

    llm = param.ClassSelector(class_=Llm)

    system_prompt = param.String()

    response_model = param.ClassSelector(
        class_=BaseModel, is_instance=False, default=String
    )

    user = param.String(default="Assistant")

    requires = param.List(default=[], readonly=True)

    fulfills = param.List(default=[], readonly=True)

    __abstract = True

    def __init__(self, **params):
        if "interface" not in params:
            params["interface"] = ChatInterface(callback=self._chat_invoke)
        super().__init__(**params)

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
        )
        placeholder.objects = [
            pn.bind(callback, code_editor.param.value, tabs.param.active)
        ]
        return tabs

    def _chat_invoke(self, contents: list | str, user: str, instance: ChatInterface):
        return self.invoke(contents)

    def __panel__(self):
        return self.interface

    def _system_prompt_with_context(
        self, messages: list | str, context: str = ""
    ) -> str:
        system_prompt = self.system_prompt
        if self.embeddings:
            context = self.embeddings.query(messages)
        if context:
            system_prompt += f"{system_prompt}\n### CONTEXT: {context}".strip()
        return system_prompt

    def _get_schema(self, source: Source, table: str):
        try:
            source.load_schema = True
            return source.get_schema(table)
        finally:
            source.load_schema = False

    def invoke(self, messages: list | str):
        message = None
        system_prompt = self._system_prompt_with_context(messages)
        for chunk in self.llm.stream(
            messages, system=system_prompt, response_model=self.response_model
        ):
            if not chunk:
                continue
            message = self.interface.stream(
                chunk, user=self.user, message=message, replace=True
            )
        return message


class SourceAgent(Agent):
    """
    The SourceAgent allows a user to provide an input source.

    Should only be used if the user explicitly requests adding a source
    or no source is in memory.
    """

    requires = param.List(default=[], readonly=True)

    fulfills = param.List(default=["current_source"], readonly=True)

    on_init = param.Boolean(default=True)

    def invoke(self, messages: list[str] | str):
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
    The ChatAgent is a general chat agent unrelated to other roles.
    """

    system_prompt = param.String(default="Be a helpful chatbot.")

    response_model = param.ClassSelector(
        default=String, class_=BaseModel, is_instance=False
    )


class LumenBaseAgent(Agent):

    user = param.String(default="Lumen")

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
                yield type(component).from_spec(load_yaml(spec))
            except Exception as e:
                yield pn.pane.Alert(
                    f"Error rendering component: {e}", alert_type="danger"
                )

        # layout widgets
        spec = yaml.safe_dump(component.to_spec())
        tabs = self._link_code_editor(spec, _render_component, "yaml")
        message_kwargs = dict(value=tabs, user=self.user)
        if message:
            self.interface.stream(message=message, **message_kwargs)
        else:
            self.interface.send(respond=False, **message_kwargs)


class TableAgent(LumenBaseAgent):
    """
    Responsible for selecting between a set of tables / datasets based on the user prompt.
    """

    system_prompt = param.String(
        default="You are an agent responsible for finding the correct table based on the user prompt."
    )

    response_model = param.ClassSelector(
        default=Table, class_=BaseModel, is_instance=False
    )

    requires = param.List(default=["current_source"], readonly=True)

    fulfills = param.List(default=["current_table", "current_pipeline"], readonly=True)

    def answer(self, messages: list | str):
        tables = tuple(memory["current_source"].get_tables())
        if len(tables) == 1:
            table = tables[0]
        else:
            system_prompt = self._system_prompt_with_context(messages)
            if self.debug:
                print(f"{self.name} is being instructed that it should {system_prompt}")
            # needed or else something with grammar issue
            tables = tuple(table.replace('"', "") for table in tables)
            table_model = create_model("Table", table=(Literal[tables], ...))
            table = self.llm.invoke(
                messages,
                system=system_prompt,
                response_model=table_model,
                allow_partial=False,
            ).table
        memory["current_table"] = table
        memory["current_pipeline"] = Pipeline(
            source=memory["current_source"], table=table
        )
        if self.debug:
            print(f"{self.name} thinks that the user is talking about {table=!r}.")
        return table

    def invoke(self, messages: list | str):
        table = self.answer(messages)
        pipeline = Pipeline(source=memory["current_source"], table=table)
        self._render_lumen(pipeline)


class TableListAgent(LumenBaseAgent):
    """
    Responsible for listing the available tables if the user's request is vague;
    do not use this if user wants a specific table.s
    """

    system_prompt = param.String(
        default="You are an agent responsible for listing the available tables in the current source."
    )

    requires = param.List(default=["current_source"], readonly=True)

    def answer(self, messages: list | str):
        tables = memory["current_source"].get_tables()
        if not tables:
            return
        tables = tuple(table.replace('"', "") for table in tables)
        table_bullets = "\n".join(f"- {table}" for table in tables)
        self.interface.send(
            f"Available tables:\n{table_bullets}", user=self.name, respond=False
        )
        return tables

    def invoke(self, messages: list | str):
        self.answer(messages)


class SQLAgent(LumenBaseAgent):
    """
    Responsible for generating and modifying SQL queries to answer user questions about statistics.
    """

    system_prompt = param.String(
        default="You are an agent responsible for writing a SQL query that will perform the data transformations the user requested."
    )

    requires = param.List(default=["current_table", "current_source"], readonly=True)

    def _render_sql(self, query):
        source = memory["current_source"]

        async def _render_sql_result(query, active):
            if active == 0:
                yield pn.indicators.LoadingSpinner(
                    value=True, name="Executing SQL query...", height=50, width=50
                )

            table = memory["current_table"]
            source.tables[table] = query
            try:
                memory["current_pipeline"] = pipeline = Pipeline(
                    source=source, table=table
                )
                yield pipeline
                tabs.active = 1
            except Exception as e:
                yield pn.pane.Alert(
                    f"Error executing SQL query: {e}", alert_type="danger"
                )

        tabs = self._link_code_editor(query, _render_sql_result, "sql")
        self.interface.stream(tabs, user="SQL", replace=True)

    def _sql_prompt(self, sql: str, table: str, schema: dict) -> str:
        prompt = f"The SQL expression for the table {table!r} is {sql!r}."
        prompt += f"\n\nThe data for table {table!r} follows the following JSON schema:\n\n```{str(schema)}```"
        return prompt

    def answer(self, messages: list | str):
        message = None
        source = memory["current_source"]
        table = memory["current_table"]
        if not hasattr(source, "get_sql_expr"):
            return None
        sql_expr = source.get_sql_expr(table)
        system_prompt = self._system_prompt_with_context(messages)
        schema = self._get_schema(source, table)
        sql_prompt = self._sql_prompt(sql_expr, table, schema)
        for chunk in self.llm.stream(
            messages,
            system=system_prompt + sql_prompt,
            response_model=Sql,
            field="query",
        ):
            if chunk:
                message = self.interface.stream(
                    f"```sql\n{chunk}\n```",
                    user="SQL",
                    message=message,
                    replace=True,
                )
        if message.object is None:
            return
        sql_out = message.object.replace("```sql", "").replace("```", "").strip()
        if f"FROM {table}" in sql_out:
            sql_in = sql_expr.replace("SELECT * from ", "")
            sql_out = sql_out.replace(f"FROM {table}", f"FROM {sql_in}")
            print(table, sql_in, sql_out)
        memory["current_sql"] = sql_out
        return sql_out

    def invoke(self, messages: list | str):
        sql = self.answer(messages)
        self._render_sql(sql)


class PipelineAgent(LumenBaseAgent):
    """
    Generates a data pipeline by applying transformations to the data.

    If the user asks to calculate, aggregate, select or perform some other operation on the data this is your best best.
    """

    system_prompt = param.String(
        default="Choose the appropriate data transformation based on the query."
    )

    requires = param.List(default=["current_table"], readonly=True)

    fulfills = param.List(default=["current_pipeline"], readonly=True)

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
        return prompt

    def _transform_prompt(
        self, model: BaseModel, transform: Transform, table: str, schema: dict
    ) -> str:
        prompt = f"{transform.__doc__}"
        if not schema:
            raise ValueError(f"No schema found for table {table!r}")
        else:
            print(f"Used schema: {schema}")
        prompt += f"\n\nThe data follows the following JSON schema:\n\n```json\n{str(schema)}\n```"
        if "current_transform" in memory:
            prompt += f"The previous transform specification was: {memory['current_transform']}"
        return prompt

    def _find_transform(
        self, messages: list | str, system_prompt: str
    ) -> Type[Transform] | None:
        picker_prompt = self._transform_picker_prompt()
        transforms = self._available_transforms
        print(
            "VALIDATOR",
            f"is the transform applicable to {messages}? if not, choose again: {picker_prompt}",
        )
        transform_model = create_model(
            "Transform",
            chain_of_thought_to_select=(str, ...),
            transform=(
                Annotated[
                    Optional[Literal[tuple(transforms)]],
                    BeforeValidator(
                        llm_validator(
                            f"is the transform partially relevant to the query {messages!r} if not, choose again: {picker_prompt}",
                            allow_override=True,
                        )
                    ),
                ],
                None,
            ),
        )
        transform = self.llm.invoke(
            messages,
            system=f"{system_prompt}\n{picker_prompt}",
            response_model=transform_model,
        )
        if transform is None or not hasattr(transform, "transform"):
            print(f"No transform found for {messages}")
            return None

        transform_name = transform.transform
        if self.debug:
            print(
                f"{self.name} thought {transform_name=!r} would be the right thing to do."
            )
        return transforms[transform_name.strip("'")] if transform_name else None

    def _construct_transform(
        self, messages: list | str, transform: Type[Transform], system_prompt: str
    ) -> Transform:
        table = memory["current_table"]
        excluded = transform._internal_params + ["controls", "type"]
        schema = self._get_schema(memory["current_source"], table)
        model = param_to_pydantic(transform, excluded=excluded, schema=schema)[
            transform.__name__
        ]
        transform_prompt = self._transform_prompt(model, transform, table, schema)
        if self.debug:
            print(f"{self.name} recalls that {transform_prompt}.")
        kwargs = self.llm.invoke(
            messages,
            system=system_prompt + transform_prompt,
            response_model=model,
            allow_partial=False,
        )
        return transform(**dict(kwargs))

    def answer(self, messages: list | str) -> Transform:
        system_prompt = self._system_prompt_with_context(messages)
        transform_type = self._find_transform(messages, system_prompt)
        if "current_pipeline" in memory:
            pipeline = memory["current_pipeline"]
        else:
            pipeline = Pipeline(
                source=memory["current_source"], table=memory["current_table"]
            )
        memory["current_pipeline"] = pipeline
        if not transform_type:
            return pipeline
        transform = self._construct_transform(messages, transform_type, system_prompt)
        memory["current_transform"] = spec = transform.to_spec()
        if self.debug:
            print(f"{self.name} settled on {spec=!r}.")
        if pipeline.transforms and type(pipeline.transforms[-1]) is type(transform):
            pipeline.transforms[-1] = transform
        else:
            pipeline.add_transform(transform)
        try:
            pipeline._update_data(force=True)
        except Exception as e:
            self.interface.send(
                f"Generated invalid transform resulting in following error: {e}",
                user="Exception",
                respond=False,
            )
            pipeline.transforms = pipeline.transforms[:-1]
            memory.pop("current_transform")
            print(f"{memory=}")
            pipeline._stale = True
        return pipeline

    def invoke(self, messages: list | str):
        pipeline = self.answer(messages)
        self._render_lumen(pipeline)


class hvPlotAgent(LumenBaseAgent):
    """
    Generates a plot of the data given a user prompt.

    If the user asks to plot, visualize or render the data this is your best best.
    """

    system_prompt = param.String(
        default="Generate the plot the user requested. Note that x, y, by and groupby arguments may not reference the same columns."
    )

    requires = param.List(default=["current_pipeline"], readonly=True)

    fulfills = param.List(default=["current_plot"], readonly=True)

    def _view_prompt(
        self, model: BaseModel, view: hvPlotUIView, table: str, schema: dict
    ) -> str:
        doc = view.__doc__.split("\n\n")[0]
        prompt = f"{doc}"
        prompt += f"\n\nThe data follows the following JSON schema:\n\n```json\n{str(schema)}\n```"
        if "current_view" in memory:
            prompt += f"The previous view specification was: {memory['current_view']}"
        return prompt

    def answer(self, messages: list | str) -> Transform:
        pipeline = memory["current_pipeline"]
        table = memory["current_table"]
        system_prompt = self._system_prompt_with_context(messages)

        # Find parameters
        view = hvPlotUIView
        schema = self._get_schema(pipeline.source, table)
        excluded = view._internal_params + [
            "controls",
            "type",
            "source",
            "pipeline",
            "transforms",
            "sql_transforms",
            "download",
            "field",
            "groupby",
            "by",
            "selection_group",
            "limit",
        ]
        model = param_to_pydantic(view, excluded=excluded, schema=schema)[view.__name__]
        view_prompt = self._view_prompt(model, view, table, schema)
        if self.debug:
            print(f"{self.name} is being instructed that {view_prompt}.")
        kwargs = self.llm.invoke(
            messages,
            system=system_prompt + view_prompt,
            response_model=model,
            allow_partial=False,
        )

        # Instantiate
        spec = dict(kwargs)
        spec["responsive"] = True
        memory["current_view"] = dict(spec, type=view.view_type)
        if self.debug:
            print(f"{self.name} settled on {spec=!r}.")
        return view(pipeline=pipeline, **spec)

    def invoke(self, messages: list | str):
        view = self.answer(messages)
        self._render_lumen(view)
