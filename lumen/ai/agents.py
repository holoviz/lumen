from typing import Literal

import panel as pn
import param
import yaml

from panel.chat import ChatInterface
from panel.viewable import Viewer
from pydantic import BaseModel, create_model

from ..base import Component
from ..dashboard import load_yaml
from ..pipeline import Pipeline
from ..transforms.sql import SQLTransform, Transform
from ..views import hvPlotUIView
from .embeddings import Embeddings
from .llm import Llm
from .memory import memory
from .models import String, Table
from .translate import param_to_pydantic


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

    response_model = param.ClassSelector(class_=BaseModel, is_instance=False, default=String)

    user = param.String(default='Assistant')

    requires = param.List(default=[], readonly=True)

    provides = param.List(default=[], readonly=True)

    __abstract = True

    def __init__(self, **params):
        if 'interface' not in params:
            params['interface'] = ChatInterface(callback=self._chat_invoke)
        super().__init__(**params)

    def _chat_invoke(self, contents: list | str, user: str, instance: ChatInterface):
        return self.invoke(contents)

    def __panel__(self):
        return self.interface

    def _system_prompt_with_context(self, messages: list | str) -> str:
        system_prompt = self.system_prompt
        if self.embeddings:
            context = self.embeddings.query(messages)
            system_prompt += f"{system_prompt}\n### CONTEXT: {context}".strip()
        return system_prompt

    def invoke(self, messages: list | str):
        message = None
        system_prompt = self._system_prompt_with_context(messages)
        for chunk in self.llm.stream(messages, system=system_prompt, response_model=self.response_model):
            if not chunk:
                continue
            message = self.interface.stream(
                chunk, user=self.user, message=message, replace=True
            )
        return message


class ChatAgent(Agent):
    """
    The ChatAgent is a general chat agent unrelated to other roles.
    """

    system_prompt = param.String(default="Be a helpful chatbot.")

    response_model = param.ClassSelector(default=String, class_=BaseModel, is_instance=False)


class LumenBaseAgent(Agent):

    user = param.String(default='Lumen')

    def _render_lumen(self, component: Component, message: pn.chat.ChatMessage = None):
        def _render_component(spec, active):
            if active == 0:
                return pn.indicators.LoadingSpinner(
                    value=True, name="Rendering component...", height=50, width=50
                )
            # store the spec in the cache instead of memory to save tokens
            memory["current_spec"] = spec
            return type(component).from_spec(load_yaml(spec))

        # layout widgets
        spec = component.to_spec()
        code_editor = pn.widgets.CodeEditor(
            value=yaml.dump(spec), language="yaml",
            min_height=300, sizing_mode="stretch_both",
        )
        dashboard_placeholder = pn.Column()
        tabs = pn.Tabs(
            ("YAML", code_editor),
            ("Dashboard", dashboard_placeholder),
        )
        dashboard_placeholder[:] = [pn.bind(
            _render_component, code_editor.param.value, tabs.param.active
        )]
        if memory.get("current_spec") is None:
            tabs.active = 1
        message_kwargs = dict(value=tabs, user=self.user)
        if message:
            self.interface.stream(message=message, **message_kwargs)
        else:
            self.interface.send(respond=False, **message_kwargs)


class TableAgent(LumenBaseAgent):
    """
    The TableAgent is responsible for selecting between a set of tables based on the user prompt.
    """

    system_prompt = param.String(default='Find the closest match to one of the available tables.')

    response_model = param.ClassSelector(default=Table, class_=BaseModel, is_instance=False)

    requires = param.List(default=['current_source'], readonly=True)

    provides = param.List(default=['current_table', 'current_pipeline'], readonly=True)

    def answer(self, messages: list | str):
        tables = tuple(memory['current_source'].get_tables())
        if len(tables) == 1:
            table = tables[0]
        else:
            system_prompt = self._system_prompt_with_context(messages)
            if self.debug:
                print(f'{self.name} is being instructed that it should {system_prompt}')
            table_model = create_model("Table", table=(Literal[tables], ...))
            table = self.llm.invoke(
                messages, system=system_prompt, response_model=table_model
            ).table
        memory["current_table"] = table
        if self.debug:
            print(f'{self.name} thinks that the user is talking about {table=!r}.')
        return table

    def invoke(self, messages: list | str):
        table = self.answer(messages)
        pipeline = Pipeline(source=memory['current_source'], table=table)
        self._render_lumen(pipeline)


class PipelineAgent(LumenBaseAgent):
    """
    The Pipeline agent generates a data pipeline by applying transformations to the data.

    If the user asks to calculate or aggregate the data this is your best best.
    """

    system_prompt = param.String(default="Generate the appropriate data transformation.")

    requires = param.List(default=['current_table'], readonly=True)

    provides = param.List(default=['current_pipeline'], readonly=True)

    @property
    def _available_transforms(self):
        transforms = param.concrete_descendents(Transform)
        return {
            name: transform for name, transform in transforms.items()
            if not isinstance(transform, SQLTransform)
        }

    def _transform_picker_prompt(self) -> str:
        prompt = 'This is a description of all available transforms:\n'
        for name, transform in self._available_transforms.items():
            if doc:= (transform.__doc__ or '').strip():
                doc = doc.split('\n\n')[0].strip().replace('\n', '')
                prompt += f"- {name}: {doc}\n"
        return prompt

    def _transform_prompt(self, model: BaseModel, transform: Transform, table: str, schema: dict) -> str:
        doc = transform.__doc__.split('\n\n')[0]
        prompt = f'{doc}'
        prompt += f'\nThe arguments must conform to the following schema:\n\n```{model.model_json_schema()}```'
        prompt += f'\n\nThe data follows the following JSON schema:\n\n```{str(schema)}```'
        if 'current_transform' in memory:
            prompt += f"The previous transform specification was: {memory['current_transform']}"
        return prompt

    def answer(self, messages: list | str) -> Transform:
        table = memory['current_table']
        system_prompt = self._system_prompt_with_context(messages)
        picker_prompt = self._transform_picker_prompt()
        if self.debug:
            print(f'{self.name} is being instructed that it should {picker_prompt}')

        # Find the transform type
        transforms = self._available_transforms
        transform_model = create_model("Transform", transform=(Literal[tuple(transforms)], ...))
        transform_type = self.llm.invoke(
            messages, system=system_prompt+picker_prompt, response_model=transform_model
        ).transform
        if self.debug:
            print(f'{self.name} thought {transform_type=!r} would be the right thing to do.')

        # Find parameters
        transform = transforms[transform_type]
        excluded = transform._internal_params+['controls', 'type']
        schema = memory['current_source'].get_schema(table)
        model = param_to_pydantic(transform, excluded=excluded, schema=schema)[transform.__name__]
        transform_prompt = self._transform_prompt(model, transform, table, schema)
        if self.debug:
            print(f'{self.name} recalls that {transform_prompt}.')
        kwargs = self.llm.invoke(
            messages, system=system_prompt+transform_prompt, response_model=model, allow_partial=False
        )

        # Instantiate
        spec = dict(kwargs)
        memory['current_transform'] = dict(
            spec, type=transform.transform_type
        )
        if self.debug:
            print(f'{self.name} settled on {spec=!r}.')
        transform = transform(**spec)
        if 'current_pipeline' in memory:
            pipeline = memory['current_pipeline']
            if pipeline.transforms and type(pipeline.transforms[-1]) is type(transform):
                pipeline.transforms[-1] = transform
            else:
                pipeline.add_transform(transform)
        else:
            memory['current_pipeline'] = pipeline = Pipeline(
                source=memory['current_source'],
                table=memory['current_table'],
                transforms=[transform]
            )
        return pipeline

    def invoke(self, messages: list | str):
        pipeline = self.answer(messages)
        self._render_lumen(pipeline)


class hvPlotAgent(LumenBaseAgent):
    """
    The hvPlot agent generates a plot of the data given a user prompt.

    If the user asks to plot, visualize or render the data this is your best best.
    """

    system_prompt = param.String(default="Generate the plot the user requested. Note that x, y, by and groupby arguments may not reference the same columns.")

    requires = param.List(default=['current_pipeline'], readonly=True)

    provides = param.List(default=['current_plot'], readonly=True)

    def _view_prompt(self, model: BaseModel, view: hvPlotUIView, table: str, schema: dict) -> str:
        doc = view.__doc__.split('\n\n')[0]
        prompt = f'{doc}'
        prompt += f'\nThe arguments must conform to the following schema:\n\n```{model.model_json_schema()}```'
        prompt += f'\n\nThe data follows the following JSON schema:\n\n```{str(schema)}```'
        if 'current_view' in memory:
            prompt += f"The previous view specification was: {memory['current_view']}"
        return prompt

    def answer(self, messages: list | str) -> Transform:
        pipeline = memory['current_pipeline']
        table = memory['current_table']
        system_prompt = self._system_prompt_with_context(messages)

        # Find parameters
        view = hvPlotUIView
        schema = memory['current_source'].get_schema(table)
        excluded = view._internal_params+[
            'controls', 'type', 'source', 'pipeline', 'transforms',
            'sql_transforms', 'download', 'field', 'groupby', 'by',
            'selection_group'
        ]
        model = param_to_pydantic(view, excluded=excluded, schema=schema)[view.__name__]
        view_prompt = self._view_prompt(model, view, table, schema)
        if self.debug:
            print(f'{self.name} is being instructed that {view_prompt}.')
        kwargs = self.llm.invoke(
            messages, system=system_prompt+view_prompt, response_model=model, allow_partial=False
        )

        # Instantiate
        spec = dict(kwargs)
        memory['current_view'] = dict(spec, type=view.view_type)
        if self.debug:
            print(f'{self.name} settled on {spec=!r}.')
        return view(pipeline=pipeline, **spec)

    def invoke(self, messages: list | str):
        view = self.answer(messages)
        self._render_lumen(view)
