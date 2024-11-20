from __future__ import annotations

from io import StringIO
from typing import TYPE_CHECKING

import param

from panel.chat import ChatInterface
from panel.config import config, panel_extension
from panel.io.resources import CSS_URLS
from panel.io.state import state
from panel.layout import (
    Column, HSpacer, Row, Tabs,
)
from panel.pane import Markdown
from panel.param import ParamMethod
from panel.theme import Material
from panel.viewable import Viewer
from panel.widgets import Button, FileDownload, MultiChoice

from ..pipeline import Pipeline
from ..sources import Source
from ..sources.duckdb import DuckDBSource
from ..transforms.sql import SQLLimit
from .agents import (
    AnalysisAgent, ChatAgent, ChatDetailsAgent, SourceAgent, SQLAgent,
    TableListAgent, VegaLiteAgent,
)
from .components import SplitJS
from .controls import SourceControls
from .coordinator import Coordinator, Planner
from .export import export_notebook
from .llm import Llm, OpenAI
from .memory import _Memory, memory

if TYPE_CHECKING:
    from .views import LumenOutput

DataT = str | Source | Pipeline


class UI(Viewer):
    """
    UI provides a baseclass and high-level entrypoint to start chatting with your data.
    """

    analyses = param.List(default=[], doc="""
        List of custom analyses. If provided the AnalysesAgent will be added."""
    )

    coordinator = param.ClassSelector(
        class_=Coordinator, default=Planner, is_instance=False, doc="""
        The Coordinator class that will be responsible for coordinating the Agents."""
    )

    agents = param.List(default=[], doc="""
        List of additional Agents to add beyond the default_agents."""
    )

    default_agents = param.List(default=[
        TableListAgent, ChatAgent, ChatDetailsAgent, SourceAgent, SQLAgent, VegaLiteAgent
    ], doc="""List of default agents which will always be added.""")

    export_functions = param.Dict(default={}, doc="""
       Dictionary mapping from name of exporter to export function.""")

    llm = param.ClassSelector(class_=Llm, default=OpenAI(), doc="""
        The LLM provider to be used by default""")

    notebook_preamble = param.String(default='', doc="""
        Preamble to add to exported notebook(s).""")

    template = param.Selector(
        default=config.param.template.names['fast'],
        objects=config.param.template.names, doc="""
        Panel template to serve the application in."""
    )

    title = param.String(default='Lumen UI', doc="Title of the app.")

    __abstract = True

    def __init__(
        self,
        data: DataT | list[DataT] | None = None,
        **params
    ):
        super().__init__(**params)
        agents = self.default_agents + self.agents
        if self.analyses:
            agents.append(AnalysisAgent(analyses=self.analyses))
        self._coordinator = self.coordinator(
            agents=agents,
            llm=self.llm
        )
        self._notebook_export = FileDownload(
            icon="notebook",
            icon_size="1.5em",
            button_type="primary",
            callback=self._export_notebook,
            filename=" ",
            stylesheets=['.bk-btn a { padding: 0 6px; }'],
        )
        self._exports = Row(
            HSpacer(),
            self._notebook_export, *(
                Button(
                    name=label, button_type='primary',
                    on_click=lambda _: e(self._coordinator.interface),
                    stylesheets=['.bk-btn { padding: 4.5px 6px;']
                )
                for label, e in self.export_functions.items()
            ),
            styles={'position': 'relative', 'right': '20px', 'top': '-5px'},
            sizing_mode='stretch_width',
            visible=False
        )
        self._resolve_data(data)
        self._main = Column(self._exports, self._coordinator, sizing_mode='stretch_both')

    def _export_notebook(self):
        nb = export_notebook(self._coordinator.interface.objects, preamble=self.notebook_preamble)
        return StringIO(nb)

    def _resolve_data(self, data: DataT | list[DataT] | None):
        if data is None:
            return
        elif not isinstance(data, list):
            data = [data]
        sources = []
        mirrors, tables = {}, {}
        remote = False
        for src in data:
            if isinstance(src, Source):
                sources.append(src)
            elif isinstance(src, Pipeline):
                mirrors[src.name] = src
            elif isinstance(src, str):
                if src.startswith('http'):
                    remote = True
                if src.endswith(('.parq', '.parquet')):
                    table = f"read_parquet('{src}')"
                elif src.endswith(".csv"):
                    table = f"read_csv('{src}')"
                elif src.endswith(".json"):
                    table = f"read_json_auto('{src}')"
                else:
                    raise ValueError(
                        "Could not determine how to load {} file."
                    )
                tables[src] = table
        if tables or mirrors:
            initializers = ["INSTALL httpfs;", "LOAD httpfs;"] if remote else []
            source = DuckDBSource(
                tables=tables, mirrors=mirrors,
                uri=':memory:', initializers=initializers,
                name='Provided'
            )
            sources.append(source)
        if not sources:
            raise ValueError(
                'Must provide at least one data source.'
            )
        memory['available_sources'] = sources
        if sources:
            memory['current_source'] = sources[0]

    def show(self, **kwargs):
        return self._create_view(server=True).show(**kwargs)

    def _create_view(self, server: bool | None = None):
        config.css_files.append(CSS_URLS['font-awesome'])
        if (state.curdoc and state.curdoc.session_context) or server is True:
            panel_extension(
                *{ext for agent in self._coordinator.agents for ext in agent._extensions}, template=self.template
            )
            config.template = self.template
            template = state.template
            template.title = self.title
            template.config.raw_css = ['#header a { font-family: Futura; font-size: 2em; font-weight: bold;}']
            template.main.append(self._main)
            return template
        return super()._create_view()

    def servable(self, title: str | None = None, **kwargs):
        if (state.curdoc and state.curdoc.session_context):
            self._create_view()
            return self
        return self._create_view().servable(title, **kwargs)

    def __panel__(self):
        return self._main

    def _repr_mimebundle_(self, include=None, exclude=None):
        panel_extension(
            *{ext for exts in self._coordinator.agents for ext in exts}, design='material', notifications=True
        )
        return self._create_view()._repr_mimebundle_(include, exclude)


class ChatUI(UI):
    """
    ChatUI provides a high-level entrypoint to start chatting with your data
    in a chat based UI.

    This high-level wrapper allows providing the data sources you will
    be chatting with and then configures the assistant and agents.

    Example:

    ```python
    import lumen.ai as lmai

    lmai.ChatUI('~/data.csv').servable()
    ```
    """

    title = param.String(default='Lumen ChatUI', doc="Title of the app.")

    def __init__(
        self,
        data: DataT | list[DataT] | None = None,
        **params
    ):
        super().__init__(data, **params)
        self._notebook_export.filename = f"{self.title.replace(' ', '_')}.ipynb"
        self._exports.visible = True


class ExplorerUI(UI):
    """
    ExplorerUI provides a high-level entrypoint to start chatting with your data
    in split UI allowing users to load tables, explore them using Graphic Walker,
    and then interrogate the data via a chat interface.

    This high-level wrapper allows providing the data sources you will
    be chatting with and then configures the assistant and agents.

    Example:

    ```python
    import lumen.ai as lmai

    lmai.ExplorerUI('~/data.csv').servable()
    ```
    """

    title = param.String(default='Lumen Explorer', doc="Title of the app.")

    def __init__(
        self,
        data: DataT | list[DataT] | None = None,
        **params
    ):
        super().__init__(data=data, **params)
        self._coordinator.interface.show_button_name = False
        cb = self._coordinator.interface.callback
        self._coordinator.render_output = False
        self._coordinator.interface.callback = self._wrap_callback(cb)
        self._explorations = Tabs(sizing_mode='stretch_both', closable=True)
        self._explorations.param.watch(self._cleanup_explorations, ['objects'])
        self._explorations.param.watch(self._set_context, ['active'])
        self._titles = []
        self._contexts = []
        self._root_conversation = self._coordinator.interface.objects
        self._conversations = []
        self._output = Tabs(
            ('Overview', self._table_explorer()),
            ('Explorations', self._explorations),
            design=Material
        )
        self._main = Column(
            SplitJS(
                left=Column(self._output, styles={'overflow-x': 'auto'}, sizing_mode='stretch_both'),
                right=Column(self._exports, self._coordinator),
                sizing_mode='stretch_both'
            )
        )
        self._output.param.watch(self._update_conversation, 'active')

    def _update_conversation(self, event):
        if event.new:
            active = self._explorations.active
            if active < len(self._conversations):
                conversation = self._conversations[active]
            else:
                conversation = list(self._coordinator.interface.objects)
            self._exports.visible = True
        else:
            self._exports.visible = False
            conversation = self._root_conversation
        self._coordinator.interface.objects = conversation

    def _cleanup_explorations(self, event):
        if len(event.new) >= len(event.old):
            return
        for i, (old, new) in enumerate(zip(event.old, event.new)):
            if old is not new:
                self._contexts.pop(i)
                self._conversations.pop(i)

    def _set_context(self, event):
        if event.new == len(self._conversations):
            return
        self._coordinator.interface.objects = self._conversations[event.new]
        self._notebook_export.param.update(
            filename = f"{self._titles[event.new].replace(' ', '_')}.ipynb"
        )
        self._exports.visible = True

    def _table_explorer(self):
        from panel_gwalker import GraphicWalker

        table_select = MultiChoice(
            sizing_mode='stretch_width', max_height=200, margin=(5, 0), max_items=5
        )
        load_button = Button(
            name='Load table(s)', icon='table-plus', button_type='primary', align='center',
            disabled=table_select.param.value.rx().rx.not_()
        )

        source_map = {}
        def update_source_map(_, __, sources, init=False):
            selected = list(table_select.value)
            deduplicate = len(sources) > 1
            new = {}
            for source in sources:
                source_tables = source.get_tables()
                for t in source_tables:
                    if deduplicate:
                        t = f'{source.name} : {t}'
                    if t.rsplit(' : ', 1)[-1] not in source_map and not init and not len(selected) > table_select.max_items and state.loaded:
                        selected.append(t)
                    new[t] = source
            source_map.clear()
            source_map.update(new)
            table_select.param.update(options=list(source_map), value=selected)
        memory.on_change('available_sources', update_source_map)
        update_source_map(None, None, memory['available_sources'], init=True)

        controls = SourceControls(select_existing=False, name='Upload')
        tabs = Tabs(controls, sizing_mode='stretch_both', design=Material)

        @param.depends(table_select, load_button, watch=True)
        def get_explorers(tables, load):
            if not load:
                return
            load_button.loading = True
            explorers = []
            for table in tables:
                source = source_map[table]
                if len(memory['available_sources']) > 1:
                    _, table = table.rsplit(' : ', 1)
                pipeline = Pipeline(
                    source=source, table=table, sql_transforms=[SQLLimit(limit=100_000)]
                )
                walker = GraphicWalker(
                    pipeline.param.data, sizing_mode='stretch_both', min_height=800,
                    kernel_computation=True, name=table, tab='data'
                )
                explorers.append(walker)
            tabs.objects = explorers + [controls]
            load_button.loading = False

        return Column(
            Markdown('### Start chatting or select an existing dataset or upload a .csv, .parquet, .xlsx file.', margin=(5, 0)),
            Row(table_select, load_button),
            tabs,
            sizing_mode='stretch_both',
        )

    def _add_exploration(self, title: str, memory: _Memory):
        self._titles.append(title)
        self._contexts.append(memory)
        self._conversations.append(self._coordinator.interface.objects)
        self._explorations.append((title, Column(name=title, sizing_mode='stretch_both', loading=True)))
        self._notebook_export.filename = f"{title.replace(' ', '_')}.ipynb"
        self._explorations.active = len(self._explorations)-1
        self._output.active = 1

    def _add_outputs(self, exploration: Column, outputs: list[LumenOutput], memory: _Memory):
        from panel_gwalker import GraphicWalker
        if 'current_sql' in memory:
            sql = memory["current_sql"]
            sql_pane = Markdown(
                f'```sql\n{sql}\n```',
                margin=0, sizing_mode='stretch_width'
            )
            if sql.count('\n') > 10:
                sql_pane = Column(sql_pane, max_height=250, scroll='y-auto')
            if len(exploration) and isinstance(exploration[0], Markdown):
                exploration[0] = sql_pane
            else:
                exploration.insert(0, sql_pane)

        content = []
        if exploration.loading:
            pipeline = memory['current_pipeline']
            content.append(
                ('Overview', GraphicWalker(
                    pipeline.param.data,
                    kernel_computation=True,
                    tab='data',
                    sizing_mode='stretch_both'
                ))
            )
        content.extend([
            (out.title or type(out).__name__.replace('Output', ''), ParamMethod(
                out.render, inplace=True,
                sizing_mode='stretch_both'
            )) for out in outputs
        ])
        if exploration.loading:
            tabs = Tabs(*content, active=len(outputs), dynamic=True)
            exploration.append(tabs)
        else:
            tabs = exploration[-1]
            tabs.extend(content)
        tabs.active = len(tabs)-1

    def _wrap_callback(self, callback):
        async def wrapper(contents: list | str, user: str, instance: ChatInterface):
            if not self._explorations:
                prev_memory = memory
            else:
                prev_memory = self._contexts[self._explorations.active]
            index = self._explorations.active if len(self._explorations) else -1
            local_memory = prev_memory.clone()
            local_memory['outputs'] = outputs = []

            def render_plan(_, old, new):
                nonlocal index
                plan = local_memory['plan']
                if any(step.expert == 'SQLAgent' for step in plan.steps):
                    self._add_exploration(plan.title, local_memory)
                    index += 1
            local_memory.on_change('plan', render_plan)

            def render_output(_, old, new):
                added = [out for out in new if out not in old]
                exploration = self._explorations[index]
                self._add_outputs(exploration, added, local_memory)
                exploration.loading = False
                outputs[:] = new
            local_memory.on_change('outputs', render_output)

            try:
                with self._coordinator.param.update(memory=local_memory):
                    await callback(contents, user, instance)
            finally:
                local_memory.remove_on_change('plan', render_plan)
                local_memory.remove_on_change('outputs', render_output)
                if not outputs:
                    prev_memory.update(local_memory)
        return wrapper
