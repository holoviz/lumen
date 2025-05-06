from __future__ import annotations

import asyncio
import logging
import traceback

from io import StringIO
from pathlib import Path
from typing import TYPE_CHECKING, Any

import param

from panel.chat.feed import PLACEHOLDER_SVG
from panel.config import config, panel_extension
from panel.io.document import hold
from panel.io.resources import CSS_URLS
from panel.io.state import state
from panel.layout import Column, HSpacer, Row
from panel.pane import SVG, Markdown
from panel.param import ParamMethod
from panel.util import edit_readonly
from panel.viewable import Child, Children, Viewer
from panel_gwalker import GraphicWalker
from panel_material_ui import (
    Button, ChatFeed, ChatInterface, ChatMessage, FileDownload, List,
    MultiChoice, Page, Paper, Tabs,
)

from ..pipeline import Pipeline
from ..sources import Source
from ..sources.duckdb import DuckDBSource
from ..transforms.sql import SQLLimit
from ..util import log
from .agents import (
    AnalysisAgent, AnalystAgent, ChatAgent, DocumentListAgent, SourceAgent,
    SQLAgent, TableListAgent, VegaLiteAgent,
)
from .components import SplitJS, StatusBadge
from .config import PROVIDED_SOURCE_NAME, SOURCE_TABLE_SEPARATOR
from .controls import SourceControls
from .coordinator import Coordinator, Planner
from .export import (
    export_notebook, make_md_cell, make_preamble, render_cells, write_notebook,
)
from .llm import Llm, OpenAI
from .memory import _Memory, memory
from .tools import TableLookup
from .vector_store import VectorStore

if TYPE_CHECKING:
    from .views import LumenOutput

DataT = str | Path | Source | Pipeline

EXPLORATIONS_INTRO = """
🧪 **Explorations**

- Explorations are analyses applied to one or more datasets.
- An exploration consists of interactive tables and visualizations.
- New explorations are launched in a new tab for each SQL query result.
- Each exploration maintains its own context, so you can branch off an existing result by navigating to that tab.
"""


class TableExplorer(Viewer):

    interface = param.ClassSelector(class_=ChatFeed, doc="""
        The interface for the Coordinator to interact with.""")

    def __init__(self, **params):
        super().__init__(**params)
        self._table_select = MultiChoice(
            placeholder="Select table(s)", sizing_mode='stretch_width',
            max_height=200, max_items=5
        )
        self._explore_button = Button(
            name='Explore table(s)', icon='chart-bar', button_type='primary', align='center',
            disabled=self._table_select.param.value.rx().rx.not_(), on_click=self._update_explorers
        )
        self._input_row = Row(self._table_select, self._explore_button)
        self._source_map = {}
        memory.on_change('sources', self._update_source_map)
        self._update_source_map(init=True)

        self._controls = SourceControls(
            select_existing=False, cancellable=False, clear_uploads=True, multiple=True, name='Upload'
        )
        self._controls.param.watch(self._explore_table_if_single, "add")
        self._tabs = Tabs(self._controls, dynamic=True, sizing_mode='stretch_both')
        self._layout = Column(
            self._input_row,self._tabs, sizing_mode='stretch_both',
        )

    def _update_source_map(self, old=None, new=None, sources=None, init=False):
        if sources is None:
            sources = memory['sources']
        selected = list(self._table_select.value)
        deduplicate = len(sources) > 1
        new = {}
        for source in sources:
            tables = source.get_tables()
            for t in tables:
                if deduplicate:
                    t = f'{source.name}{SOURCE_TABLE_SEPARATOR}{t}'
                if (t.split(SOURCE_TABLE_SEPARATOR, maxsplit=1)[-1] not in self._source_map and
                    not init and not len(selected) > self._table_select.max_items and state.loaded):
                    selected.append(t)
                new[t] = source
        self._source_map.clear()
        self._source_map.update(new)
        selected = selected if len(selected) == 1 else []
        self._table_select.param.update(options=list(self._source_map), value=selected)
        self._input_row.visible = bool(self._source_map)

    def _explore_table_if_single(self, event):
        """
        If only one table is uploaded, help the user load it
        without requiring them to click twice. This step
        only triggers when the Upload in the Overview tab is used,
        i.e. does not trigger with uploads through the SourceAgent
        """
        if len(self._table_select.options) == 1:
            self._explore_button.param.trigger("value")

    def _update_explorers(self, event):
        if not event.new:
            return

        with self._explore_button.param.update(loading=True), self.interface.param.update(loading=True):
            explorers = []
            for table in self._table_select.value:
                source = self._source_map[table]
                if SOURCE_TABLE_SEPARATOR in table:
                    _, table = table.split(SOURCE_TABLE_SEPARATOR, maxsplit=1)
                pipeline = Pipeline(
                    source=source, table=table, sql_transforms=[SQLLimit(limit=100_000, read=source.dialect)]
                )
                table_label = f"{table[:25]}..." if len(table) > 25 else table
                walker = GraphicWalker(
                    pipeline.param.data, sizing_mode='stretch_both', min_height=800,
                    kernel_computation=True, name=table_label, tab='data'
                )
                explorers.append(walker)

            self._tabs.objects = explorers + [self._controls]
            self._table_select.value = []

    def __panel__(self):
        return self._layout


class UI(Viewer):
    """
    UI provides a baseclass and high-level entrypoint to start chatting with your data.
    """

    agents = param.List(default=[], doc="""
        List of additional Agents to add beyond the default_agents."""
    )

    analyses = param.List(default=[], doc="""
        List of custom analyses. If provided the AnalysesAgent will be added."""
    )

    coordinator = param.ClassSelector(
        class_=Coordinator, default=Planner, is_instance=False, doc="""
        The Coordinator class that will be responsible for coordinating the Agents."""
    )

    coordinator_params = param.Dict(
        default={}, doc="""Keyword arguments to pass to the Coordinator."""
    )

    default_agents = param.List(default=[
        TableListAgent, ChatAgent, DocumentListAgent, AnalystAgent, SourceAgent, SQLAgent, VegaLiteAgent
    ], doc="""List of default agents which will always be added.""")

    export_functions = param.Dict(default={}, doc="""
       Dictionary mapping from name of exporter to export function.""")

    interface = param.ClassSelector(class_=ChatFeed, doc="""
        The interface for the Coordinator to interact with.""")

    llm = param.ClassSelector(class_=Llm, default=OpenAI(), doc="""
        The LLM provider to be used by default""")

    log_level = param.ObjectSelector(default='DEBUG', objects=['DEBUG', 'INFO', 'WARNING', 'ERROR'], doc="""
        The log level to use.""")

    logs_db_path = param.String(default=None, doc="""
        The path to the log file that will store the messages exchanged with the LLM.""")

    notebook_preamble = param.String(default='', doc="""
        Preamble to add to exported notebook(s).""")

    template = param.Selector(
        default=config.param.template.names['fast'],
        objects=config.param.template.names, doc="""
        Panel template to serve the application in."""
    )

    title = param.String(default='Lumen UI', doc="Title of the app.")

    tools = param.List(doc="""
       List of Tools that can be invoked by the coordinator.""")

    vector_store = param.ClassSelector(
        class_=VectorStore, default=None, doc="""
        The vector store to use for the tools. If not provided, a new one will be created
        or inferred from the tools provided."""
    )

    __abstract = True

    def __init__(
        self,
        data: DataT | list[DataT] | None = None,
        **params
    ):
        params["log_level"] = params.get("log_level", self.param["log_level"].default).upper()
        super().__init__(**params)
        log.setLevel(self.log_level)

        agents = self.agents
        agent_types = {type(agent) for agent in agents}
        for default_agent in self.default_agents:
            if default_agent not in agent_types:
                # only add default agents if they are not already set by the user
                agents.append(default_agent)

        if self.analyses:
            agents.append(AnalysisAgent(analyses=self.analyses))

        self._resolve_data(data)
        if self.interface is None:
            self.interface = ChatInterface(
                callback_exception='verbose',
                load_buffer=5,
                sizing_mode="stretch_both"
            )
        levels = logging.getLevelNamesMapping()
        if levels.get(self.log_level) < 20:
            self.interface.callback_exception = "verbose"
        self.interface.disabled = True
        self._coordinator = self.coordinator(
            agents=agents,
            interface=self.interface,
            llm=self.llm,
            tools=self.tools,
            logs_db_path=self.logs_db_path,
            within_ui=True,
            vector_store=self.vector_store,
            **self.coordinator_params
        )
        self._notebook_export = FileDownload(
            icon="notebook",
            icon_size="1.5em",
            button_type="primary",
            callback=self._export_notebook,
            filename=f"{self.title.replace(' ', '_')}.ipynb",
            stylesheets=['.bk-btn a { padding: 0 6px; }'],
            styles={'z-index': '1000'}
        )
        self._exports = Row(
            HSpacer(),
            self._notebook_export, *(
                Button(
                    name=label, button_type='primary',
                    on_click=lambda _: e(self.interface),
                    stylesheets=['.bk-btn { padding: 4.5px 6px;']
                )
                for label, e in self.export_functions.items()
            ),
            styles={'position': 'relative', 'right': '20px', 'top': '-1px'},
            sizing_mode='stretch_width'
        )
        self._sidebar = None
        self._main = self._coordinator
        self._llm_status_badge = StatusBadge(name="LLM Pending")
        self._vector_store_status_badge = StatusBadge(name="Vector Store Pending")
        if state.curdoc and state.curdoc.session_context:
            state.on_session_destroyed(self._destroy)
        state.onload(self._verify_llm)

    async def _verify_llm(self):
        try:
            await self.llm.initialize(log_level=self.log_level)
            self.interface.disabled = False
        except Exception:
            traceback.print_exc()

        for tool in self._coordinator._tools["main"]:
            if isinstance(tool, TableLookup):
                table_lookup = tool
                break

        if not table_lookup:
            self._vector_store_status_badge.param.update(
                status="success", name='Vector Store Ready')
            return

        self._vector_store_status_badge.status = "running"
        while table_lookup._ready is False:
            await asyncio.sleep(0.5)
            if table_lookup._ready:
                self._vector_store_status_badge.param.update(
                    status="success", name='Vector Store Ready')
            elif table_lookup._ready is None:
                self._vector_store_status_badge.param.update(
                    status="danger", name='Vector Store Error')

    def _destroy(self, session_context):
        """
        Cleanup on session destroy
        """

    def _export_notebook(self):
        nb = export_notebook(self.interface.objects, preamble=self.notebook_preamble)
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
            elif isinstance(src, (str, Path)):
                src = str(src)
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
                        f"Could not determine how to load {src} file."
                    )
                tables[src] = table
        if tables or mirrors:
            initializers = ["INSTALL httpfs;", "LOAD httpfs;"] if remote else []
            source = DuckDBSource(
                initializers=initializers,
                mirrors=mirrors,
                name=PROVIDED_SOURCE_NAME,
                tables=tables,
                uri=':memory:'
            )
            sources.append(source)
        memory['sources'] = sources
        if sources:
            memory['source'] = sources[0]

    def show(self, **kwargs):
        return self._create_view(server=True).show(**kwargs)

    def _create_view(self, server: bool | None = None):
        config.css_files.append(CSS_URLS['font-awesome'])
        if (state.curdoc and state.curdoc.session_context) or server is True:
            panel_extension(
                *{ext for agent in self._coordinator.agents for ext in agent._extensions}
            )
            page = Page(
                css_files=['https://fonts.googleapis.com/css2?family=Nunito:wght@700'],
                title=self.title,
                header=[self.llm.status(), self._vector_store_status_badge],
                main=[self._main],
                sidebar=[] if self._sidebar is None else [self._sidebar],
                sidebar_open=False,
                sidebar_variant='temporary',
            )
            page.servable()
            return page
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
            *{ext for agent in self._coordinator.agents for ext in agent._extensions}, notifications=True
        )
        return self._create_view()._repr_mimebundle_(include, exclude)


class ChatUI(UI):
    """
    ChatUI provides a high-level entrypoint to start chatting with your data
    in a chat based UI.

    This high-level wrapper allows providing the data sources you will
    be chatting with and then configures the assistant and agents.

    :Example:

    .. code-block:: python

        import lumen.ai as lmai

        lmai.ChatUI(data='~/data.csv').servable()
    """

    title = param.String(default='Lumen ChatUI', doc="Title of the app.")


class Exploration(param.Parameterized):

    context = param.ClassSelector(class_=_Memory)

    conversation = Children()

    title = param.String(default="")

    subtitle = param.String(default="")

    view = Child()


class ExplorerUI(UI):
    """
    ExplorerUI provides a high-level entrypoint to start chatting with your data
    in split UI allowing users to load tables, explore them using Graphic Walker,
    and then interrogate the data via a chat interface.

    This high-level wrapper allows providing the data sources you will
    be chatting with and then configures the assistant and agents.


    :Example:

    .. code-block:: python

        import lumen.ai as lmai

        lmai.ExplorerUI(data='~/data.csv').servable()
    """

    chat_ui_position = param.Selector(default='left', objects=['left', 'right'], doc="""
        The position of the chat interface panel relative to the exploration area.""")

    title = param.String(default='Lumen Explorer', doc="Title of the app.")

    def __init__(
        self,
        data: DataT | list[DataT] | None = None,
        **params
    ):
        super().__init__(data=data, **params)
        cb = self.interface.callback
        self._coordinator.render_output = False
        self.interface.callback = self._wrap_callback(cb)
        self._explorations = List(dense=True, label='Explorations', margin=0, sizing_mode='stretch_width')
        self._explorations.on_action('remove', self._delete_exploration)
        self._explorer = TableExplorer(interface=self.interface)
        self._explorations_intro = Markdown(
            EXPLORATIONS_INTRO,
            margin=(0, 0, 10, 10),
            sizing_mode='stretch_width',
            visible=self._explorations.param["items"].rx.bool().rx.not_()
        )
        self._home = Exploration(
            context=memory,
            title='Home',
            conversation=self.interface.objects,
            view=Column(self._explorations_intro, self._explorer)
        )
        home_item = {'label': 'Home', 'icon': 'home', 'view': self._home}
        self._explorations.param.update(items=[home_item], value=home_item)
        self._explorations.param.watch(self._cleanup_explorations, ['items'])
        self._explorations.param.watch(self._update_conversation, ['value'])
        self._global_notebook_export = FileDownload(
            icon="notebook",
            icon_size="1.5em",
            button_type="primary",
            callback=self._global_export_notebook,
            filename=f"{self.title.replace(' ', '_')}.ipynb"
        )
        self._exports.visible = False
        self._output = Paper(self._home.view, elevation=2, margin=5, sx={'p': 5}, sizing_mode='stretch_both')
        self._split = SplitJS(
            left=self._coordinator,
            right=self._output,
            invert=self.chat_ui_position != 'left',
            sizing_mode='stretch_both'
        )
        self._main = Column(self._split)
        self._sidebar = Column(self._explorations)
        self._idle = asyncio.Event()
        self._idle.set()
        self._last_synced = None

    def _delete_exploration(self, item):
        self._explorations.items = [it for it in self._explorations.items if it is not item]

    def _destroy(self, session_context):
        """
        Cleanup on session destroy
        """
        for c in self._explorations.items[1:]:
            c['view'].context.cleanup()

    def _global_export_notebook(self):
        cells, extensions = [], []
        for i, item in enumerate(self._explorations.items):
            exploration = item['view']
            title = exploration.title
            header = make_md_cell(f'## {title}')
            cells.append(header)
            msg_cells, msg_exts = render_cells(exploration.conversation)
            cells += msg_cells
            for ext in msg_exts:
                if ext not in extensions:
                    extensions.append(ext)
        preamble = make_preamble(self.notebook_preamble, extensions=extensions)
        return StringIO(write_notebook(preamble+cells))

    async def _update_conversation(self, event=None):
        exploration = self._explorations.value['view']
        self._output[:] = [exploration.view]

        if event is not None:
            # When user switches explorations and coordinator is running
            # wait to switch the conversation context
            await self._idle.wait()
            if self._explorations.value['view'] is not exploration:
                # If active tab has changed while we waited
                # skip the sync
                return
            # If conversation was already updated, resync conversation
            if self._last_synced is exploration:
                exploration.conversation = self.interface.objects
            else:
                exploration.conversation = self._snapshot_messages()

        with hold():
            self._set_conversation(exploration.conversation)
            self._notebook_export.param.update(
                filename = f"{exploration.title.replace(' ', '_')}.ipynb"
            )
            self._exports.visible = True
            self.interface._chat_log.scroll_to_latest()
        self._last_synced = exploration

    def _set_conversation(self, conversation: list[Any]):
        feed = self.interface._chat_log
        with edit_readonly(feed):
            feed.visible_range = None
        self.interface.objects = conversation

    async def _cleanup_explorations(self, event):
        if len(event.new) <= len(event.old):
            return
        for i, (old, new) in enumerate(zip(event.old, event.new)):
            if old is new:
                continue
            if i == self._last_synced:
                await self._update_conversation()
            break

    def _snapshot_messages(self, new=False):
        to = -3 if new else None
        messages = []
        for msg in self.interface.objects[:to]:
            if isinstance(msg, ChatMessage):
                avatar = msg.avatar
                if isinstance(avatar, SVG) and avatar.object is PLACEHOLDER_SVG:
                    continue
            messages.append(msg)
        return messages

    async def _add_exploration(self, title: str, memory: _Memory):
        # Sanpshot previous conversation
        last_exploration = self._explorations.value['view']
        last_exploration.conversation = self._snapshot_messages(new=True)

        # Create new exploration
        conversation = list(self.interface.objects)
        output = Column(sizing_mode='stretch_both', loading=True)
        exploration = Exploration(
            context=memory,
            conversation=conversation,
            title=title,
            view=output
        )
        view_item = {'label': title, 'removeable': True, 'view': exploration, 'icon': None, 'actions': [{'action': 'remove', 'label': 'Remove', 'icon': 'delete'}]}
        with hold():
            self.interface.objects = conversation
            self._idle.set()
            self._explorations.param.update(
                items=self._explorations.items+[view_item],
                value=view_item
            )
            self._idle.clear()
            self._output[:] = [output]
            self._notebook_export.filename = f"{title.replace(' ', '_')}.ipynb"
            await self._update_conversation()
        self._last_synced = exploration

    def _add_outputs(self, exploration: Exploration, outputs: list[LumenOutput] | str):
        memory = exploration.context
        view = exploration.view
        if "sql" in memory:
            sql = memory.rx("sql")
            sql_pane = Markdown(
                param.rx('```sql\n{sql}\n```').format(sql=sql),
                margin=(-15, 0, 0, 0), sizing_mode='stretch_width', name='SQL'
            )
            if sql.count('\n') > 10:
                sql_pane = Column(
                    sql_pane, max_height=325, scroll='y-auto', name='SQL'
                )
            if len(view) and view[0].name == 'SQL':
                view[0] = sql_pane
            else:
                view.insert(0, sql_pane)

        content = []
        if view.loading:
            from panel_gwalker import GraphicWalker
            pipeline = memory['pipeline']
            content.append(
                ('Overview', GraphicWalker(
                    pipeline.param.data,
                    kernel_computation=True,
                    tab='data',
                    sizing_mode='stretch_both'
                ))
            )
        for out in outputs:
            title = out.title or type(out).__name__.replace('Output', '')
            if len(title) > 25:
                title = f"{title[:25]}..."
            content.append((title, ParamMethod(out.render, inplace=True, sizing_mode='stretch_both')))
        if view.loading:
            tabs = Tabs(*content, dynamic=True, active=len(outputs), sizing_mode='stretch_both')
            view.append(tabs)
        else:
            tabs = view[-1]
            tabs.extend(content)
        tabs.active = len(tabs)-1

    def _wrap_callback(self, callback):
        async def wrapper(contents: list | str, user: str, instance: ChatInterface):
            prev = self._explorations.value
            prev_memory = prev['view'].context
            new_exploration = False
            local_memory = prev_memory.clone()
            local_memory["outputs"] = outputs = []

            async def render_plan(_, old, new):
                nonlocal new_exploration
                plan = local_memory["plan"]
                if any(step.expert_or_tool in ('SQLAgent', 'DbtslAgent') for step in plan.steps):
                    # Expand the sidebar when the first exploration is created
                    await self._add_exploration(plan.title, local_memory)
                    new_exploration = True

            def sync_available_sources_memory(_, __, sources):
                """
                For cases when the user uploads a dataset through SourceAgent
                this will update the available_sources in the global memory
                so that the overview explorer can access it
                """
                memory["sources"] += [
                    source for source in sources if source not in memory["sources"]
                ]

            local_memory.on_change('plan', render_plan)
            local_memory.on_change('sources', sync_available_sources_memory)

            async def render_output(_, old, new):
                added = [out for out in new if out not in old]
                exploration = self._explorations.value['view']
                self._add_outputs(exploration, added)
                exploration.view.loading = False
                outputs[:] = new
                if self._split.collapsed and prev['label'] == 'Home':
                    self._split.param.update(
                        collapsed=False,
                        sizes=self._split.expanded_sizes,
                    )
            local_memory.on_change('outputs', render_output)

            # Remove exploration on error if no outputs have been
            # added yet and we launched a new exploration
            async def remove_output(_, __, ___):
                nonlocal new_exploration
                if "__error__" in local_memory:
                    del memory['__error__']
                if outputs or not new_exploration:
                    return
                exploration = self._explorations.value['view']
                prev['view'].conversation = exploration.conversation
                with hold():
                    self._explorations.param.update(
                        items=self._explorations.items[:-1],
                        value=prev
                    )
                    await self._update_conversation()
                new_exploration = False
            local_memory.on_change('__error__', remove_output)

            try:
                self._idle.clear()
                with self._coordinator.param.update(memory=local_memory):
                    await callback(contents, user, instance)
            finally:
                self._explorations.value['view'].conversation = self.interface.objects
                self._idle.set()
                local_memory.remove_on_change('plan', render_plan)
                local_memory.remove_on_change('__error__', remove_output)
                if not new_exploration:
                    prev_memory.update(local_memory)
        return wrapper
