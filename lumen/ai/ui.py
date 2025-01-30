from __future__ import annotations

import asyncio
import logging
import traceback

from io import StringIO
from pathlib import Path
from typing import TYPE_CHECKING, Any

import param

from panel.chat import ChatInterface, ChatMessage
from panel.chat.feed import PLACEHOLDER_SVG
from panel.config import config, panel_extension
from panel.io.document import hold
from panel.io.resources import CSS_URLS
from panel.io.state import state
from panel.layout import (
    Column, HSpacer, Row, Tabs,
)
from panel.pane import SVG, Alert, Markdown
from panel.param import ParamMethod
from panel.template import FastListTemplate
from panel.theme import Material
from panel.util import edit_readonly
from panel.viewable import Viewer
from panel.widgets import Button, FileDownload, MultiChoice

from lumen.ai.config import SOURCE_TABLE_SEPARATOR

from ..pipeline import Pipeline
from ..sources import Source
from ..sources.duckdb import DuckDBSource
from ..transforms.sql import SQLLimit
from ..util import log
from .agents import (
    AnalysisAgent, AnalystAgent, ChatAgent, DocumentListAgent, SourceAgent,
    SQLAgent, TableListAgent, VegaLiteAgent,
)
from .components import SplitJS
from .controls import SourceControls
from .coordinator import Coordinator, Planner
from .export import (
    export_notebook, make_md_cell, make_preamble, render_cells, write_notebook,
)
from .llm import Llm, OpenAI
from .memory import _Memory, memory
from .utils import format_exception

if TYPE_CHECKING:
    from .views import LumenOutput

DataT = str | Path | Source | Pipeline


OVERVIEW_INTRO = """
👋 **Start chatting to get started!**

- Ask for summaries, ask for plots, ask for inspiration--query away!
- Select a table, or upload one, to explore the table with [Graphic Walker](https://docs.kanaries.net/graphic-walker).
- Download the chat into a reproducible notebook to share with others!
- Check out [Using Lumen AI](https://holoviz-dev.github.io/lumen/lumen_ai/getting_started/using_lumen_ai.html) for more tips & tricks!
"""

EXPLORATIONS_INTRO = """
🧪 **Explorations**

- Explorations are analyses applied to one or more datasets.
- An exploration consists of interactive tables and visualizations.
- New explorations are launched in a new tab for each SQL query result.
- Each exploration maintains its own context, so you can branch off an existing result by navigating to that tab.
"""

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
        TableListAgent, ChatAgent, DocumentListAgent, AnalystAgent, SourceAgent, SQLAgent, VegaLiteAgent
    ], doc="""List of default agents which will always be added.""")

    export_functions = param.Dict(default={}, doc="""
       Dictionary mapping from name of exporter to export function.""")

    llm = param.ClassSelector(class_=Llm, default=OpenAI(), doc="""
        The LLM provider to be used by default""")

    log_level = param.ObjectSelector(default='DEBUG', objects=['DEBUG', 'INFO', 'WARNING', 'ERROR'], doc="""
        The log level to use.""")

    logs_db_path = param.String(default=None, doc="""
        The path to the log file that will store the messages exchanged with the LLM.""")

    interface = param.ClassSelector(class_=ChatInterface, doc="""
        The ChatInterface for the Coordinator to interact with.""")

    notebook_preamble = param.String(default='', doc="""
        Preamble to add to exported notebook(s).""")

    template = param.Selector(
        default=config.param.template.names['fast'],
        objects=config.param.template.names, doc="""
        Panel template to serve the application in."""
    )

    title = param.String(default='Lumen UI', doc="Title of the app.")

    tools = param.List(default=[], doc="""
       List of Tools that can be invoked by the coordinator.""")

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
                load_buffer=5,
            )
        levels = logging.getLevelNamesMapping()
        if levels.get(self.log_level) < 20:
            self.interface.callback_exception = "verbose"
        self._coordinator = self.coordinator(
            agents=agents,
            interface=self.interface,
            llm=self.llm,
            tools=self.tools,
            logs_db_path=self.logs_db_path
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
        self._main = Column(self._exports, self._coordinator, sizing_mode='stretch_both')
        if state.curdoc and state.curdoc.session_context:
            state.on_session_destroyed(self._destroy)
        state.onload(self._verify_llm)

    async def _verify_llm(self):
        alert = Alert(
            'Initializing LLM ⌛',
            alert_type='primary',
            margin=5,
            sizing_mode='stretch_width',
            styles={'background-color': 'var(--primary-bg-subtle)'}
        )
        self.interface.send(alert, respond=False, user='System')
        try:
            await self.llm.invoke([{'role': 'user', 'content': 'Are you there? YES | NO'}])
        except Exception as e:
            traceback.print_exc()
            alert.param.update(
                alert_type='danger',
                object='❌ '+format_exception(e, limit=3 if self.log_level == 'DEBUG' else 0),
                styles={
                    'background-color': 'var(--danger-bg-subtle)',
                    'overflow-x': 'auto',
                    'padding-bottom': '0.8em'
                }
            )
        else:
            alert.param.update(
                alert_type='success',
                object='Successfully initialized LLM ✅︎',
                styles={'background-color': 'var(--success-bg-subtle)'}
            )

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
                tables=tables, mirrors=mirrors,
                uri=':memory:', initializers=initializers,
                name='ProvidedSource00000'
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
                *{ext for agent in self._coordinator.agents for ext in agent._extensions}, template=self.template
            )
            config.template = self.template
            template = state.template
            if isinstance(template, FastListTemplate):
                template.main_layout = None
            template.title = self.title
            template.config.css_files.append(
                'https://fonts.googleapis.com/css2?family=Nunito:wght@700'
            )
            template.config.raw_css = [
                "#header a { font-family: 'Nunito', sans-serif; font-size: 2em; font-weight: bold;}"
                "#main { padding: 0 } .main .card-margin.stretch_both { margin: 0; height: 100%; }"
            ]
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

    :Example:

    .. code-block:: python

        import lumen.ai as lmai

        lmai.ChatUI(data='~/data.csv').servable()
    """

    title = param.String(default='Lumen ChatUI', doc="Title of the app.")


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

    title = param.String(default='Lumen Explorer', doc="Title of the app.")

    def __init__(
        self,
        data: DataT | list[DataT] | None = None,
        **params
    ):
        super().__init__(data=data, **params)
        self.interface.show_button_name = False
        cb = self.interface.callback
        self._coordinator.render_output = False
        self.interface.callback = self._wrap_callback(cb)
        self._explorations = Tabs(
            sizing_mode='stretch_both', closable=True, tabs_location="left",
            stylesheets=[':host(.bk-left) .bk-header .bk-tab { padding-left: 5px; padding-right: 2px; text-align: left; }']
        )
        self._explorations.param.watch(self._cleanup_explorations, ['objects'])
        self._explorations.param.watch(self._set_context, ['active'])
        self._global_notebook_export = FileDownload(
            icon="notebook",
            icon_size="1.5em",
            button_type="primary",
            callback=self._global_export_notebook,
            filename=f"{self.title.replace(' ', '_')}.ipynb",
            stylesheets=['.bk-btn a { padding: 0 6px; }'],
            styles={'position': 'absolute', 'right': '0px', 'top': '-1px', 'z-index': '100'},
        )
        self._exports.visible = False
        self._titles = []
        self._contexts = []
        self._root_conversation = self.interface.objects
        self._conversations = []

        self._explorations_intro = Markdown(
            EXPLORATIONS_INTRO,
            margin=(0, 0, 10, 10),
            sizing_mode='stretch_width',
            visible=self._explorations.param["objects"].rx.bool().rx.not_()
        )
        self._output = Tabs(
            ('Overview', self._table_explorer()),
            ('Explorations', Column(self._explorations_intro, self._explorations)),
            design=Material
        )
        self._main = Column(
            SplitJS(
                left=Column(self._global_notebook_export, self._output, styles={'overflow-x': 'auto', 'overflow-y': 'clip'}, sizing_mode='stretch_both'),
                right=Column(self._exports, self._coordinator),
                sizing_mode='stretch_both'
            )
        )
        self._idle = asyncio.Event()
        self._idle.set()
        self._last_synced = None
        self._output.param.watch(self._update_conversation, 'active')

    def _destroy(self, session_context):
        """
        Cleanup on session destroy
        """
        for c in self._contexts:
            c.cleanup()

    def _global_export_notebook(self):
        cells, extensions = [], []
        for i, objects in enumerate(self._conversations):
            title = self._titles[i]
            header = make_md_cell(f'## {title}')
            cells.append(header)
            msg_cells, msg_exts = render_cells(objects)
            cells += msg_cells
            for ext in msg_exts:
                if ext not in extensions:
                    extensions.append(ext)
        preamble = make_preamble(self.notebook_preamble, extensions=extensions)
        return StringIO(write_notebook(preamble+cells))

    async def _update_conversation(self, event=None, tab=None):
        active = self._explorations.active
        if len(self._conversations) <= active:
            return

        if tab is None:
            # When user switches tabs and coordinator is running
            # wait to switch the conversation context
            await self._idle.wait()
            if self._explorations.active != active:
                # If active tab has changed while we waited
                # skip the sync
                return
            # If conversation was already updated, resync conversation
            if self._last_synced == active:
                if self._conversations:
                    self._conversations[active] = self.interface.objects
                else:
                    self._root_conversation = self.interface.objects
        if (event.new if event is not None else tab):
            # Explorations Tab
            if active < len(self._conversations):
                conversation = self._conversations[active]
            else:
                conversation = self._snapshot_messages()
            self._exports.visible = True
        else:
            # Overview tab
            self._exports.visible = False
            conversation = self._root_conversation
            # We must mark the last synced as None to ensure
            # we do not resync with the Overview tab conservation
            active = None
        self._set_conversation(conversation)
        self.interface._chat_log.scroll_to_latest()
        self._last_synced = active

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
            self._contexts.pop(i)
            self._conversations.pop(i)
            self._titles.pop(i)
            if i == self._last_synced:
                await self._update_conversation(tab=1)
            break

    async def _set_context(self, event=None, old=None, new=None):
        active = event.new if new is None else new
        if len(self._conversations) == 0:
            return
        await self._idle.wait()
        if active != self._explorations.active:
            return
        if self._last_synced == active:
            self._conversations[active] = self.interface.objects
        else:
            self._conversations[event.old if old is None else old] = self._snapshot_messages()
        conversation = self._conversations[active]
        with hold():
            self._set_conversation(conversation)
            self._notebook_export.param.update(
                filename = f"{self._titles[active].replace(' ', '_')}.ipynb"
            )
            self._exports.visible = True
        self.interface._chat_log.scroll_to_latest()
        self._last_synced = active

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

    def _table_explorer(self):
        from panel_gwalker import GraphicWalker

        table_select = MultiChoice(
            placeholder="Select table(s)", sizing_mode='stretch_width',
            max_height=200, max_items=5
        )
        explore_button = Button(
            name='Explore table(s)', icon='chart-bar', button_type='primary', align='center',
            disabled=table_select.param.value.rx().rx.not_()
        )
        input_row = Row(table_select, explore_button)

        source_map = {}
        def update_source_map(_, __, sources, init=False):
            selected = list(table_select.value)
            deduplicate = len(sources) > 1
            new = {}
            for source in sources:
                source_tables = source.get_tables()
                for t in source_tables:
                    if deduplicate:
                        t = f'{source.name}{SOURCE_TABLE_SEPARATOR}{t}'
                    if t.rsplit(SOURCE_TABLE_SEPARATOR, 1)[-1] not in source_map and not init and not len(selected) > table_select.max_items and state.loaded:
                        selected.append(t)
                    new[t] = source
            source_map.clear()
            source_map.update(new)
            selected = selected if len(selected) == 1 else []
            table_select.param.update(options=list(source_map), value=selected)
            input_row.visible = bool(source_map)
        memory.on_change('sources', update_source_map)
        update_source_map(None, None, memory['sources'], init=True)

        def explore_table_if_single(event):
            """
            If only one table is uploaded, help the user load it
            without requiring them to click twice. This step
            only triggers when the Upload in the Overview tab is used,
            i.e. does not trigger with uploads through the SourceAgent
            """
            if len(table_select.options) == 1:
                explore_button.param.trigger("value")

        controls = SourceControls(select_existing=False, cancellable=False, clear_uploads=True, multiple=True, name='Upload')
        controls.param.watch(explore_table_if_single, "add")
        tabs = Tabs(controls, dynamic=True, sizing_mode='stretch_both', design=Material)

        @param.depends(explore_button, watch=True)
        def get_explorers(load):
            if not load:
                return
            with explore_button.param.update(loading=True), self.interface.param.update(loading=True):
                explorers = []
                for table in table_select.value:
                    source = source_map[table]
                    if len(memory['sources']) > 1 and SOURCE_TABLE_SEPARATOR in table:
                        _, table = table.rsplit(SOURCE_TABLE_SEPARATOR, 1)
                    pipeline = Pipeline(
                        source=source, table=table, sql_transforms=[SQLLimit(limit=100_000)]
                    )
                    table_label = f"{table[:25]}..." if len(table) > 25 else table
                    walker = GraphicWalker(
                        pipeline.param.data, sizing_mode='stretch_both', min_height=800,
                        kernel_computation=True, name=table_label, tab='data'
                    )
                    explorers.append(walker)

                tabs.objects = explorers + [controls]
                table_select.value = []

        self._overview_intro = Markdown(
            OVERVIEW_INTRO,
            margin=(0, 0, 10, 0),
            sizing_mode='stretch_width',
            visible=self.interface.param["objects"].rx.len() <= 1
        )
        return Column(
            self._overview_intro,
            input_row,
            tabs,
            sizing_mode='stretch_both',
        )

    async def _add_exploration(self, title: str, memory: _Memory):
        n = len(self._explorations)
        active = self._explorations.active
        self._titles.append(title)
        self._contexts.append(memory)
        self.interface.objects = conversation = list(self.interface.objects)
        self._conversations.append(conversation)
        tab_title = f"{title[:25]}..." if len(title) > 25 else title
        self._explorations.append((tab_title, Column(name=title, sizing_mode='stretch_both', loading=True)))
        self._notebook_export.filename = f"{title.replace(' ', '_')}.ipynb"
        if n:
            self._conversations[active] = self._snapshot_messages(new=True)
        else:
            self._root_conversation = self._snapshot_messages(new=True)
        self._last_synced = n
        self._explorations.active = n
        self._output.active = 1
        await self._update_conversation(tab=1)

    def _add_outputs(self, exploration: Column, outputs: list[LumenOutput] | str, memory: _Memory):
        if "sql" in memory:
            sql = memory.rx("sql")
            sql_pane = Markdown(
                param.rx('```sql\n{sql}\n```').format(sql=sql),
                margin=(-15, 0, 0, 0), sizing_mode='stretch_width', name='SQL'
            )
            if sql.count('\n') > 10:
                sql_pane = Column(
                    sql_pane, max_height=275, scroll='y-auto', name='SQL'
                )
            if len(exploration) and exploration[0].name == 'SQL':
                exploration[0] = sql_pane
            else:
                exploration.insert(0, sql_pane)

        content = []
        if exploration.loading:
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
        if exploration.loading:
            tabs = Tabs(*content, dynamic=True, active=len(outputs))
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
            new_exploration = False
            prev = index = self._explorations.active if len(self._explorations) else -1
            local_memory = prev_memory.clone()
            local_memory["outputs"] = outputs = []

            async def render_plan(_, old, new):
                nonlocal index, new_exploration
                plan = local_memory["plan"]
                if any(step.expert_or_tool == 'SQLAgent' for step in plan.steps):
                    await self._add_exploration(plan.title, local_memory)
                    index = len(self._explorations)-1
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
                exploration = self._explorations[index]
                self._add_outputs(exploration, added, local_memory)
                exploration.loading = False
                outputs[:] = new
            local_memory.on_change('outputs', render_output)

            # Remove exploration on error if no outputs have been
            # added yet and we launched a new exploration
            async def remove_output(_, __, ___):
                nonlocal index, new_exploration
                del memory['__error__']
                if outputs or not new_exploration:
                    return
                # Merge the current conversation with the conversation
                # that initiated the new exploration
                conversation = self._conversations.pop()
                if prev == -1:
                    self._root_conversation = conversation
                else:
                    self._conversations[prev] = conversation
                self._last_synced = prev
                self._contexts.pop()
                self._titles.pop()
                with hold():
                    self._explorations.active = prev
                    await self._set_context(old=index, new=prev)
                    self._explorations.pop(-1)
                    if len(self._titles) == 0:
                        self._output.active = 0
                index -= 1
                new_exploration = False
            local_memory.on_change('__error__', remove_output)

            try:
                self._idle.clear()
                with self._coordinator.param.update(memory=local_memory):
                    await callback(contents, user, instance)
            finally:
                if self._conversations:
                    self._conversations[index] = self.interface.objects
                else:
                    self._root_conversation = self.interface.objects
                self._idle.set()
                local_memory.remove_on_change('plan', render_plan)
                local_memory.remove_on_change('__error__', remove_output)
                if not new_exploration:
                    prev_memory.update(local_memory)
        return wrapper
