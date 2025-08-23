from __future__ import annotations

import asyncio
import logging
import traceback

from functools import partial
from io import StringIO
from pathlib import Path
from typing import TYPE_CHECKING, Any

import param

from panel.chat.feed import PLACEHOLDER_SVG
from panel.config import config, panel_extension
from panel.io.document import hold
from panel.io.resources import CSS_URLS
from panel.io.state import state
from panel.layout import Column as PnColumn, HSpacer
from panel.pane import SVG, Markdown
from panel.param import ParamMethod
from panel.util import edit_readonly
from panel.viewable import Child, Children, Viewer
from panel_gwalker import GraphicWalker
from panel_material_ui import (
    Accordion, Button, ChatFeed, ChatInterface, ChatMessage, Column, Dialog,
    Divider, FileDownload, IconButton, MenuList, MenuToggle, MultiChoice, Page,
    Paper, Row, Switch, Tabs, ToggleIcon,
)

from ..pipeline import Pipeline
from ..sources import Source
from ..sources.duckdb import DuckDBSource
from ..transforms.sql import SQLLimit
from ..util import log
from .agents import (
    AnalysisAgent, AnalystAgent, ChatAgent, DocumentListAgent, SourceAgent,
    SQLAgent, TableListAgent, ValidationAgent, VegaLiteAgent,
)
from .components import SourceCatalog, SplitJS
from .config import PROVIDED_SOURCE_NAME, SOURCE_TABLE_SEPARATOR
from .controls import SourceControls
from .coordinator import Coordinator, Plan, Planner
from .export import (
    export_notebook, make_md_cell, make_preamble, render_cells, write_notebook,
)
from .llm import Llm, OpenAI
from .llm_dialog import LLMConfigDialog
from .memory import _Memory, memory
from .report import Report
from .vector_store import VectorStore

if TYPE_CHECKING:
    from .views import LumenOutput

DataT = str | Path | Source | Pipeline


UI_INTRO_MESSAGE = """
👋 Click a suggestion below or upload a data source to get started!

Lumen AI combines large language models (LLMs) with specialized agents to help you explore, analyze,
and visualize data without writing code.

On the chat interface...

💬 Ask questions in plain English to generate SQL queries and visualizations  
🔍 Inspect and validate results through conversation  
📝 Get summaries and key insights from your data  
🧩 Apply custom analyses with a click of a button  

If unsatisfied with the results hover over the <span class="material-icons-outlined" style="font-size: 1.2em;">add_circle</span> menu and...

<span class="material-icons">repeat</span> Use the Rerun button to re-run the last query  
<span class="material-icons">undo</span> Use the Undo butto to remove the last query  
<span class="material-icons">delete</span> Use the Clear button to start a new session  

Click the toggle, or drag the right edge, to expand the results area and...

🌐 Explore data with [Graphic Walker](https://docs.kanaries.net/graphic-walker) - filter, sort, download  
💾 Navigate, reorder and delete explorations in the contextbar  
📤 Export your session as a reproducible notebook  
"""  # noqa: W291

EXPLORATIONS_INTRO = """
🧪 **Explorations**

- Explorations are analyses applied to one or more datasets.
- An exploration consists of interactive tables and visualizations.
- New explorations are launched in a new tab for each SQL query result.
- Each exploration maintains its own context, so you can branch off an existing result by navigating to that tab.
"""


class TableExplorer(Viewer):
    """
    TableExplorer provides a high-level entrypoint to explore tables in a split UI.
    It allows users to load tables, explore them using Graphic Walker, and then
    interrogate the data via a chat interface.
    """

    interface = param.ClassSelector(class_=ChatFeed, doc="""
        The interface for the Coordinator to interact with.""")

    def __init__(self, **params):
        super().__init__(**params)
        self._table_select = MultiChoice(
            label="Select table(s) to preview", sizing_mode='stretch_width',
            max_height=200, max_items=5, margin=0
        )
        self._explore_button = Button(
            name='Explore table(s)', icon='add_chart', button_type='primary', icon_size="2em",
            disabled=self._table_select.param.value.rx().rx.not_(), on_click=self._update_explorers,
            margin=(0, 0, 0, 10), width=200, align='end'
        )
        self._input_row = Row(self._table_select, self._explore_button)
        self._source_map = {}
        memory.on_change("sources", self._update_source_map)
        self._update_source_map(init=True)

        self._tabs = Tabs(dynamic=True, sizing_mode='stretch_both')
        self._layout = Column(
            self._input_row, self._tabs, sizing_mode='stretch_both',
        )

    def _update_source_map(self, key=None, old=None, sources=None, init=False):
        if sources is None:
            sources = memory["sources"]
        selected = list(self._table_select.value)
        deduplicate = len(sources) > 1
        new = {}

        # Build the source map for UI display
        for source in sources:
            tables = source.get_tables()
            for table in tables:
                if deduplicate:
                    table = f'{source.name}{SOURCE_TABLE_SEPARATOR}{table}'

                if (table.split(SOURCE_TABLE_SEPARATOR, maxsplit=1)[-1] not in self._source_map and
                    not init and not len(selected) > self._table_select.max_items and state.loaded):
                    selected.append(table)
                new[table] = source

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

            self._tabs.objects = explorers
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
        TableListAgent, ChatAgent, DocumentListAgent, AnalystAgent, SourceAgent, SQLAgent, VegaLiteAgent, ValidationAgent
    ], doc="""List of default agents which will always be added.""")

    export_functions = param.Dict(default={}, doc="""
       Dictionary mapping from name of exporter to export function.""")

    interface = param.ClassSelector(class_=ChatFeed, doc="""
        The interface for the Coordinator to interact with.""")

    llm = param.ClassSelector(class_=Llm, default=OpenAI(), doc="""
        The LLM provider to be used by default""")

    llm_choices = param.List(default=[], doc="""
        List of available LLM model choices to show in the configuration dialog.""")

    provider_choices = param.Dict(default={}, doc="""
        Available LLM providers to show in the configuration dialog.""")

    log_level = param.ObjectSelector(default='DEBUG', objects=['DEBUG', 'INFO', 'WARNING', 'ERROR'], doc="""
        The log level to use.""")

    logs_db_path = param.String(default=None, doc="""
        The path to the log file that will store the messages exchanged with the LLM.""")

    notebook_preamble = param.String(default='', doc="""
        Preamble to add to exported notebook(s).""")

    source_controls = param.ClassSelector(
        class_=SourceControls, default=SourceControls,
        instantiate=False, is_instance=False, doc="""
        The source controls to use for managing data sources.""")

    table_upload_callbacks = param.Dict(default={}, doc="""
        Dictionary mapping from file extensions to callback function,
        e.g. {"hdf5": ...}. The callback function should accept the file bytes,
        table alias, and filename, add or modify the `source` in memory, and return a bool
        (True if the table was successfully uploaded).""")

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
        The vector store to use for tools unrelated to documents. If not provided, a new one will be created
        or inferred from the tools provided."""
    )

    document_vector_store = param.ClassSelector(
        class_=VectorStore, default=None, doc="""
        The vector store to use for document-related tools. If not provided, a new one will be created
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
        SourceAgent.source_controls = self.source_controls
        SourceControls.table_upload_callbacks = self.table_upload_callbacks
        self._source_controls = self.source_controls(memory=memory)
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
        # Create consolidated settings menu toggle
        self._settings_menu = MenuToggle(
            items=[
                {
                    'label': 'Chain of Thought',
                    'icon': 'visibility_off',
                    'active_icon': 'visibility',
                    'toggled': False
                },
                {
                    'label': 'SQL Planning',
                    'icon': 'close',
                    'active_icon': 'check',
                    'toggled': True
                }
            ],
            align="center",
            color="light",
            description="Setings",
            icon="settings",
            size="large",
            margin=(0, 5, 0, 5),
            persistent=True,
            sx={'.MuiButton-startIcon': {'mr': 0}},
            variant='text'
        )
        self._settings_menu.param.watch(self._toggle_sql_planning, 'toggled')
        self._coordinator = self.coordinator(
            agents=agents,
            interface=self.interface,
            llm=self.llm,
            tools=self.tools,
            logs_db_path=self.logs_db_path,
            within_ui=True,
            vector_store=self.vector_store,
            document_vector_store=self.document_vector_store,
            verbose=self._settings_menu.param.toggled.rx().rx.pipe(lambda toggled: 0 in toggled),
            **self.coordinator_params
        )

        self.interface = self._coordinator.interface
        levels = logging.getLevelNamesMapping()
        if levels.get(self.log_level) < 20:
            self.interface.callback_exception = "verbose"
        self.interface.disabled = True

        self._notebook_export = FileDownload(
            callback=self._export_notebook,
            color="light",
            description="Export Notebook",
            icon_size="1.8em",
            filename=f"{self.title.replace(' ', '_')}.ipynb",
            label=" ",
            margin=(12, 0, 10, 5),
            sx={"p": "6px 0", "minWidth": "32px"},
            styles={'z-index': '1000'},
            variant="text"
        )

        # Create help icon button for intro message
        self._info_button = IconButton(
            icon="help",
            description="Help & Getting Started",
            margin=(5, 10, 10, -8),
            on_click=self._open_info_dialog,
            variant="text",
            sx={"p": "6px 0", "minWidth": "32px"},
        )
        self._coordinator._main[0][0] = Row(self._coordinator._main[0][0], self._info_button)

        # Create info dialog with intro message
        self._info_dialog = Dialog(
            Markdown(UI_INTRO_MESSAGE, sizing_mode="stretch_width"),
            close_on_click=True,
            show_close_button=True,
            sizing_mode='stretch_width',
            width_option='md',
            title="Welcome to Lumen AI"
        )

        self._exports = Row(
            self._notebook_export, *(
                Button(
                    label=label,
                    on_click=lambda _, e=e: e(self.interface),
                    stylesheets=['.bk-btn { padding: 4.5px 6px;']
                )
                for label, e in self.export_functions.items()
            )
        )
        self._main = Column(self._coordinator, sizing_mode='stretch_both', align="center")
        self._source_agent = next((agent for agent in self._coordinator.agents if isinstance(agent, SourceAgent)), None)

        # Set up actions for the ChatAreaInput speed dial
        self._setup_actions()
        self._table_lookup_tool = None  # Will be set after coordinator is initialized
        self._source_catalog = SourceCatalog()
        self._source_accordion = Accordion(
            ("Add Sources", self._source_controls), ("View Sources", self._source_catalog),
            margin=(-30, 10, 0, 10), sizing_mode="stretch_both", toggle=True, active=[0]
        )
        self._sources_dialog_content = Dialog(
            self._source_accordion, close_on_click=True, show_close_button=True,
            sizing_mode='stretch_width', width_option='lg',
        )

        # Create LLM configuration dialog
        self._llm_dialog = LLMConfigDialog(
            llm=self.llm,
            llm_choices=self.llm_choices,
            provider_choices=self.provider_choices,
            on_llm_change=self._on_llm_change,
            agent_types=[type(agent) for agent in self._coordinator.agents],
        )

        # Initialize _report_toggle for compatibility with header
        self._report_toggle = None
        self._contextbar = None

        memory.on_change("sources", self._update_source_catalog)

        # LLM status is now managed through actions

        if state.curdoc and state.curdoc.session_context:
            state.on_session_destroyed(self._destroy)
        state.onload(self._setup_llm_and_watchers)

    def _setup_actions(self):
        """Set up actions for the ChatAreaInput speed dial."""
        existing_actions = self.interface.active_widget.actions
        self.interface.active_widget.actions = {
            'manage_data': {
                'icon': 'cloud_upload',
                'callback': self._open_sources_dialog,
                'label': 'Manage Data'
            },
            'manage_llm': {
                'icon': 'auto_awesome',
                'callback': self._open_llm_dialog,
                'label': 'Select LLM'
            },
            **existing_actions,
        }

    def _open_llm_dialog(self, event=None):
        """Open the LLM configuration dialog when the LLM chip is clicked."""
        self._llm_dialog.open = True

    def _on_llm_change(self, new_llm):
        """Handle LLM provider change from the dialog."""
        # Update the UI's LLM reference
        self.llm = new_llm

        # Update the coordinator's LLM reference
        self._coordinator.llm = new_llm

        # Update all agent LLMs
        for agent in self._coordinator.agents:
            agent.llm = new_llm

        # Initialize the new LLM asynchronously
        import param
        param.parameterized.async_executor(self._initialize_new_llm)

    async def _initialize_new_llm(self):
        """Initialize the new LLM after provider change."""
        try:
            await self.llm.initialize(log_level=self.log_level)
            self.interface.disabled = False
        except Exception:
            import traceback
            traceback.print_exc()

    async def _setup_llm_and_watchers(self):
        """Initialize LLM and set up reactive watchers for TableLookup readiness."""
        try:
            await self.llm.initialize(log_level=self.log_level)
            self.interface.disabled = False
        except Exception:
            traceback.print_exc()

    def _open_sources_dialog(self, event=None):
        """Open the sources dialog when the vector store badge is clicked."""
        self._sources_dialog_content.open = True

    def _open_info_dialog(self, event=None):
        """Open the info dialog when the info button is clicked."""
        self._info_dialog.open = True

    def _destroy(self, session_context):
        """
        Cleanup on session destroy
        """
        if self._table_lookup_tool:
            self._table_lookup_tool.param.unwatch(self._update_data_sources_action, '_ready')

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
                if src.endswith(('.parq', '.parquet', '.csv', '.json', '.tsv', '.jsonl', '.ndjson')):
                    table = src
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
        memory["sources"] = sources

    def show(self, **kwargs):
        return self._create_view(server=True).show(**kwargs)

    def _create_view(self, server: bool | None = None):
        config.css_files.append(CSS_URLS['font-awesome'])
        if (state.curdoc and state.curdoc.session_context) or server is True:
            panel_extension(
                *{ext for agent in self._coordinator.agents for ext in agent._extensions}
            )
            self._page = Page(
                title=self.title,
                header=[
                    HSpacer(),
                    self._exports,
                    *([self._report_toggle] if self._report_toggle else []),
                    self._settings_menu,
                    Divider(
                        orientation="vertical", height=30, margin=(17, 5, 17, 5),
                        sx={'border-color': 'white', 'border-width': '1px'}
                    )
                ],
                main=[self._main, self._sources_dialog_content, self._llm_dialog, self._info_dialog],
                contextbar=[] if self._contextbar is None else [self._contextbar],
                contextbar_open=False,
            )
            self._page.servable()
            return self._page
        return super()._create_view()

    def _trigger_source_agent(self, event=None):
        if self._source_agent:
            param.parameterized.async_executor(partial(self._source_agent.respond, []))
            self.interface._chat_log.scroll_to_latest()

    def _update_source_catalog(self, *args, **kwargs):
        """
        Update the sources dialog content when memory sources change.
        """
        self._source_catalog.sources = memory["sources"]
        self._source_accordion.active = [1]

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

    llm_choices = param.List(default=[], doc="""
        List of available LLM model choices to show in the configuration dialog.""")

    provider_choices = param.Dict(default={}, doc="""
        Available LLM providers to show in the configuration dialog.""")


class Exploration(param.Parameterized):

    context = param.ClassSelector(class_=_Memory)

    conversation = Children()

    plan = param.ClassSelector(class_=Plan)

    title = param.String(default="")

    subtitle = param.String(default="")

    view = Child()

    def __panel__(self):
        return self.view


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
        self._explorations = MenuList(
            dense=True, label='Explorations', margin=(-50, 0, 0, 0), sizing_mode='stretch_width'
        )
        self._reorder_switch = Switch(
            label='Edit', styles={'margin-left': 'auto', 'z-index': '999'},
            disabled=self._explorations.param.items.rx.len()<2
        )
        self._reorder_switch.param.watch(self._toggle_reorder, 'value')
        self._explorations.on_action('up', self._move_up)
        self._explorations.on_action('down', self._move_down)
        self._explorations.on_action('remove', self._delete_exploration)
        self._explorer = TableExplorer(interface=self.interface)
        self._explorations_intro = Markdown(
            EXPLORATIONS_INTRO,
            margin=(0, 0, 10, 10),
            sizing_mode='stretch_width',
            visible=self._explorations.param["items"].rx.bool().rx.not_()
        )
        self._report_toggle = ToggleIcon(
            icon="chat",
            active_icon="summarize",
            color="light",
            description="Toggle Report Mode",
            value=False,
            margin=(13, 0, 10, 0),
            visible=self.interface.param["objects"].rx().rx.len() > 0
        )
        self._report_toggle.param.watch(self._toggle_report_mode, ['value'])

        self._last_synced = self._home = Exploration(
            context=memory,
            title='Home',
            conversation=self.interface.objects,
            view=Column(self._explorations_intro, self._explorer)
        )
        home_item = {'label': 'Home', 'icon': 'home', 'view': self._home}
        self._explorations.param.update(items=[home_item], value=home_item)
        self._explorations.param.watch(self._cleanup_explorations, ['items'])
        self._explorations.param.watch(self._update_conversation, ['active'])
        self._global_notebook_export = FileDownload(
            callback=self._global_export_notebook,
            color="light",
            description="Export Notebook",
            icon_size="1.8em",
            filename=f"{self.title.replace(' ', '_')}.ipynb",
            label=" ",
            margin=(14, 0, 10, 5),
            styles={'z-index': '1000'},
            sx={"p": "6px 0", "minWidth": "32px"},
            variant="text"
        )
        self._exports.visible = False
        self._output = Paper(
            self._home.view, elevation=2, margin=(5, 10, 5, 5), sx={'p': 3, 'pb': 2},
            height_policy='max', sizing_mode="stretch_both"
        )
        self._split = SplitJS(
            left=self._coordinator,
            right=self._output,
            invert=self.chat_ui_position != 'left',
            sizing_mode='stretch_both'
        )
        self._report = Column()
        self._main = Column(self._split)
        self._contextbar = Column(self._reorder_switch, self._explorations)
        self._idle = asyncio.Event()
        self._idle.set()

    def _move_up(self, item):
        items = list(self._explorations.items)
        index = items.index(item)
        if index <= 1:
            return
        items.pop(index)
        items.insert(index-1, item)
        self._explorations.items = items

    def _move_down(self, item):
        items = list(self._explorations.items)
        index = items.index(item)
        items.pop(index)
        items.insert(index+1, item)
        self._explorations.items = items

    def _toggle_reorder(self, event):
        items = self._explorations.items[:1]
        reorder_actions = [
            {'action': 'up', 'label': 'Move Up', 'inline': True, 'icon': 'keyboard_arrow_up'},
            {'action': 'down', 'label': 'Move Down', 'inline': True, 'icon': 'keyboard_arrow_down'}
        ]
        for item in self._explorations.items[1:]:
            item = dict(item)
            if event.new:
                item['actions'] = item['actions'] + reorder_actions
            else:
                item['actions'] = [
                    action for action in item['actions'] if action['label'] not in ('Move Up', 'Move Down')
                ]
            items.append(item)
        self._explorations.items = items

    def _toggle_report_mode(self, event):
        """Toggle between regular and report mode."""
        if event.new:
            self._exports[0] = self._global_notebook_export
            self._main[:] = [Report(
                subtasks=[
                    exploration['view'].plan for exploration in self._explorations.items[1:]
                ]
            )]
        else:
            self._exports[0] = self._notebook_export
            self._main[:] = [self._split]

    def _toggle_sql_planning(self, event: param.Event):
        """Toggle SQL planning mode for SQLAgent."""
        planning_enabled = 1 in event.new

        # Update the SQLAgent directly if it exists
        sql_agent = next(
            (agent for agent in self._coordinator.agents if isinstance(agent, SQLAgent)),
            None
        )
        if sql_agent:
            sql_agent.planning_enabled = planning_enabled

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
        for item in self._explorations.items:
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
                exploration.conversation = list(self.interface.objects)
            elif self._last_synced is not None:
                # Otherwise snapshot the conversation
                self._last_synced.conversation = self._snapshot_messages()

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
        for i, (old, new) in enumerate(zip(event.old, event.new, strict=False)):
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
        is_home = last_exploration is self._home
        if is_home:
            last_exploration.conversation = []
        else:
            last_exploration.conversation = self._snapshot_messages(new=True)
        conversation = list(self.interface.objects)

        # Create new exploration
        output = Column(sizing_mode='stretch_both', loading=True)
        exploration = Exploration(
            context=memory,
            conversation=conversation,
            title=title,
            view=output
        )
        view_item = {
            'label': title,
            'view': exploration,
            'icon': None,
            'actions': [{'action': 'remove', 'label': 'Remove', 'icon': 'delete'}]
        }
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
                sql_pane = PnColumn(
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
                if any(step.actor in ('SQLAgent', 'DbtslAgent') for step in plan.steps):
                    # Expand the contextbar when the first exploration is created
                    await self._add_exploration(plan.title, local_memory)
                    new_exploration = True

            def sync_available_sources_memory(_, __, sources):
                """
                For cases when the user uploads a dataset through SourceAgent
                this will update the available_sources in the global memory
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
                    plan = await callback(contents, user, instance)
                self._explorations.value['view'].plan = plan
            finally:
                self._explorations.value['view'].conversation = self.interface.objects
                self._idle.set()
                local_memory.remove_on_change('plan', render_plan)
                local_memory.remove_on_change('__error__', remove_output)
                if not new_exploration:
                    prev_memory.update(local_memory)
        return wrapper
