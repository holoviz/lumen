from __future__ import annotations

import asyncio
import datetime as dt
import logging

from functools import partial
from io import StringIO
from pathlib import Path
from typing import Any

import param

from panel.chat.feed import PLACEHOLDER_SVG
from panel.config import config, panel_extension
from panel.io.document import hold
from panel.io.resources import CSS_URLS
from panel.io.state import state
from panel.layout import Column, HSpacer
from panel.pane import SVG, Markdown
from panel.param import ParamMethod
from panel.util import edit_readonly
from panel.viewable import Child, Children, Viewer
from panel_gwalker import GraphicWalker
from panel_material_ui import (
    Accordion, Button, ChatFeed, ChatInterface, ChatMessage,
    Column as MuiColumn, Dialog, Divider, FileDownload, IconButton, MenuList,
    MenuToggle, Page, Paper, Row, Switch, Tabs, ToggleIcon,
)
from panel_splitjs import VSplit

from ..pipeline import Pipeline
from ..sources import Source
from ..sources.duckdb import DuckDBSource
from ..util import log
from .agents import (
    AnalysisAgent, AnalystAgent, ChatAgent, DocumentListAgent, SourceAgent,
    SQLAgent, TableListAgent, ValidationAgent, VegaLiteAgent,
)
from .components import SplitJS
from .config import PROVIDED_SOURCE_NAME, SOURCE_TABLE_SEPARATOR
from .context import TContext
from .controls import SourceCatalog, SourceControls, TableExplorer
from .coordinator import Coordinator, Plan, Planner
from .export import (
    export_notebook, make_md_cell, make_preamble, render_cells, write_notebook,
)
from .llm import Llm, Message, OpenAI
from .llm_dialog import LLMConfigDialog
from .report import Report
from .utils import wrap_logfire
from .vector_store import VectorStore
from .views import LumenOutput, SQLOutput

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

    context = param.Dict(default={})

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
        data: DataT | list[DataT] | dict[str, DataT] | None = None,
        **params
    ):
        params["log_level"] = params.get("log_level", self.param["log_level"].default).upper()
        super().__init__(**params)
        SourceAgent.source_controls = self.source_controls
        SourceControls.table_upload_callbacks = self.table_upload_callbacks
        self._source_controls = self.source_controls(context=self.context)
        self._source_controls.param.watch(self._sync_sources, 'outputs')
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
                    'icon': 'toggle_off',
                    'active_icon': 'toggle_on',
                    'toggled': False
                },
                {
                    'label': 'SQL Planning',
                    'icon': 'toggle_off',
                    'active_icon': 'toggle_on',
                    'toggled': True
                },
                {
                    'label': 'Validation Step',
                    'icon': 'toggle_off',
                    'active_icon': 'toggle_on',
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
        self._settings_menu.param.watch(self._toggle_validation_agent, 'toggled')
        self._coordinator = self.coordinator(
            agents=agents,
            context=self.context,
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
        self._source_catalog = SourceCatalog(context=self.context)
        self._source_accordion = Accordion(
            ("Add Sources", self._source_controls), ("View Sources", self._source_catalog),
            margin=(-30, 10, 0, 10), sizing_mode="stretch_width", toggle=True, active=[0]
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

        # LLM status is now managed through actions
        if state.curdoc and state.curdoc.session_context:
            state.on_session_destroyed(self._destroy)
        state.onload(self._initialize_new_llm)

    @param.depends('context', on_init=True, watch=True)
    async def _sync_sources(self, event=None):
        context = event.new if event else self.context
        if 'sources' in context:
            old_sources = self.context.get("sources", [self.context["source"]] if "source" in self.context else [])
            new_sources = [src for src in context["sources"] if src not in old_sources]
            self.context["sources"] = old_sources + new_sources

            all_slugs = set()
            for source in new_sources:
                tables = source.get_tables()
                for table in tables:
                    table_slug = f'{source.name}{SOURCE_TABLE_SEPARATOR}{table}'
                    all_slugs.add(table_slug)

            # Update visible_slugs, preserving existing visibility where possible
            # This ensures removed tables are filtered out, new tables are added
            current_visible = self.context.get('visible_slugs', set())
            if current_visible:
                # Keep intersection of current visible and available slugs
                # Plus add any new slugs that weren't previously available
                self.context['visible_slugs'] = current_visible.intersection(all_slugs) | (all_slugs - current_visible)
            else:
                # If no visible_slugs set, make all tables visible
                self.context['visible_slugs'] = all_slugs
        if "source" in context:
            if "source" in self.context:
                old_source = self.context["source"]
                if "sources" not in self.context:
                    self.context["sources"] = [old_source]
                elif old_source not in self.context["sources"]:
                    self.context["sources"].append(old_source)
            self.context["source"] = context["source"]
        if "table" in context:
            self.context["table"] = context["table"]
        if "document_sources" in context:
            new_docs = context["document_sources"]
            if "document_sources" not in self.context:
                self.context["document_sources"] = new_docs
            else:
                self.context["document_sources"].extend(new_docs)
        await self._explorer.sync()
        await self._source_catalog.sync()
        await self._coordinator.sync(self.context)

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
        param.parameterized.async_executor(self._initialize_new_llm)

    @wrap_logfire(span_name="Initialize LLM")
    async def _initialize_new_llm(self):
        """Initialize the new LLM after provider change."""
        try:
            await self.llm.initialize(log_level=self.log_level)
            self.interface.disabled = False
        except Exception:
            import traceback
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

    def _resolve_data(self, data: DataT | list[DataT] | dict[DataT] | None):
        if data is None:
            return
        elif isinstance(data, dict):
            data = data.items()
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
            elif isinstance(src, (str, Path, tuple)):
                if isinstance(src, tuple):
                    name, src = src
                    src = str(src)
                else:
                    name = src = str(src)
                if src.startswith('http'):
                    remote = True
                if src.endswith(('.parq', '.parquet', '.csv', '.json', '.tsv', '.jsonl', '.ndjson')):
                    table = src
                else:
                    raise ValueError(
                        f"Could not determine how to load {src} file."
                    )
                tables[name] = table
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
        self.context["sources"] = sources
        self.context["source"] = sources[-1]

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
        self._source_catalog.sources = self.context["sources"]
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

    context = param.Dict()

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
        self._explorer = TableExplorer(context=self.context)
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
            context=self.context,
            title='Home',
            conversation=self.interface.objects,
            view=MuiColumn(self._explorations_intro, self._explorer)
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
            sizes=(40, 60),
            expanded_sizes=(40, 60),
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
                *(exploration['view'].plan for exploration in self._explorations.items[1:])
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

    def _toggle_validation_agent(self, event: param.Event):
        """Toggle ValidationAgent usage."""
        validation_enabled = 2 in event.new

        # Update the coordinator's validation agent setting
        self._coordinator.validation_enabled = validation_enabled

    def _delete_exploration(self, item):
        self._explorations.items = [it for it in self._explorations.items if it is not item]

    def _destroy(self, session_context):
        """
        Cleanup on session destroy
        """
        for c in self._explorations.items[1:]:
            c['view'].context.clear()

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
        now = dt.datetime.now()
        title = f'# Lumen.ai - Chat Logs {now}'
        preamble = make_preamble(self.notebook_preamble, extensions=extensions, title=title)
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

    async def _add_exploration(self, plan: Plan):
        # Sanpshot previous conversation
        last_exploration = self._explorations.value['view']
        is_home = last_exploration is self._home
        if is_home:
            last_exploration.conversation = []
        else:
            last_exploration.conversation = self._snapshot_messages(new=True)
        conversation = list(self.interface.objects)

        # Create new exploration
        output = MuiColumn(sizing_mode='stretch_both', loading=True)
        exploration = Exploration(
            context=plan.context,
            conversation=conversation,
            plan=plan,
            title=plan.title,
            view=output
        )
        view_item = {
            'label': plan.title,
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
            self._notebook_export.filename = f"{plan.title.replace(' ', '_')}.ipynb"
            await self._update_conversation()
        self._last_synced = exploration
        return exploration

    def _add_outputs(self, exploration: Exploration, outputs: list[Any], context: TContext):
        view = exploration.view
        content = []
        if view.loading and 'pipeline' in context:
            pipeline = context['pipeline']
            content.append(
                ('Overview', GraphicWalker(
                    pipeline.param.data,
                    kernel_computation=True,
                    tab='data',
                    sizing_mode='stretch_both'
                ))
            )
        for out in outputs:
            if not isinstance(out, LumenOutput):
                continue
            title = out.title or type(out).__name__.replace('Output', '')
            if len(title) > 25:
                title = f"{title[:25]}..."
            output = ParamMethod(out.render, inplace=True, sizing_mode='stretch_both')
            vsplit = VSplit(out.editor, output, sizes=(20, 80), expanded_sizes=(20, 80), sizing_mode="stretch_both")
            content.append((title, vsplit))
        if view.loading:
            tabs = Tabs(*content, dynamic=True, active=len(content), sizing_mode='stretch_both')
            view.append(tabs)
        else:
            tabs = view[-1]
            tabs.extend(content)
        tabs.active = len(tabs)-1

    def _render_view(self, out: LumenOutput) -> VSplit:
        title = out.title or type(out).__name__.replace('Output', '')
        if len(title) > 25:
            title = f"{title[:25]}..."
        output = ParamMethod(out.render, inplace=True, sizing_mode='stretch_both')
        return (title, VSplit(out.editor, output, sizes=(20, 80), expanded_sizes=(20, 80), sizing_mode="stretch_both"))

    def _add_views(self, exploration: Exploration, event: param.parameterized.Event):
        outputs = [view for view in event.new if isinstance(view, LumenOutput) and view not in event.old]

        view = exploration.view
        if len(view) == 0:
            tabs = None
            content = [('Overview', "Waiting on data...")]
        else:
            tabs = view[0]
            content = []

        for out in outputs:
            title, vsplit = self._render_view(out)
            content.append((title, vsplit))
            if tabs and isinstance(out, SQLOutput) and not isinstance(tabs[0], GraphicWalker):
                tabs[0] = ("Overview", GraphicWalker(
                    out.component.param.data,
                    kernel_computation=True,
                    tab='data',
                    sizing_mode='stretch_both'
                ))

        if tabs is None:
            tabs = Tabs(*content, dynamic=True, active=len(content), sizing_mode='stretch_both')
            view.append(tabs)
        else:
            tabs.extend(content)
        tabs.active = len(tabs)-1
        if self._split.collapsed:
            self._split.param.update(
                collapsed=False,
                sizes=self._split.expanded_sizes,
            )

    def _update_views(self, exploration: Exploration, event: param.parameterized.Event):
        current = [view for view in event.new if isinstance(view, LumenOutput)]
        old = [view for view in event.old if isinstance(view, LumenOutput)]

        idx = None
        tabs = exploration.view[0]
        content = list(zip(tabs._names, tabs, strict=False))
        for out in old:
            if out in current:
                continue
            matches = [
                i for i, (_, vsplit) in enumerate(content)
                if isinstance(vsplit, VSplit) and out.editor in vsplit
            ]
            if matches:
                idx = matches[0]
                content.pop(idx)

        for out in current:
            if out in old:
                idx = next(i for i, tab in enumerate(tabs[1:]) if out.editor in tab) + 1
                continue
            title, vsplit = self._render_view(out)
            content.insert(idx+1, (title, vsplit))
        tabs[:] = content

    def _wrap_callback(self, callback):
        async def wrapper(messages: list[Message], user: str, instance: ChatInterface):
            prev = self._explorations.value
            self._idle.clear()
            try:
                plan = await callback(messages, prev["view"].context, user, instance)
                if any('pipeline' in step[0].outputs.__annotations__ for step in plan):
                    exploration = await self._add_exploration(plan)
                    new_exploration = True
                    watcher = plan.param.watch(partial(self._add_views, exploration), "views")
                else:
                    exploration = self._explorations.value['view']
                    new_exploration = False
                    plan = exploration.plan.merge(plan)
                    watcher = None
                with plan.param.update(interface=self.interface):
                    new, out_context = await plan.execute()
                if watcher:
                    plan.param.unwatch(watcher)
                if "__error__" in out_context:
                    # On error we have to sync the conversation, unwatch the plan,
                    # and remove the exploration if it was newly created
                    prev['view'].conversation = exploration.conversation
                    del out_context['__error__']
                    if new_exploration:
                        with hold():
                            self._explorations.param.update(
                                items=self._explorations.items[:-1],
                                value=prev
                            )
                            await self._update_conversation()
                else:
                    exploration.context = out_context
                    exploration.view.loading = False
                    plan.param.watch(partial(self._update_views, exploration), "views")
            finally:
                self._explorations.value['view'].conversation = self.interface.objects
                self._idle.set()
        return wrapper
