from __future__ import annotations

import asyncio

from functools import partial
from io import StringIO
from pathlib import Path
from typing import Any

import param

from panel.chat.feed import PLACEHOLDER_SVG
from panel.config import config, panel_extension
from panel.io.document import hold
from panel.io.state import state
from panel.layout import Column, FlexBox
from panel.pane import SVG, Markdown
from panel.util import edit_readonly
from panel.viewable import (
    Child, Children, Viewable, Viewer,
)
from panel_gwalker import GraphicWalker
from panel_material_ui import (
    Button, ChatFeed, ChatInterface, ChatMessage, Column as MuiColumn, Dialog,
    FileDownload, IconButton, MenuList, NestedBreadcrumbs, Page, Paper, Popup,
    Row, Switch, Tabs, ToggleIcon, Typography,
)
from panel_splitjs import HSplit, MultiSplit, VSplit

from ..pipeline import Pipeline
from ..sources import Source
from ..sources.duckdb import DuckDBSource
from ..util import log
from .agents import (
    AnalysisAgent, AnalystAgent, ChatAgent, DocumentListAgent, SQLAgent,
    TableListAgent, ValidationAgent, VegaLiteAgent,
)
from .config import (
    DEMO_MESSAGES, GETTING_STARTED_SUGGESTIONS, PROVIDED_SOURCE_NAME,
    SOURCE_TABLE_SEPARATOR,
)
from .context import TContext
from .controls import (
    DownloadControls, SourceCatalog, TableExplorer, UploadControls,
)
from .coordinator import Coordinator, Plan, Planner
from .export import export_notebook
from .llm import Llm, Message, OpenAI
from .llm_dialog import LLMConfigDialog
from .logs import ChatLogs
from .report import ActorTask, Report
from .utils import log_debug, wrap_logfire
from .vector_store import VectorStore
from .views import AnalysisOutput, LumenOutput, SQLOutput

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
💾 Switch to report mode to see all the results and export them  
📤 Export your session as a reproducible notebook  
"""  # noqa: W291

EXPLORATIONS_INTRO = """
🧪 **Explorations**

* **Analyze** one or more datasets through interactive tables and visualizations.
* **Launch** a new exploration tab for each SQL query result.
* **Branch** from any result using the breadcrumbs menu — each exploration keeps its own context.
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
        TableListAgent, ChatAgent, DocumentListAgent, AnalystAgent, SQLAgent, VegaLiteAgent, ValidationAgent
    ], doc="""List of default agents which will always be added.""")

    demo_inputs = param.List(default=DEMO_MESSAGES, doc="""
        List of instructions to demo the Coordinator.""")

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

    logs_db_path = param.String(default=None, constant=True, doc="""
        The path to the log file that will store the messages exchanged with the LLM.""")

    notebook_preamble = param.String(default='', doc="""
        Preamble to add to exported notebook(s).""")

    suggestions = param.List(default=GETTING_STARTED_SUGGESTIONS, doc="""
        Initial list of suggestions of actions the user can take.""")

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
        interface: ChatInterface | None = None,
        **params
    ):
        params["log_level"] = params.get("log_level", self.param["log_level"].default).upper()
        super().__init__(**params)
        self._configure_context(data)
        self._configure_interface(interface)
        self._configure_coordinator()
        self._configure_session()
        self._render_page()

    @classmethod
    def _resolve_data(
        cls, data: DataT | list[DataT] | dict[DataT] | None
    ) -> list[Source]:
        """
        Resolve the data into a list of sources.

        Parameters
        ----------
        data: DataT | list[DataT] | dict[DataT] | None
            The data to resolve.

        Returns
        -------
        list[Source]
            List of sources.
        """
        if data is None:
            return []
        elif isinstance(data, dict):
            data = data.items()
        elif not isinstance(data, list):
            data = [data]

        sources = []
        remote = False
        mirrors, tables = {}, {}
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

        return sources

    @wrap_logfire(span_name="Chat Invoke")
    async def _chat_invoke(
        self, messages: list[Message], user: str, instance: ChatInterface, context: TContext | None = None
    ):
        log_debug(f"New Message: \033[91m{messages!r}\033[0m", show_sep="above")
        context = self.context if context is None else context
        await self._coordinator.respond(messages, context)

    def _configure_context(self, data: DataT | list[DataT] | dict[DataT] | None):
        """
        Configure the context with the resolved data.

        Parameters
        ----------
        data: DataT | list[DataT] | dict[DataT] | None
            The data to resolve.
        """
        self.context["sources"] = sources = self._resolve_data(data)
        if sources:
            self.context["source"] = sources[-1]

    @wrap_logfire(span_name="Initialize LLM")
    async def _initialize_new_llm(self):
        """Initialize the new LLM after provider change."""
        try:
            await self.llm.initialize(log_level=self.log_level)
            self.interface.disabled = False
        except Exception:
            import traceback
            traceback.print_exc()

    def _destroy(self, session_context):
        """
        Cleanup on session destroy.

        Parameters
        ----------
        session_context: SessionContext
            The session context.
        """

    def _configure_session(self):
        """
        Configure the Panel session state.
        """
        log_debug("New Session: \033[92mStarted\033[0m", show_sep="above")
        self._session_id = id(self)
        state.onload(self._initialize_new_llm)
        if state.curdoc and state.curdoc.session_context:
            state.on_session_destroyed(self._destroy)

    def _configure_logs(self, interface):
        if not self.logs_db_path:
            interface.message_params["show_reaction_icons"] = False
            self._logs = None
            return

        def on_message(message, instance):
            message_id = id(message)
            message_index = instance.objects.index(message)

            def update_on_reaction(event):
                self._logs.update_status(
                    message_id=message_id,
                    liked="like" in event.new,
                    disliked="dislike" in event.new,
                )

            message.param.watch(update_on_reaction, 'reactions')
            self._logs.upsert(
                session_id=self._session_id,
                message_id=message_id,
                message_index=message_index,
                message_user=message.user,
                message_content=message.serialize(),
            )

        self._logs = ChatLogs(filename=self.logs_db_path)
        interface.message_params["reaction_icons"] = {"like": "thumb-up", "dislike": "thumb-down"}
        interface.post_hook = on_message

    def _transition_to_chat(self):
        """Transition from splash screen to chat interface."""
        if self._split[0][0] is not self.interface:
            self._split[0][0] = self.interface

    def _configure_interface(self, interface):
        def on_undo(instance, _):
            if not self._logs:
                return
            count = instance._get_last_user_entry_index()
            messages = instance[-count:]
            for message in messages:
                self._logs.update_status(message_id=id(message), removed=True)

        def on_rerun(instance, _):
            if not self._logs:
                return
            count = instance._get_last_user_entry_index() - 1
            messages = instance[-count:]
            for message in messages:
                self._logs.update_status(message_id=id(message), removed=True)

        def on_clear(instance, _):
            pass

        def on_submit(event=None, instance=None):
            chat_input = self.interface.active_widget
            uploaded = chat_input.value_uploaded
            user_prompt = chat_input.value_input
            if not user_prompt and not uploaded:
                return

            with self.interface.param.update(disabled=True, loading=True):
                self._transition_to_chat()

                old_sources = self.context.get("sources", [])
                if uploaded:
                    # Process uploaded files through UploadControls if any exist
                    upload_controls = UploadControls(context=self.context, clear_uploads=True)
                    upload_controls._generate_file_cards({
                        key: value["value"] for key, value in uploaded.items()
                    })
                    upload_controls.param.trigger("add")
                    chat_input.value_uploaded = {}
                    new_sources = [
                        source for source in self.context.get("sources", [])
                        if source not in old_sources
                    ]
                    source_names = [f"**{src.name}**: {', '.join(src.get_tables())}" for src in new_sources]
                    source_view = Markdown(
                        "Added sources:\n" + "\n".join(f"- {name}" for name in source_names),
                        sizing_mode="stretch_width"
                    )
                    msg = Column(chat_input.value, source_view) if user_prompt else source_view
                else:
                    msg = Markdown(user_prompt)

                with hold():
                    self.interface.send(msg, respond=bool(user_prompt))
                    chat_input.value_input = ""

        if interface is None:
            interface = ChatInterface(
                callback=self._chat_invoke,
                callback_exception="raise",
                load_buffer=5,
                on_submit=on_submit,
                show_button_tooltips=True,
                show_button_name=False,
                sizing_mode="stretch_both"
            )
        else:
            interface = self.interface
            interface.callback = self._chat_invoke
            interface.on_submit = on_submit
        interface.button_properties = {
            "undo": {"callback": on_undo},
            "rerun": {"callback": on_rerun},
            "clear": {"callback": on_clear},
        }
        interface.disabled = False
        existing_actions = interface.active_widget.actions
        interface.active_widget.actions = {
            'manage_llm': {
                'icon': 'auto_awesome',
                'callback': self._open_llm_dialog,
                'label': 'Select LLM'
            },
            **existing_actions,
        }
        self._configure_logs(interface)
        self.interface = interface
        # Watch for drag-and-drop file uploads and open manage data dialog
        interface.active_widget.param.watch(self._handle_file_drop, 'value_uploaded')

    def _handle_file_drop(self, event):
        """Handle drag-and-drop file uploads by opening the manage data dialog."""
        if not event.new:  # No files uploaded
            return

        # Generate file cards in the upload controls
        self._upload_controls._generate_file_cards({
            key: value["value"] for key, value in event.new.items()
        })

        # Open the manage data dialog
        self._sources_dialog_content.open = True

        # Set the breadcrumbs to "Upload" to show the uploaded files
        self._source_breadcrumbs.active = (0, 1,)

    def _handle_upload_successful(self, event):
        """Handle successful file upload by switching to Source Catalog."""
        active = self._source_breadcrumbs.active
        # Navigate to Source Catalog under current selection (Upload or Download)
        self._source_breadcrumbs.active = (0, active[0], 0)

    def _update_source_content(self, event):
        """Update the source dialog content based on breadcrumb selection."""
        active = event.new
        if len(active) == 1:
            self._source_breadcrumbs.active = (0, 0)  # Default to Upload
        elif len(active) == 2:
            # Upload (0,) or Download (1,)
            if active[1] == 0:
                self._source_content[:] = [self._upload_controls]
            else:
                self._source_content[:] = [self._download_controls]
        elif len(active) == 3:
            # Upload > Source Catalog (0, 0) or Download > Source Catalog (1, 0)
            self._source_content[:] = [self._source_catalog]

    def _configure_coordinator(self):
        # Set up table upload callbacks on all control classes
        UploadControls.table_upload_callbacks = self.table_upload_callbacks
        DownloadControls.table_upload_callbacks = self.table_upload_callbacks

        log.setLevel(self.log_level)

        agents = self.agents
        agent_types = {type(agent) for agent in agents}
        for default_agent in self.default_agents:
            if default_agent not in agent_types:
                # only add default agents if they are not already set by the user
                agents.append(default_agent)

        if self.analyses:
            agents.append(AnalysisAgent(analyses=self.analyses))

        self._coordinator = self.coordinator(
            agents=agents,
            context=self.context,
            interface=self.interface,
            llm=self.llm,
            tools=self.tools,
            within_ui=True,
            vector_store=self.vector_store,
            document_vector_store=self.document_vector_store,
            **self.coordinator_params
        )

    def _render_header(self) -> list[Viewable]:
        return []

    def _render_main(self) -> list[Viewable]:
        num_sources = len(self.context.get("sources", []))
        if num_sources == 0:
            prefix_text = "Add your dataset to begin, then"
        else:
            prefix_text = f"{num_sources} source{'s' if num_sources > 1 else ''} connected;"

        # Create help icon button for intro message
        self._info_button = IconButton(
            icon="help",
            description="Help & Getting Started",
            margin=(5, 10, 10, -8),
            on_click=self._open_info_dialog,
            variant="text",
            sx={"p": "6px 0", "minWidth": "32px"},
        )

        self._cta = Typography(
            f"{prefix_text} ask any question, or select a quick action below."
        )

        self._splash = MuiColumn(
            Paper(
                Row(
                    Typography(
                        "Illuminate your data",
                        disable_anchors=True,
                        variant="h1"
                    ),
                    self._info_button
                ),
                self._cta,
                self.interface._widget,
                max_width=850,
                styles={'margin': 'auto'},
                sx={'p': '0 20px 20px 20px'}
            ),
            margin=(0, 5, 0, 0),
            sx={'display': 'flex', 'align-items': 'center'},
            height_policy='max'
        )

        self._main = Column(self._splash, sizing_mode='stretch_both', align="center")

        if self.suggestions:
            self._add_suggestions_to_footer(
                self.suggestions,
                num_objects=1,
                inplace=True,
                analysis=False,
                append_demo=True,
                hide_after_use=False
            )

        # Create info dialog with intro message
        self._info_dialog = Dialog(
            Markdown(UI_INTRO_MESSAGE, sizing_mode="stretch_width"),
            close_on_click=True,
            show_close_button=True,
            sizing_mode='stretch_width',
            width_option='md',
            title="Welcome to Lumen AI"
        )

        # Set up actions for the ChatAreaInput speed dial
        self._source_catalog = SourceCatalog(context=self.context)

        # Create separate upload and download controls with reference to catalog
        self._upload_controls = UploadControls(context=self.context, source_catalog=self._source_catalog)
        self._download_controls = DownloadControls(context=self.context, source_catalog=self._source_catalog)

        # Watch for output changes from both controls
        self._upload_controls.param.watch(self._sync_sources, 'outputs')
        self._download_controls.param.watch(self._sync_sources, 'outputs')

        # Watch for successful uploads to switch to Source Catalog
        self._upload_controls.param.watch(self._handle_upload_successful, 'upload_successful')
        self._download_controls.param.watch(self._handle_upload_successful, 'upload_successful')

        # Content container that switches based on breadcrumb selection
        self._source_content = MuiColumn(self._upload_controls, sizing_mode="stretch_width")

        self._source_breadcrumbs = NestedBreadcrumbs(
                items=[{
                    'label': 'Manage Data', 'icon': 'folder', "selectable": False, 'items': [{
                        "label": "Upload",
                        "icon": "cloud_upload",
                        "items": [
                            {"label": "Source Catalog", "icon": "storage"}
                        ]
                    },
                    {
                        "label": "Download",
                        "icon": "cloud_download",
                        "items": [
                            {"label": "Source Catalog", "icon": "storage"}
                        ]
                    }]
                }],
            active=(0, 0,),
            margin=(0, 10, 10, 10),
        )
        self._source_breadcrumbs.param.watch(self._update_source_content, 'active')

        self._sources_dialog_content = Dialog(
            MuiColumn(self._source_breadcrumbs, self._source_content, sizing_mode="stretch_width"),
            close_on_click=True, show_close_button=True, width_option='lg', title="Manage Data",
            sizing_mode='stretch_width',
        )

        self._notebook_export = FileDownload(
            callback=self._export_notebook,
            description="Export Notebook",
            icon_size="1.8em",
            filename=f"{self.title.replace(' ', '_')}.ipynb", # TODO
            label=" ",
            margin=(5, 10, 0, 0),
            sx={"p": "6px 0", "minWidth": "32px", "& .MuiButton-icon": {"ml": 0, "mr": 0}},
            styles={'z-index': '1000', 'margin-left': 'auto'},
            variant="text"
        )

        self._explorations = MenuList(
            items=[self._exploration], value=self.param._exploration, show_children=False,
            dense=True, label='Explorations', margin=0, sizing_mode='stretch_width',
            sx={".mui-light .MuiBox-root": {"backgroundColor": "var(--mui-palette-grey-100)"}},
        )
        self._explorations.param.watch(self._cleanup_explorations, 'items')
        self._explorations.param.watch(self._update_conversation, 'active')
        self._explorations.param.watch(self._sync_active, 'value')
        self._explorations.on_action('remove', self._delete_exploration)

        # Create LLM configuration dialog
        self._llm_dialog = LLMConfigDialog(
            llm=self.llm,
            llm_choices=self.llm_choices,
            provider_choices=self.provider_choices,
            on_llm_change=self._on_llm_change,
            agent_types=[type(agent) for agent in self._coordinator.agents],
        )

        return [self._main, self._sources_dialog_content, self._llm_dialog, self._info_dialog]

    def _render_contextbar(self) -> list[Viewable]:
        return []

    def _render_sidebar(self) -> list[Viewable]:
        return []

    def _render_page(self):
        self._page = Page(
            contextbar_open=False,
            contextbar=self._render_contextbar(),
            header=self._render_header(),
            main=self._render_main(),
            sidebar=self._render_sidebar(),
            title=self.title,
            sidebar_width=65,
            sidebar_resizable=False,
            sx={"&.mui-light .sidebar": {"bgcolor": "var(--mui-palette-grey-50)"}}
        )

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
        await self._coordinator.sync(self.context)
        self._source_catalog.sync(self.context)

        num_sources = len(self.context.get("sources", []))
        if num_sources == 0:
            prefix_text = "Add your dataset to begin, then"
        else:
            prefix_text = f"{num_sources} source{'s' if num_sources > 1 else ''} connected;"
        self._cta.object = f"{prefix_text} ask any question, or select a quick action below."

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

    def _open_sources_dialog(self, event=None):
        """Open the sources dialog when the vector store badge is clicked."""
        self._sources_dialog_content.open = True

    def _open_info_dialog(self, event=None):
        """Open the info dialog when the info button is clicked."""
        self._info_dialog.open = True

    def _add_suggestions_to_footer(
        self,
        suggestions: list[str],
        num_objects: int = 2,
        inplace: bool = True,
        analysis: bool = False,
        append_demo: bool = True,
        hide_after_use: bool = True,
        context: TContext | None = None
    ):
        async def hide_suggestions(_=None):
            if len(self.interface.objects) > num_objects:
                suggestion_buttons.visible = False

        async def use_suggestion(event):
            button = event.obj
            with button.param.update(loading=True), self.interface.active_widget.param.update(loading=True):
                # Find the original suggestion tuple to get the text
                button_index = suggestion_buttons.objects.index(button)
                if button_index < len(suggestions):
                    suggestion = suggestions[button_index]
                    contents = suggestion[1] if isinstance(suggestion, tuple) else suggestion
                else:
                    contents = button.name
                if hide_after_use:
                    suggestion_buttons.visible = False
                    if event.new > 1:  # prevent double clicks
                        return

                self._transition_to_chat()

                if not analysis:
                    self.interface.send(contents)
                    return

                for agent in self.agents:
                    if isinstance(agent, AnalysisAgent):
                        break
                else:
                    log_debug("AnalysisAgent not available.")
                    return

                plan = Plan(
                    ActorTask(
                        agent,
                        instruction=contents,
                        title=contents
                    ),
                    history=[{"role": "user", "content": contents}],
                    context=context,
                    coordinator=self._coordinator,
                    title=contents,
                )
                await self._execute_plan(plan)

        async def run_demo(event):
            if hide_after_use:
                suggestion_buttons.visible = False
                if event.new > 1:  # prevent double clicks
                    return
            with self.interface.active_widget.param.update(loading=True):
                for demo_message in self.demo_inputs:
                    while self.interface.disabled:
                        await asyncio.sleep(1.25)
                    self.interface.active_widget.value = demo_message
                    await asyncio.sleep(2)

        suggestion_buttons = FlexBox(
            *[
                Button(
                    label=suggestion[1] if isinstance(suggestion, tuple) else suggestion,
                    icon=suggestion[0] if isinstance(suggestion, tuple) else None,
                    variant="outlined",
                    on_click=use_suggestion,
                    margin=5,
                    disabled=self.interface.param.loading
                )
                for suggestion in suggestions
            ],
            margin=(5, 5),
        )

        if append_demo and self.demo_inputs:
            suggestion_buttons.append(Button(
                name="Show a demo",
                icon="play_arrow",
                button_type="primary",
                variant="outlined",
                on_click=run_demo,
                margin=5,
                disabled=self.interface.param.loading
            ))
        disable_js = "cb_obj.origin.disabled = true; setTimeout(() => cb_obj.origin.disabled = false, 3000)"
        for b in suggestion_buttons:
            b.js_on_click(code=disable_js)

        if len(self.interface):
            message = self.interface.objects[-1]
            if inplace:
                footer_objects = message.footer_objects or []
                message.footer_objects = footer_objects + [suggestion_buttons]
        else:
            self._splash[0].append(suggestion_buttons)

        self.interface.param.watch(hide_suggestions, "objects")

    async def _add_analysis_suggestions(self, plan: Plan):
        pipeline = plan.out_context["pipeline"]
        current_analysis = plan.out_context.get("analysis")

        # Clear current_analysis unless the last message is the same AnalysisOutput
        if current_analysis and plan.views:
            if not any(out.analysis is current_analysis for out in plan.views if isinstance(out, AnalysisOutput)):
                current_analysis = None

        allow_consecutive = getattr(current_analysis, '_consecutive_calls', True)
        applicable_analyses = []
        for analysis in self._coordinator._analyses:
            if await analysis.applies(pipeline) and (allow_consecutive or analysis is not type(current_analysis)):
                applicable_analyses.append(analysis)
        if not applicable_analyses:
            return
        self._add_suggestions_to_footer(
            [f"Apply {analysis.__name__}" for analysis in applicable_analyses],
            append_demo=False,
            analysis=True,
            context=plan.out_context,
            hide_after_use=False,
            num_objects=len(self.interface.objects),
        )

    def __panel__(self):
        return self._main

    def _create_view(self, server: bool = False):
        if server:
            panel_extension(
                *{ext for agent in self._coordinator.agents for ext in agent._extensions}
            )
            return self._page
        return super()._create_view()

    def _repr_mimebundle_(self, include=None, exclude=None):
        panel_extension(
            *{ext for agent in self._coordinator.agents for ext in agent._extensions}, notifications=True
        )
        return self._create_view()._repr_mimebundle_(include, exclude)

    def show(self, **kwargs):
        return self._create_view(server=True).show(**kwargs)

    def servable(self, title: str | None = None, **kwargs):
        server = bool(state.curdoc and state.curdoc.session_context)
        return self._create_view(server=server).servable(title, **kwargs)


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

    context = param.Dict(allow_refs=True)

    conversation = Children()

    parent = param.Parameter()

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

    _exploration = param.Dict()

    def _export_notebook(self):
        nb = export_notebook(self._exploration['view'].plan.views, preamble=self.notebook_preamble)
        return StringIO(nb)

    @hold()
    def _handle_sidebar_event(self, item):
        if item["id"] == "home":
            self._toggle_report_mode(False)
            self._exploration = self._explorations.items[0]
        elif item["id"] == "exploration":
            self._toggle_report_mode(False)
            self._sidebar_menu.update_item(item, active=True, icon="forum")
            self._sidebar_menu.update_item(self._sidebar_menu.items[3], active=False, icon="description_outlined")
            self._update_home()
        elif item["id"] == "report":
            self._toggle_report_mode(True)
            self._sidebar_menu.update_item(self._sidebar_menu.items[2], active=False, icon="forum_outlined")
            self._sidebar_menu.update_item(item, active=True, icon="description")
            self._update_home()
        elif item["id"] == "data":
            self._open_sources_dialog()
        elif item["id"] == "llm":
            self._open_llm_dialog()
        elif item["id"] == "settings":
            self._settings_popup.open = True

    @param.depends("_exploration", watch=True)
    def _update_home(self):
        is_home = self._exploration["view"] is self._home
        if not hasattr(self, '_page'):
            return
        if not is_home:
            self._transition_to_chat()
        home, _, exploration, report = self._sidebar_menu.items[:4]
        self._sidebar_menu.update_item(home, active=is_home, icon="home" if is_home and not report["active"] else "home_outlined")
        self._sidebar_menu.update_item(exploration, active=True, icon="forum_outlined" if report["active"] else "forum")

    def _render_sidebar(self) -> list[Viewable]:
        switches = []
        cot = Switch(label='Chain of Thought')
        switches.append(cot)
        sql_agent = next(
            (agent for agent in self._coordinator.agents if isinstance(agent, SQLAgent)),
            None
        )
        if sql_agent:
            sql_planning = Switch(label='SQL Planning')
            sql_agent.planning_enabled = sql_planning
            switches.append(sql_planning)
        validation = Switch(label='Validation Step')
        switches.append(validation)
        self._coordinator.param.update(
            verbose=cot,
            validation_enabled=validation
        )

        self._settings_popup = Popup(
            *switches,
            anchor_origin={"horizontal": "right", "vertical": "bottom"},
            styles={"z-index": '1300'},
            theme_config={"light": {"palette": {"background": {"paper": "var(--mui-palette-grey-50)"}}}, "dark": {}}
        )
        self._sidebar_collapse = collapse = ToggleIcon(
            value=True, active_icon="chevron_right", icon="chevron_left", styles={"margin-top": "auto", "margin-left": "auto"}, margin=5
        )
        self._sidebar_menu = menu = MenuList(
            items=[
                {"label": "Home", "icon": "home", "id": "home", "active": True},
                None,
                {"label": "Exploration", "icon": "forum", "id": "exploration", "active": True},
                {"label": "Report", "icon": "description_outlined", "id": "report", "active": False},
                None,
                {"label": "Manage Data", "icon": "drive_folder_upload_outlined", "id": "data"},
                {"label": "Configure LLM", "icon": "psychology_outlined", "id": "llm"},
                {"label": "Settings", "icon": "settings_outlined", "id": "settings"}
            ],
            attached=[self._settings_popup],
            collapsed=collapse,
            highlight=False,
            margin=0,
            on_click=self._handle_sidebar_event,
            sx={
                "& .MuiButtonBase-root.MuiListItemButton-root, & .MuiButtonBase-root.MuiListItemButton-root.collapsed": {"p": 2},
                ".MuiListItemIcon-root > .MuiIcon-root": {
                    "color": "var(--mui-palette-primary-dark)",
                    "fontSize": "28px"
                }
            }
        )

        return [menu, collapse]

    def _render_page(self):
        super()._render_page()
        self._page.sidebar_width = self._sidebar_collapse.rx().rx.where(61, 183)

    def _render_main(self) -> list[Viewable]:
        main = super()._render_main()

        # Render home page
        self._explorations_intro = Markdown(
            EXPLORATIONS_INTRO,
            margin=(0, 0, 0, 20),
            sizing_mode='stretch_width',
        )
        self._explorer = TableExplorer(context=self.context)
        self._explorer.param.watch(self._add_exploration_from_explorer, "add_exploration")
        self._home.view = MuiColumn(self._explorations_intro, self._explorer)

        # Menus
        self._breadcrumbs = NestedBreadcrumbs(
            items=[self._exploration],
            value=self.param._exploration,
            margin=(10, 0, 0, 13)
        )
        self._breadcrumbs.param.watch(self._sync_active, 'value')

        # Main Area
        self._notebook_export.disabled = self.param._exploration.rx()['view'].rx.is_(self._home)
        self._output = Paper(
            Row(self._breadcrumbs, self._notebook_export, sizing_mode="stretch_width"),
            self._home,
            elevation=2,
            margin=(5, 10, 5, 5),
            height_policy='max',
            sizing_mode="stretch_both"
        )
        self._split = HSplit(
            Column(self._splash),
            self._output,
            collapsed=1,
            expanded_sizes=(40, 60),
            show_buttons=True,
            sizing_mode='stretch_both',
            stylesheets=[".gutter-horizontal > .divider { background: unset }"]
        )
        self._main[:] = [self._split]
        return main

    async def _add_exploration_from_explorer(self, event: param.parameterized.Event):
        sql_out = self._explorer.create_sql_output()
        if sql_out is None:
            return

        table = self._explorer.table_slug.split(SOURCE_TABLE_SEPARATOR)[0]
        prev = self._explorations.value
        sql_agent = next(agent for agent in self._coordinator.agents if isinstance(agent, SQLAgent))
        sql_task = ActorTask(sql_agent, title=f"Load {table}")
        plan = Plan(sql_task, title=f"Explore {table}")
        exploration = await self._add_exploration(plan, self._home)

        with hold():
            self._transition_to_chat()
            self.interface.send(f"Add exploration for the `{table}` table", respond=False)
            out_context = await sql_out.render_context()
            watcher = plan.param.watch(partial(self._add_views, exploration), "views")
            try:
                sql_task.param.update(
                    views=[sql_out], out_context=out_context, status="success"
                )
            finally:
                plan.param.unwatch(watcher)
                await self._postprocess_exploration(plan, exploration, prev, is_new=True)

    def _configure_session(self):
        self._home = self._last_synced = Exploration(
            context=self.context,
            title='Home',
            conversation=self.interface.objects
        )
        self._exploration = {'label': 'Home', 'icon': 'home', 'view': self._home, 'items': []}
        super()._configure_session()
        self._idle = asyncio.Event()
        self._idle.set()

    def _sync_active(self, event: param.parameterized.Event):
        if event.new is not self._exploration:
            self._exploration = event.new

    def _toggle_report_mode(self, active: bool):
        """Toggle between regular and report mode."""
        if active:
            self._main[:] = [
                Row(
                    Paper(
                        self._explorations,
                        sizing_mode="stretch_height",
                        sx={"borderRadius": 0},
                        theme_config={"light": {"palette": {"background": {"paper": "var(--mui-palette-grey-100)"}}}, "dark": {}},
                        width=200,
                    ),
                    Report(
                        *(exploration['view'].plan for exploration in self._explorations.items[1:])
                    )
                )
            ]
        else:
            self._main[:] = [self._split]

    def _delete_exploration(self, item):
        self._explorations.items = [it for it in self._explorations.items if it is not item]

    def _destroy(self, session_context):
        """
        Cleanup on session destroy
        """
        for c in self._explorations.items[1:]:
            c['view'].context.clear()

    async def _update_conversation(self, event=None):
        exploration = self._explorations.value['view']
        if exploration is self._home:
            self._split[0][0] = self._splash
        self._output[1:] = [exploration]

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
        to = -2 if new else None
        messages = []
        for msg in self.interface.objects[:to]:
            if isinstance(msg, ChatMessage):
                avatar = msg.avatar
                if isinstance(avatar, SVG) and avatar.object is PLACEHOLDER_SVG:
                    continue
            messages.append(msg)
        return messages

    async def _add_exploration(self, plan: Plan, parent: Exploration) -> Exploration:
        # Sanpshot previous conversation
        is_home = parent is self._home
        if is_home:
            parent.conversation = []
        else:
            parent.conversation = self._snapshot_messages(new=True)
        conversation = list(self.interface.objects)

        tabs = Tabs(
            ('Overview', "Waiting on data..."),
            dynamic=True,
            sizing_mode='stretch_both',
            loading=plan.param.running
        )
        output = MultiSplit(tabs, sizing_mode='stretch_both')
        exploration = Exploration(
            context=plan.param.out_context,
            conversation=conversation,
            parent=parent,
            plan=plan,
            title=plan.title,
            view=output
        )

        view_item = {
            'label': plan.title,
            'view': exploration,
            'icon': "insights",
            'actions': [{'action': 'remove', 'label': 'Remove', 'icon': 'delete'}],
            'items': []
        }
        items = self._explorations.items + [view_item]
        with hold():
            self._breadcrumbs.value["items"].append(view_item)
            self._breadcrumbs.param.trigger("items")
            self.interface.objects = conversation
            self._idle.set()
            self._explorations.param.update(items=items)
            self._exploration = view_item
            self._idle.clear()
            self._output[1:] = [output]
            self._notebook_export.filename = f"{plan.title.replace(' ', '_')}.ipynb"
            await self._update_conversation()
        self._last_synced = exploration
        return exploration

    def _render_pop_out(self, exploration: Exploration, view: VSplit, title: str):
        title_view = Typography(
            title.upper(),
            color="primary",
            sx={"border-bottom": "2px solid var(--mui-palette-primary-main)"},
            variant="subtitle1"
        )
        standalone = Column(title_view, view)
        tabs = exploration.view[0]
        position = len(tabs)

        @hold()
        def pop_out(event):
            if view in tabs:
                tabs.active = max(tabs.active-1, 0)
                tabs.remove(view)
                exploration.view.append(standalone)
                pop_button.param.update(
                    description="Reattach",
                    icon="open_in_browser"
                )
            else:
                exploration.view.remove(standalone)
                tabs.insert(position, (title, view))
                tabs.active = position
                pop_button.param.update(
                    description="Open in new pane",
                    icon="open_in_new"
                )

        pop_button = IconButton(
            description="Pop-out", icon="open_in_new", icon_size="1.1em", size="small",
            margin=(5, 0, 0, 0), on_click=pop_out, styles={"margin-left": "auto"}
        )
        return pop_button

    def _render_view(self, exploration: Exploration, view: LumenOutput) -> VSplit:
        title = view.title or type(view).__name__.replace('Output', '')
        if len(title) > 25:
            title = f"{title[:25]}..."

        task = next(task for task in exploration.plan if view in task.views)
        controls = view.render_controls(task, self.interface)
        editor = Column(controls, view.editor)
        vsplit = VSplit(
            editor, view,
            expanded_sizes=(20, 80),
            sizes=(20, 80),
            sizing_mode="stretch_both",
            styles={"overflow": "auto"}
        )
        controls.append(self._render_pop_out(exploration, vsplit, title))
        return (title, vsplit)

    def _find_view_in_tabs(self, exploration: Exploration, out: LumenOutput):
        tabs = exploration.view[0]
        for i, tab in enumerate(tabs):
            content = tab[1] if isinstance(tab, tuple) and len(tab) > 1 else tab
            if isinstance(content, VSplit) and out.view in content:
                return i
        return None

    def _find_view_in_popped_out(self, exploration: Exploration, out: LumenOutput):
        for i, standalone in enumerate(exploration.view[1:], start=1):
            if isinstance(standalone, Column) and len(standalone) > 1:
                vsplit = standalone[1]
                if isinstance(vsplit, VSplit) and out.view in vsplit:
                    return i
        return None

    def _add_views(self, exploration: Exploration, event: param.parameterized.Event):
        tabs = exploration.view[0]
        content = []
        for view in event.new:
            if not isinstance(view, LumenOutput) or view in event.old:
                continue
            if tabs and isinstance(view, SQLOutput) and not isinstance(tabs[0], GraphicWalker):
                tabs[0] = ("Overview", view.render_explorer())
            title, vsplit = self._render_view(exploration, view)
            content.append((title, vsplit))

        tabs.extend(content)
        tabs.active = max(tabs.active, len(tabs)-1)
        if self._split.collapsed:
            self._split.param.update(
                sizes=self._split.expanded_sizes,
            )

    def _update_views(self, exploration: Exploration, event: param.parameterized.Event):
        current = [view for view in event.new if isinstance(view, LumenOutput)]
        old = [view for view in event.old if isinstance(view, LumenOutput)]

        idx = None
        tabs = exploration.view[0]
        content = list(zip(tabs._names, tabs, strict=False))
        popped_out_to_remove = []

        for view in old:
            if view in current:
                continue
            tab_idx = self._find_view_in_tabs(exploration, view)
            if tab_idx is not None:
                idx = tab_idx
                content.pop(tab_idx)
            popped_idx = self._find_view_in_popped_out(exploration, view)
            if popped_idx is not None:
                popped_out_to_remove.append(popped_idx)

        for popped_idx in sorted(popped_out_to_remove, reverse=True):
            exploration.view.pop(popped_idx)

        for view in current:
            if view in old:
                tab_idx = self._find_view_in_tabs(exploration, view)
                if tab_idx is not None:
                    idx = tab_idx
                continue
            title, vsplit = self._render_view(exploration, view)
            content.insert((idx or 0)+1, (title, vsplit))
        tabs[:] = content
        tabs.active = len(tabs)-1

    async def _execute_plan(self, plan: Plan, rerun: bool = False):
        prev = self._explorations.value
        parent = prev["view"]

        # Check if we are adding to existing exploration or creating a new one
        new_exploration = any("pipeline" in step.actor.output_schema.__required_keys__ for step in plan)

        if rerun:
            exploration = parent
            plan.reset()
            watcher = plan.param.watch(partial(self._add_views, exploration), "views")
        elif new_exploration:
            exploration = await self._add_exploration(plan, parent)
            watcher = plan.param.watch(partial(self._add_views, exploration), "views")
        else:
            exploration = parent
            if parent.plan is not None:
                if plan.steps_layout:
                    plan.steps_layout.header[:] = [
                        Typography(
                            "🔀 Combined tasks with previous checklist",
                            css_classes=["todos-title"],
                            margin=0,
                            styles={"font-weight": "normal", "font-size": "1.1em"}
                        )
                    ]
                partial_plan = plan
                plan = parent.plan.merge(plan)
                partial_plan.cleanup()
            watcher = None

        with plan.param.update(interface=self.interface):
            await plan.execute()

        if watcher:
            plan.param.unwatch(watcher)

        await self._postprocess_exploration(plan, exploration, prev, is_new=new_exploration)

    async def _postprocess_exploration(self, plan: Plan, exploration: Exploration, prev: dict, is_new: bool = False):
        if "__error__" in plan.out_context:
            # On error we have to sync the conversation, unwatch the plan,
            # and remove the exploration if it was newly created
            replan_button = Button(
                label="Replan", icon="alt_route", on_click=lambda _: self.interface._click_rerun(),
                description="Replan and generate a new execution strategy"
            )
            rerun_button = Button(
                label="Rerun", icon="autorenew", on_click=lambda _: state.execute(partial(self._execute_plan, plan, rerun=True)),
                description="Rerun with the same plan and context"
            )
            last_message = self.interface.objects[-1]
            footer_objects = last_message.footer_objects or []
            last_message.footer_objects = footer_objects + [rerun_button, replan_button]
            exploration.parent.conversation = exploration.conversation
            del plan.out_context['__error__']
        else:
            if "pipeline" in plan.out_context:
                await self._add_analysis_suggestions(plan)
            if is_new:
                plan.param.watch(partial(self._update_views, exploration), "views")

    @wrap_logfire(span_name="Chat Invoke")
    async def _chat_invoke(
        self, messages: list[Message], user: str, instance: ChatInterface, context: TContext | None = None
    ):
        log_debug(f"New Message: \033[91m{messages!r}\033[0m", show_sep="above")
        self._idle.clear()
        try:
            exploration = self._exploration['view']
            plan = await self._coordinator.respond(messages, exploration.context)
            if plan is not None:
                await self._execute_plan(plan)
        finally:
            self._exploration['view'].conversation = self.interface.objects
            self._idle.set()
