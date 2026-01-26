from __future__ import annotations

import asyncio

from contextlib import contextmanager
from functools import partial
from io import StringIO
from pathlib import Path
from typing import Any

import param

from panel.chat.feed import PLACEHOLDER_SVG
from panel.config import panel_extension
from panel.io.document import hold
from panel.io.state import state
from panel.layout import Column, FlexBox
from panel.pane import SVG, Markdown
from panel.util import edit_readonly
from panel.viewable import (
    Child, Children, Viewable, Viewer,
)
from panel_material_ui import (
    Alert, Breadcrumbs, Button, ChatFeed, ChatInterface, ChatMessage,
    Column as MuiColumn, Dialog, FileDownload, IconButton, MenuList, Page,
    Paper, Popup, Row, Select, Switch, Tabs, Typography,
)
from panel_splitjs import HSplit, MultiSplit, VSplit

from lumen.ai.agents.deck_gl import DeckGLAgent

from ..pipeline import Pipeline
from ..sources import Source
from ..sources.duckdb import DuckDBSource
from ..util import log
from .agents import (
    AnalysisAgent, BaseCodeAgent, ChatAgent, DocumentListAgent, SQLAgent,
    TableListAgent, ValidationAgent, VegaLiteAgent,
)
from .config import (
    DEMO_MESSAGES, GETTING_STARTED_SUGGESTIONS, PROVIDED_SOURCE_NAME,
    SOURCE_TABLE_SEPARATOR, SPLITJS_STYLESHEETS,
)
from .context import TContext
from .controls import (
    DownloadControls, SourceCatalog, TableExplorer, UploadControls,
)
from .coordinator import Coordinator, Plan, Planner
from .export import export_notebook
from .llm import (
    Llm, Message, OpenAI, get_available_llm,
)
from .llm_dialog import LLMConfigDialog
from .logs import ChatLogs
from .models import ErrorDescription
from .report import ActorTask, Report, Section
from .utils import log_debug, wrap_logfire
from .vector_store import VectorStore
from .views import AnalysisOutput, LumenOutput, SQLOutput

DataT = str | Path | Source | Pipeline


# Contextual help messages for different UI areas
# Each area has a ? icon that shows relevant help

SPLASH_HELP_HINT = "Click on Help on the left for more info."

# Help sections for breadcrumb navigation
HELP_NO_SOURCES = """**Welcome to Lumen AI!**

⚠️ **No data sources connected yet.**

**To get started:**

1. Click **Data Sources** in the left sidebar
2. Upload a file (CSV, Parquet, JSON, Excel, GeoJSON) or connect to a database
3. Once your data is loaded, come back here to start asking questions!

You can also drag & drop files directly into the chat interface.

**Once you have data, you can ask questions like:**

*Basic queries:*
- "Show me the top 10 rows"
- "What are the columns in the dataset?"
- "Give me a summary of the data"

*Visualizations:*
- "What's the average revenue by region? Plot as a bar chart"
- "Create a scatter plot of X vs Y, colored by category"

*Complex queries:*
- "Filter for records over $1000 and show the distribution by category"
- "Group by department and calculate average salary. Then plot the result."

---

➡️ After adding data, click **Navigation** above to learn how to move around the interface."""

HELP_GETTING_STARTED = """
Ask questions in plain English and Lumen AI automatically generates SQL queries and creates visualizations.

**🚀 Start exploring:**

- Type a question like "What datasets are available?"
- Click the quick action buttons below the chat
- Select a table and click **Explore** to start analyzing

**❓ Example questions:**

**Basic queries:**
- "Show me the top 10 rows"
- "What are the columns in the dataset?"
- "Give me a summary of the data"

**Visualizations:**
- "What's the average revenue by region? Plot as a bar chart"
- "Create a scatter plot of X vs Y, colored by category"
- "Show me a histogram of sales with 20 bins"

**Complex queries:**
- "Filter for records over $1000 and show the distribution by category"
- "Group by department and calculate average salary. Then plot the result."
- "Are there any outliers in the data?"
"""

HELP_INTERFACE = """
**🖥️ Panel Controls:**
- **< >** buttons - Collapse/expand the chat and results panels
- **^ v** arrows - Expand/collapse individual analyses

**📂 Sidebar:**
- **Explore** - Main chat and analysis mode (default)
- **Report** - View all explorations on one page
- **Sources** - Add or manage your data
- **Settings** - Control AI behavior settings and LLM configuration
- **Help** - Open this help dialog

**⌨️ Keyboard Shortcuts:**
- **Enter** - Send message
- **Shift + Enter** - New line in chat
"""

HELP_EXPLORATIONS = """
**🔍 What are Explorations?**

An Exploration is a persistent, contextual workspace for working with a specific dataset. It is created when a SQL query is generated and captures the full state of the interaction, including the conversation, generated analyses, visualizations, and other data artifacts.

Explorations can evolve over time, support multiple questions on the same data, and can be revisited or exported as a coherent unit.

Explorations start from the global context (available sources and metadata). If a question is a follow-up, the new exploration is nested under the parent; if it is not, Lumen creates a new top-level exploration.

**🧭 Navigation menu (left panel):**
- Shows all explorations in a persistent tree
- Click an exploration to switch context
- Follow-up explorations appear nested under their parent
- Use the item menu (⋮) to remove an exploration
"""

HELP_EDITOR = """
**📝 Two-pane editor:**

When you ask a question, you'll see two panes:

- **Top pane** = the spec (SQL query or YAML configuration)
  - Edit directly here to modify the analysis
  - Changes are applied in real-time to the preview below
  - Great for small tweaks like changing colors or sort order

- **Bottom pane** = live preview
  - Automatically updates when you change the spec
  - Shows tables, charts, or other visualizations
"""

HELP_RESULTS = """
**🛠️ Toolbar actions (top of each result):**

- **📋** - Copy YAML spec to clipboard
- **✨** - Ask the LLM to revise this analysis
- **💬** - Add highlights or callouts to visualizations (plots only)
- **↗** - Pop out into a separate pane for comparison

**📊 Table features:**
- Click **column headers** to sort (^ = ascending, v = descending)
- Use **pagination** (< >, page numbers) at the bottom to navigate large datasets
- All data is interactive - you can explore it directly

**🔄 Refining results:**
- **Rerun** (⟲ in the chat input + menu) - Re-execute the last query if there was an error
- **Undo** (↶ in the chat input + menu) - Remove the last response and try again
- **Continue the conversation** - Send a new message to refine: "Can you make that chart show only the top 5?"
- **Manual editing** - Edit the spec directly for precise control
"""

HELP_EXPORT = """
**📄 Export individual results:**

- **Export Output** ▼ (top of each result) - Download specific results:
  - **Tables**: SQL, CSV, or Excel formats
  - **Visualizations**: PNG, JPEG, PDF, SVG, or HTML

**💾 Save your work:**

- Use **Export Notebook** in the navigation menu to download everything as a Jupyter notebook
- Includes all SQL queries, visualizations, and results
- Perfect for documentation or sharing with colleagues
- Each exploration saves automatically during your session

**📋 Report Mode:**

- Click **Report** in the left sidebar
- View all explorations on one page
- Export everything as a single notebook
"""

HELP_TIPS = """
**🔗 Combine multiple requests:**
You can ask the AI to perform several steps at once.

- "Filter the data for 2023, create a bar chart of sales by region, and then summarize the top 3 regions."

**💡 Ask for suggestions:**
Not sure where to start? Ask:
- "What could be interesting to explore in this data?"
- "Give me 3 ideas for visualizations."

**✨ Refining results:**
You don't have to get it right the first time.
- "Can you make that chart show only the top 5?"
- "Change the color of the bars to green."
- "Add a trend line to the scatter plot."

**🗂️ Need more data?**
Click **Sources** in the left sidebar to add files or databases at any time.

**🧠 Chain of Thought:**
Enable **Chain of Thought** in Settings to see the AI's step-by-step reasoning.
- This is great for debugging or understanding complex queries.

---

✔️ You're all set! Close this dialog and start exploring.
"""

EXPLORATION_TOOLBAR_HELP = """
Toolbar actions for this exploration:

- **Export Output** — Download as CSV/Excel (tables) or PNG/SVG (visualizations)
- **Copy** — Copy the spec to clipboard
- **Sparkle** — Ask AI to revise the output
- **Annotations** — Add highlights to visualizations (plots only)
- **Pop-out** — Move to a separate pane

Drag the divider between editor and view to resize.
"""

REPORT_HELP = """
**Report Mode: View All Your Work**

Report mode displays all explorations in one scrollable page for review and presentation.

**Page Controls:**

- **▶** (Execute Report) - Run all analyses and generate fresh results
- **×** (Clear outputs) - Remove all output displays to start fresh
- **∨ ∧** (Collapse/Expand) - Show or hide all analysis sections at once

**Navigation:**

- Click an exploration name in the left sidebar to scroll to it
- All analyses appear in order with their results
- Scroll through the entire report or jump to specific sections

**Export:**

- Use **Export Notebook** in the navigation menu to download everything as one Jupyter notebook
- Includes all SQL queries, visualizations, and results
- Perfect for sharing findings or creating documentation

**To get back:**

- Click **Explore** in the left sidebar to return to chat mode
- Or click a specific exploration in the navigation menu to work on it
"""  # noqa: RUF001

DATA_SOURCES_HELP = """
**Add data to explore with Lumen AI.**

**Supported formats:**

- CSV, Parquet, JSON, Excel, GeoJSON files
- Remote URLs (https://example.com/data.csv)
- Databases via SQLAlchemy

**How to add data:**

1. **Upload Data** — Drag and drop files or click to browse
2. **Fetch Remote Data** — Enter URLs to fetch remote data

**Tips:**

- Upload .md or .txt files alongside data to give the AI context
- Multiple sources can be used together in queries
- Data connections persist for your session
"""

PREFERENCES_HELP = """
**Control how Lumen AI analyzes your questions.**

**Chain of Thought** — Show the AI's reasoning steps as it works

- Useful for understanding how the AI reached an answer
- Displays expandable cards with step-by-step logic
- Disabled by default

**SQL Planning** — Run discovery queries before generating final SQL

- Improves accuracy for complex questions
- The AI explores your schema before querying
- Enabled by default

**Validation Step** — Double-check if the response answered your question

- Catches data errors early
- Verifies results match your intent
- Enabled by default

**Configure AI Models** — Choose which LLM provider and models to use for different tasks
"""

EXPLORATIONS_INTRO_HELP = "**Select a table below** to start a new exploration. Click on Help on the left for more info."

EXPLORATION_VIEW_HELP = "Use < > to expand/collapse panels. Edit the spec (top) and the view (bottom) syncs. Click ✨ to ask LLM to revise."

EXPLORATION_CAPTION = """
Launch new explorations by chatting or selecting a table.

Ask follow-up questions by navigating to an existing exploration.
"""

REPORT_CAPTION = "Use ▶ to execute all, × to clear outputs, ∨∧ to collapse/expand sections. Click exploration names to jump to them."  # noqa: RUF001

CODE_EXEC_WARNING = """
### ⚠️ Security Warning

Enabling code execution allows the AI to generate and run Python code on your system.

**This poses significant security risks**, especially in environments with:

- API keys or secrets
- Database credentials
- Access to sensitive files or systems
- Network access to internal services

LLM-generated code may be unpredictable. While validation layers reduce accidental errors, they **do not provide meaningful protection** against malicious or compromised inputs.
"""


def _get_default_llm():
    """
    Get the default LLM instance, trying to find an available one first,
    falling back to OpenAI if none are available.
    """
    result = get_available_llm()
    return result() if result is not None else OpenAI()


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
        TableListAgent, ChatAgent, DocumentListAgent, SQLAgent, VegaLiteAgent, ValidationAgent, DeckGLAgent
    ], doc="""List of default agents which will always be added.""")

    demo_inputs = param.List(default=DEMO_MESSAGES, doc="""
        List of instructions to demo the Coordinator.""")

    export_functions = param.Dict(default={}, doc="""
       Dictionary mapping from name of exporter to export function.""")

    interface = param.ClassSelector(class_=ChatFeed, doc="""
        The interface for the Coordinator to interact with.""")

    llm = param.ClassSelector(class_=Llm, default=_get_default_llm(), doc="""
        The LLM provider to be used by default""")

    llm_choices = param.List(default=[], doc="""
        List of available LLM model choices to show in the configuration dialog.""")

    provider_choices = param.Dict(default={}, doc="""
        Available LLM providers to show in the configuration dialog.""")

    log_level = param.ObjectSelector(default='DEBUG', objects=['DEBUG', 'INFO', 'WARNING', 'ERROR'], doc="""
        The log level to use.""")

    logfire_tags = param.List(default=None, doc="""
        Whether to log LLM calls and responses to logfire.
        If a list of tags is provided, those tags will be used for logging.
        Suppresses streaming responses if enabled since
        logfire does not track token usage on stream.""")

    logs_db_path = param.String(default=None, constant=True, doc="""
        The path to the log file that will store the messages exchanged with the LLM.""")

    notebook_preamble = param.String(default='', doc="""
        Preamble to add to exported notebook(s).""")

    source_controls = param.List(default=[UploadControls, DownloadControls], doc="""
        List of SourceControls types to manage datasets.""")

    suggestions = param.List(default=GETTING_STARTED_SUGGESTIONS, doc="""
        Initial list of suggestions of actions the user can take.""")

    upload_handlers = param.Dict(default={}, doc="""
        Dictionary mapping from file extensions to handler function,
        e.g. {"hdf5": ...}. The callback function should accept the file bytes,
        table alias, and filename, add or modify the `sources` in memory, and return a bool
        (True if the table was successfully uploaded).""")

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

    _llm_status = param.String(default='verifying', doc="""
        The current LLM connection status. Either 'verifying', 'connected', or an error message.""")

    _cta = None  # Initialized in _render_main
    _error_alert = None  # Initialized in _render_main

    __abstract = True

    def __init__(
        self,
        data: DataT | list[DataT] | dict[str, DataT] | None = None,
        interface: ChatInterface | None = None,
        **params
    ):
        params["log_level"] = params.get("log_level", self.param["log_level"].default).upper()
        super().__init__(**params)
        if self.logfire_tags is not None:
            if self.llm._supports_logfire:
                self.llm.logfire_tags = self.logfire_tags
            else:
                log_debug(f"LLM {type(self.llm).__name__} does not support logfire logging.")
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

                # Handle "no_data" as a special case for starting without data
                if src == 'no_data':
                    continue

                # Handle explicit database connection strings first (before checking file extensions)
                if src.startswith('duckdb://'):
                    db_path = src.replace('duckdb:///', '')
                    db_path = str(Path(db_path).absolute())
                    source = DuckDBSource(uri=db_path)
                    sources.append(source)
                    continue
                elif src.startswith(('sqlite://', 'postgresql://', 'mysql://', 'mssql://', 'oracle://')):
                    try:
                        from ..sources.sqlalchemy import SQLAlchemySource
                    except ImportError as e:
                        raise ImportError(
                            "SQLAlchemy is required for database connection strings. "
                            "Install it with: pip install sqlalchemy"
                        ) from e
                    source = SQLAlchemySource(url=src)
                    sources.append(source)
                    continue

                # Handle .db files - need to detect if SQLite or DuckDB
                if src.endswith('.db'):
                    db_path = Path(src).absolute()
                    if not db_path.exists():
                        raise FileNotFoundError(f"Database file not found: {src}")

                    with open(db_path, 'rb') as f:
                        header = f.read(16)

                    # DuckDB files start with specific magic bytes
                    # The header contains "DUCK" in the first 16 bytes
                    if b'DUCK' in header:
                        source = DuckDBSource(uri=str(db_path))
                        sources.append(source)
                    else:
                        try:
                            from ..sources.sqlalchemy import SQLAlchemySource
                        except ImportError as e:
                            raise ImportError(
                                "SQLAlchemy is required to read .db files. "
                                "Install it with: pip install sqlalchemy"
                            ) from e
                        source = SQLAlchemySource(url=f'sqlite:///{db_path}')
                        sources.append(source)
                    continue

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
        self._llm_status = 'verifying'
        if self._cta is not None:
            self._cta.object = self._get_status_text()
        if self._error_alert is not None:
            self._error_alert.visible = False
        try:
            await self.llm.initialize(log_level=self.log_level)
            self._llm_status = 'connected'
            self._chat_input.disabled = False
            self.interface.disabled = False
        except Exception as e:
            import traceback
            traceback.print_exc()
            self._llm_status = str(e)
            if self._error_alert is not None:
                self._error_alert.object = "LLM connection failed; try another LLM under Preferences or check logs for more details."
                self._error_alert.visible = True
                self._chat_input.disabled = True
        finally:
            if self._cta is not None:
                self._cta.object = self._get_status_text()

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

    def _is_report_mode(self) -> bool:
        """Check if we're currently in report mode."""
        report_item = self._sidebar_menu.items[1] if len(self._sidebar_menu.items) > 1 else None
        return report_item is not None and report_item.get("active", False)

    def _has_explorations(self) -> bool:
        """Check if there are any explorations beyond home."""
        return len(self._explorations.items) > 1

    def _should_show_navigation(self) -> bool:
        """Determine if navigation pane should be visible."""
        return self._has_explorations()

    def _get_current_exploration(self) -> Exploration:
        """Get the currently active exploration."""
        return self._explorations.value['view']

    @hold()
    def _update_main_view(self, force_report_mode: bool | None = None):
        """
        Centralized method to update the main view based on current state.

        This method determines what should be displayed in self._main based on:
        - Whether we're in report mode
        - Whether there are explorations
        - Whether the current exploration has data to show
        - Whether we're on the home view

        Parameters
        ----------
        force_report_mode : bool | None
            If provided, force report mode on/off. Otherwise use current state.
        """
        in_report_mode = force_report_mode if force_report_mode is not None else self._is_report_mode()

        if in_report_mode:
            if self._navigation_title.object == "Report":
                return
            elif not self._has_explorations():
                no_explorations_msg = Markdown(
                    "### No Explorations Yet\n\nGenerate an exploration first by asking a question about your data.",
                )
                back_button = Button(
                    label="Back to Exploration",
                    icon="insights",
                    button_type="primary",
                    on_click=lambda e: self._handle_sidebar_event(self._sidebar_menu.items[0]),
                )
                main_content = Column(no_explorations_msg, back_button, styles={"margin": "auto"})
            else:
                main_content = Report(
                    *(Section(item["view"].plan, *(it["view"].plan for it in item["items"]), title=item["view"].plan.title)
                      for item in self._explorations.items[1:])
                )
            self._navigation_title.object = "Report"
            self._navigation_caption.object = REPORT_CAPTION
        else:
            exploration = self._get_current_exploration()
            if exploration is self._home:
                main_content = self.interface if self.interface.objects else self._splash
                self._output[1:] = []
            else:
                main_content = self._split
                self._output[1:] = [exploration]
            self._navigation_title.object = "Exploration"
            self._navigation_caption.object = EXPLORATION_CAPTION

        show_nav = self._should_show_navigation()
        if not self._sidebar_menu.items[6]["active"] and show_nav:
            self._toggle_navigation(True)
        if not show_nav:
            self._toggle_navigation(False)
        self._main[:] = [self._navigation, main_content]

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

        # Track pending query when files are being uploaded
        self._pending_query = None
        self._pending_sources_snapshot = None  # Also indicates dialog was opened from chat

        def on_submit(event=None, instance=None, wait=False):
            user_prompt = self._chat_input.value_input

            # Check if there are pending uploads that need to be transferred first
            uploaded = self._chat_input.value_uploaded
            if self._chat_input.pending_uploads or (wait and not uploaded):
                self._pending_query = user_prompt or None
                if not wait:
                    self._chat_input.sync()
                # Schedule a callback to await the upload
                state.add_periodic_callback(partial(on_submit, wait=True), period=100, count=1)
                return

            if not user_prompt and not uploaded:
                return

            # Open Sources dialog so user can confirm the upload
            if uploaded:
                if not self._upload_controls._file_cards:
                    self._upload_controls._generate_file_cards({
                        key: value["value"] for key, value in uploaded.items()
                    })

                # Store the pending query to execute after files are added
                if not wait:
                    self._pending_query = user_prompt or None
                self._pending_sources_snapshot = list(self.context.get("sources", []))

                with edit_readonly(self._chat_input):
                    self._chat_input.param.update(
                        value_input="",
                        value_uploaded={},
                        views=[]
                    )

                with hold():
                    self._sources_dialog_content.open = True
                    self._source_content.active = self._source_controls.index(self._upload_controls)

                async def focus():
                    await asyncio.sleep(0.25)
                    self._upload_controls._add_button.focus()

                state.execute(focus, schedule=True)
                return

            with self.interface.param.update(disabled=True, loading=True), hold():
                self._update_main_view()
                self.interface.send(user_prompt, respond=bool(user_prompt))
                with edit_readonly(self._chat_input):
                    self._chat_input.value_input = ""

        # Store as instance variable for access from other methods
        self._on_submit = on_submit

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
        self._chat_input = interface.active_widget

        existing_actions = self._chat_input.actions
        self._chat_input.enable_upload = any(
            issubclass(sc, UploadControls) for sc in self.source_controls
        )
        self._chat_input.actions = {
            **existing_actions,
        }
        self._configure_logs(interface)
        self.interface = interface

    def _handle_upload_successful(self, event):
        """Handle successful file upload by closing dialog and executing pending query if present."""
        # Only auto-close if dialog was opened from chat (indicated by snapshot being set)
        if self._pending_sources_snapshot is None:
            return

        # Capture any text currently in the input field
        current_text = self._chat_input.value_input
        if self._pending_query is None and current_text:
            self._pending_query = current_text
            with edit_readonly(self._chat_input):
                self._chat_input.value_input = ""

        # Save state before closing (close triggers _on_sources_dialog_close which resets state)
        query = self._pending_query
        sources_snapshot = self._pending_sources_snapshot
        self._sources_dialog_content.open = False

        if query:
            self._execute_pending_query_with(query, sources_snapshot)

    def _execute_pending_query_with(self, user_prompt, old_sources):
        """Execute a query after files have been uploaded."""
        # Remove views from widget
        self._chat_input.param.update(
            value_input="",
            value_uploaded={},
            views=[],
        )

        # Build message with new sources info
        new_sources = [
            source for source in self.context.get("sources", [])
            if source not in old_sources
        ]
        source_names = [f"**{src.name}**: {', '.join(src.get_tables())}" for src in new_sources]
        source_view = Markdown(
            "Added sources:\n" + "\n".join(f"- {name}" for name in source_names),
            sizing_mode="stretch_width"
        ) if new_sources else None

        self._update_main_view()
        msg = Column(user_prompt, source_view) if source_view else user_prompt
        self.interface.send(msg, respond=True)

    def _on_sources_dialog_close(self, event):
        """Handle sources dialog close - restore pending query to input if user closed without adding files."""
        if event.new:  # Only act when dialog is closing
            return

        # Restore pending query if user closed without confirming
        if self._pending_query:
            with edit_readonly(self._chat_input):
                self._chat_input.value_input = self._pending_query

        # Reset state
        self._pending_query = None
        self._pending_sources_snapshot = None
        self._chat_input.param.update(
            value_uploaded={},
            views=[],
        )

    def _configure_coordinator(self):
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

    def _get_status_text(self) -> str:
        """Generate the status text showing sources and LLM provider."""
        num_sources = len(self.context.get("sources", []))
        llm_name = type(self.llm).__name__

        # Build LLM status text
        if self._llm_status == 'verifying':
            llm_text = f"verifying **{llm_name}** connection"
        elif self._llm_status == 'connected':
            llm_text = f"using **{llm_name}** as the LLM provider"
        else:
            llm_text = f"connecting to **{llm_name}** failed"

        if num_sources == 0:
            status = f"Drag & drop your dataset here to begin; {llm_text}."
        else:
            sources_text = f"**{num_sources} source{'s' if num_sources > 1 else ''} connected**"
            status = f"{sources_text}; {llm_text}."

        return f"{status} {SPLASH_HELP_HINT}"

    def _render_header(self) -> list[Viewable]:
        return []

    def _render_main(self) -> list[Viewable]:
        self._cta = Typography(self._get_status_text(), margin=(0, 0, 10, 0))
        self._chat_splash = Column(self._cta, self._chat_input, margin=(0, 0, 0, -10))
        self._error_alert = Alert(
            object="",
            severity="error",
            sizing_mode="stretch_width",
            margin=(10, 0, 5, 0),
            visible=False
        )

        self._splash = MuiColumn(
            Paper(
                Typography(
                    "Ask questions, get insights",
                    disable_anchors=True,
                    variant="h1",
                    margin=(10, 0, 5, 0),
                ),
                self._cta,
                self._chat_splash,
                self._error_alert,
                max_width=850,
                styles={'margin': 'auto'},
                sx={'p': '0 20px 20px 20px'}
            ),
            sx={'display': 'flex', 'align-items': 'center'},
            height_policy='max'
        )

        self._main = Row(self._splash, sizing_mode='stretch_both', align="center")

        if self.suggestions:
            self._add_suggestions_to_footer(
                self.suggestions,
                num_objects=1,
                inplace=True,
                analysis=False,
                append_demo=True,
                hide_after_use=False
            )

        # Create info dialog with breadcrumb navigation
        self._help_content = MuiColumn(sizing_mode="stretch_width")

        help_items = [
            {"label": "Quickstart", "icon": "rocket_launch"},
            {"label": "Interface", "icon": "dashboard"},
            {"label": "Explorations", "icon": "layers"},
            {"label": "Editor", "icon": "code"},
            {"label": "Results", "icon": "analytics"},
            {"label": "Export", "icon": "ios_share"},
            {"label": "Tips", "icon": "lightbulb"},
        ]

        self._help_breadcrumbs = Breadcrumbs(
            items=help_items,
            value=help_items[0],
            margin=(0, 0, 10, 0),
        )

        self._next_help_button = Button(
            name="Next",
            variant="outlined",
            sizing_mode="stretch_width",
            align="end",
        )

        def next_section(event):
            current_item = self._help_breadcrumbs.value
            current_index = help_items.index(current_item)
            if current_index < len(help_items) - 1:
                self._help_breadcrumbs.value = help_items[current_index + 1]

        self._next_help_button.on_click(next_section)

        def update_help_content(event):
            selected_item = event.new
            section_index = help_items.index(selected_item)
            help_texts = [
                HELP_GETTING_STARTED, HELP_INTERFACE, HELP_EXPLORATIONS,
                HELP_EDITOR, HELP_RESULTS, HELP_EXPORT, HELP_TIPS
            ]
            text = help_texts[section_index]
            if section_index == 0:
                text = self._current_help_text
            self._help_content[:] = [Markdown(text, sizing_mode="stretch_width")]

            if section_index == len(help_items) - 1:
                self._next_help_button.visible = False
            else:
                self._next_help_button.visible = True

        self._help_breadcrumbs.param.watch(update_help_content, 'value')

        # Initialize with first section (will be updated based on data source availability)
        # Use a deferred initialization to avoid race condition
        self._help_content[:] = [Markdown("", sizing_mode="stretch_width")]

        self._info_dialog = Dialog(
            MuiColumn(
                self._help_breadcrumbs,
                self._help_content,
                self._next_help_button,
                sizing_mode="stretch_width"
            ),
            close_on_click=True,
            show_close_button=True,
            sizing_mode='stretch_width',
            width_option='md',
            title="Help Guides"
        )

        # Set up actions for the ChatAreaInput speed dial
        # Use document_vector_store if available, otherwise fall back to main vector_store
        doc_store = self.document_vector_store or self._coordinator.vector_store
        self._source_catalog = SourceCatalog(context=self.context, vector_store=doc_store)

        # Watch for visibility changes and schedule async handler
        def _schedule_visibility_change(event):
            async def _do_sync():
                await self._on_visibility_changed(event)
            param.parameterized.async_executor(_do_sync)
        self._source_catalog.param.watch(_schedule_visibility_change, 'visibility_changed')

        # Initialize source controls
        self._source_controls = []
        control_tabs = []
        for control in self.source_controls:
            control_inst = control(
                context=self.context, source_catalog=self._source_catalog,
                upload_handlers=self.upload_handlers
            )
            control_inst.param.watch(self._sync_sources, 'outputs')
            control_inst.param.watch(self._handle_upload_successful, 'upload_successful')
            if isinstance(control_inst, UploadControls):
                self._upload_controls = control_inst
            control_tabs.append((control.label, control_inst))
            self._source_controls.append(control_inst)
        self._source_content = Tabs(*control_tabs, sizing_mode="stretch_width")

        self._sources_help_caption = Typography(
            "Add data to explore with Lumen AI. Supports CSV, Parquet, JSON, Excel, and databases. "
            "Select between data / metadata and set an alias.",
            variant="body2",
            color="text.secondary",
            sizing_mode="stretch_width"
        )

        self._sources_dialog_content = Dialog(
            MuiColumn(
                self._sources_help_caption,
                Paper(self._source_content, margin=(0, 0, 10, 0)),
                Paper(self._source_catalog, max_height=400, sx={"overflowX": "auto"}),
                sizing_mode="stretch_width"
            ),
            close_on_click=True, show_close_button=True, width_option='lg',
            title="Data Sources",
            sizing_mode='stretch_width',
        )
        # Watch for dialog close to handle pending query if user closes without adding files
        self._sources_dialog_content.param.watch(self._on_sources_dialog_close, 'open')

        self._explorations = MenuList(
            items=[self._exploration], value=self.param._exploration, show_children=True,
            dense=True, margin=0, sizing_mode='stretch_width',
            sx={".mui-light .MuiBox-root": {"backgroundColor": "var(--mui-palette-grey-100)"}},
        )
        self._explorations.param.watch(self._cleanup_explorations, 'items')
        self._explorations.param.watch(self._update_conversation, 'active')
        self._explorations.param.watch(self._sync_active, 'value')
        self._explorations.on_action('remove', self._delete_exploration)
        self._explorations.on_action('export_notebook', self._export_exploration)

        # Create hidden download button for exporting notebook
        self._exploration_download = FileDownload(
            visible=False
        )

        # Create LLM configuration dialog
        self._llm_dialog = LLMConfigDialog(
            llm=self.llm,
            llm_choices=self.llm_choices,
            provider_choices=self.provider_choices,
            on_llm_change=self._on_llm_change,
            agent_types=[type(agent) for agent in self._coordinator.agents],
        )

        return [self._main, self._sources_dialog_content, self._llm_dialog, self._info_dialog, self._exploration_download]

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
            sidebar_width=62,
            sidebar_resizable=False,
            sx={
                # Hover
                ".sidebar": {"transition": "width 0.2s ease-in-out"},
                ".sidebar:hover": {"width": "140px", "transitionDelay": "0.5s"},
                "&.mui-light .sidebar": {"bgcolor": "var(--mui-palette-grey-50)"},
            }
        )
        # Unlink busy indicator
        with edit_readonly(self._page):
            self._page.busy = False

    @contextmanager
    def _busy(self):
        """
        Context manager that clears self._idle and sets self._page.busy to True,
        then restores both self._idle and self._page.busy to their previous states
        after exiting the context.
        """
        old_busy = self._page.busy
        self._idle.clear()
        with edit_readonly(self._page):
            self._page.busy = True
        try:
            yield
        finally:
            self._idle.set()
            with edit_readonly(self._page):
                self._page.busy = old_busy

    def _update_help_getting_started(self):
        """Update the Getting Started help text based on whether data sources are connected."""
        num_sources = len(self.context.get("sources", []))
        if num_sources == 0:
            self._current_help_text = HELP_NO_SOURCES
        else:
            self._current_help_text = HELP_GETTING_STARTED

    @param.depends('context', on_init=True, watch=True)
    async def _sync_sources(self, event=None):
        context = event.new if event else self.context
        if 'sources' in context:
            old_sources = self.context.get("sources", [self.context["source"]] if "source" in self.context else [])
            new_sources = [src for src in context["sources"] if src not in old_sources]
            self.context["sources"] = old_sources + new_sources

            # Compute table slugs for old, new, and all sources
            old_slugs = set()
            for source in old_sources:
                tables = source.get_tables()
                for table in tables:
                    table_slug = f'{source.name}{SOURCE_TABLE_SEPARATOR}{table}'
                    old_slugs.add(table_slug)

            all_slugs = set()
            for source in self.context["sources"]:
                tables = source.get_tables()
                for table in tables:
                    table_slug = f'{source.name}{SOURCE_TABLE_SEPARATOR}{table}'
                    all_slugs.add(table_slug)

            new_slugs = all_slugs - old_slugs

            # Update visible_slugs: preserve old table visibility, auto-check new tables
            current_visible = self.context.get("visible_slugs")
            if current_visible is not None:  # Check for None, not truthiness (empty set is valid!)
                self.context["visible_slugs"] = current_visible.intersection(old_slugs) | new_slugs
            else:
                self.context["visible_slugs"] = all_slugs

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

        # Guard against early calls during init when components don't exist yet
        if hasattr(self, '_explorer'):
            await self._explorer.sync()
        # Only sync coordinator if we have sources and coordinator exists
        if hasattr(self, '_coordinator') and self.context.get("sources"):
            await self._coordinator.sync(self.context)
        if hasattr(self, '_source_catalog'):
            self._source_catalog.sync(self.context)

        if hasattr(self, '_cta'):
            self._cta.object = self._get_status_text()

        if hasattr(self, '_input_tabs'):
            num_sources = len(self.context.get("sources", []))
            self._input_tabs.disabled = [] if num_sources else [1]

        # Update help dialog content when sources change
        if hasattr(self, '_help_content'):
            self._update_help_getting_started()
            # Only update if we're currently showing the Getting Started section
            if hasattr(self, '_help_breadcrumbs') and self._help_breadcrumbs.value['label'] == 'Getting Started':
                self._help_content[:] = [Markdown(self._current_help_text, sizing_mode="stretch_width")]

    def _open_llm_dialog(self, event=None):
        """Open the LLM configuration dialog when the LLM chip is clicked."""
        self._llm_dialog.open = True

    def _on_llm_change(self, new_llm):
        """Handle LLM provider change from the dialog."""
        # Update the UI's LLM reference
        self.llm = new_llm

        # Reset status to verifying when provider changes
        self._llm_status = 'verifying'
        if self._cta is not None:
            self._cta.object = self._get_status_text()
        if self._error_alert is not None:
            self._error_alert.visible = False

        # Update the coordinator's LLM reference
        self._coordinator.llm = new_llm

        # Update all agent LLMs
        for agent in self._coordinator.agents:
            agent.llm = new_llm

        # Initialize the new LLM asynchronously
        param.parameterized.async_executor(self._initialize_new_llm)

    def _open_sources_dialog(self, event=None):
        """Open the sources dialog when the vector store badge is clicked."""
        # Navigate to Source Catalog if sources exist, otherwise Upload
        self._sources_dialog_content.open = True

    def _open_info_dialog(self, event=None):
        """Open the info dialog when the info button is clicked."""
        # Always update help content based on current data source state
        self._update_help_getting_started()
        # Set to Getting Started section
        self._help_breadcrumbs.value = self._help_breadcrumbs.items[0]
        self._help_content[:] = [Markdown(self._current_help_text, sizing_mode="stretch_width")]
        self._info_dialog.open = True

    async def _on_visibility_changed(self, event):
        """Handle visibility changes from SourceCatalog by re-syncing the coordinator."""
        if hasattr(self, '_coordinator'):
            await self._coordinator.sync(self.context)

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
            with button.param.update(loading=True), self._chat_input.param.update(loading=True):
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
                self._update_main_view()

                if not analysis:
                    # Set the input value and trigger submit
                    # This will handle pending uploads via on_submit
                    with edit_readonly(self._chat_input):
                        self._chat_input.value_input = contents
                    self._on_submit()
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
                with self._busy():
                    await self._execute_plan(plan)

        async def run_demo(event):
            if hide_after_use:
                suggestion_buttons.visible = False
                if event.new > 1:  # prevent double clicks
                    return
            with self._chat_input.param.update(loading=True), self._busy():
                for demo_message in self.demo_inputs:
                    while self.interface.disabled:
                        await asyncio.sleep(1.25)
                    self._chat_input.value = demo_message
                    await asyncio.sleep(2)

        suggestion_buttons = FlexBox(
            *[
                Button(
                    label=suggestion[1] if isinstance(suggestion, tuple) else suggestion,
                    icon=suggestion[0] if isinstance(suggestion, tuple) else None,
                    variant="outlined",
                    on_click=use_suggestion,
                    margin=5 if i else (5, 5, 5, 5),
                    disabled=self.interface.param.loading
                )
                for i, suggestion in enumerate(suggestions)
            ],
            name="Suggestions",
            margin=5,
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
                prev_suggestions = [obj for obj in footer_objects if obj.name == "Suggestions"]
                if prev_suggestions:
                    prev_suggestions[0][:] = list(suggestion_buttons)
                else:
                    message.footer_objects = footer_objects + [suggestion_buttons]
            self.interface.param.watch(hide_suggestions, "objects")
        else:
            self._splash[0][2].append(suggestion_buttons)

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
                *{ext for agent in self._coordinator.agents for ext in agent._extensions} | {"filedropper"},
                css_files=["https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.2/css/all.min.css"]
            )
            return self._page
        return super()._create_view()

    def _repr_mimebundle_(self, include=None, exclude=None):
        panel_extension(
            *{ext for agent in self._coordinator.agents for ext in agent._extensions} | {"filedropper"},
            notifications=True,
            css_files=["https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.2/css/all.min.css"]
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

    initialized = param.Boolean(default=False)

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

    code_execution = param.Selector(
        default="hidden", objects=["hidden", "disabled", "prompt", "llm", "allow"],
        doc="""
        Code execution mode for generating Vega-Lite specs via Altair code.
        Controls whether the code execution selector appears in the UI preferences:
        - hidden: Do not show code execution option in preferences (default)
        - disabled: Show selector, but default to no code execution (Vega-Lite spec only)
        - prompt: Show selector, default to prompting user for permission to execute
        - llm: Show selector, default to LLM-validated code execution
        - allow: Show selector, default to executing code without confirmation

        WARNING: The 'prompt', 'llm', and 'allow' modes execute LLM-generated code and
        must NEVER be enabled in production environments with access to secrets, credentials,
        or sensitive data. The LLM validation layer reduces accidental errors but does NOT
        provide meaningful security against malicious or compromised inputs.
        """
    )

    title = param.String(default='Lumen AI - Data Explorer', doc="Title of the app.")

    _exploration = param.Dict()

    @hold()
    def _handle_sidebar_event(self, item):
        if item["id"] == "exploration":
            self._toggle_report_mode(False)
            self._sidebar_menu.update_item(item, active=True, icon="insert_chart")
            self._sidebar_menu.update_item(self._sidebar_menu.items[1], active=False, icon="description_outlined")
            self._update_home()
        elif item["id"] == "report":
            self._toggle_report_mode(True)
            self._sidebar_menu.update_item(self._sidebar_menu.items[0], active=False, icon="insert_chart_outlined")
            self._sidebar_menu.update_item(item, active=True, icon="description")
        elif item["id"] == "data":
            self._open_sources_dialog()
        elif item["id"] == "preferences":
            self._settings_popup.open = True
        elif item["id"] == "help":
            self._open_info_dialog()
        elif item["id"] == "nav":
            self._toggle_navigation(not item["active"])

    def _toggle_navigation(self, active: bool):
        self._navigation.visible = active
        self._sidebar_menu.update_item(self._sidebar_menu.items[6], active=active, icon="layers" if active else "layers_outlined")

    @param.depends("_exploration", watch=True)
    def _update_home(self):
        if not hasattr(self, '_page'):
            return
        exploration, report = self._sidebar_menu.items[:2]
        self._sidebar_menu.update_item(exploration, active=True, icon="timeline" if report["active"] else "insert_chart")
        self._update_main_view()

    def _render_sidebar(self) -> list[Viewable]:
        switches = []
        cot = Switch(label='Chain of Thought', description='Show AI reasoning steps')
        cot.link(self._coordinator, value='verbose', bidirectional=True)
        switches.append(cot)
        sql_agent = next(
            (agent for agent in self._coordinator.agents if isinstance(agent, SQLAgent)),
            None
        )
        if sql_agent:
            sql_planning = Switch(label='SQL Planning', description='Run discovery queries and adaptive exploration before final SQL')
            sql_planning.link(sql_agent, value='exploration_enabled', bidirectional=True)
            switches.append(sql_planning)
        validation = Switch(label='Validation Step', description='Check if the response fully answered your question')
        validation.link(self._coordinator, value='validation_enabled', bidirectional=True)
        switches.append(validation)

        # Add code execution selector if not hidden
        code_agents = False
        for agent in self._coordinator.agents:
            if isinstance(agent, BaseCodeAgent) and self.code_execution != "hidden":
                agent.code_execution = self.param.code_execution
                code_agents = True
        if code_agents:
            def on_code_exec_change(event):
                if event.new:
                    self._code_exec_warning_dialog.open = True
                    if self.code_execution == "disabled":
                        self.code_execution = "prompt"
                else:
                    self.code_execution = "disabled"
            self._code_exec_switch = Switch(
                description="Allowing code execution",
                label="Enable Code Execution",
                value=self.param.code_execution.rx() != "disabled"
            )
            self._code_exec_switch.param.watch(on_code_exec_change, "value")
            self._code_exec_select = Select.from_param(
                self.param.code_execution,
                options={"Disabled": "disabled", "Prompt": "prompt", "LLM Validation": "llm", "Always Allow": "allow"},
                description="How to handle LLM generated code execution",
                sizing_mode="stretch_width",
                margin=(10, 10, 5, 10),
                visible=self._code_exec_switch
            )
            switches.extend([self._code_exec_switch, self._code_exec_select])

        llm_config_button = Button(
            label="Configure AI Models",
            icon="auto_awesome",
            size="large",
            variant="text",
            button_type="default",
            sizing_mode="stretch_width",
            on_click=lambda e: setattr(self._llm_dialog, 'open', True),
            margin=(-5, 10, 10, 10),
            sx={
                'fontSize': '16px',
                'color': 'text.primary',
                '& .MuiIcon-root': {
                    'fontSize': '28px',
                    'marginRight': '10px',
                }
            }
        )
        switches.append(llm_config_button)

        prefs_header = Typography(
            "Settings",
            variant="subtitle1",
            sx={"fontWeight": "bold"},
            margin=(10, 10, 5, 10)
        )

        self._settings_popup = Popup(
            prefs_header,
            *switches,
            anchor_origin={"horizontal": "right", "vertical": "center"},
            transform_origin={"horizontal": "left", "vertical": "top"},
            styles={"z-index": '1300'},
            theme_config={"light": {"palette": {"background": {"paper": "var(--mui-palette-grey-50)"}}}, "dark": {}}
        )
        self._sidebar_menu = menu = MenuList(
            items=[
                {"label": "Explore", "icon": "insert_chart", "id": "exploration", "active": True},
                {"label": "Report", "icon": "description_outlined", "id": "report", "active": False},
                None,
                {"label": "Sources", "icon": "create_new_folder_outlined", "id": "data"},
                {"label": "Settings", "icon": "tune_outlined", "id": "preferences"},
                None,
                {"label": "Navigate", "icon": "layers_outlined", "id": "nav", "active": False},
                None,
                {"label": "Help", "icon": "help_outline", "id": "help"}
            ],
            active=0,
            attached=[self._settings_popup],
            collapsed=False,
            highlight=False,
            margin=0,
            on_click=self._handle_sidebar_event,
            sx={
                # Base padding
                "paddingLeft": "4px",
                # Base button styling
                "& .MuiButtonBase-root.MuiListItemButton-root": {
                    "p": "8px 14px",
                },
                # Icon styling
                ".MuiListItemIcon-root > .MuiIcon-root": {
                    "margin-right": "16px",
                    "color": "var(--mui-palette-text-primary)",
                },
                "& .MuiDivider-root": {
                    "margin-left": "-4px",
                    "margin-block": "8px",
                    "opacity": "0.3",
                    "borderColor": "var(--mui-palette-grey-500)",
                },
            },
            sizing_mode="stretch_width",
        )
        return [menu]

    def _render_page(self):
        super()._render_page()

    def _render_main(self) -> list[Viewable]:
        main = super()._render_main()

        self._explorer = TableExplorer(context=self.context)
        self._explorer.param.watch(self._add_exploration_from_explorer, "add_exploration")

        # Create tabs wrapping just the input components
        num_sources = len(self.context.get("sources", []))
        self._input_tabs = Tabs(
            ("Chat with Data", self.interface._widget),
            ("Select Data to Explore", self._explorer),
            disabled=[] if num_sources else [1],
            min_height=125,
            sizing_mode="stretch_height",
            stylesheets=[".MuiTabsPanel > .MuiBox-root { overflow: visible}"]
        )

        # Update chat_splash to use tabs instead of interface widget
        self._chat_splash[:] = [self._input_tabs]

        # Add suggestions to chat_splash (after tabs)
        if self.suggestions:
            self._add_suggestions_to_footer(
                self.suggestions,
                num_objects=0,
                inplace=False,
                analysis=False,
                append_demo=True,
                hide_after_use=False
            )

        # Initialize home as empty
        self._home.view = MuiColumn()

        # Main Area
        self._output = Paper(
            self._home,
            elevation=2,
            margin=(5, 10, 5, 5),
            height_policy="max",
            width_policy="max",
            sizing_mode="stretch_both"
        )
        self._split = HSplit(
            self.interface,
            self._output,
            collapsed=None,
            expanded_sizes=(40, 60),
            show_buttons=True,
            sizing_mode='stretch_both',
            stylesheets=SPLITJS_STYLESHEETS
        )
        self._navigation_title = Typography(
            "Exploration", variant="h6", margin=(10, 10, 0, 10)
        )
        self._navigation_caption = Typography(
            EXPLORATION_CAPTION,
            variant="body2",
            color="text.secondary",
            margin=(10, 10, 5, 10),
            sizing_mode="stretch_width"
        )
        self._navigation_footer = Typography(
            'Toggle <span class="material-icons" style="vertical-align: middle;">layers</span>**Navigate** to close',
            variant="body2",
            color="text.secondary",
            margin=20,
            styles={"margin": "auto auto 20px auto"}
        )
        self._navigation = Paper(
            self._navigation_title,
            self._navigation_caption,
            self._explorations,
            self._navigation_footer,
            height_policy="max",
            sx={"borderRadius": 0},
            theme_config={"light": {"palette": {"background": {"paper": "var(--mui-palette-grey-100)"}}}, "dark": {}},
            visible=False,
            width=250
        )
        self._main[:] = [self._navigation, self._splash]

        # Create code execution warning dialog if code execution is not hidden
        if self.code_execution != "hidden":
            main.append(self._render_code_exec_warning_dialog())

        return main

    def _render_code_exec_warning_dialog(self) -> Viewable:
        def cancel_code_execution(event):
            """Cancel the code execution change and close the warning dialog."""
            self._code_exec_select.value = "disabled"
            self._code_exec_warning_dialog.open = False

        def confirm_code_execution(event):
            self._code_exec_warning_dialog.open = False

        mode_explanation = {
            "prompt": "will prompt you for permission each time a chart is generated.",
            "llm": "will validate generated code via the LLM before executing.",
            "allow": "will execute all generated code without confirmation."
        }
        code_exec_warning = CODE_EXEC_WARNING + (
            f"\n\nYou have selected the **{self.code_execution}** execution mode. "
            f"This mode **{mode_explanation.get(self.code_execution, '')}**"
        )
        self._code_exec_warning_dialog = Dialog(
            MuiColumn(
                Markdown(code_exec_warning, sizing_mode="stretch_width"),
                Row(
                    Button(
                        label="I understand",
                        variant="outlined",
                        on_click=confirm_code_execution,
                    ),
                    Button(
                        label="Disable exec",
                        button_type="primary",
                        on_click=cancel_code_execution,
                    ),
                    align="end",
                    margin=(10, 0, 0, 0),
                ),
                sizing_mode="stretch_width",
            ),
            open=False,
            show_close_button=False,
            sizing_mode="stretch_width",
            width_option="sm",
            title="Enable Code Execution?",
        )
        return self._code_exec_warning_dialog

    async def _add_exploration_from_explorer(self, event: param.parameterized.Event | None = None):
        if self._explorer.table_slug is None:
            return

        table = self._explorer.table_slug.split(SOURCE_TABLE_SEPARATOR)[0]
        self._update_main_view()
        self.interface.send(f"Explore the `{table}` table", respond=False)

        # Flush UI
        await asyncio.sleep(0.01)

        prev = self._explorations.value
        sql_agent = next(agent for agent in self._coordinator.agents if isinstance(agent, SQLAgent))
        sql_out = self._explorer.create_sql_output()
        out_context = dict(self.context, **(await sql_out.render_context()))
        sql_task = ActorTask(sql_agent, title=f"Load {table}", views=[sql_out], out_context=out_context, status="success")
        plan = Plan(sql_task, title=f"Explore {table}", context=self.context, out_context=out_context, status="success")

        with hold(), self._busy():
            exploration = await self._add_exploration(plan, self._home)
            self._add_views(exploration, items=plan.views)
            await self._postprocess_exploration(plan, exploration, prev, is_new=True)

    def _configure_session(self):
        self._home = self._last_synced = Exploration(
            context=self.context,
            title='Home',
            conversation=self.interface.objects
        )
        self._exploration = {'label': 'New Exploration', 'icon': "add_circle_outline", 'view': self._home, 'items': []}
        super()._configure_session()
        self._idle = asyncio.Event()
        self._idle.set()

    def _sync_active(self, event: param.parameterized.Event):
        if event.new is not self._exploration:
            self._exploration = event.new

    def _toggle_report_mode(self, active: bool):
        """Toggle between regular and report mode."""
        self._update_main_view(force_report_mode=active)

    async def _delete_exploration(self, item):
        await self._idle.wait()
        with hold():
            if item in self._explorations.items:
                self._explorations.items = [it for it in self._explorations.items if it is not item]
                self._explorations.value = self._explorations.items[0]
            else:
                parent = item["parent"]
                self._explorations.update_item(parent, items=[it for it in parent["items"] if it is not item])
                self._explorations.value = parent
            self.interface.objects = []

    async def _export_exploration(self, item):
        """Export a single exploration as a Jupyter notebook."""
        exploration = item["view"]
        if exploration.plan is None:
            return

        def make_notebook():
            nb = export_notebook(exploration.plan.views, preamble=self.notebook_preamble)
            return StringIO(nb)

        self._exploration_download.param.update(
            filename=f"{exploration.title.replace(' ', '_')}.ipynb",
            callback=make_notebook
        )
        self._exploration_download.transfer()

    def _destroy(self, session_context):
        """
        Cleanup on session destroy
        """
        for c in self._explorations.items[1:]:
            c['view'].context.clear()

    async def _update_conversation(self, event=None, replan: bool = False):
        """
        Update the conversation for the current exploration.

        This method handles conversation management (switching, syncing) but
        delegates view switching to _update_main_view.
        """
        exploration = self._explorations.value['view']

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
            self._update_main_view()
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
        is_home = parent is self._home
        parent_item = self._exploration
        if is_home:
            parent.conversation = []
        else:
            parent.conversation = self._snapshot_messages(new=True)
        conversation = [msg for msg in self.interface.objects if plan.is_followup or msg not in parent.conversation]

        tabs = Tabs(
            ("Data Source", Markdown("Waiting on data...", margin=(5, 20))),
            dynamic=True,
            sizing_mode="stretch_both",
            loading=plan.param.running
        )
        output = MultiSplit(tabs, sizing_mode="stretch_both")
        exploration = Exploration(
            context=plan.param.out_context,
            conversation=conversation,
            parent=parent if plan.is_followup else self._home,
            plan=plan,
            title=plan.title,
            view=output
        )

        view_item = {
            'label': plan.title,
            'view': exploration,
            'icon': None,
            'actions': [
                {'action': 'export_notebook', 'label': 'Export Notebook', 'icon': 'download'},
                {'action': 'remove', 'label': 'Remove', 'icon': 'delete'}
            ],
            'parent': parent_item if plan.is_followup else self._explorations.items[0],
            'items': []
        }
        is_busy = not self._idle.is_set()
        with hold():
            self.interface.objects = conversation
            # Temporarily un-idle to allow exploration to be rendered
            self._idle.set()
            if is_home or not plan.is_followup:
                self._explorations.items = self._explorations.items + [view_item]
            else:
                self._explorations.update_item(
                    parent_item, items=parent_item.get('items', []) + [view_item]
                )
                self._explorations.expanded = [self._explorations._lookup_path(parent_item)]
            self._explorations.value = self._exploration = view_item
            if is_busy:
                self._idle.clear()
            self._output[1:] = [output]
            await self._update_conversation()
        self._last_synced = exploration
        return exploration

    def _render_pop_out(self, exploration: Exploration, view: Column, title: str):
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
        vsplit = VSplit(
            Column(view.editor, width_policy="max", scroll="y-auto"),
            view,
            expanded_sizes=(20, 80),
            sizes=(20, 80),
            sizing_mode="stretch_both",
            styles={"overflow": "auto"},
            stylesheets=SPLITJS_STYLESHEETS
        )
        view = Column(controls, vsplit)
        controls.append(self._render_pop_out(exploration, view, title))
        return (title, view)

    def _find_view_in_tabs(self, exploration: Exploration, out: LumenOutput):
        tabs = exploration.view[0]
        for i, tab in enumerate(tabs[1:], start=1):
            content = tab[1][1] if isinstance(tab, tuple) and len(tab) > 1 else tab[1]
            if isinstance(content, VSplit) and out.view in content:
                return i
        return None

    def _find_view_in_popped_out(self, exploration: Exploration, out: LumenOutput):
        for i, standalone in enumerate(exploration.view[1:], start=1):
            if isinstance(standalone, Column) and len(standalone) > 1:
                vsplit = standalone[1][1]
                if isinstance(vsplit, VSplit) and out.view in vsplit:
                    return i
        return None

    def _add_views(self, exploration: Exploration, event: param.parameterized.Event | None = None, items: list | None = None):
        tabs = exploration.view[0]
        content = []
        new_items = items if event is None else event.new
        for view in new_items:
            if not isinstance(view, LumenOutput) or (event and view in event.old):
                continue
            if tabs and isinstance(view, SQLOutput) and not exploration.initialized:
                tabs[0] = ("Data Source", view.render_explorer())
                exploration.initialized = True
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

    def _reset_error(self, plan: Plan, exploration: Exploration, replan: bool = False):
        plan.reset(plan._current)
        tabs = exploration.view[0]
        if replan:
            tabs[:] = [("Data Source", Markdown("Waiting on data...", margin=(5, 20)))]
            tabs.active = 0
        elif len(tabs) > 1 and isinstance(tabs[-1], Markdown):
            tabs.pop(-1)
            tabs.active = len(tabs)-1
        exploration.conversation[-1].footer_objects = []

    async def _execute_plan(self, plan: Plan, rerun: bool = False, replan: bool = False):
        prev = self._explorations.value
        parent = prev["view"]

        # Check if we are adding to existing exploration or creating a new one
        new_exploration = any("pipeline" in step.actor.output_schema.__required_keys__ for step in plan)

        partial_plan = None
        if rerun:
            exploration = parent
            self._reset_error(plan, exploration)
            watcher = plan.param.watch(partial(self._add_views, exploration), "views")
        elif new_exploration and not replan:
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
            if replan and new_exploration:
                watcher = plan.param.watch(partial(self._add_views, exploration), "views")
            else:
                watcher = None

        try:
            with plan.param.update(interface=self.interface):
                await plan.execute()
        finally:
            if watcher:
                plan.param.unwatch(watcher)
            await self._postprocess_exploration(plan, exploration, prev, is_new=new_exploration, partial_plan=partial_plan)

    async def _replan(self, plan: Plan, prev: dict, partial_plan: Plan | None = None):
        exploration = self._exploration['view']
        old_plan = plan if partial_plan is None else partial_plan
        messages = old_plan.history
        with hold():
            plan.remove(list(old_plan))
            self._reset_error(old_plan, exploration, replan=old_plan is plan)

        with self._busy():
            new_plan = await self._coordinator.respond(messages, exploration.context)
            if new_plan is not None:
                await self._execute_plan(new_plan, replan=True)

    async def _postprocess_exploration(
        self, plan: Plan, exploration: Exploration, prev: dict, is_new: bool = False, partial_plan: Plan | None = None
    ):
        self._exploration['view'].conversation = self.interface.objects
        if "__error__" not in plan.out_context and plan.status != "error":
            if "pipeline" in plan.out_context:
                await self._add_analysis_suggestions(plan)

            if is_new:
                plan.param.watch(partial(self._update_views, exploration), "views")
            return

        # On error we have to sync the conversation, unwatch the plan,
        # and remove the exploration if it was newly created
        replan_button = Button(
            label="Replan", icon="alt_route", on_click=lambda _: state.execute(partial(self._replan, plan, prev, partial_plan)),
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

        error_type = plan.out_context.pop("__error_type__", Exception)
        error = plan.out_context.pop("__error__", "Unknown error")
        _, todos = plan.render_task_history(failed=True)

        user_msg = ""
        for msg in plan.history[::-1]:
            if msg.get("role") == "user":
                user_msg = msg.get("content")
                break

        response = await self.llm.invoke(
            [{"content": (
                f"User prompt:\n\n> {user_msg}\n"
                f"Planner checklist:\n\n{todos}"
                f"Error\n\n{error_type.__name__}: {error}\n\n"
            ), "role": "user"}],
            response_model=ErrorDescription
        )
        explanation = f"❌ **Unable to complete your request**\n\n{response.explanation}"
        tabs = exploration.view[0]
        if exploration.initialized:
            tabs.append(("Error", Markdown(explanation, margin=(5, 20))))
            tabs.active = len(tabs)-1
        else:
            tabs[0].object = explanation

    @wrap_logfire(span_name="Chat Invoke")
    async def _chat_invoke(
        self, messages: list[Message], user: str, instance: ChatInterface, context: TContext | None = None
    ):
        log_debug(f"New Message: \033[91m{messages!r}\033[0m", show_sep="above")
        self._update_main_view()
        with self._busy():
            exploration = self._exploration['view']
            plan = await self._coordinator.respond(messages, exploration.context)
            if plan is not None:
                await self._execute_plan(plan)
