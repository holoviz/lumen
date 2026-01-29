# :material-application: User Interface

Configure the Lumen AI chat interface.

## ChatUI vs ExplorerUI

Lumen provides two interfaces:

- **ExplorerUI** - Split view with table explorer, multiple explorations, and a navigation tree. Best for most use cases.
- **ChatUI** - Simple chat-only interface. Best for embedded applications.

Use `ExplorerUI` unless you specifically need the simpler ChatUI.

## Basic configuration

``` py title="Minimal setup"
import lumen.ai as lmai

ui = lmai.ExplorerUI(data='penguins.csv')
ui.servable()
```

## Common parameters

### Load data

``` py title="Multiple sources"
ui = lmai.ExplorerUI(data=['customers.csv', 'orders.csv'])
```

### Configure LLM

``` py title="Change provider" hl_lines="2"
ui = lmai.ExplorerUI(
    data='penguins.csv',
    llm=lmai.llm.Anthropic()
)
```

### Add agents

``` py title="Custom agents" hl_lines="3"
ui = lmai.ExplorerUI(
    data='penguins.csv',
    agents=[MyCustomAgent()]  # Adds to 8 default agents
)
```

### Add tools

``` py title="Custom tools" hl_lines="3"
ui = lmai.ExplorerUI(
    data='penguins.csv',
    tools=[my_function]  # Functions become tools automatically
)
```

### Change title

``` py title="Custom title" hl_lines="2"
ui = lmai.ExplorerUI(
    data='penguins.csv',
    title='Sales Analytics'
)
```

### Custom suggestions

``` py title="Quick action buttons" hl_lines="3-6"
ui = lmai.ExplorerUI(
    data='penguins.csv',
    suggestions=[
        ("search", "What data is available?"),
        ("bar_chart", "Show trends"),
    ]  # (1)!
)
```

1. Tuples of (Material icon name, button text)

## Advanced parameters

### Enable chat logging

``` py title="Log conversations"
ui = lmai.ExplorerUI(
    data='penguins.csv',
    logs_db_path='logs.db'  # SQLite database for all messages
)
```

### Configure coordinator

``` py title="Coordinator options"
ui = lmai.ExplorerUI(
    data='penguins.csv',
    coordinator_params={
        'verbose': True,
        'validation_enabled': False
    }
)
```

### Custom file handlers

``` py title="Handle custom file types"
def handle_hdf5(file_bytes, alias, filename):
    # Process file and add to source
    return True

ui = lmai.ExplorerUI(
    data='penguins.csv',
    table_upload_callbacks={'hdf5': handle_hdf5}
)
```

### Provide initial context

``` py title="Pre-populate context"
ui = lmai.ExplorerUI(
    data='penguins.csv',
    context={'company': 'Acme', 'year': 2024}  # (1)!
)
```

1. Available to all agents

### Custom notebook export

``` py title="Add preamble to exports"
ui = lmai.ExplorerUI(
    data='penguins.csv',
    notebook_preamble='# Analysis by Data Team\n# Generated: 2024'
)
```

## Source controls

![Weather Control UI](../assets/configuration/controls.png)

**Source controls provide UI interfaces for loading data from external services.**

Controls let users interactively fetch data from APIs, databases, or specialized sources directly in the Lumen UI sidebar. They're essential for integrating external data that isn't available as static files or database connections.

### Why use source controls?

Source controls solve common data integration challenges:

- **External APIs** - Fetch data from REST APIs that require parameters or authentication
- **User selection** - Let users pick data subsets (years, regions, variables) before loading
- **Dynamic data** - Access real-time or frequently updated data sources
- **Complex workflows** - Handle multi-step data fetching and transformation
- **Authentication** - Manage API keys or credentials securely

### Built-in controls

| Control | Use for |
|---------|---------|
| `UploadControls` | Uploading local files (CSV, Excel, etc.) |
| `DownloadControls` | Fetching data from URLs |

### Creating custom controls

Custom controls inherit from `BaseSourceControls` and override two hooks:

| Hook | Purpose |
|------|---------|
| `_load()` | **Main hook** - fetch data and return a `SourceResult` |
| `_render_controls()` | Provide UI widgets (rendered above load button) |

The base class handles loading states, error display, progress, output registration, and event triggering automatically.

### Minimal example

```python
import asyncio
import pandas as pd
import param
from lumen.ai.controls import BaseSourceControls, SourceResult
from panel_material_ui import IntSlider


class WeatherControl(BaseSourceControls):
    """Fetch weather data from an API."""

    year = param.Integer(default=2024, bounds=(2020, 2024))

    label = '<span class="material-icons">wb_sunny</span> Weather Data'
    load_button_label = "Fetch Weather"

    def _render_controls(self):
        """Provide widgets - rendered above the load button."""
        return [IntSlider.from_param(self.param.year, label="Year")]

    async def _load(self) -> SourceResult:
        """Main hook - fetch data and return result."""
        self.progress("Fetching weather data...")

        url = f"https://api.weather.example/data?year={self.year}"
        df = await asyncio.to_thread(pd.read_csv, url)

        if df.empty:
            return SourceResult.empty("No data returned")

        return SourceResult.from_dataframe(
            df,
            table_name=f"weather_{self.year}",
            year=self.year,  # Metadata attached to table
        )


# Use in ExplorerUI
ui = lmai.ExplorerUI(source_controls=[WeatherControl])
```

### SourceResult

`SourceResult` is the return type for `_load()` with convenient factory methods:

```python
# From a DataFrame (most common)
SourceResult.from_dataframe(df, "table_name", year=2023, source="api")

# From an existing DuckDB source
SourceResult.from_source(my_source, table="users")

# Empty result (no data loaded)
SourceResult.empty("No data returned from API")
```

### Progress reporting

The `self.progress()` helper provides a simple API:

```python
# Indeterminate (spinner)
self.progress("Loading metadata...")

# Determinate with percentage (0-100)
self.progress("Downloading...", value=50)

# Determinate with current/total (auto-calculates %)
self.progress("Downloading...", current=500, total=1000)

# Increment pattern for loops
self.progress("Processing files...", total=len(files))
for f in files:
    process(f)
    self.progress.increment()

# Clear progress
self.progress.clear()
```

### Class attributes

Customize appearance with class attributes:

| Attribute | Default | Purpose |
|-----------|---------|---------|
| `label` | `""` | HTML label shown in sidebar |
| `load_button_label` | `"Load Data"` | Button text |
| `load_button_icon` | `"download"` | Material icon name |
| `load_mode` | `"button"` | `"button"` or `"manual"` |

### Manual load mode

For controls where loading is triggered by something other than a button (like clicking a table row), use `load_mode="manual"`:

```python
class CatalogBrowser(BaseSourceControls):
    """Browse and select from a catalog."""

    load_mode = "manual"  # No load button

    def __init__(self, **params):
        super().__init__(**params)
        self._layout.loading = True  # Show spinner during init
        pn.state.onload(self._load_catalog)

    def _render_controls(self):
        self._table = Tabulator(on_click=self._on_click, ...)
        return [self._table]

    def _load_catalog(self):
        self._table.value = fetch_catalog()
        self._layout.loading = False

    async def _on_click(self, event):
        # Use _run_load() for lifecycle management
        await self._run_load(self._fetch_row(event.row))

    async def _fetch_row(self, row_idx) -> SourceResult:
        self.progress("Downloading...")
        data = await download(row_idx)
        return SourceResult.from_dataframe(data, "selected_data")
```

### Best practices

- **Use `asyncio.to_thread()`** for blocking API calls to avoid freezing the UI
- **Report progress** with `self.progress()` for long operations
- **Return `SourceResult.empty()`** with a message when no data is available
- **Add metadata** to help LLM agents understand the data context
- **Validate inputs** before making expensive API calls
- **Cache API responses** when possible to avoid redundant calls
- **Use `normalize_table_name()`** to ensure DuckDB-compatible table names

### See also

- [Building a Census Data Explorer](../examples/tutorials/census_data_ai_explorer.md) — Complete walkthrough with minimal and full examples

## Complete example

``` py title="Full configuration" linenums="1"
import lumen.ai as lmai
from lumen.sources.snowflake import SnowflakeSource

source = SnowflakeSource(
    account='acme',
    database='sales',
    authenticator='externalbrowser'
)

llm = lmai.llm.OpenAI(
    model_kwargs={
        'default': {'model': 'gpt-4o-mini'},
        'sql': {'model': 'gpt-4o'},
    }
)

analysis_agent = lmai.agents.AnalysisAgent(analyses=[MyAnalysis])

ui = lmai.ExplorerUI(
    data=source,
    llm=llm,
    agents=[analysis_agent],
    tools=[my_tool],
    title='Sales Analytics',
    suggestions=[
        ("trending_up", "Revenue trends"),
        ("people", "Top customers"),
    ],
    log_level='INFO',
    logs_db_path='logs.db'
)

ui.servable()
```

## All parameters

Quick reference:

| Parameter | Type | Purpose |
|-----------|------|---------|
| `data` | str/Path/Source/list | Data sources to load |
| `llm` | Llm | LLM provider (default: OpenAI) |
| `agents` | list | Additional agents |
| `analyses` | list | Custom analyses |
| `context` | dict | Initial context |
| `coordinator` | type | Planner or DependencyResolver |
| `coordinator_params` | dict | Coordinator configuration |
| `default_agents` | list | Replace default agents |
| `demo_inputs` | list | Demo prompts for the coordinator |
| `document_vector_store` | VectorStore | Vector store for document tools |
| `export_functions` | dict | Map exporter names to export functions |
| `interface` | type | Chat interface class |
| `llm_choices` | list | LLM model choices shown in Settings |
| `log_level` | str | DEBUG/INFO/WARNING/ERROR |
| `logfire_tags` | list | Log LLM calls to Logfire with tags |
| `logs_db_path` | str | Chat logging database path |
| `notebook_preamble` | str | Export header |
| `provider_choices` | dict | LLM providers shown in Settings |
| `source_controls` | list | Source control components for data |
| `suggestions` | list | Quick action buttons |
| `title` | str | App title |
| `tools` | list | Custom tools |
| `upload_handlers` | dict | File extension upload handlers |
| `vector_store` | VectorStore | Vector store for non-doc tools |

See parameter docstrings in code for complete details.

### See also

- [Data Sources](sources.md) — File and database connections
- [Tools](tools.md) — Custom functions for agents
- [Building a Census Data Explorer](../examples/tutorials/census_data_ai_explorer.md) — Complete walkthrough with minimal and full examples
