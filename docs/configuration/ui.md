# :material-application: User Interface

Configure the Lumen AI chat interface.

## ChatUI vs ExplorerUI

Lumen provides two interfaces:

- **ExplorerUI** - Split view with table explorer, multiple explorations, and breadcrumbs. Best for most use cases.
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
| `default_agents` | list | Replace default 8 agents |
| `tools` | list | Custom tools |
| `analyses` | list | Custom analyses |
| `title` | str | App title |
| `suggestions` | list | Quick action buttons |
| `coordinator` | type | Planner or DependencyResolver |
| `coordinator_params` | dict | Coordinator configuration |
| `log_level` | str | DEBUG/INFO/WARNING/ERROR |
| `logs_db_path` | str | Chat logging database path |
| `context` | dict | Initial context |
| `notebook_preamble` | str | Export header |

See parameter docstrings in code for complete details.
