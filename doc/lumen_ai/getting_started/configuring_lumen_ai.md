# {octicon}`gear;2em;sd-mr-1` Configuring Lumen AI

Lumen AI is designed to be highly customizable and composable. Users can easily modify the default behavior of Lumen AI by providing custom data sources, LLM providers, agents, and prompts. This guide will walk you through the basic and advanced customization options available in Lumen AI.

## Data Sources

There are several ways to provide data to Lumen AI:

**Method 1: Direct data parameter (recommended):**
```python
import lumen.ai as lmai

# Single file
ui = lmai.ExplorerUI(data='https://datasets.holoviz.org/penguins/v1/penguins.csv')
ui.servable()

# Multiple files with mixed types
ui = lmai.ExplorerUI(data=[
    'https://raw.githubusercontent.com/vega/vega-datasets/master/data/cars.json',
    'https://raw.githubusercontent.com/vega/vega-datasets/master/data/stocks.csv'
])
ui.servable()
```

**Method 2: Using memory with source object:**
```python
from lumen.sources.duckdb import DuckDBSource

# Create source and add to memory
source = DuckDBSource(tables=[
    'https://raw.githubusercontent.com/vega/vega-datasets/master/data/airports.csv',
    'https://raw.githubusercontent.com/vega/vega-datasets/master/data/flights-5k.json'
])
lmai.memory["source"] = source

# Initialize UI
ui = lmai.ExplorerUI()
ui.servable()
```

:::{admonition} Tip
:class: success

Lumen AI currently supports CSV, Parquet, JSON, XLSX, GeoJSON, WKT, and ZIP files for tables, and DOC, DOCX, PDF, TXT, MD, HTML, and RST files for documents. In fact as long as [MarkItDown](https://github.com/microsoft/markitdown) any file format that is not a table will be treated as a document.
:::

## Coordinator

Lumen AI is managed by a `Coordinator` that orchestrates the actions of multiple agents to fulfill a user-defined query by generating an execution graph and executing each node along it.

The `Coordinator` is responsible for orchestrating multiple actors towards solving some goal. It can make plans, determine what additional context is needed and ensures that each actor has all the context it needs.

There are a couple options available:

- **`DependencyResolver`** chooses the agent to answer the user's query and then recursively resolves all the information required for that agent until the answer is available.

- **`Planner`** develops a plan to solve the user's query step-by-step and then executes it. This is the default coordinator used by Lumen AI.

## Agents

`Agent`s are the core components of Lumen AI that solve some sub-task in an effort to address the user query. It has certain dependencies that it requires and provides certain context for other `Agent`s to use. It may also request additional context through the use of context tools.

To provide additional agents, pass desired `Agent`s to the `ExplorerUI`. For example, Lumen AI uses VegaLite for visualizations, but you can also use hvPlot:

```python
import lumen.ai as lmai

agents = [lmai.agents.hvPlotAgent]
ui = lmai.ExplorerUI(agents=agents)
```

Or if you'd like to only use hvPlot for visualize, you may override the default list of agents:

```python
import lumen.ai as lmai
default_agents = [
    lmai.agents.TableListAgent, lmai.agents.ChatAgent, lmai.agents.SQLAgent, lmai.agents.hvPlotAgent
]
ui = lmai.ExplorerUI(default_agents=default_agents)
```

These are the `default_agents`: `TableListAgent, ChatAgent, DocumentListAgent, AnalystAgent, SourceAgent, SQLAgent, VegaLiteAgent`.

The following `Agent`s are built-in:

- **`AnalystAgent`** is a specialized agent that can perform complex analyses and provide insights based on the data available to the LLM.

- **`ChatAgent`** provides high-level information about user's datasets, including details about available tables, columns, and statistics, and can also provide suggestions on what to explore.

- **`DocumentListAgent`** lists all the documents available to the LLM, allowing users to explore and select relevant documents for their queries.

- **`TableListAgent`** provides an overview of all the tables available to the LLM.

- **`SourceAgent`** helps users upload and import their data sources so that the LLM has access to the desired datasets.

- **`SQLAgent`** translates user's natural language queries into SQL queries, returning a table view of the results.

- **`hvPlotAgent`** creates interactive visualizations of the data and query results so users can continue dive into the visuals from various perspectives.

- **`VegaLiteAgent`** generates visualizations ready to export--perfect for presentations and reports.

- **`AnalysisAgent`** allows users to plug in their own custom analyses and views.

See [Custom Agents](../how_to/ai_config/custom_agents) for more information on creating custom agents.

## LLM Providers

Lumen AI supports multiple Large Language Model (LLM) providers to ensure flexibility and access to the best models available, or specific models that are available to the user.

See the [how to guides on configuring LLM Providers](../how_to/llm/index) for more information on configuring different LLM providers and models.

## Tools

`Tool`s are used to provide additional context to `Coordinator`s and `Agent`s.

Users can provide tools to the `ExplorerUI`:

```python
ui = lmai.ExplorerUI(tools=[lmai.DocumentLookup])
```

This will grant the `Coordinator` access to the `document_lookup` function to use at its discretion.

By default, `ChatAgent` includes both the `DocumentLookup` and `TableLookup` tools, but you can also provide these tools to other, individual `Agent`s as well.

```python
ui = lmai.ExplorerUI(
    agents=[lmai.agents.SQLAgent(prompts={"main": {"tools": [lmai.DocumentLookup]}})]
)
```

However, unlike the `Coordinator`, agents will *always* use all available `tools` in their queries.

The following are built-in:

- **DocumentLookup**: provides context to the `Agent`s by looking up information in the document.

- **TableLookup**: provides context to the `Agent`s by looking up information in the table.

- **FunctionTool**: wraps arbitrary functions and makes them available as a tool for an LLM to call.

See [Custom Tools](../how_to/custom_tools) for more information on creating custom tools.

## Environment Variables

Lumen AI supports configuration through environment variables for convenient setup:

**LLM Provider API Keys:**
```bash
# Set one or more of these
export OPENAI_API_KEY='your-openai-key'
export ANTHROPIC_API_KEY='your-anthropic-key'
export MISTRAL_API_KEY='your-mistral-key'
export AZUREAI_ENDPOINT_KEY='your-azure-key'
export AZUREAI_ENDPOINT_URL='your-azure-endpoint'
```

Lumen AI will automatically detect and use the first available API key.

## Advanced Configuration

For more advanced customization options, see:

- **[LLM Configuration](../how_to/llm/index)** - Detailed LLM setup and model selection
- **[Custom Agents](../how_to/custom_agents)** - Building your own agents
- **[Custom Tools](../how_to/custom_tools)** - Creating custom function tools
- **[Custom Analyses](../how_to/custom_analyses)** - Implementing domain-specific analyses
