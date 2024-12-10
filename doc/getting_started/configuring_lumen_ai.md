# {octicon}`gear;2em;sd-mr-1` Configuring Lumen AI

Lumen AI is designed to be highly customizable and composable. Users can easily modify the default behavior of Lumen AI by providing custom data sources, LLM providers, agents, and prompts. This guide will walk users through the basic and advanced customization options available in Lumen AI.

## Coordinator

Lumen AI is managed by a `Coordinator` that orchestrates the actions of multiple agents to fulfill a user-defined query by generating an execution graph and executing each node along it.

The `Coordinator` is responsible for orchestrating multiple actors towards solving some goal. It can make plans, determine what additional context is needed and ensures that each actor has all the context it needs.

There are a couple options available:

- **`DependencyResolver`** chooses the agent to answer the user's query and then recursively resolves all the information required for that agent until the answer is available.

- **`Planner`** develops a plan to solve the user's query step-by-step and then executes it.

## Agents

`Agent`s are the core components of Lumen AI that solve some sub-task in an effort to address the user query. It has certain dependencies that it requires and provides certain context for other `Agent`s to use. It may also request additional context through the use of context tools.

To provide additional agents, pass desired `Agent`s to the `ExplorerUI` class:

```python
agents = [lmai.agents.hvPlotAgent]
ui = lmai.ExplorerUI(agents=agents)
```

Or, to override the default list of agents:

```python
default_agents = [
    lmai.agents.TableListAgent, lmai.agents.ChatAgent, lmai.agents.SQLAgent, lmai.agents.hvPlotAgent
]
```

By default, the `TableListAgent`, `ChatAgent`, `ChatDetailsAgent`, `SourceAgent`, `SQLAgent`, `VegaLiteAgent` are used in `default_agents`.

The following `Agent`s are built-in:

- **`ChatAgent`** provides high-level information about user's datasets, including details about available tables, columns, and statistics, and can also provide suggestions on what to explore.

- **`SourceAgent`** helps users upload and import their data sources so that the LLM has access to the desired datasets.

- **`TableListAgent`** provides an overview of all the tables available to the LLM.

- **`SQLAgent`** translates user's natural language queries into SQL queries, returning a table view of the results.

- **`hvPlotAgent`** creates interactive visualizations of the data and query results so users can continue dive into the visuals from various perspectives.

- **`VegaLiteAgent`** generates visualizations ready to export--perfect for presentations and reports.

- **`AnalysisAgent`** allows users to plug in their own custom analyses and views.

## LLM Providers

Lumen AI supports multiple Large Language Model (LLM) providers to ensure flexibility and access to the best models available, or specific models that are available to the user.

Users can initialize the LLM provider and pass it to the ExplorerUI class:

```python
llm = lmai.llm.OpenAI(model="gpt-4o-mini")
ui = lmai.ExplorerUI(llm=llm)
```

The following are provided:

- **OpenAI** requires `openai`. The `api_key` defaults to the environment variable `OPENAI_API_KEY` if available and the `model` defaults to `gpt-4o-mini`.
- **MistralAI** requires `mistralai`. The `api_key` defaults to the environment variable `MISTRAL_API_KEY` if available and the `model` defaults to `mistral-large-latest`.
- **Anthropic** requires `anthropic`. The `api_key` defaults to the environment variable `ANTHROPIC_API_KEY` if available and the `model` defaults to `claude-3-5-sonnet-20240620`.
- **Llama** requires `llama-cpp-python`,and `huggingface_hub`. The `model` defaults to `qwen2.5-coder-7b-instruct-q5_k_m.gguf`.

## Data Sources

In a script, users can initialize Lumen AI with specific datasets by injecting `current_source` into `lmai.memory`:

```python
lmai.memory["current_source"] = DuckDBSource(
    tables=["path/to/table.csv", "dir/data.parquet"]
)
```

:::{admonition} Tip
:class: success

Lumen AI currently supports CSV, Parquet, JSON, XLSX, GeoJSON, WKT, and ZIP files for tables, and DOC, DOCX, PDF, TXT, MD, and RST files for documents.
:::
