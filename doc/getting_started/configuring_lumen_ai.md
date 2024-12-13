# {octicon}`gear;2em;sd-mr-1` Configuring Lumen AI

Lumen AI is designed to be highly customizable and composable. Users can easily modify the default behavior of Lumen AI by providing custom data sources, LLM providers, agents, and prompts. This guide will walk users through the basic and advanced customization options available in Lumen AI.

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

## Coordinator

Lumen AI is managed by a `Coordinator` that orchestrates the actions of multiple agents to fulfill a user-defined query by generating an execution graph and executing each node along it.

The `Coordinator` is responsible for orchestrating multiple actors towards solving some goal. It can make plans, determine what additional context is needed and ensures that each actor has all the context it needs.

There are a couple options available:

- **`DependencyResolver`** chooses the agent to answer the user's query and then recursively resolves all the information required for that agent until the answer is available.

- **`Planner`** develops a plan to solve the user's query step-by-step and then executes it.

## Agents

`Agent`s are the core components of Lumen AI that solve some sub-task in an effort to address the user query. It has certain dependencies that it requires and provides certain context for other `Agent`s to use. It may also request additional context through the use of context tools.

To provide additional agents, pass desired `Agent`s to the `ExplorerUI`:

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

Users can initialize the LLM provider and pass it to the `ExplorerUI`:

```python
llm = lmai.llm.OpenAI(model="gpt-4o-mini")
ui = lmai.ExplorerUI(llm=llm)
```

The following are provided:

- **OpenAI** requires `openai`. The `api_key` defaults to the environment variable `OPENAI_API_KEY` if available and the `model` defaults to `gpt-4o-mini`.
- **MistralAI** requires `mistralai`. The `api_key` defaults to the environment variable `MISTRAL_API_KEY` if available and the `model` defaults to `mistral-large-latest`.
- **Anthropic** requires `anthropic`. The `api_key` defaults to the environment variable `ANTHROPIC_API_KEY` if available and the `model` defaults to `claude-3-5-sonnet-20240620`.
- **Llama** requires `llama-cpp-python`,and `huggingface_hub`. The `model` defaults to `qwen2.5-coder-7b-instruct-q5_k_m.gguf`.

## Tools

`Tool`s are used to provide additional context to the `Agent`s. They can be used to provide additional context to the `Agent`s, or to request additional context from the user.

Users can provide tools to the `ExplorerUI`:

```python
def duckduckgo_search(queries: list[str]) -> dict:
    results = {}
    for query in queries:
        url = f"https://duckduckgo.com/html/?q={query}"
        headers = {"User-Agent": "Mozilla/5.0"}
        response = requests.get(url, headers=headers)
        soup = BeautifulSoup(response.text, "html.parser")
        links = soup.find_all("a", {"class": "result__a"}, href=True)

        results[query] = [
            {"title": link.get_text(strip=True), "url": link["href"]} for link in links
        ]
    return results

tools = [duckduckgo_search]
ui = lmai.ExplorerUI(tools=tools)
```

This will grant the `Coordinator` access to the `duckduckgo_search` function to use at its discretion.

Alternatively, tools can be provided to individual `Agent`s:

```python
def get_wiki(articles: list[str]) -> str:
    wiki = wikipediaapi.Wikipedia("lumen-assistant", language="en")
    out = ""
    for article in articles:
        page = wiki.page(article)
        if page.exists():
            out += f"{article}:\n{page.summary}\n\n"
        else:
            out += f"The article '{article}' does not exist.\n"
    return out

tools = [get_wiki]
agents = [lmai.agents.ChatAgent(prompts={"main": {"tools": tools}})]
```

This will override the default `TableLookup` tool for the `ChatAgent` with the `get_wiki` function.

The following are built-in:

- **DocumentLookup**: provides context to the `Agent`s by looking up information in the document.

- **TableLookup**: provides context to the `Agent`s by looking up information in the table.

- **FunctionTool**: wraps arbitrary functions and makes them available as a tool for an LLM to call.
