# Lumen AI

Powered by state-of-the-art large language models (LLMs), Lumen AI lets users chat with their tabular datasets, allowing users to explore and analyze their data without the need for complex programming or technical expertise.

### Python Script

To get started with Lumen AI, copy and paste the following into a file, like `app.py`:

```python
import lumen.ai as lmai

ui = lmai.ExplorerUI()
ui.servable()
```

Then, launch it by calling:

```bash
panel serve app.py
```

This will serve the app on `localhost:5006`, where users can upload their datasets interactively and chat about them.

### Command Line Interface

Alternatively, users can also launch the Lumen AI chat interface through the command line interface (CLI) by calling:

```bash
lumen-ai serve
```

## Features

### LLM Providers

Lumen AI supports multiple Large Language Model (LLM) providers to ensure flexibility and access to the best models available, or specific models that are available to the user.

The following are provided:

- **OpenAI** requires `openai`. The `api_key` defaults to the environment variable `OPENAI_API_KEY` if available and the `model` defaults to `gpt-4o-mini`.
- **MistralAI** requires `mistralai`. The `api_key` defaults to the environment variable `MISTRAL_API_KEY` if available and the `model` defaults to `mistral-large-latest`.
- **Anthropic** requires `anthropic`. The `api_key` defaults to the environment variable `ANTHROPIC_API_KEY` if available and the `model` defaults to `claude-3-5-sonnet-20240620`.
- **Llama** requires `llama-cpp-python`,and `huggingface_hub`. The `model` defaults to `qwen2.5-coder-7b-instruct-q5_k_m.gguf`.

### Agents

Lumen AI is designed to be highly customizable and composable, allowing users to specify which agents to utilize for their tasks.

The following agents are built-in:

- **ChatAgent** provides high-level information about your datasets, including details about available tables, columns, and statistics, and can also provide suggestions on what to explore.

- **SourceAgent** helps users upload and import their data sources so that the LLM has access to the desired datasets.

- **TableListAgent** provides an overview of all the tables available to the LLM.

- **SQLAgent** translates user's natural language queries into SQL queries, returning a table view of the results.

- **hvPlotAgent** creates interactive visualizations of the data and query results so users can continue dive into the visuals from various perspectives.

- **VegaLiteAgent** generates visualizations ready to export--perfect for presentations and reports.

- **AnalysisAgent** allows users to plug in their own custom analyses and views.

In addition to these agents, users also have the flexibility to create their own custom agents, described in [#customization/agents].

### Coordinator

Lumen AI is managed by a `Coordinator` that orchestrates the actions of multiple agents to fulfill a user-defined query by generating an execution graph and executing each node along it.

There are a few options available:

- `DependencyResolver` chooses the agent to answer the user's query and then recursively resolves all the information required for that agent until the answer is available.

- `Planner` develops a plan to solve the user's query step-by-step and then executes it.

## Basic Customization

The following components can be easily modified or replaced by simply setting a few keyword arguments.

### Data Sources

In a script, you can initialize Lumen AI with specific datasets by injecting `current_source` into `lmai.memory`:

```python
lmai.memory["current_source"] = DuckDBSource(
    tables=["path/to/table.csv", "dir/data.parquet"]
)
```

In the CLI, suffix the `lumen-ai serve` calls with file paths:

```bash
lumen-ai serve path/to/table.csv dir/data.parquet
```

:::{admonition} Tip
:class: success

Lumen AI currently supports CSV, Parquet, and JSON files.
:::

### LLM Providers

In a script, you can initialize the provider and pass it to the ExplorerUI class:

```python
llm = lmai.llm.OpenAI(model="gpt-4o-mini")
ui = lmai.ExplorerUI(llm=llm)
```

In the CLI, provide the `--llm` arg:

```bash
lumen-ai serve --llm openai
```

### Agents

By default, the `TableListAgent`, `ChatAgent`, `ChatDetailsAgent`, `SourceAgent`, `SQLAgent`, `VegaLiteAgent` are used in `default_agents`.

In a script, to provide additional agents, pass them to the `ExplorerUI` class:

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

In the CLI, you can only provide additional agents by providing the `--agents` arg:

```bash
lumen-ai serve --agents TableListAgent chatagent Sql
```

:::{admonition} Tip
:class: success

Within the CLI, the names of the agents are case insensitive and the suffix `agent` can be dropped, e.g. `ChatAgent` can be specified as `chatagent`, `Chatagent`, and `Chat`.
:::

### Prompts

Each `Agent` is defined by a system prompt with modular components:

1. `instructions`
2. `context`
3. `embeddings`
4. `examples`

## Advanced Customization

The following components require a bit more effort, but offer greater flexibility and control. Users can customize these components by writing additional code.

### Analyses

Sometimes, users may want to output views of fixed analyses that the LLMs may not be able to reproduce on its own.

To achieve this, the user can write their own custom `Analysis` to perform custom actions tailored to a particular use case and/or domain.

The `AnalysisAgent` will then invoke the custom `Analysis` when needed based on relevance to the user's query and whether the current exploration's dataset contains all the required `columns`.

As a basic example, the user may be a meteorologist and want to perform a `WindAnalysis`.

```python
import param
import numpy as np
import pandas as pd
import lumen.ai as lmai
from lumen.layout import Layout
from lumen.transforms import Transform
from lumen.sources.duckdb import DuckDBSource
from lumen.views import hvPlotView, Table


class WindSpeedDirection(Transform):

    u_component = param.String(doc="Column name of the zonal component")

    v_component = param.String(doc="Column name of the meridional component")

    def apply(self, df):
        # Calculate wind speed
        df["wind_speed"] = np.sqrt(
            df[self.u_component] ** 2 + df[self.v_component] ** 2
        )

        # Calculate wind direction in degrees (from north)
        df["wind_direction"] = np.degrees(
            np.arctan2(df[self.v_component], df[self.u_component])
        )

        # Ensure wind direction is in the range [0, 360)
        df["wind_direction"] = (df["wind_direction"] + 360) % 360
        return df


class WindAnalysis(lmai.Analysis):
    """
    Calculates the wind speed and direction from the u and v components of wind,
    displaying both the table and meteogram.
    """

    columns = param.List(default=["time", "u", "v"])

    def __call__(self, pipeline):
        wind_pipeline = pipeline.chain(transforms=[WindSpeedDirection()])
        wind_speed_view = hvPlotView(
            pipeline=wind_pipeline,
            title="Wind Speed",
            x="time",
            y="wind_speed",
        )
        wind_direction_view = hvPlotView(
            pipeline=wind_pipeline,
            title="Wind Speed",
            x="time",
            y="wind_speed",
            text="wind_direction",
            kind="labels",
        )
        wind_table = Table(wind_pipeline)
        return Layout(
            views=[
                wind_speed_view,
                wind_direction_view,
                wind_table,
            ],
            layout=[[0, 1], [2]],
        )


llm = lmai.llm.Llama()
uv_df = pd.DataFrame({
    "time": pd.date_range('2024-11-11', '2024-11-22'),
    "u": np.random.rand(12),
    "v": np.random.rand(12)
})
source = lmai.memory["current_source"] = DuckDBSource.from_df({"uv_df": uv_df})
analysis_agent = lmai.agents.AnalysisAgent(analyses=[WindAnalysis])
ui = lmai.ExplorerUI(llm=llm, agents=[analysis_agent])
ui.servable()
```
