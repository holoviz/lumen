# {octicon}`zap;2em;sd-mr-1` Configuring Lumen AI

Lumen AI is designed to be highly customizable and composable. Users can easily modify the default behavior of Lumen AI by providing custom data sources, LLM providers, agents, and prompts. This guide will walk you through the basic and advanced customization options available in Lumen AI.

## Features

### Coordinator

Lumen AI is managed by a `Coordinator` that orchestrates the actions of multiple agents to fulfill a user-defined query by generating an execution graph and executing each node along it.

The `Coordinator` is responsible for orchestrating multiple actors towards solving some goal. It can make plans, determine what additional context is needed and ensures that each actor has all the context it needs.

There are a couple options available:

- **`DependencyResolver`** chooses the agent to answer the user's query and then recursively resolves all the information required for that agent until the answer is available.

- **`Planner`** develops a plan to solve the user's query step-by-step and then executes it.

### Agents

`Agent`s are the core components of Lumen AI that solve some sub-task in an effort to address the user query. It has certain dependencies that it requires and provides certain context for other `Agent`s to use. It may also request additional context through the use of context tools.

The following `Agent`s are built-in:

- **`ChatAgent`** provides high-level information about your datasets, including details about available tables, columns, and statistics, and can also provide suggestions on what to explore.

- **`SourceAgent`** helps users upload and import their data sources so that the LLM has access to the desired datasets.

- **`TableListAgent`** provides an overview of all the tables available to the LLM.

- **`SQLAgent`** translates user's natural language queries into SQL queries, returning a table view of the results.

- **`hvPlotAgent`** creates interactive visualizations of the data and query results so users can continue dive into the visuals from various perspectives.

- **`VegaLiteAgent`** generates visualizations ready to export--perfect for presentations and reports.

- **`AnalysisAgent`** allows users to plug in their own custom analyses and views.

In addition to these agents, users also have the flexibility to create their own custom agents, described in [#customization/agents].

### LLM Providers

Lumen AI supports multiple Large Language Model (LLM) providers to ensure flexibility and access to the best models available, or specific models that are available to the user.

The following are provided:

- **OpenAI** requires `openai`. The `api_key` defaults to the environment variable `OPENAI_API_KEY` if available and the `model` defaults to `gpt-4o-mini`.
- **MistralAI** requires `mistralai`. The `api_key` defaults to the environment variable `MISTRAL_API_KEY` if available and the `model` defaults to `mistral-large-latest`.
- **Anthropic** requires `anthropic`. The `api_key` defaults to the environment variable `ANTHROPIC_API_KEY` if available and the `model` defaults to `claude-3-5-sonnet-20240620`.
- **Llama** requires `llama-cpp-python`,and `huggingface_hub`. The `model` defaults to `qwen2.5-coder-7b-instruct-q5_k_m.gguf`.

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

### Template Prompts
Every `Actor` (`Agent` and `Coordinator`) operates based on a system prompt structured with modular components:

1. `instructions`
2. `context`
3. `embeddings`
4. `examples`

The base system prompt template follows this format:

```jinja2
{% block instructions %}
{% endblock %}

{% block context %}
{% endblock %}

{% block embeddings %}
{% endblock %}

{% block examples %}
{% endblock %}
```

For example, the `ChatAgent`'s prompt template uses `instructions` and `context`:

```jinja2
{% extends 'Actor/main.jinja2' %}

{% block instructions %}
Act as a helpful assistant for high-level data exploration, focusing on available datasets. If data is available, explain the purpose of each column and suggest ways to get started if necessary. Maintain factual accuracy, avoid speculation, and refrain from writing or suggesting code.
{% endblock %}

{% block context %}
{% if tables|length > 1 %}
Available tables:
{{ closest_tables }}
{% elif schema %}
{{ table }} with schema: {{ schema }}
{% endif %}
{% if 'data' in memory %}
Here's a summary of the dataset the user recently inquired about:
\```
{{ memory['data'] }}
\```
{% endif %}
{% endblock %}
```

If you'd like to override the `instructions` you can specify `template_overrides`:

```python
template_overrides = {
    "main": {
        "instructions": "Act like the user's meteorologist, and explain jargon in the format of a weather report."
    },
}
agents = [lmai.agents.ChatAgent(template_overrides=template_overrides)]
ui = lmai.ExplorerUI(agents=agents)
```

This will result in the following prompt template:

```jinja2
{% extends 'Actor/main.jinja2' %}

{% block instructions %}
Act like the user's meteorologist, and explain jargon in the format of a weather report.
{% endblock %}

{% block context %}
{% if tables|length > 1 %}
Available tables:
{{ closest_tables }}
{% elif schema %}
{{ table }} with schema: {{ schema }}
{% endif %}
{% if 'data' in memory %}
Here's a summary of the dataset the user recently inquired about:
\```
{{ memory['data'] }}
\```
{% endif %}
{% endblock %}
```

If you simply want to add a bit of text to the `instructions`, you can specify `{{ super() }}`:

```python
template_overrides = {
    "main": {
        "instructions": "{{ super() }}. Spice it up by speaking like a pirate."
    },
}

:::{admonition} Tip
:class: success

To debug prompts, you may specify `log_level="DEBUG"` in the `ExplorerUI` class to see the rendered prompts.
:::

You can also provide `examples`:

```python
template_overrides = {
    "main": {
        "instructions": "Speak like a pirate.",
        "examples": """
            Example:
            '''
            Yarr, the wind be blowin' from the north at 10 knots.
            '''
        """
    },
}
agents = [lmai.agents.ChatAgent(template_overrides=template_overrides)]
ui = lmai.ExplorerUI(agents=agents)
```

```jinja2
{% extends 'Actor/main.jinja2' %}

{% block instructions %}
Act like the user's meteorologist, and explain jargon in the format of a weather report.
{% endblock %}

{% block context %}
{% if tables|length > 1 %}
Available tables:
{{ closest_tables }}
{% elif schema %}
{{ table }} with schema: {{ schema }}
{% endif %}
{% if 'data' in memory %}
Here's a summary of the dataset the user recently inquired about:
\```
{{ memory['data'] }}
\```
{% endif %}
{% endblock %}

{% block examples %}
Example:
'''
Yarr, the wind be blowin' from the north at 10 knots.
'''
{% endblock %}
```

Alternatively, if you'd like to override the entire prompt template, you can specify the `template` key in `prompts` as a string or a valid path to a template:

```python
prompts = {
    "main": {
        "template": """
            Act like the user's meteorologist, and explain jargon in the format of a weather report.

            Available tables:
            {closest_tables}

            {table} with schema: {schema}
        """
    }
}
agents = [lmai.agents.ChatAgent(prompts=prompts)]
ui = lmai.ExplorerUI(agents=agents)
```

:::{admonition} Warning
:class: warning

If you override the prompt template, ensure that the template includes all the necessary parameters. If any parameters are missing, the LLM may lack context and provide irrelevant responses.
:::

For a listing of prompts, please see the Lumen codebase [here](https://github.com/holoviz/lumen/tree/main/lumen/ai/prompts).

### Model Prompts

Some agents employ structured Pydantic models as their response format so that the code easily use the responses. These models can also be overridden by specifying the `model` key in `prompts`.

For example, the `SQLAgent`'s `main` default model is:

```python
class Sql(BaseModel):

    chain_of_thought: str = Field(
        description="""
        You are a world-class SQL expert, and your fame is on the line so don't mess up.
        Then, think step by step on how you might approach this problem in an optimal way.
        If it's simple, just provide one sentence.
        """
    )

    expr_slug: str = Field(
        description="""
        Give the SQL expression a concise, but descriptive, slug that includes whatever transforms were applied to it,
        e.g. top_5_athletes_gold_medals
        """
    )

    query: str = Field(description="Expertly optimized, valid SQL query to be executed; do NOT add extraneous comments.")
```

To override the `chain_of_thought` field, you can subclass the `Sql` model:

```python
from lumen.ai.models import Sql

class CustomSql(Sql):
    chain_of_thought: str = Field(
        description="Think through the query like an expert DuckDB user."
    )
```

Then, you can specify the `model` key in `prompts`:

```python
prompts = {
    "main": {
        "model": CustomSql
    }
}
agents = [lmai.agents.SQLAgent(prompts=prompts)]
ui = lmai.ExplorerUI(agents=agents)
```

Note, the field names in the model must match the original model's field names, or else unexpected fields will not be used.

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

    u_component = param.String(default="u", doc="Column name of the zonal component")

    v_component = param.String(default="v", doc="Column name of the meridional component")

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
