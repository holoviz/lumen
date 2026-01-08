# :material-play-circle: Launching Lumen

Start Lumen AI from the command line or Python. Choose the method that fits your workflow.

!!! tip "Before launching"
    Make sure you've installed Lumen and configured an LLM provider. See the [Installation guide](../installation.md).

## Launch from the command line

The simplest way to start is with a single command:

```bash
lumen-ai serve
```

This opens the chat interface at `localhost:5006`.

### Pre-load a dataset

Start with data already loaded:

``` bash title="Single file"
lumen-ai serve penguins.csv
```

``` bash title="From URL"
lumen-ai serve https://datasets.holoviz.org/penguins/v1/penguins.csv
```

``` bash title="Multiple files"
lumen-ai serve penguins.csv orders.parquet
```

``` bash title="Using wildcards"
lumen-ai serve data/*.csv
```

### Configure the LLM

Configure the LLM at startup using CLI flags:

``` bash title="Specify provider"
lumen-ai serve --provider openai
```

``` bash title="Specify model"
lumen-ai serve --model-kwargs '{"default": {"model": "gpt-4o"}}'
```

``` bash title="Adjust temperature"
lumen-ai serve --temperature 0.5  # (1)!
```

1. Controls randomness; higher = more creative (0.0-2.0)

``` bash title="Combine options"
lumen-ai serve penguins.csv \
  --provider openai \
  --model-kwargs '{"default": {"model": "gpt-4o"}}' \
  --temperature 0.7
```

For a complete list of CLI options:

```bash
lumen-ai serve --help
```

## Launch from Python

For more control, use the Python API:

``` py title="Minimal Python app"
import lumen.ai as lmai

ui = lmai.ExplorerUI()
ui.servable()
```

Save as `app.py`, then launch:

```bash
panel serve app.py
```

### Pre-load data

``` py title="Load data in Python"
import lumen.ai as lmai

ui = lmai.ExplorerUI(data='penguins.csv')
ui.servable()
```

### Configure the LLM

``` py title="Configure LLM in Python" hl_lines="4-8 11"
import lumen.ai as lmai

# Configure your LLM
llm = lmai.llm.OpenAI(
    model_kwargs={
        'default': {'model': 'gpt-4o-mini'},
        'sql': {'model': 'gpt-4o'}
    },
    temperature=0.7
)

ui = lmai.ExplorerUI(data='penguins.csv', llm=llm)
ui.servable()
```

See [LLM Providers](../configuration/llm_providers.md) for advanced LLM configuration.

### Add custom components

``` py title="Custom agents and analyses" hl_lines="4-6 11-13"
import lumen.ai as lmai
from lumen.ai.agents import AnalysisAgent

# Create custom analysis
analysis_agent = AnalysisAgent(analyses=[MyAnalysis])

ui = lmai.ExplorerUI(
    data='penguins.csv',
    agents=[analysis_agent, MyCustomAgent()],
    tools=[my_custom_tool],
    suggestions=[
        ("search", "What data is available?"),
        ("bar_chart", "Show me a visualization"),
    ]
)
ui.servable()
```

## Common CLI flags

| Flag | Purpose | Example |
|------|---------|---------|
| `--provider` | Specify LLM provider | `--provider anthropic` |
| `--model-kwargs` | Configure models | `--model-kwargs '{"default": {"model": "claude-sonnet-4-5"}}'` |
| `--temperature` | Control randomness | `--temperature 0.5` |
| `--port` | Custom port | `--port 8080` |
| `--address` | Network address | `--address 0.0.0.0` |
| `--show` | Auto-open browser | `--show` |
| `--log-level` | Debug verbosity | `--log-level DEBUG` |

## Next steps

- [Navigating the UI](navigating_the_ui.md) — Learn how to use the interface
- [Using Lumen AI](using_lumen_ai.md) — Start asking questions and exploring data
- [LLM Providers](../configuration/llm_providers.md) — Configure your LLM provider and models
