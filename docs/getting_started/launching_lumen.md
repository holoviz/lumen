# Launching Lumen

Start Lumen AI from the command line or Python. Choose the method that fits your workflow.

!!! tip
    Before launching, make sure you've installed Lumen and configured an LLM provider. See the [Installation guide](../installation.md).

## Launch from the command line

The simplest way to start is with a single command:

```bash
lumen-ai serve
```

This opens the chat interface at `localhost:5006`.

### Pre-load a dataset

Start with data already loaded by passing a file path or URL:

```bash
lumen-ai serve penguins.csv
```

Load from the web:

```bash
lumen-ai serve https://datasets.holoviz.org/penguins/v1/penguins.csv
```

Load multiple files:

```bash
lumen-ai serve penguins.csv penguins.parquet
```

Use wildcards:

```bash
lumen-ai serve *.csv
```

### Configure the LLM

You can configure the LLM at startup using CLI flags.

**Specify a provider:**

```bash
lumen-ai serve --provider openai
```

**Specify a model:**

```bash
lumen-ai serve --model-kwargs '{"default": {"model": "gpt-4o"}}'
```

**Adjust temperature** (controls randomness; higher = more creative):

```bash
lumen-ai serve --temperature 0.5
```

**Combine options:**

```bash
lumen-ai serve penguins.csv --provider openai --model-kwargs '{"default": {"model": "gpt-4o"}}' --temperature 0.7
```

For a complete list of CLI options, run:

```bash
lumen-ai serve --help
```

## Launch from Python

For more control, use the Python API:

```python
import lumen.ai as lmai

ui = lmai.ExplorerUI()
ui.servable()
```

Save this as `app.py`, then launch it:

```bash
panel serve app.py
```

You can pass configuration options to `ExplorerUI`:

```python
import lumen.ai as lmai

ui = lmai.ExplorerUI(
    data='penguins.csv',
    provider='openai',
    model_kwargs={'default': {'model': 'gpt-4o'}},
    temperature=0.7,
)
ui.servable()
```

Common configuration options include:

**Specify a provider:**

```bash
provider='openai'
```

**Specify a model:**

```bash
model_kwargs={'default': {'model': 'gpt-4o'}}
```

**Adjust temperature:**

```bash
temperature=0.5
```

See [LLM Providers](../configuration/llm_providers.md) for advanced setup.

## Next steps

- [Navigating the UI](navigating_the_ui.md) — Learn how to use the interface
- [Using Lumen AI](using_lumen_ai.md) — Start asking questions and exploring data
- [LLM Providers](../configuration/llm_providers.md) — Configure your LLM provider and models
