# {octicon}`zap;2em;sd-mr-1` Using Lumen AI

Powered by state-of-the-art large language models (LLMs), Lumen AI lets users chat with their tabular datasets, allowing users to explore and analyze their data without the need for complex programming or technical expertise.

## Command Line Interface

To get started with Lumen AI, users can launch the built-in chat interface through the command line interface (CLI) by calling:

```bash
lumen-ai serve
```

This will launch the chat interface on `localhost:5006`, where users can upload their datasets interactively and chat about them.

Below are some of the options that can be passed to the `serve` command, or run `lumen-ai serve --help` for all the available options.

### Source Tables

Users can specify datasets by passing file paths:

```bash
lumen-ai serve path/to/dataset.csv path/to/another/dataset.parquet
```

Wildcards are also supported:

```bash
lumen-ai serve path/to/datasets/*.csv
```

### LLM Providers

The desired LLM provider can be chosen by specifying the `--provider` option:

```bash
lumen-ai serve --provider mistral
```

Note the provider can be auto-detected from standard environment variable names, e.g. `OPENAI_API_KEY`, if the provider option is not specified.

See [Configuring LLM Providers](configuring_lumen_ai#llm-providers) for all the supported providers.

## Python API

Users can also launch Lumen AI using the Python API:

```python
import lumen.ai as lmai

ui = lmai.ExplorerUI()
ui.servable()
```

Then, launch it by calling:

```bash
panel serve app.py
```

The benefit of using the Python API is that users have much more control in [configuring Lumen AI](configuring_lumen_ai).
