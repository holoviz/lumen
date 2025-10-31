# Command-Line Interface

Configure Lumen AI when launching from the command line using the `lumen-ai serve` command.

## Basic usage

```bash
lumen-ai serve
```

This starts the server at `localhost:5006`.

## Configuration options

### Provider

Specify which LLM provider to use. If not specified, the provider is auto-detected from environment variables.

```bash
lumen-ai serve --provider openai
```

**Supported providers:**

- `openai` — Requires `OPENAI_API_KEY`
- `google` — Requires `GOOGLE_API_KEY`
- `anthropic` — Requires `ANTHROPIC_API_KEY`
- `mistral` — Requires `MISTRAL_API_KEY`
- `azure-openai` — Requires `AZUREAI_ENDPOINT_KEY`
- `azure-mistral` — Requires `AZUREAI_ENDPOINT_KEY`
- `ai-navigator`
- `ollama`
- `llama-cpp`
- `litellm`

### Temperature

Control randomness in LLM responses. Higher values produce more creative, less deterministic results (range: typically 0.0–2.0):

```bash
lumen-ai serve --temperature 0.5
```

### Model kwargs

Pass configuration options to your LLM as a JSON string. Models are indexed by type, allowing you to specify different models for different tasks.

```bash
lumen-ai serve --model-kwargs '{"default": {"model": "gpt-4o"}}'
```

**Model types:**

- `default` — General-purpose model for most tasks
- `sql` — Used for SQL query generation
- `vega_lite` — Used for visualization generation
- `reasoning` — Used for complex reasoning tasks
- `chat` — Used for chat interactions

**Example with multiple model types:**

```bash
lumen-ai serve --model-kwargs '{"default": {"model": "gpt-4o-mini"}, "sql": {"model": "gpt-4o"}, "vega_lite": {"model": "gpt-4o"}, "reasoning": {"model": "gpt-5"}}'
```

The JSON must be valid and properly escaped. See [LLM Providers](../configuration/llm_providers.md) for provider-specific examples.

## Load data at startup

Pre-load datasets when launching:

```bash
lumen-ai serve penguins.csv
lumen-ai serve https://datasets.holoviz.org/penguins/v1/penguins.csv
lumen-ai serve *.csv
lumen-ai serve penguins.csv penguins.parquet
```

## Combine options

```bash
lumen-ai serve penguins.csv --provider openai --model-kwargs '{"default": {"model": "gpt-4o"}}' --temperature 0.7
```

## Environment variables

Set your LLM provider API key as an environment variable to avoid passing it via CLI:

```bash
export OPENAI_API_KEY="your-key-here"
lumen-ai serve
```

The provider will be auto-detected from the available environment variables.

## View all options

For a complete list of all available flags (including Panel server options):

```bash
lumen-ai serve --help
```
