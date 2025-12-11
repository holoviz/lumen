# :material-console: Command-Line Interface

Launch Lumen AI from the command line with `lumen-ai serve`.

## Basic usage

``` bash
lumen-ai serve
```

Starts the server at `localhost:5006`.

## Load data

``` bash title="Single file"
lumen-ai serve penguins.csv
```

``` bash title="Multiple files"
lumen-ai serve customers.csv orders.csv
```

``` bash title="From URL"
lumen-ai serve https://datasets.holoviz.org/penguins/v1/penguins.csv
```

``` bash title="With wildcards"
lumen-ai serve data/*.csv
```

## Configure LLM

### Choose provider

``` bash title="Specify provider"
lumen-ai serve --provider anthropic
```

If not specified, Lumen auto-detects from environment variables.

**Supported providers:**

- `openai` - Requires `OPENAI_API_KEY`
- `anthropic` - Requires `ANTHROPIC_API_KEY`
- `google` - Requires `GEMINI_API_KEY`
- `mistral` - Requires `MISTRAL_API_KEY`
- `azure-openai` - Requires `AZUREAI_ENDPOINT_KEY`
- `ollama` - Local models
- `llama-cpp` - Local models
- `litellm` - Multi-provider

### Set API key

``` bash title="Pass API key directly"
lumen-ai serve --provider openai --api-key sk-...
```

Or use environment variables:

``` bash
export OPENAI_API_KEY="sk-..."
lumen-ai serve
```

### Configure models

Use different models for different tasks:

``` bash title="Multiple models"
lumen-ai serve --model-kwargs '{
  "default": {"model": "gpt-4o-mini"},
  "sql": {"model": "gpt-4o"}
}'
```

!!! warning "Escape JSON properly"
    The JSON string must be properly quoted. Use single quotes around the entire JSON, double quotes inside.

### Adjust temperature

``` bash title="Control randomness"
lumen-ai serve --temperature 0.5
```

Lower (0.1) = deterministic. Higher (0.7) = creative. Range: 0.0-2.0

## Select agents

``` bash title="Use specific agents"
lumen-ai serve --agents SQLAgent ChatAgent VegaLiteAgent
```

Agent names are case-insensitive. The "Agent" suffix is optional: `sql` = `sqlagent` = `SQLAgent`

## Common flags

| Flag | Purpose | Example |
|------|---------|---------|
| `--provider` | LLM provider | `--provider anthropic` |
| `--api-key` | API key | `--api-key sk-...` |
| `--model-kwargs` | Model config | `--model-kwargs '{"sql": {"model": "gpt-4o"}}'` |
| `--temperature` | Randomness | `--temperature 0.5` |
| `--agents` | Active agents | `--agents SQLAgent ChatAgent` |
| `--port` | Server port | `--port 8080` |
| `--address` | Network address | `--address 0.0.0.0` |
| `--show` | Auto-open browser | `--show` |
| `--log-level` | Verbosity | `--log-level DEBUG` |

## Full example

``` bash title="Complete configuration"
lumen-ai serve penguins.csv \
  --provider openai \
  --model-kwargs '{"default": {"model": "gpt-4o-mini"}, "sql": {"model": "gpt-4o"}}' \
  --temperature 0.5 \
  --agents SQLAgent ChatAgent VegaLiteAgent \
  --port 8080 \
  --show
```

## View all options

``` bash
lumen-ai serve --help
```

Shows all available flags including Panel server options.
