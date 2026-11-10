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

### Database connections

Connect to databases using SQLAlchemy URLs:

``` bash title="SQLite database"
lumen-ai serve sqlite:///path/to/database.db
```

``` bash title="DuckDB database"
lumen-ai serve duckdb:///path/to/database.db
```

``` bash title="Auto-detect database type"
lumen-ai serve database.db
```

``` bash title="PostgreSQL"
lumen-ai serve postgresql://user:password@localhost:5432/mydb
```

``` bash title="MySQL"
lumen-ai serve mysql+pymysql://user:password@localhost:3306/mydb
```

``` bash title="Multiple sources"
lumen-ai serve data.csv postgresql://localhost/mydb
```

#### Automatic database detection

For `.db` files, Lumen automatically detects whether they're SQLite or DuckDB databases:

``` bash title="Auto-detected .db files"
lumen-ai serve test.db
```

Lumen reads the file header to determine the database type:

- SQLite files start with `"SQLite format 3"`
- DuckDB files start with `"DUCK"`

You can still use explicit URLs if preferred:

``` bash title="Explicit SQLite"
lumen-ai serve sqlite:///test.db
```

``` bash title="Explicit DuckDB"
lumen-ai serve duckdb:///test.db
```

!!! info "Tables auto-discovered"
    When connecting to a database, Lumen automatically discovers all available tables. No need to specify individual table names.

**Supported databases:**

- SQLite: `sqlite:///path/to/file.db` (or just `file.db` for auto-detection)
- DuckDB: `duckdb:///path/to/file.db` (or just `file.db` for auto-detection)
- PostgreSQL: `postgresql://user:pass@host:port/db`
- MySQL: `mysql+pymysql://user:pass@host:port/db`
- Oracle: `oracle://user:pass@host:port/db`
- SQL Server: `mssql+pyodbc://user:pass@host:port/db`

See [SQLAlchemy documentation](https://docs.sqlalchemy.org/en/20/core/engines.html#database-urls) for full URL syntax.

## Configure LLM

### Choose provider

``` bash title="Specify provider"
lumen-ai serve --provider anthropic
```

If not specified, Lumen auto-detects from environment variables. When no key is set it lists the options that work without one, such as `--provider kilo` or `--provider ollama`.

**Supported providers:**

| Provider | Requires |
|----------|----------|
| `openai` | `OPENAI_API_KEY` |
| `anthropic` | `ANTHROPIC_API_KEY` |
| `google` | `GEMINI_API_KEY` |
| `mistral` | `MISTRAL_API_KEY` |
| `azure-openai` | `AZUREAI_ENDPOINT_KEY` |
| `azure-mistral` | `AZUREAI_ENDPOINT_KEY` |
| `groq` | `GROQ_API_KEY` |
| `bedrock` | AWS credentials |
| `anthropic-bedrock` | AWS credentials |
| `openrouter` | `OPENROUTER_API_KEY` |
| `kilo` | `KILO_API_KEY` |
| `litellm` | Keys for the providers it routes to |
| `ai-catalyst` | `AI_CATALYST_API_KEY` |
| `ollama` | A running Ollama server |
| `llama-cpp` | Nothing, downloads the model on first use |
| `mlx` | Apple Silicon and `mlx-lm` |
| `ai-navigator` | A running AI Navigator API server |
| `codex-cli` | A signed-in Codex CLI |
| `claude-code` | A signed-in Claude Code CLI |
| `copilot-cli` | A signed-in GitHub Copilot CLI |
| `antigravity-cli` | A signed-in Google Antigravity CLI |

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

#### Quick model selection

For simple cases, use `--model` to set the default model:

``` bash title="Set default model"
lumen-ai serve --provider ollama --model 'qwen3.6:35b-a3b'
```

``` bash title="Different providers"
lumen-ai serve --provider openai --model 'gpt-6-luna'
lumen-ai serve --provider anthropic --model 'claude-sonnet-4-6'
lumen-ai serve --provider google --model 'gemini-3.8-flash'
```

The `--model` argument automatically sets `model_kwargs['default']['model']` for you.

#### Advanced model configuration

For multiple models or additional parameters, use `--model-kwargs` with JSON:

``` bash title="Multiple models for different tasks"
lumen-ai serve --model-kwargs '{
  "default": {"model": "gpt-6-luna"},
  "sql": {"model": "gpt-5.6-luna"}
}'
```

``` bash title="Combine --model with --model-kwargs"
lumen-ai serve --provider ollama \
  --model 'qwen3.6:35b-a3b' \
  --model-kwargs '{"edit": {"model": "qwen3.8:27b"}}'
```

This sets `qwen3.6:35b-a3b` as the default model and `qwen3.8:27b` for editing tasks.

!!! warning "Escape JSON properly"
    The JSON string must be properly quoted. Use single quotes around the entire JSON, double quotes inside.

!!! tip "Model types"
    Common model types in `model_kwargs`:
    
    - `default` - General queries and analysis
    - `sql` - SQL query generation (some providers use specialized models)
    - `edit` - Code/chart editing (may use more capable models)
    - `ui` - UI responsive check (lightweight models)

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
| `--code-execution` | [Code execution mode](../getting_started/building_lumen_apps.md#enable-code-execution): `hidden` (default), `disabled`, `prompt`, `llm` or `allow` | `--code-execution prompt` |
| `--provider` | LLM provider | `--provider anthropic` |
| `--api-key` | API key | `--api-key sk-...` |
| `--model` | Default model | `--model 'qwen3.6:35b-a3b'` |
| `--model-kwargs` | Advanced model config | `--model-kwargs '{"sql": {"model": "gpt-5.6-luna"}}'` |
| `--temperature` | Randomness | `--temperature 0.5` |
| `--agents` | Active agents | `--agents SQLAgent ChatAgent` |
| `--port` | Server port | `--port 8080` |
| `--address` | Network address | `--address 0.0.0.0` |
| `--show` | Auto-open browser | `--show` |
| `--log-level` | Verbosity | `--log-level DEBUG` |
| `--provider-endpoint` | Custom API endpoint, e.g. an OpenAI-compatible server | `--provider-endpoint http://localhost:8000/v1` |
| `--llm-model-url` | GGUF model for `llama-cpp`, with model kwargs as query params | `--llm-model-url 'https://huggingface.co/.../model.gguf'` |
| `--validation-mode` | Instructor mode for structured output | `--validation-mode tool_call` |
| `--logfire-tags` | Enable Logfire logging with these tags | `--logfire-tags dev lumen` |

!!! tip "Small models (<= 8B)"
    Models with 8B parameters or fewer likely need `--code-execution prompt` to successfully create reliable Vega-Lite specifications.

    !!! warning "Security"
        Setting `--code-execution` uses Python's `exec` function to run LLM-generated code locally. **This is insecure** as the model could generate and execute malicious code. Only use this with trusted models and in secure environments.

## Full example

``` bash title="Complete configuration"
lumen-ai serve penguins.csv \
  --provider openai \
  --model 'gpt-6-luna' \
  --model-kwargs '{"sql": {"model": "gpt-5.6-luna"}}' \
  --temperature 0.5 \
  --agents SQLAgent ChatAgent VegaLiteAgent \
  --port 8080 \
  --show
```

``` bash title="Using Ollama locally"
lumen-ai serve data/*.csv \
  --provider ollama \
  --model 'qwen3.6:35b-a3b' \
  --temperature 0.4 \
  --log-level debug \
  --show
```

## View all options

``` bash
lumen-ai serve --help
```

Shows all available flags including Panel server options.
