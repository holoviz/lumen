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

#### Quick model selection

For simple cases, use `--model` to set the default model:

``` bash title="Set default model"
lumen-ai serve --provider ollama --model 'qwen3:32b'
```

``` bash title="Different providers"
lumen-ai serve --provider openai --model 'gpt-4o-mini'
lumen-ai serve --provider anthropic --model 'claude-sonnet-4-5'
lumen-ai serve --provider google --model 'gemini-2.0-flash'
```

The `--model` argument automatically sets `model_kwargs['default']['model']` for you.

#### Advanced model configuration

For multiple models or additional parameters, use `--model-kwargs` with JSON:

``` bash title="Multiple models for different tasks"
lumen-ai serve --model-kwargs '{
  "default": {"model": "gpt-4o-mini"},
  "sql": {"model": "gpt-4o"},
  "edit": {"model": "gpt-4o"}
}'
```

``` bash title="Combine --model with --model-kwargs"
lumen-ai serve --provider ollama \
  --model 'qwen3:32b' \
  --model-kwargs '{"edit": {"model": "mistral-small3.2:24b"}}'
```

This sets `qwen3:32b` as the default model and `mistral-small3.2:24b` for editing tasks.

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
| `--code-execution` | [Code execution mode](../getting_started/using_lumen_ai.md#code-execution-for-visualizations) | `--code-execution prompt` |
| `--provider` | LLM provider | `--provider anthropic` |
| `--api-key` | API key | `--api-key sk-...` |
| `--model` | Default model | `--model 'qwen3:32b'` |
| `--model-kwargs` | Advanced model config | `--model-kwargs '{"sql": {"model": "gpt-4o"}}'` |
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
  --model 'gpt-4o-mini' \
  --model-kwargs '{"sql": {"model": "gpt-4o"}}' \
  --temperature 0.5 \
  --agents SQLAgent ChatAgent VegaLiteAgent \
  --port 8080 \
  --show
```

``` bash title="Using Ollama locally"
lumen-ai serve data/*.csv \
  --provider ollama \
  --model 'qwen3:32b' \
  --temperature 0.4 \
  --log-level debug \
  --show
```

## View all options

``` bash
lumen-ai serve --help
```

Shows all available flags including Panel server options.
