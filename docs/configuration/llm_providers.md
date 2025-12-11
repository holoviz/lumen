# :material-cloud-circle: LLM Providers

Configure which AI model powers Lumen.

## Quick start

Set your API key and launch:

``` bash
export OPENAI_API_KEY="sk-..."
lumen-ai serve penguins.csv
```

Lumen auto-detects the provider from environment variables.

## Different models per agent

Use cheap models for simple tasks, powerful models for complex tasks:

``` py title="Cost-optimized configuration" hl_lines="4-7"
import lumen.ai as lmai

model_config = {
    "default": {"model": "gpt-4o-mini"},  # Cheap for most agents
    "sql": {"model": "gpt-4o"},           # Powerful for SQL
    "vega_lite": {"model": "gpt-4o"},     # Powerful for charts
    "analyst": {"model": "gpt-4o"},       # Powerful for analysis
}

llm = lmai.llm.OpenAI(model_kwargs=model_config)
ui = lmai.ExplorerUI(data='penguins.csv', llm=llm)
ui.servable()
```

Agent names map to model types: `SQLAgent` → `"sql"`, `VegaLiteAgent` → `"vega_lite"`, etc.

## Configure temperature

Lower temperature = more deterministic. Higher = more creative.

``` py title="Temperature by task" hl_lines="4 8"
model_config = {
    "sql": {
        "model": "gpt-4o",
        "temperature": 0.1,  # Deterministic SQL
    },
    "chat": {
        "model": "gpt-4o-mini",
        "temperature": 0.4,  # Natural conversation
    },
}
```

Recommended ranges: 0.1 (SQL) to 0.4 (chat).

## Provider setup

### OpenAI

=== "CLI"

    ``` bash
    export OPENAI_API_KEY="sk-..."
    lumen-ai serve penguins.csv
    ```

=== "Python"

    ``` py
    llm = lmai.llm.OpenAI(api_key='sk-...')
    ui = lmai.ExplorerUI(data='penguins.csv', llm=llm)
    ui.servable()
    ```

### Anthropic

=== "CLI"

    ``` bash
    export ANTHROPIC_API_KEY="sk-ant-..."
    lumen-ai serve penguins.csv --provider anthropic
    ```

=== "Python"

    ``` py
    llm = lmai.llm.Anthropic(api_key='sk-ant-...')
    ui = lmai.ExplorerUI(data='penguins.csv', llm=llm)
    ui.servable()
    ```

### Google Gemini

=== "CLI"

    ``` bash
    export GEMINI_API_KEY="..."
    lumen-ai serve penguins.csv --provider google
    ```

=== "Python"

    ``` py
    llm = lmai.llm.Google(api_key='...')
    ui = lmai.ExplorerUI(data='penguins.csv', llm=llm)
    ui.servable()
    ```

### Mistral

=== "CLI"

    ``` bash
    export MISTRAL_API_KEY="..."
    lumen-ai serve penguins.csv --provider mistral
    ```

=== "Python"

    ``` py
    llm = lmai.llm.MistralAI(api_key='...')
    ui = lmai.ExplorerUI(data='penguins.csv', llm=llm)
    ui.servable()
    ```

### Azure OpenAI

=== "CLI"

    ``` bash
    export AZUREAI_ENDPOINT_KEY="..."
    export AZUREAI_ENDPOINT_URL="https://your-resource.openai.azure.com/"
    lumen-ai serve penguins.csv --provider azure-openai
    ```

=== "Python"

    ``` py
    llm = lmai.llm.AzureOpenAI(
        api_key='...',
        endpoint='https://your-resource.openai.azure.com/'
    )
    ui = lmai.ExplorerUI(data='penguins.csv', llm=llm)
    ui.servable()
    ```

### Ollama (local)

Pull a model first:

``` bash
ollama pull qwen2.5:7b
```

=== "CLI"

    ``` bash
    lumen-ai serve penguins.csv --provider ollama
    ```

=== "Python"

    ``` py
    llm = lmai.llm.Ollama(
        model_kwargs={"default": {"model": "qwen2.5:7b"}}
    )
    ui = lmai.ExplorerUI(data='penguins.csv', llm=llm)
    ui.servable()
    ```

### Llama.cpp (local)

=== "CLI"

    ``` bash
    lumen-ai serve penguins.csv \
      --provider llama-cpp \
      --llm-model-url 'https://huggingface.co/unsloth/Qwen2.5-7B-Instruct-GGUF/blob/main/Qwen2.5-7B-Instruct-Q4_K_M.gguf'
    ```

=== "Python"

    ``` py
    llm = lmai.llm.LlamaCpp(
        model_kwargs={
            "default": {
                "repo_id": "unsloth/Qwen2.5-7B-Instruct-GGUF",
                "filename": "Qwen2.5-7B-Instruct-Q4_K_M.gguf"
            }
        }
    )
    ui = lmai.ExplorerUI(data='penguins.csv', llm=llm)
    ui.servable()
    ```

### LiteLLM (multi-provider)

Route between multiple providers:

``` py title="Multi-provider routing"
llm = lmai.llm.LiteLLM(
    model_kwargs={
        "default": {"model": "gpt-4o-mini"},
        "sql": {"model": "anthropic/claude-sonnet-4-5"},  # (1)!
    }
)
```

1. Use provider prefix for non-OpenAI models

### OpenAI-compatible endpoints

``` py title="Custom endpoint"
llm = lmai.llm.OpenAI(
    api_key='...',
    endpoint='https://your-endpoint.com/v1'
)
```

## Model types

Agent class names convert to model types automatically:

| Agent | Model type |
|-------|------------|
| SQLAgent | `sql` |
| VegaLiteAgent | `vega_lite` |
| ChatAgent | `chat` |
| AnalystAgent | `analyst` |
| AnalysisAgent | `analysis` |
| (others) | `default` |

Conversion rule: remove "Agent" suffix, convert to snake_case.

Additional model types:

- `edit` - Used when fixing errors
- `ui` - Used for UI initialization

## Troubleshooting

**"API key not found"** - Set environment variable or pass `api_key=` in Python.

**Wrong model used** - Model type names must be snake_case: `"sql"` not `"SQLAgent"`.

**High costs** - Use `gpt-4o-mini` for `default`, reserve `gpt-4o` for critical tasks (`sql`, `vega_lite`, `analyst`).

**Slow responses** - Local models are slower than cloud APIs. Use cloud providers when speed matters.

## Best practices

**Use powerful models for critical tasks:**
- `sql` - SQL generation needs strong reasoning
- `vega_lite` - Visualizations need design understanding  
- `analyst` - Analysis needs statistical knowledge

**Use efficient models elsewhere:**
- `default` - Simple tasks don't need expensive models
- `chat` - Conversation works with smaller models

**Set temperature by task:**
- 0.1 for SQL (deterministic)
- 0.3-0.4 for analysis and chat
- 0.5-0.7 for creative tasks

**Test before deploying** - Different models behave differently. Test with real queries.
