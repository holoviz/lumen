# :material-cloud-circle: LLM Providers

Configure which AI model powers Lumen.

!!! tip "First time setup?"
    See the [Installation guide](../installation.md) for step-by-step instructions on setting up API keys and environment variables for each provider.

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
    "default": {"model": "gpt-4.1-mini"},  # Cheap for most agents
    "sql": {"model": "gpt-4.1"},           # Powerful for SQL
    "vega_lite": {"model": "gpt-4.1"},     # Powerful for charts
    "analyst": {"model": "gpt-4.1"},       # Powerful for analysis
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
        "model": "gpt-4.1",
        "temperature": 0.1,  # Deterministic SQL
    },
    "chat": {
        "model": "gpt-4.1-mini",
        "temperature": 0.4,  # Natural conversation
    },
}
```

Recommended ranges: 0.1 (SQL) to 0.4 (chat).

## Supported providers

For installation and API key setup instructions, see the [Installation guide](../installation.md).

### Cloud providers

| Provider | Default Model | Popular Models |
|----------|---------------|----------------|
| **OpenAI** | `gpt-4.1-mini` | `gpt-4.1`, `gpt-4-turbo`, `gpt-4` |
| **Anthropic** | `claude-haiku-4-5` | `claude-sonnet-4-5`, `claude-opus-4-5` |
| **Google** | `gemini-3-flash-preview` | `gemini-3-pro-preview`, `gemini-2.5-flash`, `gemini-2.0-flash` |
| **Mistral** | `mistral-small-latest` | `mistral-large-latest`, `ministral-8b-latest` |
| **Azure OpenAI** | `gpt-4.1-mini` | `gpt-4.1`, `gpt-4-turbo`, `gpt-4` |
| **Azure Mistral** | `azureai` | `mistral-large`, `mistral-small` |

!!! warning "Reasoning Models Not Suitable for Dialog"
    Reasoning models like `gpt-5`, `o4-mini`, and `gemini-2.0-flash-thinking` are **significantly slower** than standard models. They are designed for single, complex queries that require deep thinking, not interactive chat interfaces. For dialog-based applications like Lumen, use standard models for better user experience.

### Local providers

| Provider | Default Model | Notes |
|----------|---------------|-------|
| **Ollama** | `qwen3:32b` | Requires Ollama installed, models pulled locally |
| **Llama.cpp** | `unsloth/Qwen3-32B-GGUF` | Auto-downloads models on first use |

**Recommended local models:**

- **General purpose:** `qwen3:32b`, `llama3.3:70b`, `qwen3:30b-a3b`, `nemotron-3-nano:30b`
- **Coding:** `qwen3-coder:32b`, `qwen2.5-coder:32b`
- **Reasoning:** `nemotron-3-nano:30b`

### Router / Gateway providers

| Provider | Purpose | Setup |
|----------|---------|-------|
| **AWS Bedrock** | Gateway to Anthropic, Meta, Mistral, Amazon models | [Installation guide](../installation.md#aws-bedrock-gateway) |
| **LiteLLM** | Router for 100+ models across all providers | [Installation guide](../installation.md#router-multi-provider) |

**AWS Bedrock options:**

- **AnthropicBedrock** - Optimized for Claude models using Anthropic's SDK
- **Bedrock** - Universal access to all Bedrock models (Claude, Llama, Mistral, Titan, etc.)

## Advanced configuration

### Custom endpoints

Override default API endpoints:

``` py title="Custom endpoint"
llm = lmai.llm.OpenAI(
    api_key='...',
    endpoint='https://your-custom-endpoint.com/v1'
)
```

### Managed Identity (Azure)

Use Azure Active Directory authentication:

``` py title="Azure Managed Identity"
from azure.identity import DefaultAzureCredential, get_bearer_token_provider

token_provider = get_bearer_token_provider(
    DefaultAzureCredential(),
    "https://cognitiveservices.azure.com/.default"
)

llm = lmai.llm.AzureOpenAI(
    api_version="2024-02-15-preview",
    endpoint="https://your-resource.openai.azure.com/",
    model_kwargs={
        "default": {
            "model": "gpt4o-mini",
            "azure_ad_token_provider": token_provider
        }
    }
)
```

### Fallback models (LiteLLM)

Automatically retry with backup models if primary fails:

``` py title="Fallback configuration"
llm = lmai.llm.LiteLLM(
    model_kwargs={
        "default": {"model": "gpt-4.1-mini"}
    },
    fallback_models=[
        "gpt-4.1-mini",
        "claude-haiku-4-5",
        "gemini/gemini-2.5-flash"
    ]
)
```

### Remote Ollama server

Connect to Ollama running on another machine:

``` py title="Remote Ollama"
llm = lmai.llm.Ollama(
    endpoint='http://your-server:11434/v1',
    model_kwargs={"default": {"model": "qwen3:32b"}}
)
```

## Model types

Agent class names convert to model types automatically:

| Agent | Model type |
|-------|------------|
| SQLAgent | `sql` |
| VegaLiteAgent | `vega_lite` |
| ChatAgent | `chat` |
| AnalysisAgent | `analysis` |
| (others) | `default` |

Conversion rule: remove "Agent" suffix, convert to snake_case.

Additional model types:

- `edit` - Used when fixing errors
- `ui` - Used for UI initialization

## Model string formats

Different providers use different model string formats:

- **OpenAI**: `"gpt-4.1"`, `"gpt-4.1-mini"`, `"gpt-4-turbo"`, `"gpt-4"`
- **Anthropic**: `"claude-sonnet-4-5"`, `"claude-haiku-4-5"`, `"claude-opus-4-5"`
- **Google**: `"gemini-3-flash-preview"`, `"gemini-2.5-flash"`
- **Mistral**: `"mistral-large-latest"`, `"mistral-small-latest"`
- **Azure**: `"your-deployment-name"` (use your Azure deployment name)
- **Bedrock**: `"us.anthropic.claude-sonnet-4-5-20250929-v1:0"`, `"meta.llama3-70b-instruct-v1:0"`
- **LiteLLM**: `"gpt-4.1-mini"` (OpenAI), `"anthropic/claude-sonnet-4-5"` (Anthropic), `"gemini/gemini-2.5-flash"` (Google)

For LiteLLM, use the `provider/model` format for non-OpenAI models.

## Troubleshooting

**"API key not found"** - Set environment variable or pass `api_key=` in Python.

**Wrong model used** - Model type names must be snake_case: `"sql"` not `"SQLAgent"`.

**High costs** - Use `gpt-4.1-mini` or `claude-haiku-4-5` for `default`, reserve `gpt-4.1` or `claude-sonnet-4-5` for critical tasks (`sql`, `vega_lite`, `analyst`).

**Slow responses** - Local models are slower than cloud APIs. Use cloud providers when speed matters.

**AWS credentials not found** - For Bedrock, ensure AWS credentials are configured. See [Installation guide](../installation.md#aws-bedrock-gateway).

## Best practices

**Use efficient models elsewhere:**

- `default` - Simple tasks work well with `gpt-4.1-mini` or `claude-haiku-4-5`
- `chat` - Conversation works with smaller models

**Avoid reasoning models in dialog interfaces:**

- Reasoning models (`o1`, `o1-mini`, `gemini-3.0-pro`) are **significantly slower**
- They're designed for single, complex queries, not interactive chat
- For Lumen's dialog interface, use standard models (`mistral-small-latest`, `gpt-4.1-mini`, `claude-sonnet-4-5`)
- Reserve reasoning models for batch processing or one-off complex analyses

**Set temperature by task:**

- 0.1 for SQL (deterministic)
- 0.3-0.4 for analysis and chat
- 0.5-0.7 for creative tasks

**Test before deploying:**

- Different models behave differently. Test with real queries.
