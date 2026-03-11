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

### Specify provider explicitly

To use a specific provider, pass the `--provider` flag:

``` bash
lumen-ai serve penguins.csv --provider anthropic
lumen-ai serve penguins.csv --provider google --model 'gemini-2.5-flash'
lumen-ai serve penguins.csv --provider ollama --model 'qwen3:32b'
```

For more CLI options, see the [CLI guide](../configuration/cli.md#configure-llm).

## Different models per agent

Use cheap models for simple tasks, powerful models for complex tasks:

``` py title="Cost-optimized configuration" hl_lines="4-7"
import lumen.ai as lmai

model_config = {
    "default": {"model": "gpt-4.1-mini"},  # Cheap for most agents
    "sql": {"model": "gpt-4.1"},           # Powerful for SQL
    "vega_lite": {"model": "gpt-4.1"},     # Powerful for charts
    "deck_gl": {"model": "gpt-4.1"},       # Powerful for 3D maps
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

| Provider | Default Model | Description |
|----------|---------------|-------------|
| **Ollama** | `qwen3:32b` | External service; requires `ollama pull` to manage and run models. |
| **Llama.cpp** | `unsloth/Qwen3-32B-GGUF` | Embedded runner; automatically downloads GGUF models from HuggingFace. |
| **AI Navigator** | `server-model` | Desktop GUI; provides a local OpenAI-compatible API once the API server is started. |

**Recommended local models:**

- **General purpose:** `qwen3:32b`, `llama3.3:70b`, `qwen3:30b-a3b`, `nemotron-3-nano:30b`
- **Coding:** `qwen3-coder:32b`, `qwen2.5-coder:32b`
- **Reasoning:** `nemotron-3-nano:30b`

!!! tip "Small models (<= 8B)"
    Models with 8B parameters or fewer likely need [`--code-execution prompt`](cli.md#common-flags) to successfully create reliable Vega-Lite specifications.

    !!! warning "Security"
        Setting `--code-execution` uses Python's `exec` function to run LLM-generated code locally. **This is insecure** as the model could generate and execute malicious code. Only use this with trusted models and in secure environments.

### Router / Gateway providers

| Provider | Default Model | Description |
|----------|---------------|-------------|
| **AWS Bedrock** | `us.anthropic.claude-sonnet-4-5-20250929-v1:0` | Enterprise gateway providing access to models from Anthropic, Meta, Mistral, and more. |
| **LiteLLM** | `gpt-4.1-mini` | Unified router to access 100+ models across all supported LLM providers. |
| **AI Catalyst** | `ai_catalyst` | Enterprise model server; provides access to validated, governed open-source models. |

**AWS Bedrock options:**

- **AnthropicBedrock** - Optimized for Claude models using Anthropic's SDK
- **Bedrock** - Universal access to all Bedrock models (Claude, Llama, Mistral, Titan, etc.)

## Advanced configuration

### API keys

Each provider has three ways to supply an API key, resolved in this order:

1. **`api_key` param** — pass it directly at instantiation
2. **`api_key_env_var`** — a class variable naming the environment variable to read from; defaults to the value from `PROVIDER_ENV_VARS` for that provider
3. **`PROVIDER_ENV_VARS`** — the module-level dict that is the source of truth for env var names; changing it updates both `api_key_env_var` defaults and `get_available_llm()` auto-detection

``` py title="Pass api_key directly (highest priority)"
llm = lmai.llm.OpenAI(api_key="sk-...")
```

``` py title="Set via environment variable (default behaviour)"
import os
os.environ["OPENAI_API_KEY"] = "sk-..."
llm = lmai.llm.OpenAI()  # api_key is auto-populated from OPENAI_API_KEY
```

If your key is stored under a different environment variable name, there are
three ways to override it:

``` py title="1. Override on the class"
lmai.llm.OpenAI.api_key_env_var = "MY_OPENAI_KEY"
llm = lmai.llm.OpenAI()  # now reads from MY_OPENAI_KEY
```

``` py title="2. Override on a subclass"
class MyOpenAI(lmai.llm.OpenAI):
    api_key_env_var = "MY_OPENAI_KEY"

llm = MyOpenAI(model_kwargs={"default": {"model": "gpt-4.1-mini"}})
```

``` py title="3. Override the module-level PROVIDER_ENV_VARS dict"
import lumen.ai.llm as llm_module

llm_module.PROVIDER_ENV_VARS["openai"] = "MY_OPENAI_KEY"
```

Option 3 also affects `get_available_llm()`, which uses `PROVIDER_ENV_VARS` to
auto-detect which provider to use at startup.

!!! note
    `api_key_env_var` is a plain class variable, not a param. It must be set at
    the **class** level — changing it on an instance has no effect, because
    `models()` is a classmethod and resolves it from the class.

The default values for each provider are:

| Provider | Default `api_key_env_var` |
|----------|---------------------------|
| `OpenAI` | `OPENAI_API_KEY` |
| `Anthropic` | `ANTHROPIC_API_KEY` |
| `Google` | `GEMINI_API_KEY` |
| `MistralAI` | `MISTRAL_API_KEY` |
| `AzureOpenAI` / `AzureMistralAI` | `AZUREAI_ENDPOINT_KEY` |
| `Bedrock` / `AnthropicBedrock` | `AWS_ACCESS_KEY_ID` |
| `AICatalyst` | `AI_CATALYST_API_KEY` |
| `Ollama`, `LlamaCpp`, `AINavigator` | *(none required)* |

### Checking available models

`models()` is a classmethod you can call directly to inspect what models are
available from a provider without instantiating the LLM:

``` py title="Check available models"
print(lmai.llm.OpenAI.models())
# {'gpt-4.1-mini', 'gpt-4.1', 'gpt-4.1-nano', ...}

print(lmai.llm.Anthropic.models())
# {'claude-haiku-4-5', 'claude-sonnet-4-5', 'claude-opus-4-5', ...}
```

Providers that don't support listing models (e.g. `AzureOpenAI`, `LiteLLM`) return
an empty set.

If you implement a custom provider subclass, override `models()` to support it:

``` py title="Custom provider with models()"
class MyProvider(lmai.llm.Llm):
    api_key_env_var = "MY_PROVIDER_API_KEY"

    @classmethod
    def models(cls) -> set[str]:
        import requests
        api_key = os.environ.get(cls.api_key_env_var)
        if not api_key:
            return set()
        resp = requests.get("https://my-provider.com/models",
                            headers={"Authorization": f"Bearer {api_key}"})
        return {m["id"] for m in resp.json()["data"]}
```

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

### AI Navigator (Local)

Connect to [Anaconda AI Navigator](https://www.anaconda.com/products/ai-navigator) running on your machine:

``` py title="AI Navigator"
llm = lmai.llm.AINavigator()
```

By default, it uses `http://localhost:8080/v1`.

### AI Catalyst (Enterprise)

Connect to [Anaconda AI Catalyst](https://www.anaconda.com/platform/ai-catalyst) running on your company's infrastructure:

``` py title="AI Catalyst"
# Requires AI_CATALYST_BASE_URL and AI_CATALYST_API_KEY env vars
llm = lmai.llm.AICatalyst()
```

Or configure explicitly:

``` py title="Explicit AI Catalyst"
llm = lmai.llm.AICatalyst(
    endpoint='https://your-company.anacondaconnect.com/api/v1/model-servers/your-server/v1',
    api_key='...'
)
```

## Model types

Agent class names convert to model types automatically:

| Agent | Model type |
|-------|------------|
| SQLAgent | `sql` |
| VegaLiteAgent | `vega_lite` |
| DeckGLAgent | `deck_gl` |
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
