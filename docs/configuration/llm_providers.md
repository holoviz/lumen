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
    "default": {"model": "gpt-5-mini"},  # Cheap for most agents
    "sql": {"model": "gpt-5.2"},         # Powerful for SQL
    "vega_lite": {"model": "gpt-5.2"},   # Powerful for charts
    "analyst": {"model": "gpt-5.2"},     # Powerful for analysis
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
        "model": "gpt-5.2",
        "temperature": 0.1,  # Deterministic SQL
    },
    "chat": {
        "model": "gpt-5-mini",
        "temperature": 0.4,  # Natural conversation
    },
}
```

Recommended ranges: 0.1 (SQL) to 0.4 (chat).

## Provider setup

### OpenAI

**Prerequisites:**

- Lumen AI installed in your Python environment
- An OpenAI API Key from [the OpenAI Dashboard](https://platform.openai.com/api-keys)

**Default models:**

- `default`: `gpt-4.1-mini`
- `sql`: `gpt-4.1-mini`
- `vega_lite`: `gpt-4.1-mini`
- `edit`: `gpt-4.1-mini`
- `ui`: `gpt-4.1-nano`

**Popular models:**

- **`gpt-5.2`** - Best model for coding and agentic tasks
- **`gpt-5-mini`** - Faster, cost-efficient for well-defined tasks
- **`gpt-5-nano`** - Fastest, most cost-efficient
- **`gpt-4.1`** - Smartest non-reasoning model

**Environment variables:**

- `OPENAI_API_KEY`: Your OpenAI API key (required)
- `OPENAI_ORGANIZATION`: Your OpenAI organization ID (optional)

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

**Prerequisites:**

- Lumen AI installed in your Python environment
- An Anthropic API Key from [the Anthropic Console](https://console.anthropic.com/settings/keys)

**Default models:**

- `default`: `claude-haiku-4-5`
- `edit`: `claude-sonnet-4-5`

**Popular models:**

- **`claude-sonnet-4-5`** - Smart model for complex agents and coding
- **`claude-haiku-4-5`** - Fastest with near-frontier intelligence
- **`claude-opus-4-5`** - Premium model with maximum intelligence

**Environment variables:**

- `ANTHROPIC_API_KEY`: Your Anthropic API key (required)

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

**Prerequisites:**

- Lumen AI installed in your Python environment
- A Google AI API Key from [Google AI Studio](https://aistudio.google.com/app/apikey)

**Default models:**

- `default`: `gemini-2.5-flash` (best price-performance with thinking)
- `edit`: `gemini-2.5-pro` (state-of-the-art thinking model)

**Environment variables:**

- `GEMINI_API_KEY`: Your Google AI API key (required)

**Popular models:**

- **`gemini-2.5-pro`** - State-of-the-art thinking model
- **`gemini-2.5-flash`** - Best price-performance with thinking
- **`gemini-2.5-flash-lite`** - Lightweight with thinking capabilities
- **`gemini-2.0-flash`** - Latest general-purpose model

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

**Prerequisites:**

- Lumen AI installed in your Python environment
- A Mistral API Key from [the Mistral Dashboard](https://console.mistral.ai/api-keys/)

**Default models:**

- `default`: `mistral-small-latest`
- `edit`: `mistral-medium-latest`

**Environment variables:**

- `MISTRAL_API_KEY`: Your Mistral API key (required)

**Popular models:**

- **`mistral-small-latest`** - Cost-effective for general tasks
- **`mistral-large-latest`** - Advanced reasoning and complex queries
- **`ministral-8b-latest`** - Lightweight edge model

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

**Prerequisites:**

- Lumen AI installed in your Python environment
- An Azure OpenAI Service resource from [Azure Portal](https://portal.azure.com/)
- Azure Inference API Key and Endpoint URL from your Azure OpenAI resource

**Environment variables:**

- `AZUREAI_ENDPOINT_KEY`: Your Azure API key (required)
- `AZUREAI_ENDPOINT_URL`: Your Azure endpoint URL (required)

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

=== "Managed Identity"

    ``` py
    from azure.identity import DefaultAzureCredential, get_bearer_token_provider
    
    token_provider = get_bearer_token_provider(
        DefaultAzureCredential(),
        "https://cognitiveservices.azure.com/.default"
    )
    
    llm = lmai.llm.AzureOpenAI(
        api_version="...",
        endpoint="...",
        model_kwargs={
            "default": {
                "model": "gpt4o-mini",
                "azure_ad_token_provider": token_provider
            }
        }
    )
    ui = lmai.ExplorerUI(data='penguins.csv', llm=llm)
    ui.servable()
    ```

### AWS Bedrock

**Prerequisites:**

- Lumen AI installed in your Python environment
- `boto3` installed: `pip install boto3`
- AWS credentials with access to Bedrock foundation models

**Default models:**

- `default`: `us.anthropic.claude-sonnet-4-5`
- `edit`: `us.anthropic.claude-opus-4-5`

**Popular models:**

- **`us.anthropic.claude-sonnet-4-5`** - High intelligence cross-region inference profile
- **`us.anthropic.claude-haiku-4-5`** - High performance cross-region inference profile
- **`us.anthropic.claude-opus-4-5`** - Highest intelligence cross-region inference profile
- **`us.anthropic.claude-3-5-sonnet-20241022-v2:0`** - Performance cross-region inference profile

**Environment variables:**

- `AWS_ACCESS_KEY_ID`: Your AWS access key ID (required if not using SSO)
- `AWS_SECRET_ACCESS_KEY`: Your AWS secret access key (required if not using SSO)
- `AWS_SESSION_TOKEN`: Your AWS session token (optional)
- `AWS_PROFILE`: Your AWS profile name (for SSO or named profiles)
- `AWS_DEFAULT_REGION`: Your AWS region (e.g. `us-east-1`)

=== "CLI"

    ``` bash
    export AWS_ACCESS_KEY_ID="..."
    export AWS_SECRET_ACCESS_KEY="..."
    lumen-ai serve penguins.csv --provider bedrock
    ```

=== "SSO Login"

    ``` bash
    aws sso login --profile your-profile
    export AWS_PROFILE=your-profile
    lumen-ai serve penguins.csv --provider bedrock
    ```

=== "Python"

    ``` py
    llm = lmai.llm.Bedrock(
        aws_access_key_id='...',
        api_key='...', # Secret Access Key
        region_name='us-east-1'
    )
    ui = lmai.ExplorerUI(data='penguins.csv', llm=llm)
    ui.servable()
    ```

#### Anthropic on Bedrock

For specialized support of Anthropic models on AWS Bedrock, Lumen provides the `AnthropicBedrock` provider. This provider uses the `anthropic` library's native Bedrock integration and is restricted to Anthropic models.

**Prerequisites:**

- `anthropic` installed: `pip install anthropic`

=== "Python"

    ``` py
    llm = lmai.llm.AnthropicBedrock(
        aws_access_key_id='...',
        api_key='...', # Secret Access Key
        region_name='us-east-1'
    )
    ui = lmai.ExplorerUI(data='penguins.csv', llm=llm)
    ui.servable()
    ```

### Ollama (local)

**Prerequisites:**

- Lumen AI installed in your Python environment
- Ollama installed from [ollama.com](https://ollama.com/)
- At least one model pulled locally

**Default models:**

- `default`: `qwen3:8b`

**Setup Ollama:**

=== "Unix/Linux/macOS"

    ``` bash
    # Install Ollama
    curl -fsSL https://ollama.com/install.sh | sh
    
    # Start the service
    ollama serve
    
    # Pull a model
    ollama pull qwen3:8b
    ```

=== "Windows"

    Download and run the installer from [ollama.com](https://ollama.com/), then:
    
    ``` bash
    # Pull a model
    ollama pull qwen3:8b
    ```

**Recommended models:**

| Use Case | Model | Notes |
|----------|-------|-------|
| **General purpose** | `qwen3:8b` | Default - comprehensive capabilities |
| | `llama3.3:70b` | State of the art 70B performance |
| | `gemma3:12b` | Google's efficient model |
| **Coding** | `qwen3-coder:30b` | Specialized for code generation |
| | `qwen2.5-coder:7b` | Smaller coding model |
| **Lightweight** | `gemma3:12b` | High-performing, efficient |
| | `phi4:14b` | Microsoft's lightweight SOTA |
| **Reasoning** | `deepseek-r1:7b` | Advanced reasoning model |
| **Latest** | `llama4:latest` | Cutting edge model |

**Run Lumen:**

=== "CLI"

    ``` bash
    lumen-ai serve penguins.csv --provider ollama
    ```

=== "Python"

    ``` py
    llm = lmai.llm.Ollama(
        model_kwargs={"default": {"model": "qwen3:8b"}}
    )
    ui = lmai.ExplorerUI(data='penguins.csv', llm=llm)
    ui.servable()
    ```

=== "Remote Server"

    ``` py
    # Connect to Ollama on another machine
    llm = lmai.llm.Ollama(endpoint='http://your-server:11434/v1')
    ui = lmai.ExplorerUI(data='penguins.csv', llm=llm)
    ui.servable()
    ```

!!! info "Model String Formats"
    - **OpenAI**: `"gpt-5.2"`, `"gpt-5-mini"`, `"gpt-5-nano"`, `"gpt-4.1"`
    - **Anthropic**: `"claude-sonnet-4-5"`, `"claude-haiku-4-5"`, `"claude-opus-4-5"`
    - **Google**: `"gemini/gemini-2.0-flash"`, `"gemini/gemini-2.0-flash-thinking-exp"`
    - **Azure**: `"azure/your-deployment-name"`
    - **Bedrock**: `"us.anthropic.claude-sonnet-4-5"`
    - **Cohere**: `"command-r-plus"`
    
    See [LiteLLM providers](https://docs.litellm.ai/docs/providers) for complete list.

### Llama.cpp (local)

**Prerequisites:**

- Lumen AI installed in your Python environment
- Llama.cpp installed - [Installation Guide](https://llama-cpp-python.readthedocs.io/en/latest/)
- Decent hardware (modern GPU, ARM Mac, or high-core CPU)

**Default models:**

- `default`: `Qwen/Qwen2.5-Coder-7B-Instruct-GGUF`
- `reasoning`: Uses default if not specified

!!! note "First Run Downloads Model"
    The first time you use Llama.cpp, it will download the specified model, which may take some time depending on model size and your internet connection.

**Run Lumen:**

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

=== "Custom Model"

    ``` py
    # DeepSeek R1 with custom settings
    llm = lmai.llm.LlamaCpp(
        model_kwargs={
            "default": {
                "repo_id": "bartowski/DeepSeek-R1-Distill-Qwen-32B-GGUF",
                "filename": "DeepSeek-R1-Distill-Qwen-32B-Q4_K_M.gguf",
                "chat_format": "qwen",
                "n_ctx": 131072
            }
        }
    )
    ui = lmai.ExplorerUI(data='penguins.csv', llm=llm)
    ui.servable()
    ```

=== "URL Shortcut"

    ``` bash
    # Pass model URL with query params for config
    lumen-ai serve penguins.csv \
      --llm-model-url 'https://huggingface.co/bartowski/DeepSeek-R1-Distill-Qwen-32B-GGUF/blob/main/DeepSeek-R1-Distill-Qwen-32B-Q4_K_M.gguf?chat_format=qwen&n_ctx=131072'
    ```

!!! tip "For Larger Models"
    For working with larger models, consider using the [Llama.cpp server](https://llama-cpp-python.readthedocs.io/en/latest/server/) with OpenAI-compatible endpoints for better performance.

### LiteLLM (multi-provider)

**Prerequisites:**

- Lumen AI installed in your Python environment
- `litellm` package: `pip install litellm`
- API keys for providers you want to use

**Default models:**

- `default`: `gpt-4.1-mini` (cost-effective for general tasks)
- `edit`: `anthropic/claude-sonnet-4-5` (advanced reasoning)
- `sql`: `gpt-4.1-mini` (SQL query generation)

**Environment variables:**

Set environment variables for any provider you want to use:

- `OPENAI_API_KEY` - For OpenAI models
- `ANTHROPIC_API_KEY` - For Anthropic models
- `GEMINI_API_KEY` - For Google models
- `AZURE_API_KEY` + `AZURE_API_BASE` - For Azure
- `AWS_ACCESS_KEY_ID` + `AWS_SECRET_ACCESS_KEY` - For AWS Bedrock
- `COHERE_API_KEY` - For Cohere models

**Route between providers:**

=== "Multi-Provider"

    ``` py
    # Mix different providers for different tasks
    llm = lmai.llm.LiteLLM(
        model_kwargs={
            "default": {"model": "gpt-4.1-mini"},                    # OpenAI
            "edit": {"model": "anthropic/claude-sonnet-4-5"},        # Anthropic
            "sql": {"model": "gpt-4.1-mini"}                         # OpenAI
        }
    )
    ui = lmai.ExplorerUI(data='penguins.csv', llm=llm)
    ui.servable()
    ```

=== "With Fallbacks"

    ``` py
    # Automatic fallback if primary model fails
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
    ui = lmai.ExplorerUI(data='penguins.csv', llm=llm)
    ui.servable()
    ```

=== "Azure Config"

    ``` py
    # Azure OpenAI with LiteLLM
    llm = lmai.llm.LiteLLM(
        model_kwargs={
            "default": {"model": "azure/gpt-4.1-mini"}
        },
        litellm_params={
            "api_base": "https://your-resource.openai.azure.com/",
            "api_version": "2024-02-15-preview"
        }
    )
    ui = lmai.ExplorerUI(data='penguins.csv', llm=llm)
    ui.servable()
    ```

**Supported providers:**

OpenAI • Anthropic • Google Gemini • Azure • AWS Bedrock • Cohere • Hugging Face • Vertex AI • And 100+ more

!!! info "Model String Formats"
    - **OpenAI**: `"gpt-4.1-mini"`, `"gpt-4.1-nano"`, `"gpt-5-mini"`, `"gpt-4o"`
    - **Anthropic**: `"anthropic/claude-sonnet-4-5"`, `"anthropic/claude-haiku-4-5"`, `"anthropic/claude-opus-4-1"`
    - **Google**: `"gemini/gemini-2.5-flash"`, `"gemini/gemini-2.5-pro"`
    - **Mistral**: `"mistral/mistral-medium-latest"`, `"mistral/mistral-small-latest"`
    - **Azure**: `"azure/your-deployment-name"`
    - **Bedrock**: `"bedrock/us.anthropic.claude-sonnet-4-5"`
    - **Cohere**: `"command-r-plus"`
    
    See [LiteLLM providers](https://docs.litellm.ai/docs/providers) for complete list.

### OpenAI-compatible endpoints

**Prerequisites:**

- Lumen AI installed in your Python environment
- Server endpoint URL of your OpenAI API-compliant server
- API key (if required by your endpoint)

**Environment variables:**

- `OPENAI_API_BASE_URL`: Your custom endpoint URL
- `OPENAI_API_KEY`: Your API key (if required)

=== "CLI"

    ``` bash
    export OPENAI_API_BASE_URL='https://your-server-endpoint.com/v1'
    export OPENAI_API_KEY='your-api-key'
    lumen-ai serve penguins.csv --provider openai
    ```

=== "Python"

    ``` py
    llm = lmai.llm.OpenAI(
        api_key='...',
        endpoint='https://your-endpoint.com/v1'
    )
    ui = lmai.ExplorerUI(data='penguins.csv', llm=llm)
    ui.servable()
    ```

!!! tip "Local Llama.cpp Server"
    For a local OpenAI-compatible server, see the [Llama.cpp server documentation](https://llama-cpp-python.readthedocs.io/en/latest/server/).

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

**High costs** - Use `gpt-5-mini` or `claude-haiku-4-5` for `default`, reserve `gpt-5.2` or `claude-sonnet-4-5` for critical tasks (`sql`, `vega_lite`, `analyst`).

**Slow responses** - Local models are slower than cloud APIs. Use cloud providers when speed matters.

## Best practices

**Use powerful models for critical tasks:**

- `sql` - SQL generation needs strong reasoning
- `vega_lite` - Visualizations need design understanding  
- `analyst` - Analysis needs statistical knowledge

**Use efficient models elsewhere:**

- `default` - Simple tasks work well with `gpt-5-mini` or `claude-haiku-4-5`
- `chat` - Conversation works with smaller models

**Set temperature by task:**

- 0.1 for SQL (deterministic)
- 0.3-0.4 for analysis and chat
- 0.5-0.7 for creative tasks

**Test before deploying**

- Different models behave differently. Test with real queries.
