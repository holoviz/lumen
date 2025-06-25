# Use LiteLLM with Lumen AI

Lumen AI ships with the {py:class}`lumen.ai.llm.LiteLLM` wrapper that provides unified access to 100+ LLM providers through a single interface. LiteLLM allows you to easily switch between different providers like OpenAI, Anthropic, Google, Cohere, Azure, Vertex AI, and many more using the same input/output format.

By default, Lumen uses the following models with LiteLLM:

- **`default`**: gpt-4o-mini (Cost-effective for general tasks)
- **`reasoning`**: claude-3-5-sonnet-latest (Advanced reasoning capabilities)
- **`sql`**: gpt-4o-mini (SQL query generation)

## Prerequisites

- Lumen AI installed in your Python environment
- API keys for the LLM providers you want to use
- `litellm` package installed: `pip install litellm`

## Using Environment Variables

LiteLLM automatically detects environment variables for different providers. Set the appropriate environment variable for your chosen provider:

::::{tab-set}

:::{tab-item} OpenAI
```bash
export OPENAI_API_KEY='your-openai-api-key'
```
:::

:::{tab-item} Anthropic
```bash
export ANTHROPIC_API_KEY='your-anthropic-api-key'
```
:::

:::{tab-item} Google
```bash
export GEMINI_API_KEY='your-gemini-api-key'
```
:::

:::{tab-item} Azure
```bash
export AZURE_API_KEY='your-azure-api-key'
export AZURE_API_BASE='your-azure-endpoint'
```
:::

::::

Once environment variables are set, simply run your Lumen AI application:

```bash
lumen-ai serve <your-data-file-or-url> --provider litellm
```

## Using CLI Arguments

You can also provide API keys directly via CLI arguments:

```bash
lumen-ai serve <your-data-file-or-url> --provider litellm --api-key <your-api-key>
```

## Using Python

In Python, import the LiteLLM wrapper and pass it to the {py:class}`lumen.ai.ui.ExplorerUI`:

```python
import lumen.ai as lmai

# Basic usage - will use environment variables
litellm_llm = lmai.llm.LiteLLM()

ui = lmai.ui.ExplorerUI('<your-data-file-or-url>', llm=litellm_llm)
ui.servable()
```

## Configuring Models

Override the default model configuration by providing custom `model_kwargs`. LiteLLM supports various model string formats:

```python
import lumen.ai as lmai

# Mix different providers for different tasks
config = {
    "default": {"model": "gpt-4o-mini"},                    # OpenAI
    "reasoning": {"model": "claude-3-5-sonnet-latest"},     # Anthropic
    "sql": {"model": "gemini/gemini-1.5-flash"}            # Google
}

llm = lmai.llm.LiteLLM(model_kwargs=config)

ui = lmai.ui.ExplorerUI('<your-data-file-or-url>', llm=llm)
ui.servable()
```

## Model String Formats

LiteLLM supports various model string formats depending on the provider:

- **OpenAI**: `"gpt-4"`, `"gpt-4o-mini"`, `"gpt-3.5-turbo"`
- **Anthropic**: `"claude-3-5-sonnet-latest"`, `"claude-3-haiku"`
- **Google**: `"gemini/gemini-pro"`, `"gemini/gemini-1.5-flash"`
- **Azure**: `"azure/your-deployment-name"`
- **Cohere**: `"command-r-plus"`, `"command-r"`
- **Hugging Face**: `"huggingface/microsoft/DialoGPT-medium"`

## Advanced Configuration Options

The LiteLLM wrapper supports extensive configuration options:

```python
import lumen.ai as lmai

llm = lmai.llm.LiteLLM(
    temperature=0.7,           # Controls randomness (0-2)
    mode='TOOLS',              # Instructor mode for structured outputs
    enable_caching=True,       # Enable built-in caching
    fallback_models=[          # Fallback models if primary fails
        "gpt-4o-mini",
        "claude-3-haiku",
        "gemini/gemini-1.5-flash"
    ],
    litellm_params={           # Additional LiteLLM parameters
        "timeout": 30,
        "max_retries": 3,
        "api_base": "custom-endpoint"
    },
    router_settings={          # Load balancing settings
        "routing_strategy": "least-busy",
        "num_retries": 3
    }
)

ui = lmai.ui.ExplorerUI('<your-data-file-or-url>', llm=llm)
ui.servable()
```

## Provider-Specific Configuration

For providers requiring special configuration, use the `litellm_params` parameter:

```python
import lumen.ai as lmai

# Azure OpenAI configuration
azure_config = {
    "default": {"model": "azure/gpt-4o-mini"},
    "reasoning": {"model": "azure/gpt-4"}
}

llm = lmai.llm.LiteLLM(
    model_kwargs=azure_config,
    litellm_params={
        "api_base": "https://your-resource.openai.azure.com/",
        "api_version": "2024-02-15-preview"
    }
)

# Vertex AI configuration
vertex_config = {
    "default": {"model": "vertex_ai/gemini-pro"},
    "reasoning": {"model": "vertex_ai/gemini-pro-vision"}
}

llm = lmai.llm.LiteLLM(
    model_kwargs=vertex_config,
    litellm_params={
        "vertex_project": "your-project-id",
        "vertex_location": "us-central1"
    }
)
```

## Caching and Performance

Enable caching to improve performance for repeated queries:

```python
import lumen.ai as lmai

llm = lmai.llm.LiteLLM(
    enable_caching=True,
    litellm_params={
        "timeout": 60,
        "max_retries": 2
    }
)
```

## Error Handling and Fallbacks

Configure fallback models for improved reliability:

```python
import lumen.ai as lmai

llm = lmai.llm.LiteLLM(
    model_kwargs={
        "default": {"model": "gpt-4o-mini"},
        "reasoning": {"model": "claude-3-5-sonnet-latest"}
    },
    fallback_models=[
        "gpt-4o-mini",           # Fallback 1
        "claude-3-haiku",        # Fallback 2
        "gemini/gemini-1.5-flash" # Fallback 3
    ]
)
```

## Supported Providers

LiteLLM supports 100+ LLM providers including:

- **OpenAI**: GPT-4, GPT-3.5, etc.
- **Anthropic**: Claude 3.5 Sonnet, Claude 3 Haiku, etc.
- **Google**: Gemini Pro, Gemini Flash, etc.
- **Azure**: Azure OpenAI, Azure Anthropic, etc.
- **AWS**: Bedrock models (Claude, Titan, etc.)
- **Cohere**: Command R+, Command R, etc.
- **Hugging Face**: Open source models
- **Vertex AI**: Google Cloud models
- **And many more...**

For the complete list of supported providers and their model strings, see the [LiteLLM documentation](https://docs.litellm.ai/docs/providers).

## Environment Variables Reference

Common environment variables for popular providers:

```bash
# OpenAI
export OPENAI_API_KEY='your-key'

# Anthropic
export ANTHROPIC_API_KEY='your-key'

# Google AI
export GEMINI_API_KEY='your-key'

# Azure OpenAI
export AZURE_API_KEY='your-key'
export AZURE_API_BASE='your-endpoint'
export AZURE_API_VERSION='2024-02-15-preview'

# AWS Bedrock
export AWS_ACCESS_KEY_ID='your-key'
export AWS_SECRET_ACCESS_KEY='your-secret'
export AWS_REGION_NAME='us-east-1'

# Cohere
export COHERE_API_KEY='your-key'

# Hugging Face
export HUGGINGFACE_API_KEY='your-key'
```

Refer to the [LiteLLM provider documentation](https://docs.litellm.ai/docs/providers) for specific setup instructions for each provider.
