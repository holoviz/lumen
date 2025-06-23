# Use OpenAI with Lumen AI

Lumen AI ships with the {py:class}`lumen.ai.llm.OpenAI` wrapper for OpenAI LLM models. Lumen picks between different models to solve different tasks, by default it will use the following models:

- **`default`**: gpt-4o-mini
- **`sql`**: gpt-4.1-mini
- **`vega_lite`**: gpt-4.1-mini
- **`reasoning`**: gpt-4.1-mini

## Prerequisites

- Lumen AI installed in your Python environment.
- An OpenAI API Key. You can obtain this from [the OpenAI Dashboard](https://platform.openai.com/api-keys).

## Using Environment Variables

1. Set the Environment Variable

Set the OPENAI_API_KEY environment variable in your system. This allows Lumen AI to automatically detect and use OpenAI as the provider.

::::{tab-set}

:::{tab-item} Unix/Linux/macOS
```bash
export OPENAI_API_KEY='your-openai-api-key'
```
:::

:::{tab-item} Windows
```bash
set OPENAI_API_KEY=your-openai-api-key
```
:::

::::

2. Simply run your Lumen AI application, and it will automatically use OpenAI as the LLM provider. If you have environment variables for multiple providers override it.

```bash
lumen-ai serve <your-data-file-or-url> [--provider openai]
```

## Using CLI Arguments

Alternatively you can also provide the API key as a CLI argument:

```bash
lumen-ai serve <your-data-file-or-url> --provider openai --api-key <your-openai-api-key>
```

## Using Python

In Python, simply import the LLM wrapper {py:class}`lumen.ai.llm.OpenAI` and pass it to the {py:class}`lumen.ai.ui.ExplorerUI`:

```python
import lumen.ai as lmai

openai_llm = lmai.llm.OpenAI(api_key='your-openai-api-key')

ui = lmai.ui.ExplorerUI('<your-data-file-or-url>', llm=openai_llm)
ui.servable()
```

## Configuring the Model

If you want to override the default models, a model configuration can be provided via the `model_kwargs` parameter. Lumen will use different models for different scenarios:

- **`default`**: Used for general tasks
- **`sql`**: Used for SQL generation tasks
- **`vega_lite`**: Used for visualization generation
- **`reasoning`**: Used for complex reasoning tasks

You can find the latest list of OpenAI models [in their documentation](https://platform.openai.com/docs/models). To swap different models for specific tasks, override the default `model_kwargs`:

```python
import lumen.ai as lmai

config = {
    "default": {"model": "gpt-4o-mini"},
    "sql": {"model": "gpt-4o"},
    "vega_lite": {"model": "gpt-4o"},
    "reasoning": {"model": "o1-mini"}
}

llm = lmai.llm.OpenAI(model_kwargs=config)

lmai.ui.ExplorerUI('<your-data-file-or-url>', llm=llm).servable()
```

## Additional Configuration Options

The OpenAI wrapper supports several additional configuration parameters:

```python
import lumen.ai as lmai

llm = lmai.llm.OpenAI(
    api_key='your-openai-api-key',
    endpoint='https://api.openai.com/v1',  # Custom endpoint if needed
    organization='your-org-id',  # OpenAI organization ID
    temperature=0.25,  # Controls randomness (0-1)
    mode='TOOLS',  # Instructor mode: 'TOOLS', 'JSON_SCHEMA', etc.
    use_logfire=False,  # Enable logfire logging
    create_kwargs={}  # Additional kwargs for chat.completions.create
)

ui = lmai.ui.ExplorerUI('<your-data-file-or-url>', llm=llm)
ui.servable()
```

### Environment Variables

The OpenAI wrapper will automatically detect these environment variables if they are set:

- `OPENAI_API_KEY`: Your OpenAI API key
- `OPENAI_ORGANIZATION`: Your OpenAI organization ID (optional)

If these are set, you can create the LLM without explicitly passing parameters:

```python
import lumen.ai as lmai

# Will use OPENAI_API_KEY from environment
llm = lmai.llm.OpenAI()

ui = lmai.ui.ExplorerUI('<your-data-file-or-url>', llm=llm)
ui.servable()
```
