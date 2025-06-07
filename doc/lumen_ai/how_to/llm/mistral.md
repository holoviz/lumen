# Use Mistral with Lumen AI

Lumen AI ships with the {py:class}`lumen.ai.llm.MistralAI` wrapper for Mistral LLM models. Lumen picks between a default and a reasoning model to solve different tasks, by default it will invoke the following models:

- **`default`**: mistral-small-latest
- **`reasoning`**: mistral-large-latest

For information on configuring the models being used find the [configuring LLMs how-to guide](configure.md).

## Prerequisites

- Lumen AI installed in your Python environment.
- An Mistral API Key. You can obtain this from [the Mistral Dashboard](https://console.mistral.ai/api-keys/).

## Using Environment Variables

1. Set the Environment Variable

Set the MISTRAL_API_KEY environment variable in your system. This allows Lumen AI to automatically detect and use Mistral as the provider.

::::{tab-set}

:::{tab-item} Unix/Linux/macOS
```bash
export MISTRAL_API_KEY='your-mistral-api-key'
```
:::

:::{tab-item} Windows
```bash
set MISTRAL_API_KEY=your-mistral-api-key
```
:::

::::

2. Simply run your Lumen AI application, and it will automatically use Mistral as the LLM provider. If you have environment variables for multiple providers override it.

```bash
lumen-ai serve <your-data-file-or-url> [--provider mistral]
```

## Using CLI Arguments

Alternatively you can also provide the API key as a CLI argument:

```bash
lumen-ai serve <your-data-file-or-url> --provider mistral --api-key <your-mistral-api-key>
```

## Using Python

In Python, simply import the LLM wrapper {py:class}`lumen.ai.llm.MistralAI` and pass it to the {py:class}`lumen.ai.ui.ExplorerUI`:

```python
import lumen.ai as lmai

mistral_llm = lmai.llm.MistralAI(api_key='your-mistral-api-key')

ui = lmai.ui.ExplorerUI('<your-data-file-or-url>', llm=mistral_llm)
ui.servable()
```

## Configuring the Model

If you want to override the `default` and/or `reasoning` models, which use `mistral-small-latest` and `mistral-large-latest` respectively, a model configuration can be provided via the `model_kwargs` parameter. Lumen will use different models for different scenarios, currently a `default` and a `reasoning` model can be provided. If no reasoning model is provided it will always use the `default` model.

You can find the latest list of Mistral models [in their documentation](https://docs.mistral.ai/getting-started/models/models_overview/). To swap `mistral-small-latest` for `ministral-8b-latest` as the default model simply override the default `model_kwargs`:

```python
import lumen.ai as lmai

config = {
    "default": {"model": "ministral-8b-latest"},
    "reasoning": {"model": "mistral-large-latest"}
}

llm = lmai.llm.MistralAI(model_kwargs=config)

lmai.ui.ExplorerUI('<your-data-file-or-url>', llm=llm).servable()
```

## Additional Configuration Options

The MistralAI wrapper supports several additional configuration parameters:

```python
import lumen.ai as lmai

llm = lmai.llm.MistralAI(
    api_key='your-mistral-api-key',
    temperature=0.7,  # Controls randomness (0-1)
    mode='MISTRAL_TOOLS',  # Instructor mode: 'MISTRAL_TOOLS' or 'JSON_SCHEMA'
    create_kwargs={}  # Additional kwargs for chat.completions.create
)

ui = lmai.ui.ExplorerUI('<your-data-file-or-url>', llm=llm)
ui.servable()
```

### Environment Variables

The MistralAI wrapper will automatically detect the `MISTRAL_API_KEY` environment variable if it is set:

```python
import lumen.ai as lmai

# Will use MISTRAL_API_KEY from environment
llm = lmai.llm.MistralAI()

ui = lmai.ui.ExplorerUI('<your-data-file-or-url>', llm=llm)
ui.servable()
```

### Streaming Support

**Note:** The MistralAI wrapper has limited streaming support. While it supports streaming for text generation, it does not support streaming for structured output (Pydantic models) due to instructor limitations with Mistral's streaming API.
