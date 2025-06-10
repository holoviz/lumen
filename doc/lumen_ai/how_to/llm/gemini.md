# Use Google Gemini with Lumen AI

Lumen AI ships with the {py:class}`lumen.ai.llm.GoogleAI` wrapper for Google Gemini LLM models. Lumen picks between different models to solve different tasks, by default it will use the following models:

- **`default`**: gemini-2.0-flash (Cost-optimized, low latency)
- **`reasoning`**: gemini-2.5-flash-preview-05-20 (Thinking model, balanced price/performance)

## Prerequisites

- Lumen AI installed in your Python environment.
- A Google AI API Key. You can obtain this from [Google AI Studio](https://aistudio.google.com/app/apikey).

## Using Environment Variables

1. Set the Environment Variable

Set the GEMINI_API_KEY environment variable in your system. This allows Lumen AI to automatically detect and use Google Gemini as the provider.

::::{tab-set}

:::{tab-item} Unix/Linux/macOS
```bash
export GEMINI_API_KEY='your-gemini-api-key'
```
:::

:::{tab-item} Windows
```bash
set GEMINI_API_KEY=your-gemini-api-key
```
:::

::::

2. Simply run your Lumen AI application, and it will automatically use Google Gemini as the LLM provider. If you have environment variables for multiple providers override it.

```bash
lumen-ai serve <your-data-file-or-url> [--provider gemini]
```

## Using CLI Arguments

Alternatively you can also provide the API key as a CLI argument:

```bash
lumen-ai serve <your-data-file-or-url> --provider gemini --api-key <your-gemini-api-key>
```

## Using Python

In Python, simply import the LLM wrapper {py:class}`lumen.ai.llm.GoogleAI` and pass it to the {py:class}`lumen.ai.ui.ExplorerUI`:

```python
import lumen.ai as lmai

gemini_llm = lmai.llm.GoogleAI(api_key='your-gemini-api-key')

ui = lmai.ui.ExplorerUI('<your-data-file-or-url>', llm=gemini_llm)
ui.servable()
```

## Configuring the Model

If you want to override the default models, a model configuration can be provided via the `model_kwargs` parameter. Lumen will use different models for different scenarios:

- **`default`**: Used for general tasks
- **`reasoning`**: Used for complex reasoning tasks

You can find the latest list of Google AI models [in their documentation](https://ai.google.dev/gemini-api/docs/models/gemini). To swap different models for specific tasks, override the default `model_kwargs`:

```python
import lumen.ai as lmai

config = {
    "default": {"model": "gemini-2.0-flash"},
    "reasoning": {"model": "gemini-2.0-flash-thinking-exp"}
}

llm = lmai.llm.GoogleAI(model_kwargs=config)

lmai.ui.ExplorerUI('<your-data-file-or-url>', llm=llm).servable()
```

## Additional Configuration Options

The GoogleAI wrapper supports several additional configuration parameters:

```python
import lumen.ai as lmai

llm = lmai.llm.GoogleAI(
    api_key='your-gemini-api-key',
    temperature=1.0,  # Controls randomness (0-1)
    mode='GENAI_TOOLS',  # Instructor mode: 'GENAI_TOOLS' or 'GENAI_STRUCTURED_OUTPUTS'
    create_kwargs={}  # Additional kwargs for generate_content
)

ui = lmai.ui.ExplorerUI('<your-data-file-or-url>', llm=llm)
ui.servable()
```

### Environment Variables

The GoogleAI wrapper will automatically detect the `GEMINI_API_KEY` environment variable if it is set:

```python
import lumen.ai as lmai

# Will use GEMINI_API_KEY from environment
llm = lmai.llm.GoogleAI()

ui = lmai.ui.ExplorerUI('<your-data-file-or-url>', llm=llm)
ui.servable()
```

### Available Models

Some popular Gemini models you can use:

- **gemini-2.0-flash**: Latest general-purpose model, fast and efficient
- **gemini-2.5-flash-preview-05-20**: Advanced reasoning model with thinking capabilities
- **gemini-2.5-pro-preview-06-05**: High-performance model for complex tasks

Refer to [Google AI's model documentation](https://ai.google.dev/gemini-api/docs/models/gemini) for the most up-to-date list and specifications.
