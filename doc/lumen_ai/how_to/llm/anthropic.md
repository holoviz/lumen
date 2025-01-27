# Using Anthropic with Lumen AI

Lumen AI ships with the {py:class}`lumen.ai.llm.AnthropicAI` wrapper for Anthropic LLM models. Lumen picks between a default and a reasoning model to solve different tasks, by default it will use the following models:

- **`default`**: claude-3-haiku-latest
- **`reasoning`**: claude-3-5-sonnet-latest

## Prerequisites

- Lumen AI installed in your Python environment.
- An Anthropic API Key. You can obtain this from [the Anthropic Console](https://console.anthropic.com/settings/keys).

## Using Environment Variables

By default Lumen will pick the first LLM provider it can find an API key for. By setting it you can give it access to that provider.

1. Set the Environment Variable

Set the OPENAI_API_KEY environment variable in your system. This allows Lumen AI to automatically detect and use Anthropic as the provider.

::::{tab-set}

:::{tab-item} Unix/Linux/macOS
```bash
export ANTHROPIC_API_KEY='your-anthropic-api-key'
```
:::

:::{tab-item} Windows
```bash
set ANTHROPIC_API_KEY=your-anthropic-api-key
```
:::

::::

2. Simply run your Lumen AI application, and it will automatically use Anthropic as the LLM provider. If you have environment variables for multiple providers override it.

```python
lumen-ai serve <your-data-file-or-url> [--provider anthropic]
```

## Using CLI Arguments

Alternatively you can also provide the API key as a CLI argument:

```bash
lumen-ai serve <your-data-file-or-url> --provider anthropic --api-key <your-anthropic-api-key>
```

## Python

In Python, simply import the LLM wrapper {py:class}`lumen.ai.llm.AnthropicAI` and pass it to the {py:class}`lumen.ai.ui.ExplorerUI`:

```python
import lumen.ai as lmai

anthropic_llm = lmai.llm.AnthropicAI(api_key='your-anthropic-api-key')

ui = lmai.ui.ExplorerUI('<your-data-file-or-url>', llm=anthropic_llm)
ui.servable()
```

## Configuring the Model

If you want to override the `default` and/or `reasoning` models, which use Claude 3 Haiku and Claude 3.5 Sonnet respectively, a model configuration can be provided via the `model_kwargs` parameter. Lumen will use different models for different scenarios, currently a `default` and a `reasoning` model can be provided. If no reasoning model is provided it will always use the `default` model.

You can find the latest list of Anthropic models [in their documentation](https://docs.anthropic.com/en/docs/about-claude/models). To swap Opus out for Haiku as the default model simply override the default `model_kwargs`:

```python
import lumen.ai as lmai

config = {
    "default": "claude-3-opus-latest",
    "reasoning": "claude-3-5-sonnet-latest"
}

llm = lmai.llm.AnthropicAI(model_kwargs=config)

lmai.ui.ExplorerUI('<your-data-file-or-url>', llm=llm).servable()
```
