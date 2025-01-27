# Use OpenAI with Lumen AI

Lumen AI ships with the {py:class}`lumen.ai.llm.OpenAI` wrapper for OpenAI LLM models. Lumen usually picks between a default and a reasoning model to solve different tasks, by default it will use the following models:

- **`default`**: gpt-4o-mini
- **`reasoning`**: gpt-4o

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

openai_llm = lmai.llm.OpenAI('<your-data-file-or-url>', api_key='your-openai-api-key')

ui = lmai.ui.ExplorerUI(llm=openai_llm)
ui.servable()
```

## Configuring the Model

If you want to override the `default` and/or `reasoning` models, which use GPT-4o-mini and GPT-4o respectively, a model configuration can be provided via the `model_kwargs` parameter. Lumen will use different models for different scenarios, currently a `default` and a `reasoning` model can be provided. If no reasoning model is provided it will always use the `default` model.

You can find the latest list of OpenAI models [in their documentation](https://platform.openai.com/docs/models). To swap GPT-4o for o1-mini as the reasoning model simply override the default `model_kwargs`:

```python
import lumen.ai as lmai

config = {
    "default": "gpt-4o-mini",
    "reasoning": "o1-mini"
}

llm = lmai.llm.OpenAI(model_kwargs=config)

lmai.ui.ExplorerUI('<your-data-file-or-url>', llm=llm).servable()
```
