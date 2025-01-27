# Use OpenAI compatible endpoints

The OpenAI API standard makes it possible to use Lumen with providers other than the default cloud providers.

## Prerequisites

- Lumen AI installed in your Python environment.
- Server Endpoint URL of your OpenAI API compliant server.

If you want to deploy an OpenAI compliant server locally [see the LLama.cpp server documentation](https://llama-cpp-python.readthedocs.io/en/latest/server/).

## Using Environment Variables

1. Set the Environment Variable

Set the `OPENAI_API_BASE_URL` environment variablein your system (and if necessary the `OPENAI_API_KEY` variable as well):

::::{tab-set}

:::{tab-item} Unix/Linux/macOS
```bash
export OPENAI_API_BASE_URL='https://your-server-endpoint.com/v1'
export OPENAI_API_KEY='your-openai-api-key'
```
:::

:::{tab-item} Windows
```bash
set OPENAI_API_BASE_URL=https://your-server-endpoint.com/v1
set OPENAI_API_KEY=your-openai-api-key
```
:::

::::

2. Simply run your Lumen AI application and set openai as the LLM provider:

```bash
lumen-ai serve <your-data-file-or-url> --provider openai
```

## Using CLI Arguments

Alternatively you can also provide the endpoint as a CLI argument (note the API key may not be necessary):

```bash
lumen-ai serve --provider openai \
         --api-key <your-api-key> \
         --provider-endpoint <https://your-server-endpoint.com/v1>
```

## Using Python

In Python, simply import the LLM wrapper {py:class}`lmai.llm.OpenAI` and pass it to the {py:class}`llm.ui.ExplorerUI`:

```python
import lumen.ai as lmai

openai_llm = lmai.llm.OpenAI(api_key='your-openai-api-key', endpoint='https://your-server-endpoint.com/v1')

ui = lmai.ui.ExplorerUI(llm=openai_llm)
ui.servable()
```
