# Use Azure AI with Lumen AI

Lumen AI ships with the {py:class}`lumen.ai.llm.AzureOpenAI` and {py:class}`lumen.ai.llm.AzureMistralAI` wrappers for LLM models.

## Prerequisites

- Lumen AI installed in your Python environment.
- An Azure OpenAI Service resource. You can create one through the [Azure Portal](https://portal.azure.com/).
- Azure Inference API Key and Endpoint URL. These can be obtained from the Azure Portal under your Azure OpenAI resource.

## Using Environment Variables

1. Set the Environment Variable

Set the `AZUREAI_ENDPOINT_KEY` and `AZUREAI_ENDPOINT_URL` environment variables in your system. This allows Lumen AI to automatically detect and use OpenAI as the provider.

::::{tab-set}

:::{tab-item} Unix/Linux/macOS
```bash
export AZUREAI_ENDPOINT_KEY='your-azure-api-key'
export AZUREAI_ENDPOINT_URL='your-azure-endpoint'
```
:::

:::{tab-item} Windows
```bash
set AZUREAI_ENDPOINT_KEY=your-azure-api-key
set AZUREAI_ENDPOINT_URL=your-azure-endpoint
```
:::

2. Now run `lumen-ai` serve and select whether you want to use OpenAI or Mistral based models:

```bash
lumen-ai serve <your-data-file-or-url> --provider <azure-openai | azure-mistral>
```

## Using CLI Arguments

Alternatively you can also provide the API key and endpoint as CLI arguments:

```bash
lumen-ai serve <your-data-file-or-url> --provider <azure-openai | azure-mistral> --api-key <your-azure-api-key> --provider-endpoint <your-azure-endpoint>
```

## Using Python

In Python, simply import the LLM wrapper {py:class}`lumen.ai.llm.AzureOpenAI` or {py:class}`lmai.llm.AzureMistralAI` and pass it to the {py:class}`lumen.ai.ui.ExplorerUI`:

```python
import lumen.ai as lmai

azure_llm = lmai.llm.AzureOpenAI(api_key='your-azure-api-key', endpoint='your-azure-endpoint')

ui = lmai.ui.ExplorerUI('<your-data-file-or-url>', llm=azure_llm)
ui.servable()
```
