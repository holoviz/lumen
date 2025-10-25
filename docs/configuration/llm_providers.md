# LLM Providers

Lumen AI supports a wide range of LLM providers and models. Configure your preferred provider either in the cloud or locally.

## Supported providers

- **OpenAI** — GPT-5, GPT-4o, GPT-4o-mini
- **Anthropic** — Claude 4-5 Sonnet, Claude 4-5 Haiku
- **Google Gemini** — Gemini Pro, Gemini Ultra, Gemini Flash
- **Mistral** — Mistral Large, Mistral 8x7B, Mistral 7B
- **Azure OpenAI** — OpenAI models hosted on Azure
- **Ollama** — Run open-source models locally
- **Llama.cpp** — Lightweight local model inference
- **LiteLLM** — Proxy to multiple providers
- **OpenAI-compatible endpoints** — Any OpenAI-compatible API

## Quick start

Set your provider's API key as an environment variable, then launch:

```bash
export OPENAI_API_KEY="your-key-here"
lumen-ai serve
```

The provider is auto-detected. Alternatively, specify it explicitly:

```bash
lumen-ai serve --provider openai
```

## Configure models for agents

Lumen AI uses different models for different tasks to optimize performance and cost. Specify different models for different agent types using `model_kwargs`.

### Model types

Agents automatically determine which model to use based on their task:

- `default` — General-purpose tasks
- `sql` — SQL query generation and database operations
- `vega_lite` — Visualization generation
- `reasoning` — Complex reasoning tasks
- `chat` — Chat interactions
- `analysis` — Data analysis

### Basic configuration

Configure different models for specific tasks:

```bash
lumen-ai serve --model-kwargs '{"default": {"model": "gpt-4o-mini"}, "sql": {"model": "gpt-4o"}, "vega_lite": {"model": "gpt-4o"}}'
```

In Python:

```python
import lumen.ai as lmai

config = {
    "default": {"model": "gpt-4o-mini"},
    "sql": {"model": "gpt-4o"},
    "vega_lite": {"model": "gpt-4o"},
    "reasoning": {"model": "gpt-5"}
}

llm = lmai.llm.OpenAI(model_kwargs=config)
ui = lmai.ExplorerUI(data='penguins.csv', llm=llm)
ui.servable()
```

### Advanced configuration

Configure temperature and other parameters per model type:

```python
config = {
    "default": {
        "model": "gpt-4o-mini",
        "temperature": 0.3,
    },
    "sql": {
        "model": "gpt-4o",
        "temperature": 0.1,  # Lower for deterministic SQL
    },
    "vega_lite": {
        "model": "gpt-4o",
        "temperature": 0.2,  # Moderate for creative visualizations
    },
    "reasoning": {
        "model": "gpt-5",
        "temperature": 0.4,  # Higher for complex reasoning
    },
}

llm = lmai.llm.OpenAI(model_kwargs=config)
```

## Provider-specific configuration

### OpenAI

**Environment variable:**

```bash
export OPENAI_API_KEY="your-openai-api-key"
```

**Python:**

```python
import lumen.ai as lmai

llm = lmai.llm.OpenAI(api_key='your-openai-api-key')
ui = lmai.ExplorerUI(data='penguins.csv', llm=llm)
ui.servable()
```

**Default models:**

- `default`: gpt-4o-mini
- `sql`: gpt-4o
- `vega_lite`: gpt-4o
- `reasoning`: gpt-5

### Anthropic

**Environment variable:**

```bash
export ANTHROPIC_API_KEY="your-anthropic-api-key"
```

**Python:**

```python
import lumen.ai as lmai

llm = lmai.llm.AnthropicAI(api_key='your-anthropic-api-key')
ui = lmai.ExplorerUI(data='penguins.csv', llm=llm)
ui.servable()
```

### Google Gemini

**Environment variable:**

```bash
export GOOGLE_API_KEY="your-google-api-key"
```

**Python:**

```python
import lumen.ai as lmai

llm = lmai.llm.GoogleAI(api_key='your-google-api-key')
ui = lmai.ExplorerUI(data='penguins.csv', llm=llm)
ui.servable()
```

### Mistral

**Environment variable:**

```bash
export MISTRAL_API_KEY="your-mistral-api-key"
```

**Python:**

```python
import lumen.ai as lmai

llm = lmai.llm.MistralAI(api_key='your-mistral-api-key')
ui = lmai.ExplorerUI(data='penguins.csv', llm=llm)
ui.servable()
```

### Azure OpenAI

**Environment variable:**

```bash
export AZUREAI_ENDPOINT_KEY="your-azure-endpoint-key"
```

**Python:**

```python
import lumen.ai as lmai

llm = lmai.llm.AzureOpenAI(api_key='your-azure-api-key')
ui = lmai.ExplorerUI(data='penguins.csv', llm=llm)
ui.servable()
```

### Ollama (Local)

Run open-source models locally with Ollama.

**Install and run Ollama:**

```bash
# Install from https://ollama.ai
# Pull a model
ollama pull llama2

# Ollama runs on localhost:11434 by default
```

**Python:**

```python
import lumen.ai as lmai

llm = lmai.llm.Ollama(model='llama2')
ui = lmai.ExplorerUI(data='penguins.csv', llm=llm)
ui.servable()
```

### Llama.cpp (Local)

**Load from HuggingFace:**

```bash
lumen-ai serve --provider llama-cpp --llm-model-url 'https://huggingface.co/RE/PO/blob/main/FILE.gguf'
```

**Python:**

```python
import lumen.ai as lmai

llm = lmai.llm.LlamaCpp(
    repo='model-repo/model-name',
    model_file='model.gguf'
)
ui = lmai.ExplorerUI(data='penguins.csv', llm=llm)
ui.servable()
```

### OpenAI-compatible endpoints

Use any API compatible with OpenAI's interface:

```bash
lumen-ai serve --provider openai --provider-endpoint https://your-endpoint.com/v1
```

**Python:**

```python
import lumen.ai as lmai

llm = lmai.llm.OpenAI(
    api_key='your-api-key',
    endpoint='https://your-endpoint.com/v1'
)
ui = lmai.ExplorerUI(data='penguins.csv', llm=llm)
ui.servable()
```

## Agent-specific model recommendations

Choose models based on the agent's task:

**SQL Agent** — Generates SQL queries

- Recommended: Capable models (gpt-4o, claude-4-5-sonnet)
- Why: SQL requires strong logical reasoning and syntax knowledge
- Temperature: 0.1 (low, deterministic)

**VegaLiteAgent** — Creates visualizations

- Recommended: Creative, capable models (gpt-4o, claude-4-5-sonnet)
- Why: Requires design principles and JSON structure understanding
- Temperature: 0.2 (moderate)

**ChatAgent** — General conversation

- Recommended: Cost-effective models (gpt-4o-mini, claude-4-5-haiku)
- Why: Simple interactions don't need advanced reasoning
- Temperature: 0.3 (moderate, natural conversation)

**AnalysisAgent** — Data analysis

- Recommended: Advanced models (gpt-4o, claude-4-5-sonnet)
- Why: Requires statistical and domain knowledge
- Temperature: 0.2 (moderate)

**DbtslAgent** — Business metric queries

- Recommended: SQL-capable models (gpt-4o, claude-4-5-sonnet)
- Why: Requires semantic modeling understanding
- Temperature: 0.1 (low, deterministic)
