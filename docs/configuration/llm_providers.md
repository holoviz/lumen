# LLM Providers

**Configure which AI model powers Lumen.**

Lumen works with OpenAI, Anthropic, Google, local models, and more.

## Quick start

Set your API key and launch:

```bash
export OPENAI_API_KEY="sk-..."
lumen-ai serve penguins.csv
```

Lumen auto-detects the provider from your environment variables.

## Supported providers

- OpenAI (GPT-4o, GPT-4o-mini)
- Anthropic (Claude 4-5 Sonnet, Claude 4-5 Haiku)
- Google Gemini
- Mistral AI
- Azure OpenAI
- Ollama (local models)
- Llama.cpp (local models)

## Use different models per agent

Agents automatically use different models based on their task. Configure models by agent type:

```python
import lumen.ai as lmai

model_config = {
    "default": {"model": "gpt-4o-mini"},  # Cheap model for most agents
    "sql": {"model": "gpt-4o"},           # Powerful model for SQL
    "vega_lite": {"model": "gpt-4o"},     # Powerful model for charts
    "analyst": {"model": "gpt-4o"},       # Powerful model for analysis
}

llm = lmai.llm.OpenAI(model_kwargs=model_config)
ui = lmai.ExplorerUI(data='penguins.csv', llm=llm)
ui.servable()
```

**How model types work:**

Agent names map to model types:

- SQLAgent → `"sql"`
- VegaLiteAgent → `"vega_lite"`
- ChatAgent → `"chat"`
- AnalystAgent → `"analyst"`

Use `"default"` as fallback for any agent without specific configuration.

## Configure temperature

Control randomness in responses:

```python
model_config = {
    "default": {
        "model": "gpt-4o-mini",
        "temperature": 0.3,  # Moderate
    },
    "sql": {
        "model": "gpt-4o",
        "temperature": 0.1,  # Low = deterministic SQL
    },
    "chat": {
        "model": "gpt-4o-mini",
        "temperature": 0.4,  # Higher = natural conversation
    },
}
```

Lower temperature (0.1-0.3) for precise tasks. Higher temperature (0.4-0.7) for creative tasks.

## Model recommendations

**Use powerful models for:**

- `sql` - SQL generation requires strong reasoning
- `vega_lite` - Visualizations need design understanding  
- `analyst` - Analysis needs statistical knowledge

**Use efficient models for:**

- `chat` - Simple conversations
- `table_list` - Listing tables
- `document_list` - Managing documents

**Example balanced configuration:**

```python
model_config = {
    "default": {"model": "gpt-4o-mini"},      # Fallback
    "sql": {"model": "gpt-4o"},               # Critical
    "vega_lite": {"model": "gpt-4o"},         # Critical
    "analyst": {"model": "gpt-4o"},           # Critical
    "edit": {"model": "gpt-4o"},              # Error fixing
    # Everything else uses default (gpt-4o-mini)
}
```

## Provider setup

### OpenAI

```bash
export OPENAI_API_KEY="sk-..."
lumen-ai serve
```

Or in Python:

```python
llm = lmai.llm.OpenAI(api_key='sk-...')
ui = lmai.ExplorerUI(data='penguins.csv', llm=llm)
```

### Anthropic

```bash
export ANTHROPIC_API_KEY="sk-ant-..."
lumen-ai serve
```

Or in Python:

```python
llm = lmai.llm.AnthropicAI(api_key='sk-ant-...')
```

### Google Gemini

```bash
export GOOGLE_API_KEY="..."
lumen-ai serve
```

Or in Python:

```python
llm = lmai.llm.GoogleAI(api_key='...')
```

### Mistral

```bash
export MISTRAL_API_KEY="..."
lumen-ai serve
```

Or in Python:

```python
llm = lmai.llm.MistralAI(api_key='...')
```

### Azure OpenAI

```bash
export AZUREAI_ENDPOINT_KEY="..."
lumen-ai serve --provider azure-openai
```

Or in Python:

```python
llm = lmai.llm.AzureOpenAI(api_key='...')
```

### Ollama (local)

Install Ollama and pull a model:

```bash
ollama pull llama2
```

Then:

```python
llm = lmai.llm.Ollama(model='llama2')
ui = lmai.ExplorerUI(data='penguins.csv', llm=llm)
```

Ollama runs on `localhost:11434` by default.

### Llama.cpp (local)

```bash
lumen-ai serve --provider llama-cpp --llm-model-url 'https://huggingface.co/REPO/MODEL.gguf'
```

Or in Python:

```python
llm = lmai.llm.LlamaCpp(repo='model-repo', model_file='model.gguf')
ui = lmai.ExplorerUI(data='penguins.csv', llm=llm)
```

### OpenAI-compatible endpoints

Any API with OpenAI's format:

```python
llm = lmai.llm.OpenAI(
    api_key='...',
    endpoint='https://your-endpoint.com/v1'
)
```

## All model types

| Model type | Agent | Use |
|------------|-------|-----|
| `default` | Fallback | Any agent without specific config |
| `sql` | SQLAgent | SQL queries |
| `vega_lite` | VegaLiteAgent | Charts |
| `chat` | ChatAgent | Conversation |
| `analyst` | AnalystAgent | Data analysis |
| `analysis` | AnalysisAgent | Custom analyses |
| `table_list` | TableListAgent | List tables |
| `document_list` | DocumentListAgent | List documents |
| `source` | SourceAgent | Data uploads |
| `validation` | ValidationAgent | Validate results |
| `edit` | (any agent) | Fix errors |

## Troubleshooting

**"API key not found":**

Set your API key as an environment variable or pass it directly in Python.

**Wrong model used:**

Check model type names match agent names. SQLAgent uses `"sql"`, not `"SQLAgent"`.

**High costs:**

Use cheaper models for `default` and non-critical agents. Only use powerful models for `sql`, `vega_lite`, and `analyst`.

**Slow responses:**

Local models (Ollama, Llama.cpp) are slower than cloud APIs. Use cloud providers for production.

## Best practices

- Use powerful models (`gpt-4o`) for SQL, visualization, and analysis
- Use efficient models (`gpt-4o-mini`) for chat and simple tasks
- Set lower temperature (0.1) for SQL and data tasks
- Set moderate temperature (0.3-0.4) for conversational tasks
- Test with your production model before deploying
