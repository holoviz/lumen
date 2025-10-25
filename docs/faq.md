# Frequently Asked Questions

Common questions about Lumen AI.

---

## General

### What is Lumen AI?

Lumen AI is an open-source agent framework for exploring data conversationally using large language models. Ask questions in plain English, and AI agents generate SQL queries, visualizations, and analyses automatically.

### Why use Lumen AI instead of ChatGPT?

**ChatGPT:**

- Can't see or query your data
- Requires manual SQL writing
- No persistent data connection
- Can't generate real visualizations

**Lumen AI:**

- Connects directly to your data
- Auto-generates SQL from natural language
- Maintains query history and context
- Creates interactive visualizations
- Customizable with your own agents
- Keeps your data private (no external API access needed if using local LLM)

### Is Lumen AI free?

Yes! Lumen AI is fully open source under the Apache 2.0 license. Costs are optional:

- **LLM API calls** (OpenAI, Anthropic, etc.) — Pay per token
- **Local LLM** — Free, runs offline
- **Hosting** — Only if you deploy to the cloud

---

## Installation & Setup

### What are the system requirements?

- Python 3.10 or later
- Linux, macOS, or Windows
- 4GB RAM minimum (more for large datasets)

### Can I use Lumen AI offline?

Yes! Use a local LLM:

```bash
pip install 'lumen[ai-llama]'
```

See [LLM Providers](configuration/llm_providers.md) for local LLM setup instructions.

### Which Python versions are supported?

Python 3.10, 3.11, and 3.12. We recommend 3.11+.

---

## LLM Setup

### Which LLM providers does Lumen AI support?

- **OpenAI** (GPT-5, GPT-4o)
- **Anthropic** (Claude 4-5 Sonnet, Claude 4-5 Haiku)
- **Google Gemini**
- **Mistral AI**
- **Azure OpenAI**
- **Local models** (Llama, Mistral via llama.cpp, Ollama)

See [LLM Providers](configuration/llm_providers.md) for setup details.

### How much do LLM API calls cost?

Costs vary by provider:

| Provider | Cost (Approx) |
|----------|--------------|
| OpenAI GPT-4o | $0.015 per 1K input tokens |
| OpenAI GPT-4o-mini | $0.00015 per 1K input tokens |
| Anthropic Claude 4-5 Sonnet | $0.003 per 1K input tokens |
| Mistral | $0.0002 per 1K input tokens |
| Local LLM | Free (after download) |

Typical data exploration query: 1-5K tokens = $0.01–$0.15

### Can I use Lumen AI with my own LLM?

Yes! Use any OpenAI-compatible endpoint:

```python
import lumen.ai as lmai

llm = lmai.llm.OpenAI(
    api_key='your-key',
    endpoint='https://your-endpoint.com/v1'
)

ui = lmai.ExplorerUI(data='data.csv', llm=llm)
ui.servable()
```

---

## Chat & Data Analysis

### How do I ask better questions?

Structure your questions clearly:

- **Good**: "Show me the top 5 products by revenue in Q4 2024"
- **Bad**: "What about the data?"

Tips:

1. Specify what you want (top/bottom, count, average, etc.)
2. Include filters (date ranges, regions, categories)
3. Ask for visualizations if you want charts
4. Combine multiple steps in one query

### Why are my SQL queries wrong?

The LLM doesn't understand your data schema. Provide more context:

```python
ui = lmai.ExplorerUI(
    data='sales.csv',
)
```

Then ask specific questions that hint at the data structure.

### Can I see the generated SQL?

Yes! The SQL panel shows every query generated. You can:

- Review it for correctness
- Edit it directly for fine-tuning
- Copy it for reuse

### How do I use different models for different tasks?

Lumen AI automatically routes tasks to different models:

```python
import lumen.ai as lmai

config = {
    "default": {"model": "gpt-4o-mini"},      # General tasks
    "sql": {"model": "gpt-4o"},               # SQL generation
    "vega_lite": {"model": "gpt-4o"},         # Visualization
    "reasoning": {"model": "gpt-5"}           # Complex analysis
}

llm = lmai.llm.OpenAI(model_kwargs=config)
ui = lmai.ExplorerUI(data='data.csv', llm=llm)
ui.servable()
```

See [LLM Providers](configuration/llm_providers.md) for full configuration.

---

## Agents & Customization

### What are agents?

Agents are specialized AI workers. Built-in agents handle:

- **TableListAgent** — Describes available data
- **ChatAgent** — Answers general questions
- **SQLAgent** — Generates SQL queries
- **VegaLiteAgent** — Creates visualizations
- **AnalystAgent** — Analyzes results
- **ValidationAgent** — Checks for errors

See [Agents](configuration/agents.md) for details.

### Can I build custom agents?

Yes! Create agents for domain-specific tasks:

```python
import lumen.ai as lmai

class CustomAgent(lmai.agents.Agent):
    purpose = "Analyzes customer sentiment"
    requires = ["customer_feedback"]
    provides = ["sentiment_summary"]
    
    async def respond(self, messages, **kwargs):
        feedback = self._memory["customer_feedback"]
        # Your logic here
        return result

ui = lmai.ExplorerUI(data='data.csv', agents=[CustomAgent()])
ui.servable()
```

See [Agents](configuration/agents.md) for full guide.

### Can I customize how agents behave?

Yes! Override agent prompts:

```python
template_overrides = {
    "main": {
        "instructions": "Speak concisely and focus on key metrics."
    }
}

agent = lmai.agents.ChatAgent(template_overrides=template_overrides)
ui = lmai.ExplorerUI(data='data.csv', agents=[agent])
ui.servable()
```

See [Prompts](configuration/prompts.md) for full customization guide.

---

## Data & Integration

### What data sources does Lumen AI support?

- **Files**: CSV, Parquet, JSON, Excel
- **Databases**: PostgreSQL, MySQL, SQLite, DuckDB, Snowflake
- **APIs**: REST endpoints
- **Cloud storage**: S3, GCS
- **Custom**: Write your own data source

### Can I use Lumen AI with multiple data sources?

Yes! Load multiple files or tables:

```python
ui = lmai.ExplorerUI(data=[
    'sales.csv',
    'customers.csv',
    'products.csv'
])
ui.servable()
```

Agents can query across all sources.

### Can I upload data through the UI?

Yes! Users can upload CSV, Parquet, JSON files directly in the chat interface.

---

## Deployment & Sharing

### How do I share my Lumen AI app?

Deploy as a web app:

```bash
lumen-ai serve data.csv --show
```

This opens a browser window. Share the URL with others.

For persistent deployment, use:

- Panel server
- Docker
- AWS/GCP/Azure

### Can I add authentication?

Yes! Lumen AI supports basic auth and OAuth through Panel.

---

## Performance

### Why is my query slow?

Common causes and fixes:

1. **Large CSV files** — Use a database instead
2. **Bad SQL** — Check the generated query
3. **Slow LLM** — Use a faster model (GPT-4o-mini vs GPT-4o)
4. **Network latency** — Use local LLM or nearby API endpoint

### How can I cache results?

Lumen AI caches SQL queries automatically for performance.

---

## Getting Help

### Where do I ask questions?

- 💬 [Discourse](https://discourse.holoviz.org/c/lumen/14)
- 🐙 [GitHub Discussions](https://github.com/holoviz/lumen/discussions)
- 📱 [Discord](https://discord.com/invite/rb6gPXbdAr)

### How do I report bugs?

Report on [GitHub Issues](https://github.com/holoviz/lumen/issues):

1. Check if it's already reported
2. Include Python version, Lumen version, traceback
3. Provide minimal reproducible example

### Can I contribute?

Yes! We welcome contributions:

- 🐛 Bug reports
- 📚 Documentation improvements
- ✨ New features
- 🔧 Custom agents and tools

See [Contributing](contributing.md).
