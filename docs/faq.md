# FAQ

Common questions about Lumen AI.

## General

**What is Lumen AI?**

Open-source framework for exploring data with AI. Ask questions in plain English, get SQL queries, visualizations, and analysis automatically.

**Why use Lumen instead of ChatGPT?**

ChatGPT can't see your data or run queries. Lumen connects directly to your data, generates SQL, creates visualizations, and keeps history. Works with any LLM provider or local models.

**Is Lumen free?**

Yes, fully open source (Apache 2.0). Costs:

- LLM API calls (OpenAI, Anthropic) - pay per token
- Local LLMs - free after download
- Hosting - only if deploying to cloud

## Installation

**System requirements?**

Python 3.10+, 4GB RAM minimum. Works on Linux, macOS, Windows.

**Can I use Lumen offline?**

Yes. Install with local LLM:

```bash
pip install 'lumen[ai-llama]'
```

See [LLM Providers](configuration/llm_providers.md) for setup.

**Which Python versions?**

Python 3.10, 3.11, 3.12. We recommend 3.11+.

## LLM Setup

**Which LLM providers?**

- OpenAI (GPT-4o, GPT-4o-mini)
- Anthropic (Claude 4-5 Sonnet, Haiku)
- Google Gemini
- Mistral AI
- Azure OpenAI
- Local models (Ollama, Llama.cpp)

**How much do API calls cost?**

Typical query: 1-5K tokens = $0.01-$0.15

| Provider | Cost per 1K tokens |
|----------|-------------------|
| GPT-4o-mini | $0.00015 |
| GPT-4o | $0.015 |
| Claude Sonnet | $0.003 |
| Mistral | $0.0002 |
| Local | Free |

**Can I use my own LLM endpoint?**

Yes:

```python
llm = lmai.llm.OpenAI(
    api_key='...',
    endpoint='https://your-endpoint.com/v1'
)
```

## Using Lumen

**How do I ask better questions?**

Be specific:

- Good: "Show top 5 products by revenue in Q4 2024"
- Bad: "What about the data?"

Include what you want (top/bottom, count, average), filters (dates, categories), and whether you want charts.

**Why are SQL queries wrong?**

LLM doesn't know your schema. Ask questions that hint at column names. Or check generated SQL and provide feedback for retry.

**Can I see the generated SQL?**

Yes. SQL panel shows every query. You can review, edit, or copy it.

**How do I use different models for different tasks?**

```python
config = {
    "default": {"model": "gpt-4o-mini"},  # Cheap for most tasks
    "sql": {"model": "gpt-4o"},           # Powerful for SQL
    "vega_lite": {"model": "gpt-4o"},     # Powerful for charts
}

llm = lmai.llm.OpenAI(model_kwargs=config)
```

See [LLM Providers](configuration/llm_providers.md).

## Customization

**What are agents?**

Specialized workers. SQLAgent writes queries, VegaLiteAgent creates charts, AnalystAgent explains results. See [Agents](configuration/agents.md).

**Can I build custom agents?**

Yes:

```python
class CustomAgent(lmai.agents.Agent):
    purpose = "Analyzes sentiment"
    
    async def respond(self, messages, context, **kwargs):
        # Your logic
        return outputs, context
```

See [Agents](configuration/agents.md).

**Can I customize agent behavior?**

Yes, with prompt overrides:

```python
template_overrides = {
    "main": {
        "instructions": "Be concise. Focus on key metrics."
    }
}

agent = lmai.agents.ChatAgent(template_overrides=template_overrides)
```

See [Prompts](configuration/prompts.md).

## Data Sources

**What data sources are supported?**

Files (CSV, Parquet, JSON), databases (PostgreSQL, MySQL, Snowflake, DuckDB), cloud storage (S3, GCS), APIs, or custom sources.

**Can I use multiple data sources?**

Yes:

```python
ui = lmai.ExplorerUI(data=['sales.csv', 'customers.csv', 'products.csv'])
```

**Can I upload data through the UI?**

Yes. Click "Manage Data" to upload CSV, Parquet, or JSON files.

## Deployment

**How do I share my app?**

```bash
lumen-ai serve data.csv --show
```

Share the URL. For production, deploy with Panel server, Docker, or cloud platforms.

**Can I add authentication?**

Yes. Lumen supports basic auth and OAuth through Panel.

## Performance

**Why is my query slow?**

Common causes:

- Large CSV files - use a database instead
- Slow LLM - use faster model (gpt-4o-mini)
- Bad SQL - check generated query
- Network latency - use local LLM or nearby endpoint

**Does Lumen cache results?**

Yes, SQL queries are cached automatically.

## Getting Help

**Where can I ask questions?**

- [Discord](https://discord.com/invite/rb6gPXbdAr)
- [Discourse](https://discourse.holoviz.org/c/lumen/14)
- [GitHub Discussions](https://github.com/holoviz/lumen/discussions)

**How do I report bugs?**

[GitHub Issues](https://github.com/holoviz/lumen/issues). Include:

- Python version
- Lumen version
- Minimal reproducible example
- Error traceback

**Can I contribute?**

Yes! Bug reports, documentation, features, custom agents. See [Contributing](reference/contributing.md).
