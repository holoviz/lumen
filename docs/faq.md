# :material-help-circle: FAQ

Common questions about Lumen AI.

## What is Lumen AI?

Lumen is an **open-source Python framework for building domain-aware, extensible AI data applications**.

It uses natural language as an interface, but all work is executed through explicit, inspectable components:
- SQL generated against real data sources
- Deterministic Python analyses
- Reproducible specifications for visualizations and layouts

Lumen is designed to be configured and extended with your own data sources, analysis code, tools, and agents, so results reflect your domain logic rather than generic assumptions.

## Why use Lumen instead of ChatGPT?

ChatGPT is a general-purpose conversational model. It does not have access to your data, cannot execute queries, and cannot enforce domain rules or reproducibility.

Lumen provides the missing execution layer:
- Connects directly to your databases and files
- Generates and runs SQL and Python, not just text
- Produces inspectable query history and artifacts
- Encodes domain terminology, constraints, and business logic
- Works with any LLM provider, including fully local models

ChatGPT can help you think about data. Lumen lets you **work with data**, using AI as an interface rather than a black box.

## Is Lumen free?

Yes, fully open source (BSD 3-Clause). Costs:

- LLM API calls (OpenAI, Anthropic) - pay per token
- Local LLMs - free after download
- Hosting - only if deploying to cloud

## Can I use Lumen offline?

Yes. Install with local LLM:

```bash
pip install 'lumen[ai-llama]'
```

See [LLM Providers](configuration/llm_providers.md) for setup.

## Which LLM providers?

- OpenAI (GPT-4o, GPT-4o-mini)
- Anthropic (Claude 4-5 Sonnet, Haiku)
- Google Gemini
- Mistral AI
- Azure OpenAI
- Local models (Ollama, Llama.cpp)

## Can I use my own LLM endpoint?

Yes:

```python
llm = lmai.llm.OpenAI(
    api_key='...',
    endpoint='https://your-endpoint.com/v1'
)
```

## How do I load data from an in-memory DataFrame?

```python
import lumen.ai as lmai
from lumen.sources.duckdb import DuckDBSource

df = pd.DataFrame({
    'revenue': [1200, 2400, 3600, 4800],
    'churned': [0, 1, 0, 1]
})
source = DuckDBSource.from_df(tables={"data": df})
ui = lmai.ExplorerUI(data=source)
ui.servable()
```

## I launched the UI, but it only shows a loading indicator?

Make sure to call `.servable()` on the UI object to make it available in a server context:

```python
ui = lmai.ExplorerUI(data=source)
ui.servable()
```

## Can I add authentication?

Yes. Lumen supports basic auth and OAuth through Panel. See [Panel's authentication docs](https://panel.holoviz.org/how_to/authentication/index.html).
