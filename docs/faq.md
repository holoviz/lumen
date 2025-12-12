# :material-help-circle: FAQ

Common questions about Lumen AI.

## What is Lumen AI?

Open-source framework for exploring data with AI. Ask questions in plain English, get SQL queries, visualizations, and analysis automatically.

## Why use Lumen instead of ChatGPT?

ChatGPT can't see your data or run queries. Lumen connects directly to your data, generates SQL, creates visualizations, and keeps history. Works with any LLM provider or local models.

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

## Can I add authentication?

Yes. Lumen supports basic auth and OAuth through Panel. See [Panel's authentication docs](https://panel.holoviz.org/how_to/authentication/index.html).
