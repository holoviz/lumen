# Finance AI Example (Massive)

This example sets up an opinionated finance workflow for Lumen AI:

- `MassiveDataSourceAgent` fetches OHLCV bars from Massive directly, then creates a DuckDB source with table `prices`.
- The app starts with no pre-registered source; data is loaded through the agent on demand.
- Massive market tools are provided directly as **LLM native tools** (`Llm(tools=[...])`):
  - `fetch_stock_news`
  - `fetch_upcoming_market_holidays` (`GET /v1/marketstatus/upcoming`)
  - `get_next_market_holiday`
  - `check_market_holiday`
- Three `Analysis` classes produce finance-focused plots:
  - Rolling return
  - Volatility regime
  - Price/volume distribution

## Prerequisites

Set API keys in your environment:

```bash
export MASSIVE_API_KEY="..."
export OPENAI_API_KEY="..."   # or ANTHROPIC_API_KEY / GEMINI_API_KEY
```

## Run

From the repository root:

```bash
panel serve examples/finance/app.py --show
```

## Example prompts

- "Load stock data for AAPL, TSLA, and META."
- "Use the news tool for NVDA and summarize key headlines."
- "List upcoming NASDAQ market holidays."
- "Is 2026-11-27 an early-close day on NYSE?"
- "Run the volatility regime analysis on this dataset."
