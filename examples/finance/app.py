from __future__ import annotations

import os

import panel as pn

from agent import MassiveDataSourceAgent
from tools import (
    check_market_holiday, fetch_stock_news, fetch_upcoming_market_holidays,
    get_next_market_holiday,
)

import lumen.ai as lmai

from analyses import (
    OhlcChartAnalysis, PriceVolumeDistributionAnalysis, RollingReturnAnalysis,
    VolatilityRegimeAnalysis,
)

pn.config.css_files = ["https://use.typekit.net/ngy3wov.css"]

def _build_llm():
    tools = [
        fetch_stock_news,
        fetch_upcoming_market_holidays,
        get_next_market_holiday,
        check_market_holiday,
    ]
    # Keep this opinionated but robust: use available key and attach native tool-calling.
    if os.getenv("OPENAI_API_KEY"):
        return lmai.llm.OpenAI(tools=tools)
    if os.getenv("ANTHROPIC_API_KEY"):
        return lmai.llm.Anthropic(tools=tools)
    if os.getenv("GEMINI_API_KEY"):
        return lmai.llm.Google(tools=tools)
    # Fallback still includes the native tool; user must provide compatible credentials.
    return lmai.llm.OpenAI(tools=tools)


ui = lmai.ExplorerUI(
    llm=_build_llm(),
    agents=[MassiveDataSourceAgent],
    page_config={
        "theme_config": {
            "palette": {
                "primary": {
                    "main": "#000000",
                    "contrastText": "#ffffff",
                }
            },
            "typography": {
                "fontFamily": "Indivisible, Inter, Arial, sans-serif",
            },
        },
    },
    analyses=[
        OhlcChartAnalysis,
        RollingReturnAnalysis,
        VolatilityRegimeAnalysis,
        PriceVolumeDistributionAnalysis,
    ],
    title="AI Assistant",
    suggestions=[
        ("dataset_selection", "Fetch AAPL, TSLA, and META data."),
        ("question_answer", "Summarize news about NVDA."),
        ("vertical_align_top", "Show an OHLC chart for TSLA."),
        ("vertical_align_top", "Show a volatility regime analysis of the active stock data."),
    ],
)

ui.servable()
