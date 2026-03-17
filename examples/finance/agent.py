from __future__ import annotations

import os
import re
import time

from datetime import date, timedelta
from typing import Any
from urllib.parse import urlencode, urlparse

import pandas as pd
import param
import requests

from pydantic import BaseModel, Field
from typing_extensions import NotRequired

from lumen.ai.agents.base import Agent
from lumen.ai.context import ContextModel, TContext
from lumen.ai.editors import SQLEditor
from lumen.ai.llm import Message
from lumen.pipeline import Pipeline
from lumen.sources.base import Source
from lumen.sources.duckdb import DuckDBSource


class MassiveSourceInputs(ContextModel):
    sources: NotRequired[list[Source]]


class MassiveSourceOutputs(ContextModel):
    source: Source
    sources: list[Source]
    table: str


class MassiveFetchArgs(BaseModel):
    """
    LLM-selected arguments for Massive aggregate bar retrieval.
    """

    tickers: list[str] = Field(default_factory=list, description="Ticker symbols, e.g. ['AAPL', 'MSFT'].")
    start_date: str = Field(description="Start date in YYYY-MM-DD format.")
    end_date: str = Field(description="End date in YYYY-MM-DD format.")
    multiplier: int = Field(default=1, ge=1, description="Range multiplier for aggregate bars.")
    timespan: str = Field(default="day", description="One of: minute, hour, day, week, month.")


class MassiveDataSourceAgent(Agent):
    """
    Agent that fetches Massive stock ticket data.
    """

    conditions = param.List(default=[
        "Use when the user asks for stock, equities, ticker, OHLCV, or market price data.",
        "Do not use for fetching news or anything other than OHLCV data."
    ])

    purpose = param.String(default="""
        Fetches market data from Massive making it available for SQL, plotting and further analysis.""")

    user = param.String(default="Finance")

    default_tickers = param.List(default=["AAPL", "MSFT", "NVDA"], item_type=str)

    api_key = param.String(default=os.getenv("MASSIVE_API_KEY", ""))
    base_url = param.String(default="https://api.massive.com")
    start_date = param.String(default=(date.today() - timedelta(days=180)).isoformat())
    end_date = param.String(default=date.today().isoformat())
    multiplier = param.Integer(default=1, bounds=(1, None))
    timespan = param.Selector(default="day", objects=["minute", "hour", "day", "week", "month"])
    adjusted = param.Boolean(default=True)
    sort = param.Selector(default="asc", objects=["asc", "desc"])
    limit = param.Integer(default=5000, bounds=(1, 50000))
    max_pages = param.Integer(default=5, bounds=(1, None))
    timeout = param.Number(default=20.0, bounds=(1, None))

    input_schema = MassiveSourceInputs
    output_schema = MassiveSourceOutputs

    _ticker_pattern = re.compile(r"\b[A-Z]{1,5}\b")
    _ticker_stopwords = {
        "AND", "THE", "WITH", "FROM", "FOR", "SHOW", "GET", "ALL", "OHLCV",
        "API", "USD", "USA", "ETF", "AI",
    }

    def _extract_tickers(self, text: str) -> list[str]:
        tickers = []
        for token in self._ticker_pattern.findall(text.upper()):
            if token in self._ticker_stopwords:
                continue
            tickers.append(token)
        return list(dict.fromkeys(tickers))

    def _request_json(self, url: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
        if not self.api_key:
            raise ValueError("Missing Massive API key. Set MASSIVE_API_KEY or configure api_key.")
        request_params = dict(params or {})
        request_params.setdefault("apiKey", self.api_key)
        response = requests.get(url, params=request_params, timeout=float(self.timeout))
        response.raise_for_status()
        return response.json()

    def _append_api_key(self, url: str) -> str:
        parsed = urlparse(url)
        if "apiKey=" in parsed.query:
            return url
        joiner = "&" if parsed.query else "?"
        return f"{url}{joiner}{urlencode({'apiKey': self.api_key})}"

    def _fetch_aggregates(self, ticker: str) -> pd.DataFrame:
        endpoint = (
            f"{self.base_url}/v2/aggs/ticker/{ticker.upper()}/range/"
            f"{self.multiplier}/{self.timespan}/{self.start_date}/{self.end_date}"
        )
        params = {
            "adjusted": str(bool(self.adjusted)).lower(),
            "sort": self.sort,
            "limit": int(self.limit),
        }
        payload = self._request_json(endpoint, params=params)
        rows = list(payload.get("results", []))
        next_url = payload.get("next_url")
        pages = 1

        while next_url and pages < int(self.max_pages):
            payload = self._request_json(self._append_api_key(next_url))
            rows.extend(payload.get("results", []))
            next_url = payload.get("next_url")
            pages += 1
            time.sleep(0.1)

        if not rows:
            return pd.DataFrame(
                columns=[
                    "timestamp",
                    "ticker",
                    "open",
                    "high",
                    "low",
                    "close",
                    "volume",
                    "vwap",
                    "transactions",
                ]
            )

        prices = pd.DataFrame(rows).rename(
            columns={
                "t": "timestamp",
                "o": "open",
                "h": "high",
                "l": "low",
                "c": "close",
                "v": "volume",
                "vw": "vwap",
                "n": "transactions",
            }
        )
        prices["timestamp"] = pd.to_datetime(prices["timestamp"], unit="ms", utc=True).dt.tz_localize(None)
        prices["ticker"] = ticker.upper()
        keep = ["timestamp", "ticker", "open", "high", "low", "close", "volume", "vwap", "transactions"]
        return prices[[column for column in keep if column in prices.columns]]

    async def _select_fetch_args(
        self, messages: list[Message], default_tickers: list[str]
    ) -> MassiveFetchArgs:
        today = date.today()
        default_start = today - timedelta(days=180)
        # Keep extraction deterministic by forcing a strict structured output.
        system = (
            "Extract stock market data fetch parameters from the conversation.\n"
            "Return valid JSON matching the schema.\n"
            "Rules:\n"
            "- Dates must be YYYY-MM-DD.\n"
            "- If user gives no date period, use start_date="
            f"{default_start.isoformat()} and end_date={today.isoformat()}.\n"
            "- Use only timespan in: minute, hour, day, week, month.\n"
            "- If user gives no interval/granularity, use timespan=day and multiplier=1.\n"
            f"- If user gives no tickers, use these defaults: {default_tickers}.\n"
            "- Always output uppercase tickers.\n"
            "- Clamp nonsensical ranges by keeping end_date >= start_date."
        )
        try:
            args = await self.llm.invoke(
                messages=messages,
                system=system,
                response_model=MassiveFetchArgs,
                model_spec=self.llm_spec_key,
            )
        except Exception:
            return MassiveFetchArgs(
                tickers=default_tickers,
                start_date=default_start.isoformat(),
                end_date=today.isoformat(),
                multiplier=1,
                timespan="day",
            )

        tickers = [ticker.upper() for ticker in args.tickers if ticker]
        if not tickers:
            tickers = default_tickers

        allowed_timespans = {"minute", "hour", "day", "week", "month"}
        timespan = args.timespan if args.timespan in allowed_timespans else "day"
        multiplier = max(int(args.multiplier), 1)

        # Normalize date strings and enforce ordering.
        start = pd.to_datetime(args.start_date, errors="coerce")
        end = pd.to_datetime(args.end_date, errors="coerce")
        if pd.isna(start):
            start = pd.Timestamp(default_start)
        if pd.isna(end):
            end = pd.Timestamp(today)
        if end < start:
            start, end = end, start

        return MassiveFetchArgs(
            tickers=tickers,
            start_date=start.date().isoformat(),
            end_date=end.date().isoformat(),
            multiplier=multiplier,
            timespan=timespan,
        )

    async def respond(
        self,
        messages: list[Message],
        context: TContext,
        step_title: str | None = None,
    ) -> tuple[list[Any], dict[str, Any]]:
        message_text = " ".join(
            m.get("content", "")
            for m in messages
            if m.get("role") == "user"
        )
        tickers = self._extract_tickers(message_text)
        if not tickers:
            tickers = [ticker.upper() for ticker in self.default_tickers]
        llm_args = await self._select_fetch_args(messages, tickers)

        original_start, original_end = self.start_date, self.end_date
        original_multiplier, original_timespan = self.multiplier, self.timespan
        self.start_date = llm_args.start_date
        self.end_date = llm_args.end_date
        self.multiplier = llm_args.multiplier
        self.timespan = llm_args.timespan
        try:
            prices_frames = [self._fetch_aggregates(ticker) for ticker in llm_args.tickers]
        finally:
            self.start_date = original_start
            self.end_date = original_end
            self.multiplier = original_multiplier
            self.timespan = original_timespan
        prices = pd.concat(prices_frames, ignore_index=True) if prices_frames else pd.DataFrame()
        if not prices.empty:
            prices = prices.sort_values(["ticker", "timestamp"]).reset_index(drop=True)

        duck_name = f"massive_{'_'.join(llm_args.tickers).lower()}_duckdb"
        source = DuckDBSource.from_df({"prices": prices}, name=duck_name)
        pipeline = Pipeline(source=source, table="prices")
        editor = SQLEditor(component=pipeline, spec="SELECT * FROM prices")

        existing_sources = list(context.get("sources", []))
        filtered = [existing for existing in existing_sources if getattr(existing, "name", None) != duck_name]
        updated_sources = [*filtered, source]

        context = await editor.render_context()
        context["source"] = source
        context["sources"] = updated_sources
        return [editor], context
