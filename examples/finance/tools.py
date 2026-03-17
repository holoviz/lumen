from __future__ import annotations

import os

from datetime import date
from typing import Any

import requests

from lumen.ai.tools import define_tool

_MASSIVE_API_BASE = "https://api.massive.com"


def _get_massive_api_key() -> str | None:
    return os.getenv("MASSIVE_API_KEY")


def _get_json(url: str, params: dict[str, Any]) -> dict[str, Any]:
    response = requests.get(url, params=params, timeout=20)
    response.raise_for_status()
    return response.json()


@define_tool(
    purpose="Fetch the most recent stock-related news headlines from Massive.",
)
def fetch_stock_news(ticker: str, limit: int = 5) -> dict[str, Any]:
    """
    Fetch recent stock news for a ticker from Massive returning news as JSON.
    Can provide context for the ChatAgent to summarize.

    Parameters
    ----------
    ticker : str
        Stock ticker symbol, e.g. AAPL.
    limit : int
        Number of articles to fetch (max 50).

    Returns
    -------
    dict[str, Any]
        A dictionary containing ticker, article_count, and a list of article summaries.
    """
    api_key = _get_massive_api_key()
    if not api_key:
        return {"error": "Missing MASSIVE_API_KEY environment variable."}

    safe_limit = max(1, min(int(limit), 50))
    url = f"{_MASSIVE_API_BASE}/v2/reference/news"
    params = {
        "ticker": ticker.upper(),
        "limit": safe_limit,
        "sort": "published_utc",
        "order": "desc",
        "apiKey": api_key,
    }
    try:
        response = requests.get(url, params=params, timeout=20)
        response.raise_for_status()
    except Exception as exc:
        return {"error": f"News request failed: {exc}"}

    payload = response.json()
    articles = []
    for item in payload.get("results", []):
        publisher = item.get("publisher") or {}
        articles.append(
            {
                "published_utc": item.get("published_utc"),
                "title": item.get("title"),
                "source": publisher.get("name"),
                "summary": item.get("description"),
                "url": item.get("article_url"),
            }
        )
    return {
        "ticker": ticker.upper(),
        "article_count": len(articles),
        "articles": articles,
    }


@define_tool(
    purpose="Retrieve upcoming US market holidays and early closes from Massive.",
)
def fetch_upcoming_market_holidays(
    exchange: str = "",
    status: str = "",
    limit: int = 50,
) -> dict[str, Any]:
    """
    Fetch upcoming market holidays from Massive market operations API.

    Parameters
    ----------
    exchange : str
        Optional exchange filter (e.g. NYSE, NASDAQ, OTC).
    status : str
        Optional holiday status filter (e.g. closed, early-close).
    limit : int
        Max number of entries to return after filtering.

    Returns
    -------
    dict[str, Any]
        Upcoming holiday entries and summary metadata.
    """
    api_key = _get_massive_api_key()
    if not api_key:
        return {"error": "Missing MASSIVE_API_KEY environment variable."}

    url = f"{_MASSIVE_API_BASE}/v1/marketstatus/upcoming"
    params = {"apiKey": api_key}
    try:
        payload = _get_json(url, params=params)
    except Exception as exc:
        return {"error": f"Upcoming market holiday request failed: {exc}"}

    entries = payload if isinstance(payload, list) else payload.get("results", [])
    filtered = []
    wanted_exchange = exchange.strip().upper()
    wanted_status = status.strip().lower()
    for item in entries:
        if wanted_exchange and str(item.get("exchange", "")).upper() != wanted_exchange:
            continue
        if wanted_status and str(item.get("status", "")).lower() != wanted_status:
            continue
        filtered.append(item)

    safe_limit = max(1, min(int(limit), 500))
    filtered = filtered[:safe_limit]
    return {
        "count": len(filtered),
        "exchange": wanted_exchange or None,
        "status": wanted_status or None,
        "holidays": filtered,
    }


@define_tool(
    purpose="Find the next market holiday/early-close from Massive.",
)
def get_next_market_holiday(exchange: str = "") -> dict[str, Any]:
    """
    Get the next upcoming market holiday event.

    Parameters
    ----------
    exchange : str
        Optional exchange filter.

    Returns
    -------
    dict[str, Any]
        Next upcoming holiday event, or a message when none are available.
    """
    data = fetch_upcoming_market_holidays(exchange=exchange, limit=500)
    if "error" in data:
        return data
    holidays = data.get("holidays", [])
    if not holidays:
        return {"message": "No upcoming market holidays found."}

    holidays = sorted(holidays, key=lambda item: item.get("date", "9999-12-31"))
    return {"next_holiday": holidays[0]}


@define_tool(
    purpose="Check whether a specific date has a market holiday event on Massive.",
)
def check_market_holiday(date_str: str, exchange: str = "") -> dict[str, Any]:
    """
    Check if a specific date appears in upcoming market holiday events.

    Parameters
    ----------
    date_str : str
        Date in YYYY-MM-DD format.
    exchange : str
        Optional exchange filter.

    Returns
    -------
    dict[str, Any]
        Matching holiday events and a boolean indicator.
    """
    try:
        target = date.fromisoformat(date_str).isoformat()
    except ValueError:
        return {"error": "Invalid date_str. Expected YYYY-MM-DD."}

    data = fetch_upcoming_market_holidays(exchange=exchange, limit=500)
    if "error" in data:
        return data

    matches = [item for item in data.get("holidays", []) if item.get("date") == target]
    return {
        "date": target,
        "exchange": exchange.upper() if exchange else None,
        "is_holiday_event": bool(matches),
        "events": matches,
    }
