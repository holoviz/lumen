"""Provider-reported token usage and optional per-model cost accounting."""

from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from threading import Lock
from typing import Any


@dataclass(frozen=True)
class Usage:
    model: str
    input_tokens: int
    output_tokens: int
    cached_tokens: int = 0
    cost_usd: float | None = None
    cache_write_tokens: int = 0


class UsageCollector:
    def __init__(self):
        self._records: list[Usage] = []
        self._lock = Lock()

    @property
    def records(self) -> list[Usage]:
        with self._lock:
            return list(self._records)

    @property
    def cost_usd(self) -> float | None:
        records = self.records
        if not records or any(record.cost_usd is None for record in records):
            return None
        return sum(record.cost_usd for record in records if record.cost_usd is not None)

    @property
    def input_tokens(self) -> int:
        return sum(record.input_tokens for record in self.records)

    @property
    def output_tokens(self) -> int:
        return sum(record.output_tokens for record in self.records)

    @property
    def cached_tokens(self) -> int:
        return sum(record.cached_tokens for record in self.records)

    def add(self, usage: Usage):
        with self._lock:
            self._records.append(usage)


_scopes: ContextVar[tuple[tuple[object, UsageCollector], ...]] = ContextVar("lumen_llm_usage_scopes", default=())


@contextmanager
def capture_usage(llm: object):
    """Collect usage for calls made in this async context, including child tasks."""
    collector = UsageCollector()
    token = _scopes.set((*_scopes.get(), (llm, collector)))
    try:
        yield collector
    finally:
        _scopes.reset(token)


def record_usage(llm: object, collector: UsageCollector, usage: Usage,
                 scopes: tuple[tuple[object, UsageCollector], ...] | None = None):
    collector.add(usage)
    for owner, scope in _scopes.get() if scopes is None else scopes:
        if owner is llm:
            scope.add(usage)


def _field(value: Any, name: str, default: Any = None) -> Any:
    if isinstance(value, dict):
        parts = name.split('_')
        camel = parts[0] + ''.join(part.title() for part in parts[1:])
        return value.get(name, value.get(camel, default))
    return getattr(value, name, default)


def parse_usage(response: Any, provider: str, requested_model: str, pricing: dict) -> Usage | None:
    """Normalize provider totals; never infer token counts from generated text."""
    if provider == "mistral":
        response = _field(response, "data", response)
    raw = _field(response, "usage") or _field(response, "usage_metadata")
    if raw is None and provider == "bedrock":
        raw = _field(response, "metadata")
        raw = _field(raw, "usage")
    if raw is None:
        return None

    input_tokens = _field(raw, "input_tokens")
    if input_tokens is None:
        input_tokens = _field(raw, "prompt_tokens")
    if input_tokens is None:
        input_tokens = _field(raw, "prompt_token_count")
    output_tokens = _field(raw, "output_tokens")
    if output_tokens is None:
        output_tokens = _field(raw, "completion_tokens")
    if output_tokens is None:
        output_tokens = _field(raw, "candidates_token_count")
    if input_tokens is None or output_tokens is None:
        return None

    details = _field(raw, "input_tokens_details") or _field(raw, "prompt_tokens_details")
    cached = (_field(details, "cached_tokens") or _field(raw, "cache_read_input_tokens")
              or _field(raw, "cached_content_token_count") or 0)
    # Anthropic reports cache writes separately from input_tokens.
    cache_write = _field(raw, "cache_creation_input_tokens") or 0
    if provider == "anthropic":
        input_tokens += cached + cache_write
    model = _field(response, "model") or _field(response, "model_version") or requested_model
    rates = pricing.get(model)
    cost = None
    if rates is not None:
        cost = ((input_tokens - cached - cache_write) * rates["input"]
                + cached * rates.get("cached", rates["input"])
                + cache_write * rates.get("cache_write", rates["input"])
                + output_tokens * rates["output"]) / 1_000_000
    return Usage(model, input_tokens, output_tokens, cached, cost, cache_write)


def meter_method(llm: Any, resource: Any, name: str, provider: str, stream_method: bool = False):
    """Meter the SDK method so instructor retries and tool rounds are counted."""
    create = getattr(resource, name)

    async def metered(*args, **kwargs):
        model = kwargs.get("model") or kwargs.get("modelId") or "unknown"
        streaming = stream_method or kwargs.get("stream", False)
        scopes = _scopes.get()
        if provider == "openai" and streaming and name == "create" and getattr(llm, "api", None) != "responses":
            kwargs = {**kwargs, "stream_options": {**kwargs.get("stream_options", {}), "include_usage": True}}
        result = await create(*args, **kwargs)
        if not streaming:
            usage = parse_usage(result, provider, model, llm.usage_pricing)
            if usage is not None:
                record_usage(llm, llm.usage, usage, scopes)
            return result

        async def events():
            last = None
            anthropic_input = None
            async for event in result:
                if provider == "anthropic":
                    if _field(event, "type") == "message_start":
                        anthropic_input = _field(_field(event, "message"), "usage")
                    elif _field(event, "type") == "message_delta":
                        delta_usage = _field(event, "usage")
                        if delta_usage is not None and anthropic_input is not None:
                            last = {"model": model, "usage": {
                                "input_tokens": _field(anthropic_input, "input_tokens", 0),
                                "cache_read_input_tokens": _field(anthropic_input, "cache_read_input_tokens", 0),
                                "cache_creation_input_tokens": _field(anthropic_input, "cache_creation_input_tokens", 0),
                                "output_tokens": _field(delta_usage, "output_tokens", 0),
                            }}
                else:
                    candidate = _field(event, "response") if _field(event, "type") == "response.completed" else event
                    if parse_usage(candidate, provider, model, llm.usage_pricing) is not None:
                        last = candidate
                yield event
            if last is not None:
                usage = parse_usage(last, provider, model, llm.usage_pricing)
                if usage is not None:
                    record_usage(llm, llm.usage, usage, scopes)

        return events()

    setattr(resource, name, metered)
