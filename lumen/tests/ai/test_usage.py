import asyncio

from copy import deepcopy
from types import SimpleNamespace as Obj

import pytest

from lumen.ai.llm import OpenAI
from lumen.ai.tool_trace import ModelCall, ToolCall
from lumen.ai.tools.base import FunctionTool
from lumen.ai.usage import (
    Usage, meter_method, parse_usage, record_usage,
)

PRICING = {"m": {"input": 2, "cached": 0.2, "cache_write": 3, "output": 10}}


@pytest.mark.parametrize("provider,response,expected", [
    ("openai", Obj(model="m", usage=Obj(prompt_tokens=100, completion_tokens=20,
                                 prompt_tokens_details=Obj(cached_tokens=40))), (100, 20, 40)),
    ("openai", Obj(model="m", usage=Obj(input_tokens=100, output_tokens=20,
                                 input_tokens_details=Obj(cached_tokens=40))), (100, 20, 40)),
    ("anthropic", Obj(model="m", usage=Obj(input_tokens=50, output_tokens=20,
                                    cache_read_input_tokens=40, cache_creation_input_tokens=10)), (100, 20, 40)),
    ("google", Obj(model_version="m", usage_metadata=Obj(prompt_token_count=100,
                                      candidates_token_count=20, cached_content_token_count=40)), (100, 20, 40)),
    ("mistral", Obj(data=Obj(model="m", usage=Obj(prompt_tokens=100, completion_tokens=20))), (100, 20, 0)),
    ("bedrock", {"metadata": {"usage": {"inputTokens": 100, "outputTokens": 20}}}, (100, 20, 0)),
])
def test_parse_provider_usage(provider, response, expected):
    usage = parse_usage(response, provider, "m", PRICING)
    assert (usage.input_tokens, usage.output_tokens, usage.cached_tokens) == expected
    write = usage.cache_write_tokens
    assert usage.cost_usd == pytest.approx(((expected[0] - expected[2] - write) * 2
                                            + expected[2] * .2 + write * 3 + expected[1] * 10) / 1e6)


def test_unknown_model_does_not_infer_price():
    usage = parse_usage(Obj(model="other", usage=Obj(prompt_tokens=2, completion_tokens=3)), "openai", "m", PRICING)
    assert usage.model == "other"
    assert usage.cost_usd is None


def test_dated_snapshot_uses_alias_price():
    usage = parse_usage(Obj(model="m-2025-04-14", usage=Obj(prompt_tokens=100, completion_tokens=20)), "openai", "m", PRICING)
    assert usage.model == "m-2025-04-14"
    assert usage.cost_usd == pytest.approx((100 * 2 + 20 * 10) / 1e6)


def test_llm_usage_survives_param_deepcopy():
    llm = OpenAI(api_key="unused", model_kwargs={"default": {"model": "m"}})
    llm.usage.add(parse_usage(Obj(model="m", usage=Obj(prompt_tokens=2, completion_tokens=3)), "openai", "m", {}))

    copied = deepcopy(llm)

    assert copied.usage.records == llm.usage.records
    assert copied.usage is not llm.usage
    copied.usage.add(parse_usage(Obj(model="m", usage=Obj(prompt_tokens=5, completion_tokens=7)), "openai", "m", {}))
    assert (llm.usage.input_tokens, copied.usage.input_tokens) == (2, 7)


def test_default_llm_can_be_copied_by_param():
    from lumen.ai.ui import UI

    original = UI.param["llm"].default
    copied = deepcopy(original)
    assert copied.usage is not original.usage


async def test_concurrent_streams_keep_usage_in_request_scope():
    llm = OpenAI(api_key="unused", model_kwargs={"default": {"model": "m"}}, usage_pricing=PRICING)

    async def create(**kwargs):
        async def stream():
            await asyncio.sleep(0)
            yield Obj(usage=None)
            yield Obj(usage=Obj(prompt_tokens=kwargs["tokens"], completion_tokens=1), model=kwargs["model"])
        return stream()

    resource = Obj(create=create)
    meter_method(llm, resource, "create", "openai")

    async def request(tokens):
        with llm.capture_usage() as scope:
            events = await resource.create(model="m", stream=True, tokens=tokens)
        async for _ in events:
            pass
        return scope.records

    first, second = await asyncio.gather(request(10), request(20))
    assert [u.input_tokens for u in first] == [10]
    assert [u.input_tokens for u in second] == [20]
    assert sorted(u.input_tokens for u in llm.usage.records) == [10, 20]


async def test_responses_stream_records_completion_once():
    llm = OpenAI(api_key="unused", api="responses", model_kwargs={"default": {"model": "m"}})

    async def create(**kwargs):
        async def stream():
            yield Obj(type="response.output_text.delta", delta="text")
            yield Obj(type="response.completed", response=Obj(model="m", usage=Obj(input_tokens=5, output_tokens=2)))
        return stream()

    resource = Obj(create=create)
    meter_method(llm, resource, "create", "openai")
    with llm.capture_usage() as scope:
        async for _ in await resource.create(model="m", stream=True):
            pass
    assert len(scope.records) == 1
    assert (scope.records[0].input_tokens, scope.records[0].output_tokens) == (5, 2)


async def test_anthropic_stream_accounts_for_cache_write():
    llm = OpenAI(api_key="unused", model_kwargs={"default": {"model": "m"}}, usage_pricing=PRICING)

    async def create(**kwargs):
        async def stream():
            yield Obj(type="message_start", message=Obj(usage=Obj(input_tokens=50,
                      cache_read_input_tokens=40, cache_creation_input_tokens=10)))
            yield Obj(type="message_delta", usage=Obj(output_tokens=20))
        return stream()

    resource = Obj(create=create)
    meter_method(llm, resource, "create", "anthropic")
    with llm.capture_usage() as scope:
        async for _ in await resource.create(model="m", stream=True):
            pass
    assert scope.records[0].input_tokens == 100
    assert scope.records[0].cache_write_tokens == 10
    assert scope.records[0].cost_usd == pytest.approx((50 * 2 + 40 * .2 + 10 * 3 + 20 * 10) / 1e6)


async def test_nested_scopes_collect_only_their_llm():
    first = OpenAI(api_key="unused", model_kwargs={"default": {"model": "m"}})
    second = OpenAI(api_key="unused", model_kwargs={"default": {"model": "m"}})

    async def create(**kwargs):
        return Obj(model="m", usage=Obj(prompt_tokens=2, completion_tokens=3))

    resource = Obj(create=create)
    meter_method(first, resource, "create", "openai")
    with first.capture_usage() as outer, first.capture_usage() as inner, second.capture_usage() as unrelated:
        await resource.create(model="m")
    assert len(outer.records) == len(inner.records) == 1
    assert unrelated.records == []
    assert outer.cost_usd is None


async def test_openai_sdk_client_records_usage_through_invoke(monkeypatch):
    llm = OpenAI(api_key="unused", model_kwargs={"default": {"model": "m"}}, usage_pricing=PRICING)

    async def create(**kwargs):
        return Obj(model=kwargs["model"], usage=Obj(prompt_tokens=8, completion_tokens=4),
                   choices=[Obj(message=Obj(content="done"))])

    client = Obj(chat=Obj(completions=Obj(create=create)))
    monkeypatch.setattr(llm, "_instantiate_client", lambda **kwargs: client)
    with llm.capture_usage() as scope:
        result = await llm.invoke([{"role": "user", "content": "hello"}])
    assert result.choices[0].message.content == "done"
    assert (scope.input_tokens, scope.output_tokens) == (8, 4)
    assert scope.cost_usd == pytest.approx((8 * 2 + 4 * 10) / 1e6)


async def test_trace_only_captures_calls_inside_scope():
    llm = OpenAI(api_key="unused", model_kwargs={"default": {"model": "m"}})

    def uppercase(value: str) -> str:
        return value.upper()

    tool = FunctionTool(uppercase)
    calls = [{"id": "1", "function": {"name": "uppercase", "arguments": '{"value": "first"}'}}]

    await llm._run_tool_calls({"uppercase": tool}, calls, {}, [])
    with llm.trace() as traces:
        await llm._run_tool_calls({"uppercase": tool}, calls, {}, [])
    await llm._run_tool_calls({"uppercase": tool}, calls, {}, [])

    assert [(call.name, call.arguments, call.result) for call in traces if isinstance(call, ToolCall)] == [
        ("uppercase", {"value": "first"}, "FIRST"),
    ]
    assert not hasattr(llm, "tool_calls")


async def test_nested_trace_scopes_isolate_llms_and_child_tasks():
    first = OpenAI(api_key="unused", model_kwargs={"default": {"model": "m"}})
    second = OpenAI(api_key="unused", model_kwargs={"default": {"model": "m"}})

    async def lookup(value):
        await asyncio.sleep(0)
        return value

    tool = FunctionTool(lookup)

    async def execute(llm, value):
        calls = [{"id": value, "function": {"name": "lookup", "arguments": f'{{"value": "{value}"}}'}}]
        await llm._run_tool_calls({"lookup": tool}, calls, {}, [])

    with first.trace() as outer, first.trace() as inner, second.trace() as unrelated:
        await asyncio.gather(execute(first, "one"), execute(first, "two"), execute(second, "other"))

    assert {call.result for call in outer if isinstance(call, ToolCall)} == {"one", "two"}
    assert {call.result for call in inner if isinstance(call, ToolCall)} == {"one", "two"}
    assert [call.result for call in unrelated if isinstance(call, ToolCall)] == ["other"]


async def test_trace_records_model_round_trip_and_failure(monkeypatch):
    llm = OpenAI(api_key="unused", model_kwargs={"default": {"model": "m"}})

    async def run_client(model_spec, messages, **kwargs):
        record_usage(llm, llm.usage, Usage("m", 5, 2, 0, 0.001))
        if messages[0]["content"] == "fail":
            raise ValueError("provider failed")
        return Obj(choices=[Obj(message=Obj(content="done"))])

    monkeypatch.setattr(llm, "run_client", run_client)
    with llm.trace() as events:
        await llm.invoke([{"role": "user", "content": "hello"}])
        with pytest.raises(ValueError, match="provider failed"):
            await llm.invoke([{"role": "user", "content": "fail"}])

    model_calls = [event for event in events if isinstance(event, ModelCall)]
    assert len(model_calls) == 2
    assert model_calls[0].model == "m"
    assert model_calls[0].messages[0]["content"] == "hello"
    assert model_calls[0].response.choices[0].message.content == "done"
    assert model_calls[0].duration >= 0
    assert model_calls[0].usage[0].input_tokens == 5
    assert model_calls[1].error == "ValueError('provider failed')"
    assert model_calls[1].usage[0].output_tokens == 2


async def test_trace_includes_structured_tool_round_trips(monkeypatch):
    from pydantic import BaseModel

    class Answer(BaseModel):
        value: str

    llm = OpenAI(api_key="unused", model_kwargs={"default": {"model": "m"}})

    def lookup(value: str) -> str:
        return value.upper()

    async def run_client(model_spec, messages, **kwargs):
        if kwargs.get("response_model"):
            return Answer(value="done")
        if messages[-1]["role"] == "tool":
            return Obj(choices=[Obj(message=Obj(content="done", tool_calls=None))])
        return Obj(choices=[Obj(message=Obj(content=None, tool_calls=[{
            "id": "1", "function": {"name": "lookup", "arguments": '{"value": "one"}'},
        }]))])

    monkeypatch.setattr(llm, "run_client", run_client)
    with llm.trace() as events:
        result = await llm.invoke([{"role": "user", "content": "hello"}], tools=[FunctionTool(lookup)], response_model=Answer)

    assert result.value == "done"
    assert [type(event) for event in events] == [ModelCall, ToolCall, ModelCall, ModelCall]
    assert events[1].result == "ONE"


async def test_trace_stream_records_final_response(monkeypatch):
    llm = OpenAI(api_key="unused", model_kwargs={"default": {"model": "m"}})

    async def run_client(model_spec, messages, **kwargs):
        async def chunks():
            yield Obj(choices=[Obj(delta=Obj(content="first"))])
            record_usage(llm, llm.usage, Usage("m", 7, 3, 0, 0.001))
            yield Obj(choices=[Obj(delta=Obj(content="done"))])
        return chunks()

    monkeypatch.setattr(llm, "run_client", run_client)
    with llm.trace() as events:
        async for _ in llm.stream([{"role": "user", "content": "hello"}]):
            pass

    calls = [event for event in events if isinstance(event, ModelCall)]
    assert len(calls) == 1
    assert calls[0].response.choices[0].delta.content == "done"
    assert calls[0].usage[0].input_tokens == 7
