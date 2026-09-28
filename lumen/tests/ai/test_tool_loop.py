"""Tests for the LLM tool loop: error results, argument validation, dedup, budget and submission."""

import json

from types import SimpleNamespace

import pytest

try:
    from lumen.ai.config import MissingContextError
    from lumen.ai.llm import (
        DUPLICATE_TOOL_CALL_NOTE, OpenAI, SubmitTool, inline_schema_refs,
    )
    from lumen.ai.tools import FunctionTool
except ModuleNotFoundError:
    pytest.skip("lumen.ai could not be imported, skipping tests.", allow_module_level=True)

from pydantic import BaseModel


class Answer(BaseModel):
    value: int


class Pair(BaseModel):
    source: str
    table: str


class Nested(BaseModel):
    pairs: list[Pair]


def _response(*calls, content=None):
    tool_calls = [
        {
            "id": f"call_{i}",
            "type": "function",
            "function": {"name": name, "arguments": args if isinstance(args, str) else json.dumps(args)},
        }
        for i, (name, args) in enumerate(calls)
    ]
    message = SimpleNamespace(content=content, tool_calls=tool_calls or None)
    return SimpleNamespace(choices=[SimpleNamespace(message=message)])


class Script:
    """Fake ``run_client`` that replays responses and records every request."""

    def __init__(self, *responses):
        self.responses = list(responses)
        self.requests = []

    async def __call__(self, model_spec, messages, **kwargs):
        self.requests.append((list(messages), kwargs))
        response = self.responses.pop(0)
        if callable(response):
            return response(messages, kwargs)
        return response

    def tool_messages(self, request):
        return [m for m in self.requests[request][0] if m.get("role") == "tool"]


@pytest.fixture
def llm():
    return OpenAI(model_kwargs={"default": {"model": "gpt-test"}})


@pytest.fixture
def calls():
    return []


@pytest.fixture
def lookup(calls):
    def lookup(key: str, limit: int = 5) -> str:
        """Look up a key."""
        calls.append((key, limit))
        return f"value of {key}"
    return FunctionTool(lookup)


def _final(messages, kwargs):
    assert kwargs["response_model"] is Answer
    return Answer(value=1)


async def test_unknown_tool_is_answered_with_closest_match(llm, lookup, monkeypatch):
    script = Script(_response(("lokup", {"key": "a"}), ("lookup", {"key": "b"})), _response(), _final)
    monkeypatch.setattr(llm, "run_client", script)

    assert await llm.invoke([{"role": "user", "content": "hi"}], response_model=Answer, tools=[lookup]) == Answer(value=1)

    unknown, known = script.tool_messages(1)
    assert unknown["tool_call_id"] == "call_0"
    assert "Unknown tool 'lokup'" in unknown["content"]
    assert "Closest matches: lookup" in unknown["content"]
    assert known["content"].startswith("value of b")


async def test_invalid_json_arguments_are_reported_without_running_the_tool(llm, lookup, calls, monkeypatch):
    script = Script(_response(("lookup", '{"key": "a"')), _response(), _final)
    monkeypatch.setattr(llm, "run_client", script)

    await llm.invoke([{"role": "user", "content": "hi"}], response_model=Answer, tools=[lookup])

    (message,) = script.tool_messages(1)
    assert "not valid JSON" in message["content"]
    assert "key (required)" in message["content"]
    assert calls == []


@pytest.mark.parametrize("arguments, expected", [
    ({"key": "a", "limit": "many"}, "Invalid arguments for 'lookup': limit"),
    ({}, "Invalid arguments for 'lookup': key: Field required"),
    ({"key": "a", "extra": 1}, "Unexpected argument(s) 'extra'"),
])
async def test_arguments_are_validated_against_the_tool_model(llm, lookup, calls, monkeypatch, arguments, expected):
    script = Script(_response(("lookup", arguments)), _response(), _final)
    monkeypatch.setattr(llm, "run_client", script)

    await llm.invoke([{"role": "user", "content": "hi"}], response_model=Answer, tools=[lookup])

    (message,) = script.tool_messages(1)
    assert expected in message["content"]
    assert calls == []


async def test_tool_exception_is_returned_to_the_model(llm, monkeypatch):
    def boom(x: int) -> str:
        """Always fails."""
        raise RuntimeError("nope")

    script = Script(_response(("boom", {"x": 1})), _response(), _final)
    monkeypatch.setattr(llm, "run_client", script)

    result = await llm.invoke([{"role": "user", "content": "hi"}], response_model=Answer, tools=[FunctionTool(boom)])

    assert result == Answer(value=1)
    (message,) = script.tool_messages(1)
    assert message["content"].startswith("Tool 'boom' failed: RuntimeError: nope")


async def test_unrecoverable_tool_error_propagates(llm, monkeypatch):
    def needs_context(x: int) -> str:
        """Requires context."""
        raise MissingContextError("no table")

    monkeypatch.setattr(llm, "run_client", Script(_response(("needs_context", {"x": 1}))))

    with pytest.raises(MissingContextError):
        await llm.invoke([{"role": "user", "content": "hi"}], response_model=Answer, tools=[FunctionTool(needs_context)])


async def test_identical_calls_are_executed_once(llm, lookup, calls, monkeypatch):
    script = Script(
        _response(("lookup", {"key": "a", "limit": 3}), ("lookup", {"key": "a", "limit": 3})),
        _response(("lookup", {"limit": 3, "key": "a"})),
        _response(),
        _final,
    )
    monkeypatch.setattr(llm, "run_client", script)

    await llm.invoke([{"role": "user", "content": "hi"}], response_model=Answer, tools=[lookup])

    assert calls == [("a", 3)]
    first, repeated = script.tool_messages(1)
    assert first["content"].startswith("value of a")
    assert repeated["content"].startswith(DUPLICATE_TOOL_CALL_NOTE + "value of a")
    later = script.tool_messages(2)[-1]
    assert later["content"].startswith(DUPLICATE_TOOL_CALL_NOTE)


async def test_tool_rounds_are_capped_and_the_budget_is_stated(llm, lookup, calls, monkeypatch):
    script = Script(*(_response(("lookup", {"key": str(i)})) for i in range(3)), _final)
    monkeypatch.setattr(llm, "run_client", script)

    result = await llm.invoke(
        [{"role": "user", "content": "hi"}], response_model=Answer, tools=[lookup], max_tool_rounds=2,
    )

    assert result == Answer(value=1)
    assert [key for key, _ in calls] == ["0", "1"]
    # First call, two tool rounds, the unanswered third request, then the structured call.
    assert len(script.requests) == 4
    assert "[Tool round 1 of 2; 1 remaining.]" in script.tool_messages(1)[-1]["content"]
    assert "Tool budget exhausted after 2 rounds" in script.tool_messages(2)[-1]["content"]
    assert "response_model" in script.requests[-1][1]


async def test_submit_tool_answers_in_one_call(llm, lookup, monkeypatch):
    script = Script(_response(("submit_answer", {"value": 7})))
    monkeypatch.setattr(llm, "run_client", script)

    result = await llm.invoke(
        [{"role": "user", "content": "hi"}], response_model=Answer, tools=[lookup],
        submit_tool=SubmitTool("submit_answer", "Submit the answer."),
    )

    assert result == Answer(value=7)
    assert len(script.requests) == 1
    names = [spec["function"]["name"] for spec in script.requests[0][1]["tools"]]
    assert names == ["lookup", "submit_answer"]


async def test_submit_tool_rejection_is_returned_for_correction(llm, monkeypatch):
    def validate(answer):
        if answer.value < 0:
            raise ValueError("value must be positive")

    script = Script(
        _response(("submit_answer", {"value": "x"})),
        _response(("submit_answer", {"value": -1})),
        _response(("submit_answer", {"value": 3})),
    )
    monkeypatch.setattr(llm, "run_client", script)

    result = await llm.invoke(
        [{"role": "user", "content": "hi"}], response_model=Answer,
        submit_tool=SubmitTool("submit_answer", "Submit.", validate),
    )

    assert result == Answer(value=3)
    assert "Invalid submit_answer arguments: value" in script.tool_messages(1)[-1]["content"]
    rejected = script.tool_messages(2)[-1]["content"]
    assert "submit_answer rejected: ValueError: value must be positive" in rejected


async def test_submission_is_accepted_after_the_budget_is_spent(llm, lookup, monkeypatch):
    script = Script(
        _response(("lookup", {"key": "a"})),
        _response(("lookup", {"key": "b"}), ("submit_answer", {"value": 2})),
    )
    monkeypatch.setattr(llm, "run_client", script)

    result = await llm.invoke(
        [{"role": "user", "content": "hi"}], response_model=Answer, tools=[lookup],
        submit_tool=SubmitTool("submit_answer", "Submit."), max_tool_rounds=1,
    )

    assert result == Answer(value=2)
    assert "call `submit_answer`" in script.tool_messages(1)[-1]["content"]


async def test_text_reply_with_submit_tool_keeps_the_text_for_the_structured_call(llm, monkeypatch):
    script = Script(_response(content="The answer is 4."), _final)
    monkeypatch.setattr(llm, "run_client", script)

    await llm.invoke(
        [{"role": "user", "content": "hi"}], response_model=Answer,
        submit_tool=SubmitTool("submit_answer", "Submit."),
    )

    final_messages = script.requests[-1][0]
    assert final_messages[-2] == {"role": "assistant", "content": "The answer is 4."}
    assert final_messages[-1]["role"] == "user"
    assert "submit_answer" in final_messages[-1]["content"]


def test_submit_tool_spec_inlines_definitions():
    spec = OpenAI._submit_tool_spec(SubmitTool("submit", "Submit.", model=Nested))
    parameters = spec["function"]["parameters"]
    assert "$defs" not in json.dumps(parameters)
    assert parameters["properties"]["pairs"]["items"]["properties"]["table"]["type"] == "string"


def test_inline_schema_refs_keeps_sibling_keys():
    schema = {
        "$defs": {"A": {"type": "object", "properties": {"x": {"type": "integer"}}}},
        "properties": {"a": {"$ref": "#/$defs/A", "description": "An A"}},
    }
    assert inline_schema_refs(schema) == {
        "properties": {"a": {"type": "object", "properties": {"x": {"type": "integer"}}, "description": "An A"}},
    }


async def test_responses_api_accepts_submission(monkeypatch):
    llm = OpenAI(api="responses", model_kwargs={"default": {"model": "gpt-test"}})
    output = SimpleNamespace(
        id="resp_1",
        output=[SimpleNamespace(type="function_call", call_id="call_1", name="submit_answer", arguments='{"value": 5}')],
    )
    script = Script(output)
    monkeypatch.setattr(llm, "run_client", script)

    result = await llm.invoke(
        [{"role": "user", "content": "hi"}], response_model=Answer,
        submit_tool=SubmitTool("submit_answer", "Submit."),
    )

    assert result == Answer(value=5)
    assert len(script.requests) == 1


def _stream_chunks(tool_call=None, text=""):
    async def gen():
        if tool_call:
            yield SimpleNamespace(choices=[SimpleNamespace(delta=SimpleNamespace(content="", tool_calls=[tool_call]))])
        else:
            yield SimpleNamespace(choices=[SimpleNamespace(delta=SimpleNamespace(content=text, tool_calls=None))])
    return gen()


async def test_stream_tool_recursion_is_capped(llm, lookup, calls, monkeypatch):
    requests = []

    async def run_client(model_spec, messages, **kwargs):
        requests.append(messages)
        key = str(len(requests))
        return _stream_chunks({
            "index": 0, "id": f"call_{key}", "type": "function",
            "function": {"name": "lookup", "arguments": json.dumps({"key": key})},
        })

    monkeypatch.setattr(llm, "run_client", run_client)

    async for _ in llm.stream([{"role": "user", "content": "hi"}], tools=[lookup], max_tool_rounds=2):
        pass

    assert [key for key, _ in calls] == ["1", "2"]
    assert len(requests) == 3
    assert "Tool budget exhausted" in requests[-1][-1]["content"]
