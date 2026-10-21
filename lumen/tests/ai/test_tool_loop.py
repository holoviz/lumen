"""Tests for the LLM tool loop: error results, argument validation, dedup, budget and submission."""

import json

from types import SimpleNamespace

import pytest

try:
    from lumen.ai.config import MissingContextError
    from lumen.ai.llm import (
        DEFAULT_MAX_TOOL_ROUNDS, DUPLICATE_TOOL_CALL_NOTE, OpenAI, SubmitTool,
        inline_schema_refs,
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
    return FunctionTool(lookup, read_only=True)


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


async def test_tool_error_is_truncated_and_redacted(llm, monkeypatch):
    def fetch(x: int) -> str:
        """Always fails."""
        raise RuntimeError(f"403 for https://user:pw@api.example.com/v1?key=SECRET&q=1 {'x' * 5000}")

    script = Script(_response(("fetch", {"x": 1})), _response(), _final)
    monkeypatch.setattr(llm, "run_client", script)

    await llm.invoke([{"role": "user", "content": "hi"}], response_model=Answer, tools=[FunctionTool(fetch)])

    (message,) = script.tool_messages(1)
    assert "SECRET" not in message["content"] and "pw@" not in message["content"]
    assert "https://<redacted>@api.example.com/v1?<redacted>" in message["content"]
    assert len(message["content"]) < 1200


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


async def test_failed_calls_are_not_cached(llm, monkeypatch):
    attempts = []

    def flaky(key: str) -> str:
        """Fails the first time."""
        attempts.append(key)
        if len(attempts) == 1:
            raise ConnectionError("timeout")
        return "ok"

    script = Script(_response(("flaky", {"key": "a"})), _response(("flaky", {"key": "a"})), _response(), _final)
    monkeypatch.setattr(llm, "run_client", script)

    await llm.invoke(
        [{"role": "user", "content": "hi"}], response_model=Answer, tools=[FunctionTool(flaky, read_only=True)],
    )

    assert attempts == ["a", "a"]
    assert script.tool_messages(2)[-1]["content"].startswith("ok")


async def test_identical_calls_to_tools_with_side_effects_run_again(llm, monkeypatch):
    applied = []

    def apply_filter(year: int) -> str:
        """Filter the data."""
        applied.append(year)
        return f"filtered to {year}"

    script = Script(
        *(_response(("apply_filter", {"year": year})) for year in (2020, 2021, 2020)), _response(), _final,
    )
    monkeypatch.setattr(llm, "run_client", script)

    await llm.invoke([{"role": "user", "content": "hi"}], response_model=Answer, tools=[FunctionTool(apply_filter)])

    assert applied == [2020, 2021, 2020]


def test_default_tool_budget_is_not_the_sql_budget():
    assert DEFAULT_MAX_TOOL_ROUNDS == 16


async def test_tool_rounds_are_capped_and_the_budget_is_stated(llm, lookup, calls, monkeypatch):
    script = Script(*(_response(("lookup", {"key": str(i)})) for i in range(4)), _final)
    monkeypatch.setattr(llm, "run_client", script)

    result = await llm.invoke(
        [{"role": "user", "content": "hi"}], response_model=Answer, tools=[lookup], max_tool_rounds=2,
    )

    assert result == Answer(value=1)
    assert [key for key, _ in calls] == ["0", "1"]
    # First call, two tool rounds, one turn answered as not run, then the structured call.
    assert len(script.requests) == 5
    assert "[Tool round 1 of 2; 1 remaining.]" in script.tool_messages(1)[-1]["content"]
    assert "Tool budget exhausted after 2 rounds" in script.tool_messages(2)[-1]["content"]
    assert script.tool_messages(3)[-1]["content"].startswith("Not run")
    # The structured call answers every call id, including the last unexecuted one.
    final_messages = script.requests[-1][0]
    assert final_messages[-1]["role"] == "tool" and final_messages[-1]["content"].startswith("Not run")
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


async def test_accepted_submission_still_runs_the_other_calls(llm, monkeypatch):
    applied = []

    def apply_filter(year: int) -> str:
        """Filter the data."""
        applied.append(year)
        return "filtered"

    script = Script(_response(("apply_filter", {"year": 2020}), ("submit_answer", {"value": 2})))
    monkeypatch.setattr(llm, "run_client", script)

    result = await llm.invoke(
        [{"role": "user", "content": "hi"}], response_model=Answer, tools=[FunctionTool(apply_filter)],
        submit_tool=SubmitTool("submit_answer", "Submit."),
    )

    assert result == Answer(value=2)
    assert applied == [2020]


async def test_structured_fallback_answer_is_validated(llm, monkeypatch):
    def validate(answer):
        if answer.value == 42:
            raise ValueError("hardcoded answer")

    script = Script(_response(content="Done."), lambda messages, kwargs: Answer(value=42))
    monkeypatch.setattr(llm, "run_client", script)

    with pytest.raises(ValueError, match="submit_answer rejected: ValueError: hardcoded answer"):
        await llm.invoke(
            [{"role": "user", "content": "hi"}], response_model=Answer,
            submit_tool=SubmitTool("submit_answer", "Submit.", validate),
        )


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


async def test_responses_api_answers_pending_calls_before_the_structured_call(lookup, monkeypatch):
    llm = OpenAI(api="responses", model_kwargs={"default": {"model": "gpt-test"}})

    def call(i):
        return SimpleNamespace(
            id=f"resp_{i}",
            output=[SimpleNamespace(type="function_call", call_id=f"call_{i}", name="lookup", arguments=f'{{"key": "{i}"}}')],
        )

    script = Script(call(0), call(1), call(2), _final)
    monkeypatch.setattr(llm, "run_client", script)

    result = await llm.invoke(
        [{"role": "user", "content": "hi"}], response_model=Answer, tools=[lookup], max_tool_rounds=1,
    )

    assert result == Answer(value=1)
    final_inputs, final_kwargs = script.requests[-1]
    assert final_kwargs["previous_response_id"] == "resp_2"
    outputs = [item for item in final_inputs if item.get("type") == "function_call_output"]
    assert [item["call_id"] for item in outputs] == ["call_2"]


async def test_stateless_responses_api_resends_the_conversation(lookup, monkeypatch):
    llm = OpenAI(api="responses", stateless_responses=True, model_kwargs={"default": {"model": "gpt-test"}})

    def call(i):
        return SimpleNamespace(
            id=f"resp_{i}",
            output=[
                SimpleNamespace(type="reasoning", id=f"rs_{i}", summary=[]),
                SimpleNamespace(type="function_call", call_id=f"call_{i}", name="lookup", arguments=f'{{"key": "{i}"}}'),
            ],
        )

    done = SimpleNamespace(id="resp_2", output=[], output_text="Found it.")
    script = Script(call(0), call(1), done, _final)
    monkeypatch.setattr(llm, "run_client", script)

    result = await llm.invoke([{"role": "user", "content": "hi"}], response_model=Answer, tools=[lookup])

    assert result == Answer(value=1)
    assert all("previous_response_id" not in kwargs for _, kwargs in script.requests)
    final_inputs = script.requests[-1][0]
    assert final_inputs[0] == {"role": "user", "content": "hi"}
    assert [(item.get("type"), item.get("call_id") or item.get("id")) for item in final_inputs[1:]] == [
        ("reasoning", "rs_0"), ("function_call", "call_0"), ("function_call_output", "call_0"),
        ("reasoning", "rs_1"), ("function_call", "call_1"), ("function_call_output", "call_1"),
    ]


async def test_stateless_responses_stream_resends_the_conversation(lookup, calls, monkeypatch):
    llm = OpenAI(api="responses", stateless_responses=True, model_kwargs={"default": {"model": "gpt-test"}})
    requests = []

    async def tool_call_events():
        yield SimpleNamespace(type="response.output_item.added", output_index=0,
                              item=SimpleNamespace(type="function_call", call_id="call_1", name="lookup"))
        yield SimpleNamespace(type="response.function_call_arguments.done", output_index=0,
                              name="lookup", arguments='{"key": "a"}')

    async def text_events():
        yield SimpleNamespace(type="response.output_text.delta", delta="done")

    async def run_client(model_spec, messages, **kwargs):
        requests.append((list(messages), kwargs))
        return tool_call_events() if len(requests) == 1 else text_events()

    monkeypatch.setattr(llm, "run_client", run_client)

    chunks = [chunk async for chunk in llm.stream([{"role": "user", "content": "hi"}], tools=[lookup])]

    assert chunks[-1] == "done"
    assert calls == [("a", 5)]
    inputs, kwargs = requests[1]
    assert "previous_response_id" not in kwargs
    assert inputs[0] == {"role": "user", "content": "hi"}
    assert inputs[1] == {"type": "function_call", "call_id": "call_1", "name": "lookup", "arguments": '{"key": "a"}'}
    assert inputs[2]["type"] == "function_call_output"
    assert inputs[2]["call_id"] == "call_1"


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
    assert len(requests) == 4
    assert "Tool budget exhausted" in requests[2][-1]["content"]
    assert requests[3][-1]["content"].startswith("Not run")
