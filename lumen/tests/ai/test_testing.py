import json

from typing import get_args

import pandas as pd
import pytest

from openai.types.chat import ChatCompletion

try:
    import lumen.ai  # noqa
except ModuleNotFoundError:
    pytest.skip("lumen.ai could not be imported, skipping tests.", allow_module_level=True)

from lumen.ai.agents import ChatAgent, SQLAgent
from lumen.ai.agents.sql import make_sql_model
from lumen.ai.answer import aask
from lumen.ai.coordinator.planner import make_plan_model
from lumen.ai.testing import (
    MissingRecording, ReplayLLM, cassette, record,
)
from lumen.sources.data import read
from lumen.tests.ai.conftest import MockLLM


def _plan(*steps):
    PlanModel = make_plan_model(["ChatAgent", "SQLAgent"], [])
    (StepModel,) = get_args(PlanModel.__annotations__["steps"])
    return PlanModel(
        chain_of_thought="Query then explain", title="Total value",
        steps=[StepModel(actor=actor, instruction=title, title=title) for actor, title in steps],
    )


def _source():
    # A new Source each time, so its generated name differs, as in a new process
    return read({"tiny": pd.DataFrame({"category": ["A", "B", "A"], "value": [10.5, 20.0, 7.5]})})


def _live(source):
    """A stand-in for a real provider, answering the total value question."""
    llm = MockLLM()
    SQLModel = make_sql_model([(source.name, "tiny")])
    llm.set_responses([
        _plan(("SQLAgent", "Sum value"), ("ChatAgent", "Explain")),
        SQLModel(query="SELECT SUM(value) AS total FROM tiny", table_slug="tiny_total", tables=["tiny"]),
        "The total value is 38.",
    ])
    return llm


async def _ask(source, llm):
    return await aask(source, "Total value?", llm=llm, agents=[ChatAgent(), SQLAgent()])


async def test_a_recorded_question_replays_without_the_llm(tmp_path):
    path = tmp_path / "total.json"
    source = _source()
    with record(_live(source), path) as llm:
        recorded = await _ask(source, llm)

    replayed = await _ask(_source(), ReplayLLM(path))
    assert replayed.status == "success" and replayed.value == recorded.value == "The total value is 38."
    assert replayed.sql == recorded.sql and replayed.pipeline.data.iloc[0, 0] == 38.0


async def test_a_cassette_is_the_same_file_however_often_it_is_recorded(tmp_path):
    for name in ("first.json", "second.json"):
        source = _source()
        with record(_live(source), tmp_path / name) as llm:
            await _ask(source, llm)
    first = (tmp_path / "first.json").read_text()
    assert first == (tmp_path / "second.json").read_text()
    calls = json.loads(first)["calls"]
    assert len(calls) == 3 and all(len(call["responses"]) == 1 for call in calls.values())


async def test_a_call_that_was_not_recorded_names_the_closest_one(tmp_path):
    path = tmp_path / "total.json"
    source = _source()
    with record(_live(source), path) as llm:
        await _ask(source, llm)
    with pytest.raises(MissingRecording, match="(?s)record again with LUMEN_AI_RECORD=1.*closest recorded call asked:\n.*Total value"):
        await aask(_source(), "Total weight?", llm=ReplayLLM(path), agents=[ChatAgent(), SQLAgent()])


def test_recording_turns_streaming_off_and_back_on(tmp_path):
    llm = MockLLM()
    llm._supports_stream = llm._supports_model_stream = True
    with record(llm, tmp_path / "c.json"):
        assert not llm._supports_stream and not llm._supports_model_stream
        assert llm.invoke.__name__ == "recording"
    assert llm._supports_stream and llm._supports_model_stream
    assert llm.invoke.__name__ == "invoke"


class CompletionLLM(MockLLM):
    """Answers a text call with the SDK's response object, as OpenAI-compatible providers do."""

    async def invoke(self, messages, system="", response_model=None, **kwargs):
        if response_model is not None:
            return await super().invoke(messages, system=system, response_model=response_model, **kwargs)
        return ChatCompletion(
            id="1", created=0, model="m", object="chat.completion", choices=[{
                "index": 0, "finish_reason": "stop", "message": {"role": "assistant", "content": "Hi there."},
            }],
        )


async def test_a_text_call_answered_with_an_sdk_object_replays_as_text(tmp_path):
    path = tmp_path / "text.json"
    question = [{"role": "user", "content": "Hello?"}]
    with record(CompletionLLM(), path) as llm:
        await llm.invoke(question)
    assert await ReplayLLM(path).invoke(question) == "Hi there."


async def test_the_same_call_twice_replays_both_responses_in_order(tmp_path):
    path = tmp_path / "twice.json"
    llm = MockLLM()
    llm.set_responses(["first", "second"])
    question = [{"role": "user", "content": "Hello?"}]
    with record(llm, path):
        assert [await llm.invoke(question), await llm.invoke(question)] == ["first", "second"]
    replay = ReplayLLM(path)
    assert [await replay.invoke(question), await replay.invoke(question)] == ["first", "second"]
    with pytest.raises(MissingRecording):
        await replay.invoke(question)


async def test_cassette_replays_unless_asked_to_record(tmp_path, monkeypatch):
    path = tmp_path / "c.json"
    live = MockLLM()
    live.set_responses(["hi"])
    monkeypatch.setenv("LUMEN_AI_RECORD", "1")
    with cassette(path, live) as llm:
        assert await llm.invoke([{"role": "user", "content": "Hello?"}]) == "hi"
    monkeypatch.delenv("LUMEN_AI_RECORD")
    with cassette(path) as llm:
        assert isinstance(llm, ReplayLLM)
        assert await llm.invoke([{"role": "user", "content": "Hello?"}]) == "hi"


async def test_recording_again_replaces_the_old_calls_unless_appending(tmp_path):
    path = tmp_path / "c.json"
    for answer in ("old", "new"):
        llm = MockLLM()
        llm.set_responses([answer])
        with record(llm, path):
            await llm.invoke([{"role": "user", "content": f"{answer} prompt?"}])
    calls = json.loads(path.read_text())["calls"]
    assert [call["prompt"] for call in calls.values()] == ["new prompt?"]
    llm = MockLLM()
    llm.set_responses(["more"])
    with record(llm, path, append=True):
        await llm.invoke([{"role": "user", "content": "more prompt?"}])
    assert len(json.loads(path.read_text())["calls"]) == 2


async def test_a_recording_can_leave_the_prompts_out(tmp_path):
    path = tmp_path / "c.json"
    llm = MockLLM()
    llm.set_responses(["hi"])
    with record(llm, path, preview=False):
        await llm.invoke([{"role": "user", "content": "Revenue of ACME Corp?"}])
    assert "ACME" not in path.read_text()
    assert await ReplayLLM(path).invoke([{"role": "user", "content": "Revenue of ACME Corp?"}]) == "hi"


async def test_a_changed_response_model_misses_instead_of_failing_to_validate(tmp_path):
    from pydantic import BaseModel

    class Reply(BaseModel):
        text: str

    path = tmp_path / "c.json"
    llm = MockLLM()
    llm.set_responses([Reply(text="hi")])
    question = [{"role": "user", "content": "Hello?"}]
    with record(llm, path):
        await llm.invoke(question, response_model=Reply)

    class Reply(BaseModel):  # noqa: F811
        text: str
        count: int

    with pytest.raises(MissingRecording):
        await ReplayLLM(path).invoke(question, response_model=Reply)


def test_a_block_that_raises_writes_no_cassette(tmp_path):
    path = tmp_path / "c.json"
    with pytest.raises(RuntimeError), record(MockLLM(), path):
        raise RuntimeError("test failed")
    assert not path.exists()


def test_nested_recording_restores_the_outer_invoke(tmp_path):
    llm = MockLLM()
    with record(llm, tmp_path / "outer.json"):
        outer = llm.invoke
        with record(llm, tmp_path / "inner.json"):
            assert llm.invoke is not outer
        assert llm.invoke is outer
    assert llm.invoke.__name__ == "invoke"


@pytest.mark.parametrize("text, kept", [("MetadataLookup00348", False), ("CA94110", True), ("id_10234", True)])
def test_only_generated_names_lose_their_counter(text, kept):
    from lumen.ai.testing import _normalize
    assert (_normalize(text) == text) is kept


def test_a_cassette_from_another_version_is_refused(tmp_path):
    path = tmp_path / "old.json"
    path.write_text(json.dumps({"version": 0, "calls": {}}))
    with pytest.raises(ValueError, match="record it again"):
        ReplayLLM(path)


_PLANNER_PROMPT = '''
import asyncio, os, sys
import pandas as pd
from lumen.ai.agents import ChatAgent, SQLAgent, VegaLiteAgent
from lumen.ai.answer import aask
from lumen.sources.data import read
from lumen.tests.ai.conftest import MockLLM

class Capture(MockLLM):
    async def invoke(self, messages, system="", **kwargs):
        sys.stdout.write(system + str(VegaLiteAgent()))
        sys.stdout.flush()
        # The question runs on Lumen's loop thread, where SystemExit would not end the process
        os._exit(0)

source = read({"tiny": pd.DataFrame({"value": [1.0]})})
asyncio.run(aask(source, "Total?", llm=Capture(), agents=[ChatAgent(), SQLAgent(), VegaLiteAgent(n_doc_pages=0)]))
'''


@pytest.mark.parametrize("seeds", [("1", "2"), ("3", "4")])
def test_the_planner_prompt_is_the_same_in_every_process(seeds):
    """Sets iterate in a different order per process; a prompt built from one would not replay."""
    import os
    import subprocess
    import sys
    prompts = [
        subprocess.run(
            [sys.executable, "-c", _PLANNER_PROMPT], capture_output=True, text=True, check=True, timeout=300,
            env={**os.environ, "PYTHONHASHSEED": seed},
        ).stdout
        for seed in seeds
    ]
    assert "Blocked, requires: `data`, `pipeline`, `table`" in prompts[0]
    assert prompts[0] == prompts[1]


async def test_a_plain_language_check_replays_its_query(tmp_path):
    from lumen.ai.checks import acheck

    class RuleLLM(MockLLM):
        async def invoke(self, messages, system="", response_model=None, **kwargs):
            return response_model(query="SELECT * FROM tiny WHERE value < 10")

    path = tmp_path / "check.json"
    with record(RuleLLM(), path) as llm:
        recorded = await acheck(_source(), "no value under 10", llm=llm)
    replayed = await acheck(_source(), "no value under 10", llm=ReplayLLM(path))
    assert replayed.sql == recorded.sql and replayed.count == recorded.count == 1
