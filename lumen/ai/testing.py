"""
Record a real LLM's answers once and replay them in CI, so tests of code
built on lumen.ai run fast, offline and the same every time.

    from lumen.ai.testing import cassette

    def test_heaviest_species():
        with cassette("tests/cassettes/heaviest.json") as llm:
            answer = source.ask("Which species is heaviest?", llm=llm)
        assert answer.result == "Gentoo"

The cassette replays by default. With LUMEN_AI_RECORD=1 set it records
with the LLM given, or the one lumen.ai would pick, and writes the file.
"""
from __future__ import annotations

import difflib
import hashlib
import json
import os
import re

from collections import defaultdict
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import param

from pydantic import BaseModel

from .llm import Llm, Message

#: Bumped when the key or the file layout changes, so old cassettes miss
CASSETTE_VERSION = 2

# What changes from run to run without changing the question: the counter
# param gives unnamed objects (MetadataLookup00348), UUIDs and addresses.
# Only a class-style name takes the counter, so data such as CA94110 or
# id_10234 is left alone.
_VOLATILE = [
    (re.compile(r"\b([A-Z][a-z]+(?:[A-Z][a-z]*)*)\d{5}\b"), r"\1#"),
    (re.compile(r"\b[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}\b"), "<uuid>"),
    (re.compile(r"\b0x[0-9a-f]{6,}\b"), "0x#"),
]


class MissingRecording(LookupError):
    """A replayed call that the cassette has no response for."""


def _normalize(text: str) -> str:
    for pattern, replacement in _VOLATILE:
        text = pattern.sub(replacement, text)
    return text


def _key(messages: list[Message], system: str, response_model: type | None, model_spec: Any) -> str:
    """What identifies a call: its prompts, the type it returns and the model it asks."""
    if response_model is None:
        returns = None
    elif isinstance(response_model, type) and issubclass(response_model, BaseModel):
        # The fields, so a changed model misses instead of failing to validate
        returns = response_model.model_json_schema()
    else:
        returns = response_model.__name__
    payload = json.dumps({
        "messages": messages, "system": system, "response_model": returns,
        # A routing config hands invoke the resolved settings, which a
        # replay never sees, so only a named spec is part of the key
        "model_spec": model_spec if isinstance(model_spec, str) else "default",
    }, sort_keys=True, default=str)
    return hashlib.sha256(_normalize(payload).encode()).hexdigest()


def _prompt(messages: list[Message]) -> str:
    """The last message as text, kept beside each call so a miss can name the closest one."""
    content = messages[-1].get("content", "") if messages else ""
    return _normalize(content if isinstance(content, str) else json.dumps(content, default=str))[:500]


def _load(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"version": CASSETTE_VERSION, "calls": {}}
    cassette = json.loads(path.read_text(encoding="utf-8"))
    if cassette.get("version") != CASSETTE_VERSION:
        raise ValueError(f"{path} was recorded by another version of lumen.ai.testing; record it again.")
    return cassette


@contextmanager
def record(llm: Llm, path: str | Path, *, append: bool = False, preview: bool = True) -> Iterator[Llm]:
    """
    Record every call llm makes inside the block into the cassette at
    path, written once the block finishes without an error. The cassette
    starts empty, so calls a changed prompt no longer makes do not linger;
    with append the calls already in it are kept. Streaming is turned off
    while recording, so each call is stored as one whole response.

    The cassette holds every response and, with preview, the first 500
    characters of each call's last message, which name the closest call
    when a replay misses. Both can hold values from your data; pass
    preview=False to keep the prompts out.
    """
    path = Path(path)
    cassette = _load(path) if append else {"version": CASSETTE_VERSION, "calls": {}}
    invoke = llm.invoke
    # An invoke set on the instance already, e.g. by an outer record
    previous = llm.__dict__.get("invoke")

    async def recording(messages, system="", response_model=None, model_spec="default", **kwargs):
        output = await invoke(messages, system=system, response_model=response_model, model_spec=model_spec, **kwargs)
        # A text call can return the SDK's response object, itself a pydantic model
        if response_model is not None and isinstance(output, BaseModel):
            response = {"model": output.model_dump(mode="json")}
        else:
            response = {"text": output if isinstance(output, str) else llm._get_content(output)}
        call = cassette["calls"].setdefault(
            _key(messages, system, response_model, model_spec),
            {"prompt": _prompt(messages) if preview else "", "responses": []},
        )
        call["responses"].append(response)
        return output

    streams = llm._supports_stream, llm._supports_model_stream
    llm.invoke = recording
    llm._supports_stream = llm._supports_model_stream = False
    try:
        yield llm
    finally:
        if previous is None:
            del llm.invoke
        else:
            llm.invoke = previous
        llm._supports_stream, llm._supports_model_stream = streams
    # Only a block that finished: a partial recording would replay a half test
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(cassette, indent=1, sort_keys=True) + "\n", encoding="utf-8")


class ReplayLLM(Llm):
    """
    An LLM that answers from a cassette written by record, calling no
    provider. A call that was not recorded raises MissingRecording naming the
    closest recorded prompt. The same call made more than once gets the
    recorded responses in order.
    """

    model_kwargs = param.Dict(default={"default": {"model": "replay"}})

    path = param.Path(check_exists=False, doc="The cassette to replay.")

    # One whole response per call, as recorded
    _supports_stream = False
    _supports_model_stream = False

    def __init__(self, path: str | Path, **params):
        super().__init__(path=str(path), **params)
        self._calls = _load(Path(path))["calls"]
        self._used: defaultdict[str, int] = defaultdict(int)

    async def initialize(self, log_level: str):
        self._ready = True

    async def invoke(
        self, messages: list[Message], system: str = "", response_model: type | None = None,
        allow_partial: bool = False, model_spec: str | dict = "default", **kwargs,
    ) -> Any:
        key = _key(messages, system, response_model, model_spec)
        call = self._calls.get(key)
        if call is None or self._used[key] >= len(call["responses"]):
            prompts = [c["prompt"] for c in self._calls.values() if c["prompt"]]
            closest = difflib.get_close_matches(_prompt(messages), prompts, n=1, cutoff=0)
            hint = f"\n\nThe closest recorded call asked:\n{closest[0]}" if closest else ""
            raise MissingRecording(
                f"No recorded response for this call in {self.path}; record again with LUMEN_AI_RECORD=1. "
                f"It asked:\n{_prompt(messages)}{hint}"
            )
        response = call["responses"][self._used[key]]
        self._used[key] += 1
        if "model" in response:
            return response_model.model_validate(response["model"])
        return response["text"]


@contextmanager
def cassette(path: str | Path, llm: Llm | None = None, *, preview: bool = True) -> Iterator[Llm]:
    """
    An LLM for a test: one replaying the cassette at path, or with
    LUMEN_AI_RECORD=1 set, llm (or the one lumen.ai would pick) recording
    into it. preview is passed on to record.
    """
    if os.environ.get("LUMEN_AI_RECORD", "").lower() in ("1", "true"):
        # Deferred: answer pulls in the Panel UI stack, which replaying never needs
        from .answer import _default_llm
        with record(llm or _default_llm(), path, preview=preview) as recording:
            yield recording
    else:
        yield ReplayLLM(path)
