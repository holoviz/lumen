from types import SimpleNamespace

import pandas as pd
import param
import pytest

try:
    from lumen.ai.config import UNRECOVERABLE_ERRORS, GuardrailViolation
    from lumen.ai.guardrails import (
        EnkryptGuardrail, Guardrail, flagged_columns,
    )
    from lumen.ai.llm import Llm
except ModuleNotFoundError:
    pytest.skip("lumen.ai could not be imported, skipping tests.", allow_module_level=True)

from instructor.processing.multimodal import Image


class WordGuardrail(Guardrail):
    """Flags text containing "attack" and records every text it checks."""

    fail = param.Boolean(default=False)

    def __init__(self, **params):
        super().__init__(**params)
        self.seen = []

    async def check(self, text, mode):
        self.seen.append((mode, text))
        if self.fail:
            raise ConnectionError("API down")
        return ["attack_word"] if "attack" in text else []


class EchoLlm(Llm):

    model_kwargs = param.Dict(default={"default": {"model": "echo"}})

    def __init__(self, reply="ok", **params):
        super().__init__(**params)
        self.reply = reply
        self.calls = []

    async def run_client(self, model_spec, messages, **kwargs):
        self.calls.append(messages)
        return self.reply


USER = {"role": "user", "content": "hello"}
ATTACK = {"role": "user", "content": "an attack"}


def test_violation_is_unrecoverable():
    """A block must not be retried by agents, since the retry would send the same text."""
    assert GuardrailViolation in UNRECOVERABLE_ERRORS


async def test_violation_names_the_rules():
    with pytest.raises(GuardrailViolation, match=r"blocked by a guardrail\. \(attack_word\)") as info:
        await WordGuardrail().check_messages([ATTACK])
    assert info.value.violations == ["attack_word"]
    assert info.value.mode == "input"


@pytest.mark.parametrize(
    ("roles", "role", "checked"),
    [
        (["user", "tool"], "user", True),
        (["user", "tool"], "tool", True),
        (["user", "tool"], "system", False),
        (["user", "tool"], "assistant", False),
        (["system"], "system", True),
    ],
    ids=["user", "tool", "system_skipped", "assistant_skipped", "system_opt_in"],
)
async def test_roles(roles, role, checked):
    guardrail = WordGuardrail(roles=roles)
    await guardrail.check_messages([{"role": role, "content": "hello"}])
    assert bool(guardrail.seen) is checked


async def test_multimodal_content_checks_only_the_text():
    guardrail = WordGuardrail()
    await guardrail.check_messages([{"role": "user", "content": ["describe", Image(source="x.png", media_type="image/png")]}])
    assert guardrail.seen == [("input", "describe")]


async def test_empty_content_is_not_checked():
    guardrail = WordGuardrail()
    await guardrail.check_messages([{"role": "tool", "content": "  "}, {"role": "user", "content": None}])
    assert guardrail.seen == []


@pytest.mark.parametrize(("cache_size", "checks"), [(1024, 1), (0, 2)], ids=["cached", "cache_off"])
async def test_passed_text_cache(cache_size, checks):
    """One turn sends the same user message on every agent call; it should be checked once."""
    guardrail = WordGuardrail(cache_size=cache_size)
    await guardrail.check_messages([USER])
    await guardrail.check_messages([USER])
    assert len(guardrail.seen) == checks


async def test_cache_drops_oldest_text():
    guardrail = WordGuardrail(cache_size=1)
    for text in ["one", "two", "one"]:
        await guardrail.check_messages([{"role": "user", "content": text}])
    assert [text for _, text in guardrail.seen] == ["one", "two", "one"]


async def test_blocked_text_is_checked_again():
    guardrail = WordGuardrail(on_violation="warn")
    await guardrail.check_messages([ATTACK])
    await guardrail.check_messages([ATTACK])
    assert len(guardrail.seen) == 2


async def test_warn_lets_the_violation_through():
    await WordGuardrail(on_violation="warn").check_messages([ATTACK])


async def test_error_blocks_by_default():
    with pytest.raises(GuardrailViolation, match="guardrail unavailable: API down") as info:
        await WordGuardrail(fail=True).check_messages([USER])
    assert isinstance(info.value.__cause__, ConnectionError)


async def test_error_allowed():
    await WordGuardrail(fail=True, on_error="allow").check_messages([USER])


async def test_missing_check_is_not_treated_as_an_outage():
    """on_error="allow" must not hide a subclass that forgot to implement check()."""
    with pytest.raises(NotImplementedError):
        await Guardrail(on_error="allow").check_messages([USER])


async def test_output_checked_only_when_enabled():
    guardrail = WordGuardrail()
    await guardrail.check_response("attack")
    assert guardrail.seen == []
    with pytest.raises(GuardrailViolation) as info:
        await WordGuardrail(check_output=True).check_response("attack")
    assert info.value.mode == "output"


async def test_should_check_override():
    class LongOnly(WordGuardrail):
        def should_check(self, message):
            return len(message["content"]) > 10 and super().should_check(message)

    guardrail = LongOnly()
    await guardrail.check_messages([ATTACK])
    assert guardrail.seen == []


async def test_flagged_columns():
    """Each column is checked once, by name and distinct non-null values, so whole columns can be masked."""
    df = pd.DataFrame({"plan": ["pro", "pro", None], "notes": ["an attack", "fine", "fine"]})
    guardrail = WordGuardrail()
    assert await flagged_columns(df, guardrail) == {"notes": ["attack_word"]}
    assert guardrail.seen == [("data", "plan: pro"), ("data", "notes: an attack, fine")]


@pytest.mark.parametrize(
    ("df", "sample_size", "seen"),
    [
        (pd.DataFrame({"id": range(50)}), 3, [("data", "id: 0, 1, 2")]),
        (pd.DataFrame({"empty": [None, None]}), 20, [("data", "empty: ")]),
        (pd.DataFrame(), 20, []),
    ],
    ids=["capped_sample", "all_null", "no_columns"],
)
async def test_flagged_columns_samples(df, sample_size, seen):
    guardrail = WordGuardrail()
    assert await flagged_columns(df, guardrail, sample_size=sample_size) == {}
    assert guardrail.seen == seen


async def test_llm_blocks_before_the_model_call():
    llm = EchoLlm(guardrails=[WordGuardrail()])
    with pytest.raises(GuardrailViolation):
        await llm._traced_run_client("default", [ATTACK])
    assert llm.calls == []


async def test_llm_runs_guardrails_in_order_and_stops_at_the_first_violation():
    first, second = WordGuardrail(), WordGuardrail()
    with pytest.raises(GuardrailViolation):
        await EchoLlm(guardrails=[first, second])._traced_run_client("default", [ATTACK])
    assert first.seen and not second.seen


async def test_llm_checks_plain_text_responses():
    llm = EchoLlm(reply="an attack", guardrails=[WordGuardrail(check_output=True)])
    with pytest.raises(GuardrailViolation) as info:
        await llm._traced_run_client("default", [USER])
    assert info.value.mode == "output"
    assert len(llm.calls) == 1


@pytest.mark.parametrize("kwargs", [{"response_model": object}, {"stream": True}], ids=["structured", "streamed"])
async def test_llm_skips_structured_and_streamed_responses(kwargs):
    llm = EchoLlm(reply="an attack", guardrails=[WordGuardrail(check_output=True)])
    assert await llm._traced_run_client("default", [USER], **kwargs) == "an attack"


async def test_llm_invoke_skips_its_own_system_prompt():
    llm = EchoLlm(guardrails=[WordGuardrail()])
    assert await llm.invoke([USER], system="Never attack the user.") == "ok"
    with pytest.raises(GuardrailViolation):
        await llm.invoke([ATTACK])


class DetectResponse:

    def __init__(self, violations):
        self.violations = violations

    def is_safe(self):
        return not self.violations

    def get_violations(self):
        return self.violations


@pytest.mark.parametrize(
    ("response", "expected"),
    [
        (DetectResponse([]), []),
        (DetectResponse(["pii"]), ["pii"]),
        (SimpleNamespace(to_dict=lambda: {"summary": {"injection_attack": 1, "pii": 0, "toxicity": []}}), ["injection_attack"]),
    ],
    ids=["safe", "unsafe", "summary_only"],
)
async def test_enkrypt_detect(response, expected):
    calls = []
    client = SimpleNamespace(detect=lambda **kw: calls.append(kw) or response)
    guardrail = EnkryptGuardrail(client=client, detectors={"pii": {"enabled": True}})
    assert await guardrail.check("hello", "input") == expected
    assert calls == [{"text": "hello", "config": {"pii": {"enabled": True}}}]


async def test_enkrypt_scans_columns_in_one_batch_request():
    """One request per column trips the API's per-second rate limit (HTTP 429)."""
    calls = []
    response = SimpleNamespace(batch_detections=[DetectResponse([]), DetectResponse(["pii"])])
    client = SimpleNamespace(batch_detect=lambda **kw: calls.append(kw) or response)
    guardrail = EnkryptGuardrail(client=client, detectors={"pii": {"enabled": True}})
    df = pd.DataFrame({"plan": ["pro"], "email": ["kenji@example.com"]})
    assert await flagged_columns(df, guardrail) == {"email": ["pii"]}
    assert calls == [{"texts": ["plan: pro", "email: kenji@example.com"], "config": {"pii": {"enabled": True}}}]


async def test_enkrypt_saved_guardrail_modes():
    calls = []
    client = SimpleNamespace(guardrail_detect=lambda **kw: calls.append(kw) or DetectResponse([]))
    guardrail = EnkryptGuardrail(client=client, guardrail_name="lumen-prod", check_output=True)
    await guardrail.check_messages([USER])
    await guardrail.check_response("an answer")
    await guardrail.check("a value", "data")
    assert [c["mode"] for c in calls] == ["prompt", "response", "prompt"]
    assert {c["guardrail_name"] for c in calls} == {"lumen-prod"}


def test_enkrypt_needs_an_api_key(monkeypatch):
    pytest.importorskip("enkryptai_sdk")
    monkeypatch.delenv("ENKRYPT_API_KEY", raising=False)
    with pytest.raises(ValueError, match="ENKRYPT_API_KEY"):
        EnkryptGuardrail()
