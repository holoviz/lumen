"""
Guardrails check the text Lumen sends to an LLM and, optionally, what it returns.

Every model call goes through ``Llm._traced_run_client``, so guardrails set on an
``Llm`` cover all agents, tool loops and providers::

    llm = lmai.llm.OpenAI(guardrails=[EnkryptGuardrail()])

To add a provider, subclass ``Guardrail`` and implement ``check``. To mask data
before the table or the model sees it, pass guardrails to the source too, which
masks the columns ``flagged_columns`` finds in every table, schema and query::

    DuckDBSource(tables=..., guardrails=[guardrail], mask_columns=["ssn"])
"""
from __future__ import annotations

import asyncio
import os

from typing import TYPE_CHECKING, Literal

import pandas as pd
import param

from .config import GuardrailViolation
from .utils import content_to_text, log_debug

if TYPE_CHECKING:
    from .llm import Message

Mode = Literal["input", "output", "data"]


async def flagged_columns(df: pd.DataFrame, guardrail: Guardrail, sample_size: int = 20) -> dict[str, list[str]]:
    """
    Return the columns of ``df`` that ``guardrail`` flags, with the rules each breaks.

    Each column is checked once, by its name and up to ``sample_size`` distinct
    non-null values, in a single ``check_many`` call, so callers can mask whole
    columns instead of checking every cell. If the guardrail itself fails, its
    ``on_error`` decides: 'block' raises GuardrailViolation, 'allow' flags nothing.
    """
    texts = [
        f"{column}: " + ", ".join(df[column].dropna().astype(str).unique()[:sample_size])
        for column in df.columns
    ]
    if not texts:
        return {}
    try:
        results = await guardrail.check_many(texts, "data")
    except NotImplementedError:
        raise
    except Exception as error:  # cleanup: ignore[blind-except] providers raise their own errors; on_error decides
        guardrail._on_check_error(error, "data")
        return {}
    return {column: rules for column, rules in zip(df.columns, results, strict=True) if rules}


class Guardrail(param.Parameterized):
    """
    Base class for checks on LLM inputs and outputs.

    Subclasses implement ``check``, returning the rules a text breaks.
    """

    cache_size = param.Integer(default=1024, bounds=(0, None), doc="""
        Number of passed texts to remember, so text repeated across the agent
        calls of one turn is checked once. 0 disables the cache.""")

    check_output = param.Boolean(default=False, doc="""
        Whether to also check plain-text, non-streamed responses. Structured
        outputs and streamed text are not checked.""")

    message = param.String(default="This request was blocked by a guardrail.", doc="""
        User-facing text of the GuardrailViolation raised on a block.""")

    on_error = param.Selector(default="block", objects=["block", "allow"], doc="""
        Whether to block or allow the call when check() itself fails, e.g.
        because its API is unreachable.""")

    on_violation = param.Selector(default="raise", objects=["raise", "warn"], doc="""
        Whether a violation raises GuardrailViolation or is only logged.""")

    roles = param.List(default=["user", "tool"], item_type=str, doc="""
        Roles of the messages to check. System prompts are left out because
        Lumen's own prompts are instructions an injection detector may flag;
        tool results are in because they can carry injected text.""")

    def __init__(self, **params):
        super().__init__(**params)
        self._passed: dict[tuple[str, str], None] = {}

    async def check(self, text: str, mode: Mode) -> list[str]:
        """
        Return the rules ``text`` breaks; an empty list lets it pass.

        ``mode`` is 'input' for messages sent to the model, 'output' for
        its response and 'data' for values read from a data source.
        """
        raise NotImplementedError

    async def check_many(self, texts: list[str], mode: Mode) -> list[list[str]]:
        """Check several texts one at a time; override to send them in one request."""
        return [await self.check(text, mode) for text in texts]

    def should_check(self, message: Message) -> bool:
        """Override to skip messages that ``roles`` alone cannot pick out."""
        return message["role"] in self.roles

    async def handle_violation(self, text: str, violations: list[str], mode: Mode) -> None:
        """Override to act on a violation, e.g. to write an audit log."""
        if self.on_violation == "warn":
            log_debug(f"{type(self).__name__} flagged {mode}: {violations}")
            return
        raise GuardrailViolation(self.message, violations, mode)

    async def check_messages(self, messages: list[Message]) -> None:
        for message in messages:
            if self.should_check(message):
                await self._run(content_to_text(message["content"]), "input")

    async def check_response(self, text: str) -> None:
        if self.check_output:
            await self._run(text, "output")

    def _on_check_error(self, error: Exception, mode: Mode) -> None:
        """Return to allow the call, or raise to block it, when a check itself failed."""
        if self.on_error == "allow":
            log_debug(f"{type(self).__name__} failed, allowing: {error!r}")
            return
        raise GuardrailViolation(self.message, [f"guardrail unavailable: {error}"], mode) from error

    async def _run(self, text: str, mode: Mode) -> None:
        key = (mode, text)
        if not text.strip() or key in self._passed:
            return
        try:
            violations = await self.check(text, mode)
        except NotImplementedError:
            raise
        except Exception as error:  # cleanup: ignore[silent-except] on_error="allow" fails open on purpose
            self._on_check_error(error, mode)
            return
        if violations:
            await self.handle_violation(text, violations, mode)
        elif self.cache_size:
            self._passed[key] = None
            if len(self._passed) > self.cache_size:
                del self._passed[next(iter(self._passed))]


class EnkryptGuardrail(Guardrail):
    """
    Checks text with the Enkrypt AI Guardrails API.

    Pass inline ``detectors`` in the format of ``GuardrailsClient.detect()``,
    or the ``guardrail_name`` of a guardrail saved in Enkrypt AI so detector
    changes need no code change::

        EnkryptGuardrail(detectors={"pii": {"enabled": True, "entities": ["pii", "secrets"]}})
        EnkryptGuardrail(guardrail_name="lumen-prod", check_output=True)
    """

    api_key = param.String(default=None, doc="""
        Enkrypt AI API key; defaults to the ENKRYPT_API_KEY environment variable.""")

    base_url = param.String(default=None, doc="""
        Enkrypt AI API URL, if not the SDK's default.""")

    client = param.Parameter(default=None, doc="""
        The GuardrailsClient to call; created from api_key and base_url if not given.""")

    detectors = param.Dict(default={"injection_attack": {"enabled": True}}, doc="""
        Detector configuration for GuardrailsClient.detect(); ignored when
        guardrail_name is set.""")

    guardrail_name = param.String(default=None, doc="""
        Name of a guardrail saved in Enkrypt AI: its input detectors check
        messages and its output detectors check responses.""")

    message = param.String(default="This request was blocked by Enkrypt AI guardrails.")

    def __init__(self, **params):
        super().__init__(**params)
        if self.client is not None:
            return
        from enkryptai_sdk import GuardrailsClient
        api_key = self.api_key or os.environ.get("ENKRYPT_API_KEY")
        if not api_key:
            raise ValueError("EnkryptGuardrail needs api_key or the ENKRYPT_API_KEY environment variable.")
        kwargs = {"base_url": self.base_url} if self.base_url else {}
        self.client = GuardrailsClient(api_key=api_key, **kwargs)

    async def check(self, text: str, mode: Mode) -> list[str]:
        if self.guardrail_name:
            response = await asyncio.to_thread(
                self.client.guardrail_detect, guardrail_name=self.guardrail_name, text=text,
                mode="response" if mode == "output" else "prompt",
            )
        else:
            response = await asyncio.to_thread(self.client.detect, text=text, config=self.detectors)
        return self._violations(response)

    async def check_many(self, texts: list[str], mode: Mode) -> list[list[str]]:
        if self.guardrail_name:
            return await super().check_many(texts, mode)
        # One request for all texts stays under the API's per-second rate limit.
        response = await asyncio.to_thread(self.client.batch_detect, texts=texts, config=self.detectors)
        return [self._violations(item) for item in response.batch_detections]

    @staticmethod
    def _violations(response) -> list[str]:
        if hasattr(response, "is_safe"):
            return [] if response.is_safe() else ([str(v) for v in response.get_violations()] or ["unsafe"])
        # The SDK documents is_safe() only for detect(); otherwise read the summary.
        return [name for name, hit in response.to_dict()["summary"].items() if hit]
