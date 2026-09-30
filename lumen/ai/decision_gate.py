"""Gate closed-choice decisions through an optional DecisionModel."""

from __future__ import annotations

import asyncio
import math
import time

from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any

import param

from .decisions import (
    Choice, ChoiceAnswer, DecisionModel, Noul, NoulAnswer,
)
from .tool_trace import DecisionCall, record_trace
from .utils import log_debug

DEFAULT_DECISION_THRESHOLD = 0.90


@dataclass
class DecisionOutcome:
    """A gated answer: ``value`` is only meaningful when ``accepted``."""

    accepted: bool
    value: Any = None
    certainty: float | None = None
    call: DecisionCall | None = None


def _gate(question: Choice | Noul, answer: Any) -> tuple[bool, float, Any]:
    if isinstance(question, Choice) and isinstance(answer, ChoiceAnswer):
        valid = (answer.choice in question.criteria and set(answer.probabilities) == set(question.criteria)
                 and all(math.isfinite(p) and 0 <= p <= 1 for p in answer.probabilities.values()))
        return valid, answer.confidence, answer.choice
    if isinstance(question, Noul) and isinstance(answer, NoulAnswer):
        # Symmetric, so an uncertain "no" falls back as readily as an uncertain "yes".
        return True, 2 * abs(answer.noul - 0.5), answer.noul > 0.5
    return False, 0.0, None


class DecisionUser(param.Parameterized):
    """Mixin for components that ask a DecisionModel before (or instead of) an LLM."""

    decision_model = param.ClassSelector(default=None, class_=DecisionModel, doc="""
        Optional model for closed-choice decisions before LLM fallback.""")

    decision_thresholds = param.Dict(default={}, doc="""
        Minimum certainty for each decision site, defaulting to 0.90.""")

    def _threshold(self, site: str) -> float:
        return self.decision_thresholds.get(site, DEFAULT_DECISION_THRESHOLD)

    async def _decide(
        self, site: str, state: dict, questions: dict[str, Choice | Noul]
    ) -> dict[str, DecisionOutcome]:
        """
        Ask every question in one request and gate each answer independently
        against the site's threshold. Returns an unaccepted outcome for every
        question when there is no model or the request fails.
        """
        outcomes = {key: DecisionOutcome(False) for key in questions}
        if self.decision_model is None or not questions:
            return outcomes
        names = {key: site if len(questions) == 1 and key == site else f"{site}:{key}" for key in questions}
        calls = {key: DecisionCall(names[key], "error") for key in questions}
        for key, call in calls.items():
            outcomes[key].call = call
        started = time.perf_counter()
        try:
            result = await self.decision_model.invoke(state, {names[key]: q for key, q in questions.items()})
            threshold = self._threshold(site)
            for key, question in questions.items():
                valid, certainty, value = _gate(question, result.answers.get(names[key]))
                accepted = valid and math.isfinite(certainty) and certainty >= threshold - 1e-12
                outcomes[key] = DecisionOutcome(accepted, value, certainty, calls[key])
                calls[key].value, calls[key].certainty = value, certainty
                calls[key].route = "accepted" if accepted else "fallback"
            # One request: attribute its usage to the first question only.
            next(iter(calls.values())).usage = dict(result.usage)
            log_debug(f"Decision {site}: {sum(o.accepted for o in outcomes.values())}/{len(outcomes)} accepted")
        except asyncio.CancelledError:
            raise
        except Exception as e:
            log_debug(f"Decision {site}: {type(e).__name__}; falling back")
        finally:
            duration = time.perf_counter() - started
            llm = getattr(self, "llm", None)
            for call in calls.values():
                call.duration = duration
                record_trace(llm, call)
        return outcomes

    async def _decide_or_fallback(
        self, site: str, decision_state: dict, question: Choice | Noul, fallback: Callable[[], Awaitable[Any]]
    ) -> Any:
        """Return the accepted decision, or the value of the existing LLM path."""
        if self.decision_model is None:
            return await fallback()
        outcome = (await self._decide(site, decision_state, {site: question}))[site]
        if outcome.accepted:
            return outcome.value
        value = await fallback()
        outcome.call.fallback_value = value
        return value
