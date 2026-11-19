"""
Record the sessions in ``tests/evals/sessions`` again, live, with
``python -m tests.evals.record_sessions [--model gpt-6-luna]``, or through OpenRouter with
``--provider openrouter --model openai/gpt-6-luna`` and ``OPENROUTER_API_KEY`` set.

Run it whenever the agent's instructions or tool schemas change, which the
cassette tests report as a fingerprint mismatch. Review the recorded
answers before committing them: each session's expectations are computed
from what it recorded.
"""

import argparse
import asyncio
import os

from pathlib import Path

from lumen.agent import AgentSession
from lumen.agent.evals import build_model
from lumen.agent.sessions import save_session
from lumen.ai.evals.harness import Inputs

from .cases import source

SESSIONS = Path(__file__).parent / "sessions"
KEYS = {"openai": "OPENAI_API_KEY", "openrouter": "OPENROUTER_API_KEY"}

QUESTIONS = {
    "sales_chart": ("sales", ["What is the total amount by category?", "Show that as a bar chart."]),
    "paid_by_region": ("commerce", ["What is the total paid order amount per region?", "Only keep the west region."]),
    "top_customer": ("commerce", ["Which customer has the most paid orders? Show only that customer and the count."]),
}


def fixture(inputs: Inputs):
    data = source(inputs)
    # Tool results show rows in the engine's order, which only one thread keeps stable.
    data._connection.execute("SET threads TO 1")
    return data


async def record(model: str, names: list[str], provider: str = "openai"):
    for name in names:
        name_fixture, questions = QUESTIONS[name]
        trace = SESSIONS / f"{name}.trace.jsonl"
        trace.unlink(missing_ok=True)
        session = AgentSession(fixture(Inputs(questions, fixture=name_fixture)), build_model(model, provider=provider, api_key=os.environ.get(KEYS[provider])), trace=trace)
        try:
            turns = [await session.ask(question) for question in questions]
        finally:
            session.close()
        for turn in turns:
            print(f"{name}: {turn.prompt}\n  {turn.output}\n  nodes={turn.nodes} changed={turn.changed} error={turn.error}")
        save_session(SESSIONS / f"{name}.yaml", turns, name_fixture)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="gpt-6-luna")
    parser.add_argument("--provider", choices=list(KEYS), default="openai")
    parser.add_argument("names", nargs="*", default=list(QUESTIONS))
    args = parser.parse_args()
    asyncio.run(record(args.model, args.names, args.provider))


if __name__ == "__main__":
    main()
