"""
Record the sessions in ``tests/evals/sessions`` again, live, with
``python -m tests.evals.record_sessions [--model gpt-6-luna]``.

Run it whenever the agent's instructions or tool schemas change, which the
cassette tests report as a fingerprint mismatch. Review the recorded
answers before committing them: each session's expectations are computed
from what it recorded.
"""

import argparse
import asyncio

from pathlib import Path

from lumen.agent import AgentSession
from lumen.agent.evals import build_model
from lumen.agent.sessions import save_session
from lumen.ai.evals.harness import Inputs

from .cases import metrics, source

SESSIONS = Path(__file__).parent / "sessions"

QUESTIONS = {
    "sales_chart": ("sales", ["What is the total amount by category?", "Show that as a bar chart."]),
    "paid_by_region": ("commerce", ["What is the total paid order amount per region?", "Only keep the west region."]),
    "top_customer": ("commerce", ["Which customer has the most paid orders? Show only that customer and the count."]),
    "capacity_by_state": ("turbine_metrics", ["What is the installed capacity in MW by state, counting projects since 2005?"]),
    "turbines_map": ("turbines", ["Map the turbines, sized by capacity."]),
}


def fixture(inputs: Inputs):
    data = source(inputs)
    # Tool results show rows in the engine's order, which only one thread keeps stable.
    data._connection.execute("SET threads TO 1")
    return data


async def record(model: str, names: list[str]):
    for name in names:
        name_fixture, questions = QUESTIONS[name]
        trace = SESSIONS / f"{name}.trace.jsonl"
        trace.unlink(missing_ok=True)
        inputs = Inputs(questions, fixture=name_fixture)
        session = AgentSession(fixture(inputs), build_model(model), trace=trace, metrics=metrics(inputs))
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
    parser.add_argument("names", nargs="*", default=list(QUESTIONS))
    args = parser.parse_args()
    asyncio.run(record(args.model, args.names))


if __name__ == "__main__":
    main()
