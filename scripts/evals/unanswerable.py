"""
Questions the data cannot answer, next to questions it can, asked through
lumen.ai on two hvsampledata datasets.

    python scripts/evals/unanswerable.py --llm kilo --repeat 2

An impossible question is flagged when the answer comes back with status
"unanswerable", and declined when it is flagged or its text says the data
does not hold it. An answerable question passes when it is not flagged, so
a planner that says "cannot answer" too often fails here.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import time

from pathlib import Path

import hvsampledata

import lumen

from lumen.ai.answer import aask, configure
from lumen.ai.llm import ClaudeCode, Kilo

QUESTIONS = {
    "penguins": {
        "impossible": [
            "What is the average beak colour of each penguin species?",
            "How many penguins were counted in 2030?",
            "What is the elevation of each island?",
            "What is the name of each penguin?",
            "How old is each penguin in years?",
        ],
        "answerable": [
            "Which species is heaviest on average?",
            "How many penguins are on each island?",
            "What is the longest flipper length?",
            "How many penguins were measured in 2008?",
            "What is the average bill length of male and female penguins?",
        ],
    },
    "quakes": {
        "impossible": [
            "How many people died in each earthquake?",
            "Which earthquakes caused a tsunami?",
            "How many earthquakes were there in 2019?",
            "What was the economic damage of the strongest earthquake?",
        ],
        "answerable": [
            "What is the largest magnitude?",
            "How many earthquakes are in each depth class?",
            "What was the average depth of the earthquakes in June 2024?",
            "How many earthquakes were stronger than magnitude 6?",
        ],
    },
}
DECLINED = ("no ", "not ", "n't", "cannot", "unavailable", "does not")


def declined(answer) -> bool:
    if answer.status == "unanswerable":
        return True
    return answer.status == "success" and any(word in answer.value.lower().replace("’", "'") for word in DECLINED)


async def main(args):
    llm = {"kilo": Kilo, "claude-code": ClaudeCode}[args.llm]()
    configure(share=args.share)
    sources = {
        "penguins": lumen.read({"penguins": hvsampledata.penguins("pandas")}),
        "quakes": lumen.read({"quakes": hvsampledata.earthquakes("pandas")}),
    }
    results = []
    for repeat in range(args.repeat):
        for name, kinds in QUESTIONS.items():
            for kind, questions in kinds.items():
                for question in questions:
                    start = time.monotonic()
                    try:
                        answer = await aask(sources[name], question, llm=llm)
                        status, is_declined = answer.status, declined(answer)
                    except Exception as e:
                        status, is_declined = f"{type(e).__name__}: {e}", False
                    results.append({
                        "repeat": repeat, "dataset": name, "kind": kind, "question": question, "status": status,
                        "flagged": status == "unanswerable", "declined": is_declined,
                        "seconds": round(time.monotonic() - start, 1),
                    })
                    print(f"{repeat} {kind:10} {status:13} {question}", flush=True)
    impossible = [r for r in results if r["kind"] == "impossible"]
    answerable = [r for r in results if r["kind"] == "answerable"]
    print(f"impossible flagged: {sum(r['flagged'] for r in impossible)}/{len(impossible)}")
    print(f"impossible declined: {sum(r['declined'] for r in impossible)}/{len(impossible)}")
    print(f"answerable answered: {sum(not r['flagged'] and r['status'] == 'success' for r in answerable)}/{len(answerable)}")
    if args.out:
        Path(args.out).write_text(json.dumps({"llm": args.llm, "share": args.share, "results": results}, indent=1))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--llm", choices=["kilo", "claude-code"], default="kilo")
    parser.add_argument("--share", choices=["sample", "schema", "none"], default="sample",
                        help="What the LLM may see of the data; below sample it cannot rule on a missing value")
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--out", help="Write the results as JSON")
    asyncio.run(main(parser.parse_args()))
