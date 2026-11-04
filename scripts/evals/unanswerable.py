"""
Questions the data cannot answer, next to questions it can, asked through
lumen.ai.

    python scripts/evals/unanswerable.py --llm kilo

An impossible question passes when the answer is unanswerable, or when it
declines in its text without inventing numbers. An answerable question
passes when the answer is not unanswerable, so a planner that says "cannot
answer" too often fails here.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import time

from pathlib import Path

import hvsampledata

import lumen

from lumen.ai.answer import aask
from lumen.ai.llm import ClaudeCode, Kilo

IMPOSSIBLE = [
    "What is the average beak colour of each penguin species?",
    "How many penguins were counted in 2030?",
    "Which penguin has the longest wingspan in centimetres?",
    "What is the elevation of each island?",
]
ANSWERABLE = [
    "Which species is heaviest on average?",
    "How many penguins are on each island?",
    "What is the longest flipper length?",
    "How many penguins were measured in 2008?",
]
DECLINED = ("no ", "not ", "n't", "cannot", "unavailable", "does not")


def declined(answer) -> bool:
    if answer.status == "unanswerable":
        return True
    return answer.status == "success" and any(word in answer.value.lower().replace("’", "'") for word in DECLINED)


async def main(args):
    llm = {"kilo": Kilo, "claude-code": ClaudeCode}[args.llm]()
    source = lumen.read({"penguins": hvsampledata.penguins("pandas")})
    results = []
    for kind, questions in (("impossible", IMPOSSIBLE), ("answerable", ANSWERABLE)):
        for question in questions:
            start = time.monotonic()
            try:
                answer = await aask(source, question, llm=llm)
                status, passed = answer.status, declined(answer) if kind == "impossible" else answer.status != "unanswerable"
            except Exception as e:
                status, passed = f"{type(e).__name__}: {e}", False
            results.append({"kind": kind, "question": question, "status": status, "passed": passed,
                            "seconds": round(time.monotonic() - start, 1)})
            print(f"{'PASS' if passed else 'FAIL'} {kind:10} {status:13} {question}")
    for kind in ("impossible", "answerable"):
        rows = [r for r in results if r["kind"] == kind]
        print(f"{kind}: {sum(r['passed'] for r in rows)}/{len(rows)} passed")
    if args.out:
        Path(args.out).write_text(json.dumps({"llm": args.llm, "results": results}, indent=1))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--llm", choices=["kilo", "claude-code"], default="kilo")
    parser.add_argument("--out", help="Write the results as JSON")
    asyncio.run(main(parser.parse_args()))
