"""
Run the dashboard assistant's scenarios and print how many turns did what
was asked.

    python -m lumen.agent.scenarios --live                   # record new cassettes with gpt-6-luna
    python -m lumen.agent.scenarios                          # replay the recorded cassettes
    python -m lumen.agent.scenarios --scenario kpi_top_bar --themes
    python -m lumen.agent.scenarios --output run.json --diff tests/evals/scenarios/baseline.json

Live runs read the OpenRouter key from ``$OPENROUTER_API_KEY`` or a
``--env-file`` and write each scenario's model trace to the cassette
directory, so the run can be replayed offline.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import subprocess
import sys
import tempfile

from datetime import UTC, datetime
from pathlib import Path

from . import (
    agent_fingerprint, cassette_path, compress, diff, load_cassette,
    recorded_fingerprints, run_scenario, table, to_json,
)
from .catalog import BY_NAME, SCENARIOS

CASSETTES = Path("tests/evals/scenarios")
# Next to the cassettes: the commit of the agent code they were recorded against.
RECORDED = "recorded.json"
DEFAULT_MODEL = "openai/gpt-6-luna"


def api_key(env_file: str | None) -> str | None:
    """The OpenRouter key from the environment or an env file, never echoed."""
    if os.environ.get("OPENROUTER_API_KEY"):
        return os.environ["OPENROUTER_API_KEY"]
    if env_file and Path(env_file).expanduser().exists():
        for line in Path(env_file).expanduser().read_text().splitlines():
            if line.startswith("OPENROUTER_API_KEY="):
                return line.split("=", 1)[1].strip().strip("'\"")
    return None


def echo(text: str = "", *, err: bool = False):
    (sys.stderr if err else sys.stdout).write(f"{text}\n")


def commit() -> str | None:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
    except Exception:
        return None


def stale_message(names: list[str], everything: bool) -> str:
    command = "python -m lumen.agent.scenarios --live" + ("" if everything else "".join(f" --scenario {n}" for n in names))
    return (
        f"The agent's instructions or tools changed since {len(names)} cassette(s) were recorded "
        f"({', '.join(names)}), so replaying them would score another agent. Re-record them with:\n  {command}"
    )


async def run(args) -> tuple[list, list[str]]:
    from ..evals import build_model

    names = args.scenario or [s.name for s in SCENARIOS]
    cassettes = Path(args.cassettes)
    results, stale = [], []
    current = None if args.live else agent_fingerprint()
    key = api_key(args.env_file) if args.live else None
    if args.live and not key:
        sys.exit("No OpenRouter key: set OPENROUTER_API_KEY or pass --env-file.")
    for name in names:
        scenario = BY_NAME[name]
        if args.live:
            cassettes.mkdir(parents=True, exist_ok=True)
            with tempfile.TemporaryDirectory() as tmp:
                trace = Path(tmp) / f"{name}.trace.jsonl"
                result = await run_scenario(scenario, build_model(args.model, provider="openrouter", api_key=key),
                                            trace=trace)
                if args.record and trace.exists():
                    compress(trace, cassette_path(cassettes, name))
        else:
            path = cassette_path(cassettes, name)
            if not path.exists():
                echo(f"{name}: no cassette at {path}", err=True)
                continue
            if recorded_fingerprints(path) != {current}:
                stale.append(name)
                continue
            result = await run_scenario(scenario, load_cassette(path))
            if result.stale:
                stale.append(name)
                continue
        results.append(result)
        verdicts = " ".join(t.verdict for t in result.turns)
        echo(f"{name}: {verdicts}{' ERROR ' + result.error[:200] if result.error else ''}", err=True)
        for divergence in result.divergences:
            echo(f"  replay diverged at request {divergence['request']}: {divergence['kind']}"
                 + (f" of {divergence['tool']}" if divergence["tool"] else ""), err=True)
    return results, stale


def recorded_commit(cassettes: Path) -> str | None:
    try:
        return json.loads((cassettes / RECORDED).read_text()).get("commit")
    except (OSError, ValueError):
        return None


def recorded_model(results: list) -> str | None:
    """The model the replayed cassettes were recorded with."""
    return next((t.model for r in results for t in r.turns if t.model), None)


def main(argv: list[str] | None = None):
    parser = argparse.ArgumentParser(prog="python -m lumen.agent.scenarios", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--live", action="store_true", help="Call the model instead of replaying cassettes.")
    parser.add_argument("--no-record", dest="record", action="store_false",
                        help="In a live run, keep the existing cassettes.")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="OpenRouter model id of a live run.")
    parser.add_argument("--env-file", help="Env file holding OPENROUTER_API_KEY, if it is not set.")
    parser.add_argument("--cassettes", default=str(CASSETTES), help="Directory of the recorded cassettes.")
    parser.add_argument("--scenario", action="append", choices=sorted(BY_NAME), help="Run only this scenario.")
    parser.add_argument("--themes", action="store_true", help="Print the table by theme as well.")
    parser.add_argument("--output", help="Write the results as JSON to this path.")
    parser.add_argument("--diff", help="A JSON report to compare the run with.")
    args = parser.parse_args(argv)

    results, stale = asyncio.run(run(args))
    if stale:
        echo(stale_message(stale, everything=not args.scenario), err=True)
        sys.exit(1)
    cassettes = Path(args.cassettes)
    head = commit()
    if args.live and args.record:
        (cassettes / RECORDED).write_text(json.dumps({"commit": head, "model": args.model}, indent=1) + "\n")
    report = to_json(
        results, mode="live" if args.live else "replay", model=args.model if args.live else recorded_model(results),
        commit=head if args.live else recorded_commit(cassettes), replayed_on=None if args.live else head,
        fingerprint=agent_fingerprint(), date=datetime.now(UTC).isoformat(timespec="seconds"),
    )
    echo(table(report["summary"]))
    if args.themes:
        echo()
        echo(table(report["summary"], "themes"))
    if args.output:
        Path(args.output).write_text(json.dumps(report, indent=1, default=str) + "\n")
    if args.diff:
        echo()
        echo("\n".join(diff(json.loads(Path(args.diff).read_text()), report)) or "No verdicts changed.")


if __name__ == "__main__":
    main()
