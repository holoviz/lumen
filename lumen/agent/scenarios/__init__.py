"""
Scenarios of the dashboard assistant, drawn from its playtests: a fixture,
a prompt per turn and checks on the dashboard each turn leaves, so every
change to the assistant can be measured against a recorded baseline.

A turn did what was asked when all its functional checks pass (nodes,
tiles, numbers, wiring, layout), partly when some do and not when none
do; its quality checks (chart titles, bar counts, the request budget) are
scored apart. Checks a later workstream of the plan is meant to
fix are listed in a scenario's ``xfail`` with that workstream, so a replay
of the recorded cassettes stays green and reports them as they flip.

Run ``python -m lumen.agent.scenarios --help`` for the runner.
"""
from __future__ import annotations

import gzip
import json
import os

from collections.abc import Iterable
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from .checks import (
    Check, Tile, TurnState, parse_graph,
)

__all__ = ["Check", "Scenario", "ScenarioResult", "Turn", "TurnOutcome", "run_scenario", "summarize"]


@dataclass
class Turn:
    prompt: str
    checks: list[Check] = field(default_factory=list)


@dataclass
class Scenario:
    name: str
    theme: str
    fixture: str
    turns: list[Turn]
    seed: str | None = None
    """Dashboard YAML the scenario starts from; its source node is ``data``."""
    metrics: str | None = None
    """The name of the fixture's project file in ``fixtures.METRICS``, for the app's metrics."""
    playtest: str = ""
    """The playtest turns it is drawn from, e.g. ``"C s1 t4"``."""
    xfail: dict[str, str] = field(default_factory=dict)
    """Checks expected to fail today, as ``"<turn>/<check id>"`` (turns from 1), with the workstream that fixes them."""


@dataclass
class CheckOutcome:
    id: str
    description: str
    workstream: str
    passed: bool
    detail: str = ""
    xfail: str | None = None
    quality: bool = False


@dataclass
class TurnOutcome:
    prompt: str
    checks: list[CheckOutcome]
    requests: int
    input_tokens: int
    output_tokens: int
    cached_tokens: int
    refusals: int
    tools: list[str]
    added: list[str]
    changed: list[str]
    removed: list[str]
    output: str | None = None
    error: str | None = None
    model: str | None = None

    @property
    def verdict(self) -> str:
        """Whether the turn did what was asked, by its functional checks: nodes, tiles, numbers, wiring and layout."""
        passed = [c.passed for c in self.checks if not c.quality]
        if all(passed):
            return "did"
        return "partly" if any(passed) else "not"

    @property
    def clean(self) -> bool:
        """Whether the turn did what was asked and also passed its quality checks: titles, bar counts, request budget."""
        return all(c.passed for c in self.checks)


@dataclass
class ScenarioResult:
    name: str
    theme: str
    playtest: str
    turns: list[TurnOutcome]
    error: str | None = None
    stale: bool = False
    """Replay only: the cassette was recorded against another agent surface."""


def scenario_agent(model: Any) -> Any:
    """The agent of the app's dashboard drawer, as ``lumen serve --flowdash`` builds it."""
    from ..assistant import create_agent

    return create_agent(model, dashboard_tools="layout")


def tiles_of(editor: Any) -> tuple[dict[str, Tile], list[str]]:
    """Each node's tile as the editor shows it, and the visible grid tiles in their order."""
    from panel_flowdash.ui import tile_layout

    entries = tile_layout._entries(editor)
    tiles = {}
    for item in editor._tile_items:
        name = item["instance_id"]
        entry = entries.get(name, {})
        area = tile_layout.area_of(editor, item, entries)
        tiles[name] = Tile(area, area is not None and entry.get("visible") is not False,
                           entry.get("width"), entry.get("height"))
    grid = [item["instance_id"] for item in tile_layout.area_items(editor, tile_layout.GRID, entries)
            if tiles[item["instance_id"]].visible]
    return tiles, grid


def turn_state(session: Any, result: Any, before: str) -> TurnState:
    """What the checks see of a turn ``session`` just ran, from its ``TurnResult``."""
    from ..evals import tool_calls

    tiles, grid = tiles_of(session.editor)

    def port(ref: str) -> Any:
        try:
            return session.ops.node_output(ref)
        except Exception:
            return None

    def reference(sql: str):
        return session.source.execute(sql)

    usage = [r.usage for r in result.responses]
    return TurnState(
        prompt=result.prompt, graph=parse_graph(result.graph), before=parse_graph(before),
        added=list(result.nodes), changed=list(result.changed),
        calls=tool_calls(result.messages, result.tool_results), errors=len(set(result.tool_errors)),
        requests=len(result.responses), input_tokens=sum(u.input_tokens for u in usage),
        output_tokens=sum(u.output_tokens for u in usage), cached_tokens=sum(u.cache_read_tokens for u in usage),
        output=result.output, error=result.error, model=next((r.model_name for r in result.responses), None), tiles=tiles, grid=grid, port=port, reference=reference,
    )


def evaluate(scenario: Scenario, index: int, state: TurnState) -> TurnOutcome:
    """Run turn ``index``'s checks (from 0) on ``state``."""
    outcomes = []
    for check in scenario.turns[index].checks:
        passed, detail = check.run(state)
        outcomes.append(CheckOutcome(check.id, check.description, check.workstream, passed, detail,
                                     scenario.xfail.get(f"{index + 1}/{check.id}"), check.quality))
    return TurnOutcome(
        prompt=state.prompt, checks=outcomes, requests=state.requests, input_tokens=state.input_tokens,
        output_tokens=state.output_tokens, cached_tokens=state.cached_tokens, refusals=state.refusals,
        tools=[call.name for call in state.calls], added=state.added, changed=state.changed,
        removed=state.removed, output=state.output, error=state.error, model=state.model,
    )


def recorded_usage(outcome: TurnOutcome, recorded: list[Any] | None, played: int):
    """
    The tokens and model of a turn played back from a cassette, from the
    responses it recorded: a cassette answers with its own estimate.
    """
    if not recorded:
        return
    responses = recorded[played:played + outcome.requests]
    outcome.input_tokens = sum(r.usage.input_tokens for r in responses)
    outcome.output_tokens = sum(r.usage.output_tokens for r in responses)
    outcome.cached_tokens = sum(r.usage.cache_read_tokens for r in responses)
    outcome.model = next((r.model_name for r in responses), outcome.model)


def seed(session: Any, text: str):
    """Load a scenario's starting dashboard into ``session``, whatever it holds."""
    issues = [issue for issue in session.ops.replace_yaml(text, revision=session.ops.revision())
              if issue.severity == "error"]
    if issues:
        raise ValueError(f"The seed dashboard has errors: {issues}")


async def run_scenario(scenario: Scenario, model: Any, *, trace: str | os.PathLike | None = None) -> ScenarioResult:
    """
    Run every turn of ``scenario`` with ``model``, e.g. a live model or a
    ``Cassette``, checking the dashboard after each; ``trace`` records the
    model trace a cassette is played back from.
    """
    from ..headless import AgentSession
    from .fixtures import fixture_metrics, fixture_source

    source = fixture_source(scenario.fixture)
    session = AgentSession(source, model, agent=scenario_agent(model), trace=trace,
                           metrics=fixture_metrics(scenario.metrics))
    result = ScenarioResult(scenario.name, scenario.theme, scenario.playtest, [])
    try:
        if scenario.seed:
            seed(session, scenario.seed)
        for index, turn in enumerate(scenario.turns):
            before = session.graph()
            ran = await session.ask(turn.prompt)
            outcome = evaluate(scenario, index, turn_state(session, ran, before))
            played = sum(t.requests for t in result.turns)
            recorded_usage(outcome, getattr(model, "responses", None), played)
            result.turns.append(outcome)
            if ran.error and ("CassetteError" in ran.error or "fingerprint" in ran.error):
                result.stale = "fingerprint" in ran.error
                result.error = ran.error
                break
    except Exception as exc:
        result.error = f"{type(exc).__name__}: {exc}"
    finally:
        session.close()
    return result


# -- cassettes -------------------------------------------------------------------------------------


def cassette_path(directory: str | os.PathLike, name: str) -> Path:
    return Path(directory) / f"{name}.trace.jsonl.gz"


def compress(trace: Path, target: Path):
    """Gzip a recorded trace into ``target`` with a fixed header, so a re-recording only differs by its content."""
    with open(trace, "rb") as raw, open(target, "wb") as out, gzip.GzipFile(fileobj=out, mode="wb", mtime=0, filename="") as gz:
        gz.write(raw.read())


def load_cassette(path: str | os.PathLike, **kwargs) -> Any:
    from panel_agents.adapters.cassette import Cassette

    with gzip.open(path, "rt") as f:
        records = [json.loads(line) for line in f if line.strip()]
    return Cassette.from_trace(records, **kwargs)


def recorded_fingerprints(path: str | os.PathLike) -> set[str]:
    with gzip.open(path, "rt") as f:
        return {
            stamp for line in f if line.strip()
            for stamp in [((json.loads(line).get("metadata") or {}).get("panel_agents") or {}).get("fingerprint")]
            if stamp
        }


def agent_fingerprint() -> str:
    """The fingerprint the scenario agent stamps on its responses, to tell a stale cassette."""
    from ..evals import agent_fingerprint as fingerprint

    return fingerprint(dashboard_tools="layout")


# -- reports ---------------------------------------------------------------------------------------


def _per_turn(turns: list[TurnOutcome], attr: str) -> float:
    return sum(getattr(t, attr) for t in turns) / len(turns) if turns else 0.0


def summarize(results: Iterable[ScenarioResult]) -> dict[str, Any]:
    """Counts of turns that did what was asked, partly or not, by scenario, theme and in total."""
    results = list(results)

    def tally(turns: list[TurnOutcome]) -> dict[str, Any]:
        verdicts = [t.verdict for t in turns]
        checks = [c for t in turns for c in t.checks]
        quality = [c for c in checks if c.quality]
        return {
            "turns": len(turns), "did": verdicts.count("did"), "partly": verdicts.count("partly"),
            "not": verdicts.count("not"), "did_with_quality": sum(t.clean for t in turns),
            "checks": len(checks), "checks_passed": sum(c.passed for c in checks),
            "quality_checks": len(quality), "quality_passed": sum(c.passed for c in quality),
            "refusals_per_turn": round(_per_turn(turns, "refusals"), 2),
            "requests_per_turn": round(_per_turn(turns, "requests"), 2),
            "tokens_per_turn": round(_per_turn(turns, "input_tokens") + _per_turn(turns, "output_tokens")),
            "max_requests": max((t.requests for t in turns), default=0),
        }

    themes: dict[str, list[TurnOutcome]] = {}
    for result in results:
        themes.setdefault(result.theme, []).extend(result.turns)
    return {
        "scenarios": {r.name: tally(r.turns) for r in results},
        "themes": {theme: tally(turns) for theme, turns in themes.items()},
        "total": tally([t for r in results for t in r.turns]),
    }


def to_json(results: list[ScenarioResult], **meta: Any) -> dict[str, Any]:
    scenarios = []
    for result in results:
        data = asdict(result)
        for turn, outcome in zip(data["turns"], result.turns, strict=True):
            turn["verdict"] = outcome.verdict
            turn["did_with_quality"] = outcome.clean
        scenarios.append(data)
    return {**meta, "summary": summarize(results), "results": scenarios}


def table(summary: dict[str, Any], group: str = "scenarios") -> str:
    """The summary as a plain text table, one row per scenario (or theme) and a total."""
    header = ["", "turns", "did", "partly", "not", "did+quality", "quality", "refusals/turn", "requests/turn",
              "tokens/turn"]
    rows = [[name, *(_cells(row))] for name, row in summary[group].items()]
    rows.append(["total", *_cells(summary["total"])])
    widths = [max(len(str(r[i])) for r in [header, *rows]) for i in range(len(header))]
    lines = ["  ".join(str(c).ljust(w) if i == 0 else str(c).rjust(w) for i, (c, w) in enumerate(zip(r, widths, strict=True)))
             for r in [header, *rows]]
    lines.insert(1, "  ".join("-" * w for w in widths))
    return "\n".join(lines)


def _cells(row: dict[str, Any]) -> list[Any]:
    return [row["turns"], row["did"], row["partly"], row["not"], row["did_with_quality"],
            f"{row['quality_passed']}/{row['quality_checks']}", f"{row['refusals_per_turn']:.2f}",
            f"{row['requests_per_turn']:.1f}", f"{row['tokens_per_turn']:,}"]


def diff(base: dict[str, Any], new: dict[str, Any]) -> list[str]:
    """The turn verdicts and checks that differ between two JSON reports."""
    def index(report):
        return {
            (s["name"], i + 1): t for s in report["results"] for i, t in enumerate(s["turns"])
        }

    old, current = index(base), index(new)
    lines = []
    for key in sorted(set(old) | set(current)):
        a, b = old.get(key), current.get(key)
        if a is None or b is None:
            lines.append(f"{key[0]} t{key[1]}: {'added' if a is None else 'gone'}")
            continue
        flips = [
            f"{c['id']} {'pass' if c['passed'] else 'FAIL'}"
            for c in b["checks"]
            if any(o["id"] == c["id"] and o["passed"] != c["passed"] for o in a["checks"])
        ]
        if a["verdict"] != b["verdict"] or flips:
            lines.append(f"{key[0]} t{key[1]}: {a['verdict']} -> {b['verdict']}" + (f" ({', '.join(flips)})" if flips else ""))
    return lines
