"""Isolated worker for a single BIRD evaluation case."""

import asyncio
import json
import time

from pydantic_evals import Dataset

from lumen.ai.evals.bird import BirdExecution, bird_source, database_path
from lumen.ai.evals.harness import (
    DEFAULT_SCORING, CheckResult, artifacts_path, evaluate, git_state,
    model_metadata, resolve_subject,
)

CASE_TIMEOUT = 240


def default_api(provider):
    return "chat_completions" if provider == "openrouter" else "responses"


def run_bird_case(args):
    (index, case, databases, model, provider, key, output, instructions, instruction_version,
     disable_sql_cleanup, settings, api, spec) = args
    subject = resolve_subject(spec, **({"sql_cleanup": False} if disable_sql_cleanup else {}))
    llm = subject.build_model(model, provider=provider, api_key=key, api=api, instructions=instructions)
    source_factory = lambda inputs: bird_source(database_path(databases, inputs.fixture.removeprefix("bird:")))
    one = Dataset(name="bird_case", cases=[case], evaluators=[CheckResult(), BirdExecution(databases)])
    part = output.with_name(f"{output.stem}.{index}.current.json")
    try:
        report = asyncio.run(asyncio.wait_for(evaluate(llm, one, source_factory, part,
                                                     instruction_version=instruction_version,
                                                     settings=settings, subject=subject,
                                                     scoring=settings.get("scoring", DEFAULT_SCORING),
                                                     instructions=instructions),
                                              timeout=CASE_TIMEOUT))
        result = json.loads(part.read_text(encoding="utf-8"))
        sidecar = artifacts_path(part)
        if sidecar.exists():
            result["artifacts"] = [json.loads(line) for line in sidecar.read_text(encoding="utf-8").splitlines()]
        if not result["cases"]:
            failure = report.failures[0].error_message if report.failures else "No case result returned"
            result["cases"] = [{"name": case.name, "turns": [], "usage": None, "assertions": {},
                                "duration": None, "error": failure}]
    except TimeoutError:
        result = _harness_failure(case, one, llm, model, provider, f"Case exceeded the {CASE_TIMEOUT}s evaluation limit")
    except (ValueError, OSError) as exc:
        # Subject errors become error turns and gold SQL errors evaluator
        # failures inside evaluate(), so only harness errors reach this point.
        result = _harness_failure(case, one, llm, model, provider, f"Harness failed: {type(exc).__name__}: {exc}")
    finally:
        part.unlink(missing_ok=True)
        artifacts_path(part).unlink(missing_ok=True)
    return result


def _harness_failure(case, dataset, llm, model, provider, message):
    commit, dirty = git_state()
    return {
        "run": {"commit": commit, "dirty": dirty, "dataset": dataset.name, **model_metadata(llm), "model": model,
                "provider": provider},
        "cases": [{"name": case.name, "turns": [], "usage": None, "assertions": {},
                   "duration": None, "error": message}],
        "failures": [],
    }


def run_bird_case_process(args, sender):
    try:
        sender.send(run_bird_case(args))
    finally:
        sender.close()


def simulated_bird_case_process(args, sender):
    index, case, *_ = args
    if index == 2:
        time.sleep(600)
    else:
        time.sleep(0.1)
    try:
        sender.send({"run": {"dataset": "bird_test", "model": "test", "api": "responses", "provider": "openai"},
                     "cases": [{"name": case.name, "turns": [], "usage": None,
                                "assertions": {"execution_accuracy": True}, "duration": 0.1, "error": None}],
                     "failures": []})
    finally:
        sender.close()
