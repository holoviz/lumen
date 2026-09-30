"""Isolated worker for a single BIRD evaluation case."""

import asyncio
import json
import time

from pydantic_evals import Dataset

from lumen.ai.evals.bird import BirdExecution, bird_source, database_path
from lumen.ai.evals.harness import CheckResult, EvalOpenAI, evaluate


def run_bird_case(args):
    index, case, databases, model, provider, key, output, instructions, instruction_version, disable_sql_cleanup = args
    llm = EvalOpenAI(
        suite_instructions=instructions,
        api="chat_completions" if provider == "openrouter" else "responses",
        temperature=None, api_key=key,
        endpoint="https://openrouter.ai/api/v1" if provider == "openrouter" else None,
        model_kwargs={"default": {"model": model}, "ui": {"model": model}},
    )
    llm.disable_sql_cleanup = disable_sql_cleanup
    source_factory = lambda inputs: bird_source(database_path(databases, inputs.fixture.removeprefix("bird:")))
    one = Dataset(name="bird_case", cases=[case], evaluators=[CheckResult(), BirdExecution(databases)])
    part = output.with_name(f"{output.stem}.{index}.current.json")
    try:
        report = asyncio.run(asyncio.wait_for(evaluate(llm, one, source_factory, part,
                                                     instruction_version=instruction_version), timeout=240))
        result = json.loads(part.read_text(encoding="utf-8"))
        if not result["cases"]:
            failure = report.failures[0].error_message if report.failures else "No case result returned"
            result["cases"] = [{"name": case.name, "turns": [], "usage": None, "assertions": {},
                                "duration": None, "error": failure}]
    except (TimeoutError, ValueError, OSError) as exc:
        result = {
            "run": {"commit": "", "dirty": True, "dataset": one.name, "model": model,
                    "api": llm.api, "provider": provider},
            "cases": [{"name": case.name, "turns": [], "usage": None, "assertions": {},
                       "duration": None, "error": f"Gold SQL or scoring failed: {type(exc).__name__}: {exc}"}],
            "failures": [],
        }
    finally:
        part.unlink(missing_ok=True)
    return result


def run_bird_case_process(args, sender):
    try:
        sender.send(run_bird_case(args))
    finally:
        sender.close()


def simulated_bird_case_process(args, sender):
    index, case, *_ = args
    if index == 2:
        time.sleep(8)
    else:
        time.sleep(0.1)
    try:
        sender.send({"run": {"dataset": "bird_test", "model": "test", "api": "responses", "provider": "openai"},
                     "cases": [{"name": case.name, "turns": [], "usage": None,
                                "assertions": {"execution_accuracy": True}, "duration": 0.1, "error": None}],
                     "failures": []})
    finally:
        sender.close()
