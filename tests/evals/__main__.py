"""Run the development-only evaluation suites with ``python -m tests.evals``."""

import argparse
import asyncio
import hashlib
import json
import logging
import multiprocessing
import os
import time

from datetime import UTC, datetime
from multiprocessing.connection import wait
from pathlib import Path

from lumen.ai.evals.bird import bird_source, database_path, download_questions
from lumen.ai.evals.harness import MeteredOpenAI, case_fingerprint, evaluate

from .bird_runner import run_bird_case_process
from .cases import (
    BIRD_STRATIFIED_IDS, DATASET, QUESTION_IDS, QUESTION_URL,
    SUITE_INSTRUCTIONS, bird_dataset, documents, source,
)

RESULTS = Path(__file__).parent / "results"
LOG = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Run Lumen's AI evaluations")
    parser.add_argument("--model", default="gpt-4.1-mini")
    parser.add_argument("--provider", choices=["openai", "openrouter"], default="openai")
    parser.add_argument("--suite", choices=["local", "bird"], default="local")
    parser.add_argument("--bird-questions", type=Path, help="Pinned BIRD Mini-Dev SQLite JSON file")
    parser.add_argument("--bird-databases", type=Path, help="Extracted BIRD Mini-Dev database directory")
    parser.add_argument("--bird-all", action="store_true", help="Run all 500 Mini-Dev SQLite questions (requires --suite bird)")
    parser.add_argument("--bird-stratified", action="store_true", help="Run 33 fixed all-pass, mixed, and all-fail BIRD cases")
    parser.add_argument("--download-bird-questions", type=Path, help="Download pinned Mini-Dev question JSON and exit")
    parser.add_argument("--case", help="Run one case by name")
    parser.add_argument("--bird-ids", type=int, nargs="+", help="Run specified Mini-Dev question IDs through the parallel BIRD runner")
    parser.add_argument("--no-sql-cleanup", action="store_true", help="Disable automatic SQL cleanup in evaluation cases")
    parser.add_argument("--output", type=Path, help="Result file (defaults to a timestamped file in tests/evals/results)")
    parser.add_argument("--resume", action="store_true", help="Resume an interrupted parallel BIRD run using --output")
    parser.add_argument("--workers", type=int, default=18, help="Concurrent isolated BIRD workers (default: 18)")
    args = parser.parse_args()
    if args.download_bird_questions:
        download_questions(args.download_bird_questions, QUESTION_URL)
        return
    if (args.bird_all or args.bird_stratified or args.bird_ids) and args.suite != "bird":
        parser.error("BIRD case selection requires --suite bird")
    if sum(bool(option) for option in (args.bird_all, args.bird_stratified, args.bird_ids)) > 1:
        parser.error("--bird-all, --bird-stratified and --bird-ids are mutually exclusive")
    if args.resume and (not (args.bird_all or args.bird_stratified or args.bird_ids) or not args.output):
        parser.error("--resume requires parallel BIRD case selection and --output")
    if not 1 <= args.workers <= 64:
        parser.error("--workers must be between 1 and 64")

    instructions, instruction_version = SUITE_INSTRUCTIONS[args.suite]
    if args.no_sql_cleanup:
        instruction_version += "-no-sql-cleanup-v1"
    dataset = DATASET
    source_factory = source
    documents_factory = documents
    fixtures = None
    if args.suite == "bird":
        if not args.bird_questions or not args.bird_databases:
            parser.error("--bird-questions and --bird-databases are required for BIRD")
        if not args.bird_questions.is_file():
            parser.error(f"BIRD question file not found: {args.bird_questions}; download it with --download-bird-questions")
        if not args.bird_databases.is_dir():
            parser.error(f"BIRD database directory not found: {args.bird_databases}; extract Mini-Dev first")
        if args.bird_all:
            ids = ()
        elif args.bird_stratified:
            ids = tuple(question_id for group in BIRD_STRATIFIED_IDS.values() for question_id in group)
        elif args.bird_ids:
            ids = tuple(args.bird_ids)
        else:
            ids = QUESTION_IDS
        dataset = bird_dataset(args.bird_questions, args.bird_databases, ids)
        source_factory = lambda inputs: bird_source(database_path(args.bird_databases, inputs.fixture.removeprefix("bird:")))
        documents_factory = None
    else:
        fixtures = {case.inputs.fixture: source(case.inputs).tables for case in dataset.cases}

    key = os.environ.get("OPENROUTER_API_KEY" if args.provider == "openrouter" else "OPENAI_API_KEY")
    if not key:
        parser.error(f"A {args.provider} API key is required for live evaluations")

    llm = MeteredOpenAI(
        suite_instructions=instructions,
        api="chat_completions" if args.provider == "openrouter" else "responses",
        temperature=None, api_key=key,
        endpoint="https://openrouter.ai/api/v1" if args.provider == "openrouter" else None,
        model_kwargs={"default": {"model": args.model}, "ui": {"model": args.model}},
    )
    llm.disable_sql_cleanup = args.no_sql_cleanup
    output = args.output or RESULTS / f"{args.model.replace('/', '-')}-{datetime.now(UTC):%Y%m%d-%H%M%S}.json"
    if args.bird_all or args.bird_stratified or args.bird_ids:
        summary = run_all_bird(dataset, output, args.bird_questions, args.bird_databases,
                               args.model, args.provider, key, args.resume, args.workers,
                               instructions=instructions, instruction_version=instruction_version,
                               disable_sql_cleanup=args.no_sql_cleanup)
        LOG.warning("BIRD run summary: %s", summary)
        if summary["failed"] or summary["unscorable"]:
            raise SystemExit(1)
        return
    report = asyncio.run(evaluate(llm, dataset, source_factory, output, args.case, documents_factory,
                                  fixtures, instruction_version=instruction_version))
    report.print()
    if report.failures or any(not all(result.value for result in case.assertions.values()) for case in report.cases):
        raise SystemExit(1)


def run_all_bird(dataset, output, questions, databases, model, provider, key, resume, workers,
                 worker=run_bird_case_process, case_timeout=240, instructions="", instruction_version="",
                 disable_sql_cleanup=False):
    """Checkpoint a contiguous prefix while evaluating cases in separate workers."""
    fingerprint = case_fingerprint(dataset.cases, {"bird_questions_sha256": hashlib.sha256(questions.read_bytes()).hexdigest()},
                                   instructions, instruction_version)
    previous = None
    if output.exists():
        if not resume:
            raise FileExistsError(f"Result exists: {output}; pass --resume to continue")
        previous = json.loads(output.read_text(encoding="utf-8"))
        meta = previous["run"]
        api = "chat_completions" if provider == "openrouter" else "responses"
        if meta["case_fingerprint"] != fingerprint or meta["model"] != model or meta["api"] != api or meta.get("provider") != provider:
            raise ValueError("Existing result has a different case set, model, or API")
        expected = [case.name for case in dataset.cases]
        recorded = [case["name"] for case in previous["cases"]]
        if recorded != expected[:len(recorded)]:
            raise ValueError("Existing result cases are not a prefix of the requested suite")

    completed = len(previous["cases"]) if previous else 0
    context = multiprocessing.get_context("spawn")
    active = {}
    ready = {}
    next_start = completed + 1
    next_save = completed + 1
    try:
        while next_save <= len(dataset.cases):
            while len(active) < workers and next_start <= len(dataset.cases):
                receiver, sender = context.Pipe(duplex=False)
                case = dataset.cases[next_start - 1]
                args = (next_start, case, databases, model, provider, key, output, instructions, instruction_version,
                        disable_sql_cleanup)
                process = context.Process(target=worker, args=(args, sender))
                process.start()
                sender.close()
                active[next_start] = (process, receiver, time.monotonic())
                next_start += 1

            readable = wait([receiver for _, receiver, _ in active.values()], timeout=1)
            for index, (process, receiver, started) in list(active.items()):
                if receiver not in readable and time.monotonic() - started < case_timeout and process.is_alive():
                    continue
                if receiver in readable:
                    try:
                        result = receiver.recv()
                    except EOFError:
                        result = _failed_bird_case(dataset.cases[index - 1].name, "Worker exited without a result")
                else:
                    result = _failed_bird_case(dataset.cases[index - 1].name, f"Case exceeded the {case_timeout:g}s wall-clock limit" if process.is_alive() else "Worker exited without a result")
                    if process.is_alive():
                        process.terminate()
                receiver.close()
                process.join(timeout=2)
                if process.is_alive():
                    process.kill()
                    process.join()
                ready[index] = result
                del active[index]

            while next_save in ready:
                result = ready.pop(next_save)
                index = next_save
                next_save += 1
                if previous is None:
                    previous = {"run": dict(result["run"]), "cases": [], "failures": []}
                    previous["run"].update(dataset=dataset.name, case_fingerprint=fingerprint,
                                           question_count=len(dataset.cases), suite_instructions=instructions,
                                           instruction_version=instruction_version, model=model,
                                           api="chat_completions" if provider == "openrouter" else "responses",
                                           provider=provider)
                previous["cases"].extend(result["cases"])
                previous["failures"].extend(result["failures"])
                output.parent.mkdir(parents=True, exist_ok=True)
                checkpoint = output.with_suffix(".checkpoint.json")
                checkpoint.write_text(json.dumps(previous, indent=2), encoding="utf-8")
                checkpoint.replace(output)
                last = result["cases"][0]
                status = "pass" if last["assertions"].get("execution_accuracy") else "unscorable" if not last["assertions"] else "fail"
                LOG.warning("BIRD %s/%s: %s %s", index, len(dataset.cases), last["name"], status)
    finally:
        for process, receiver, _ in active.values():
            receiver.close()
            if process.is_alive():
                process.terminate()
            process.join(timeout=2)
            if process.is_alive():
                process.kill()
                process.join()
    cases = previous["cases"] if previous else []
    return {"completed": len(cases), "total": len(dataset.cases),
            "correct": sum(case["assertions"].get("execution_accuracy", False) for case in cases),
            "unscorable": sum(not case["assertions"] for case in cases),
             "failed": len(previous["failures"]) if previous else 0}


def _failed_bird_case(name, message):
    return {"run": {}, "cases": [{"name": name, "turns": [], "usage": None, "assertions": {},
                                    "duration": None, "error": message}], "failures": []}


if __name__ == "__main__":
    main()
