"""Run the development-only evaluation suites with ``python -m tests.evals``."""

import argparse
import asyncio
import os

from datetime import UTC, datetime
from pathlib import Path

from lumen.ai.evals.bird import bird_source, database_path, download_questions
from lumen.ai.evals.harness import MeteredOpenAI, evaluate

from .cases import (
    DATASET, QUESTION_URL, bird_dataset, documents, source,
)

RESULTS = Path(__file__).parent / "results"


def main():
    parser = argparse.ArgumentParser(description="Run Lumen's AI evaluations")
    parser.add_argument("--model", default="gpt-4.1-mini")
    parser.add_argument("--provider", choices=["openai", "openrouter"], default="openai")
    parser.add_argument("--suite", choices=["local", "bird"], default="local")
    parser.add_argument("--bird-questions", type=Path, help="Pinned BIRD Mini-Dev SQLite JSON file")
    parser.add_argument("--bird-databases", type=Path, help="Extracted BIRD Mini-Dev database directory")
    parser.add_argument("--download-bird-questions", type=Path, help="Download pinned Mini-Dev question JSON and exit")
    parser.add_argument("--case", help="Run one case by name")
    parser.add_argument("--output", type=Path, help="Result file (defaults to a timestamped file in tests/evals/results)")
    args = parser.parse_args()
    if args.download_bird_questions:
        download_questions(args.download_bird_questions, QUESTION_URL)
        return

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
        dataset = bird_dataset(args.bird_questions, args.bird_databases)
        source_factory = lambda inputs: bird_source(database_path(args.bird_databases, inputs.fixture.removeprefix("bird:")))
        documents_factory = None
    else:
        fixtures = {case.inputs.fixture: source(case.inputs).tables for case in dataset.cases}

    key = os.environ.get("OPENROUTER_API_KEY" if args.provider == "openrouter" else "OPENAI_API_KEY")
    if not key:
        parser.error(f"A {args.provider} API key is required for live evaluations")

    llm = MeteredOpenAI(
        api="chat_completions" if args.provider == "openrouter" else "responses",
        temperature=None, api_key=key,
        endpoint="https://openrouter.ai/api/v1" if args.provider == "openrouter" else None,
        model_kwargs={"default": {"model": args.model}, "ui": {"model": args.model}},
    )
    output = args.output or RESULTS / f"{args.model.replace('/', '-')}-{datetime.now(UTC):%Y%m%d-%H%M%S}.json"
    report = asyncio.run(evaluate(llm, dataset, source_factory, output, args.case, documents_factory, fixtures))
    report.print()
    if report.failures or any(not all(result.value for result in case.assertions.values()) for case in report.cases):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
