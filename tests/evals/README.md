# Lumen AI evaluations

The evaluation suites run Lumen's agents against live models and score the answers. They are
for development: the unit tests in this directory run offline as part of `pixi run test-unit`,
while the suites themselves call a model and need an API key.

The harness lives in `lumen.ai.evals` and needs the `evals` extra (`pip install "lumen[evals]"`);
the `evals` and `test-*` pixi environments include it.

## Running the local suite

The local suite asks questions about small fixtures (in-memory tables, `fixtures/orders.csv` and
`fixtures/refunds.json`) and checks actors, SQL results, charts, tool calls and answers.

```bash
export OPENAI_API_KEY=...
pixi run -e evals python -m tests.evals                   # all local cases, gpt-4.1-mini
pixi run -e evals python -m tests.evals --case sales_total
pixi run -e evals python -m tests.evals --api chat_completions
```

To go through OpenRouter, set `OPENROUTER_API_KEY` and pass `--provider openrouter` with an
OpenRouter model id, e.g. `--model openai/gpt-4.1-mini`. OpenRouter defaults to the Chat
Completions API; pass `--api responses` to use its Responses API instead.

The command exits with status 1 when any case fails or errors. `python -m tests.evals --help`
lists every option.

## Other subjects and outcome scoring

By default the suites evaluate Lumen's `ExplorerUI`. `--subject` selects another registered
subject, or any class implementing the subject protocol given as `package.module:Class`.
`--scoring outcome` scores only the result (status, answer, SQL, rows, charts, errors) and keeps
Lumen-specific architecture checks such as actors and tool calls as diagnostics, which is how to
compare a subject that does not share Lumen's agents. The unit tests ship an offline subject with
canned answers:

```bash
pixi run -e evals python -m tests.evals --subject tests.evals.test_harness:FakeSubject --scoring outcome
```

## Running BIRD Mini-Dev

The BIRD suite scores text-to-SQL execution accuracy on
[BIRD Mini-Dev](https://github.com/bird-bench/mini_dev) SQLite questions. It needs the
question file and the SQLite databases, which are not checked in:

```bash
# Revision-pinned question JSON from Hugging Face
pixi run -e evals python -m tests.evals --download-bird-questions tests/evals/results/bird-mini-dev-sqlite.json

# Databases: download minidev.zip from the Mini-Dev README and extract it
unzip minidev.zip -d tests/evals/results
```

Any directory that contains `<db_id>/<db_id>.sqlite`, `dev_databases/<db_id>/<db_id>.sqlite` or
`minidev/MINIDEV/dev_databases/<db_id>/<db_id>.sqlite` works for `--bird-databases`.

```bash
BIRD="--suite bird --bird-questions tests/evals/results/bird-mini-dev-sqlite.json --bird-databases tests/evals/results"
pixi run -e evals python -m tests.evals $BIRD                    # six default questions
pixi run -e evals python -m tests.evals $BIRD --bird-stratified  # 33 questions across all databases
pixi run -e evals python -m tests.evals $BIRD --bird-sample      # 100 questions, stratified the same way
pixi run -e evals python -m tests.evals $BIRD --bird-all --workers 8 --output tests/evals/results/full.json
pixi run -e evals python -m tests.evals $BIRD --bird-all --resume --output tests/evals/results/full.json
```

`--bird-all`, `--bird-stratified`, `--bird-sample` and `--bird-ids` run each case in its own process with a
240 second limit and checkpoint after every case, so an interrupted run can continue with
`--resume`. A case whose gold SQL cannot run is reported as unscorable rather than as a miss.

## Results and the dashboard

Each run writes a JSON file to `tests/evals/results/` (ignored by git) unless `--output` names
another path. The file records the commit, whether the working tree was dirty, the model, API,
provider and a fingerprint of the cases and instructions, followed by every turn's answer, SQL,
rows, tool calls, usage and assertion results.

To compare runs, serve the dashboard:

```bash
pixi run -e evals panel serve tests/evals/dashboard.py --show
```

It groups runs by suite and case set, plots pass rate against estimated cost and runtime as well
as token usage per run and per case, and shows each case's turns, SQL and errors for every run
that attempted it.
