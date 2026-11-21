# AGENTS.md

Guidance for AI coding agents working in this repository.

## What Lumen is

Lumen is a general-purpose, open-source framework for chatting with data
and building data apps — not a tool for one dataset, one chart type, or
one workflow. It has two faces that share one declarative data model:

- **Lumen AI** (`lumen-ai serve data.csv`): an agent-based chat interface
  that turns natural-language questions into data pipelines, charts,
  tables, and dashboards, with validation and editing of every LLM output.
- **Declarative dashboards**: YAML specifications that describe sources,
  pipelines, filters, variables, and views, rendered as interactive
  Panel apps (see `docs/getting_started/`).

Because the data model is declarative, a pipeline or dashboard can be
serialized, shared, reproduced in a notebook, or composed into a larger
app. When adding a feature, keep it general: prefer a reusable component,
agent, or option over a special case for one data source or one question
type.

## Repository layout

- `lumen/` — the Python package (Python >= 3.11, built on
  [Panel](https://panel.holoviz.org), `param`, and Narwhals).
  - `pipeline.py` — `Pipeline`, the core object: a source plus a chain of
    transforms, filters, and variables producing a queryable table.
  - `dashboard.py` — dashboard specification loading and rendering.
  - `config.py`, `state.py`, `schema.py`, `validation.py`, `base.py` —
    global configuration, app state, spec schemas, validation, and shared
    base classes.
  - `sources/` — data sources (`base.py`, `duckdb.py`, `intake.py`,
    `sqlalchemy.py`, `snowflake.py`, `bigquery.py`, `ae5.py`, ...).
  - `transforms/` — data transforms (`base.py`, `sql.py`).
  - `views/` — visualizations and tables, all defined in `base.py`
    (`VegaLiteView`, `DeckGLView`, `AltairView`, `hvPlotBaseView`,
    `StringView`, `IndicatorView`, `PerspectiveView`, ...).
  - `filters/`, `variables/` — interactive filter and variable components.
  - `data/` — data catalog, contracts, quality checks, sharing, SQL
    helpers, and statistics (`catalog.py`, `contracts.py`, `quality.py`,
    `sharing.py`, `sql.py`, `stats.py`).
  - `ui/` — the Explorer/builder UI served by `lumen-ai serve`.
  - `ai/` — the Lumen AI agent system:
    - `agents/` — the agents (`base.py`, `sql.py`, `source.py`,
      `vega_lite.py`, `hvplot.py`, `validation.py`, `chat.py`,
      `analysis.py`, ...). Each agent handles one step, e.g. planning a
      query, writing SQL, or building a chart.
    - `llm.py` — LLM provider abstraction; `models/` — structured-output
      models; `coordinator/` — agent orchestration; `controls/` — source
      loaders and other UI controls agents can drive.
    - `validation.py`, `checks.py`, `decisions.py` — checking LLM outputs
      before they are shown; `embeddings.py` — document context (RAG);
      `export.py`, `notebook.py` — exporting answers to notebooks.
    - `evals/` — evaluation harnesses for the agent system.
  - `auth.py`, `rest.py`, `downloads.py`, `serializers.py` — auth, REST
    API, downloads, and spec serialization.
- `lumen/tests/` — unit tests, mirroring the package layout
  (`lumen/tests/ai/`, `lumen/tests/sources/`, `lumen/tests/views/`, ...).
- `tests/evals/` — agent evaluation tests.
- `docs/` — documentation (built with Zensical): `getting_started/`,
  `configuration/`, `reference/`, `examples/`, plus `contributing.md`,
  `extending.md`, and `faq.md`. User-facing docs live here, not in
  docstrings alone.
- `examples/` — example dashboards and specs.
- `scripts/` — developer scripts (`evals/`, Vega-Lite doc embeddings).
- `homepage/` — the project homepage sources.
- `pixi.toml`, `pyproject.toml` — environment and packaging metadata.

## Development workflow

Pixi is the recommended setup (see `docs/contributing.md`):

```bash
pixi install                        # create the dev environment
pixi run -e test-312 test-unit      # run the test suite
pixi run -e lint lint               # run pre-commit linting on all files
pixi run -e docs docs-serve         # serve docs locally with live reload
pixi run -e docs docs-build         # build docs
```

Without pixi: `pip install -e ".[tests]"` and run
`pytest lumen/tests tests/evals -n logical --dist loadgroup`.

Key environments in `pixi.toml`: `test-311`–`test-314` (full test suite
with AI/SQL extras), `test-core` (minimal), `docs`, `lint`, `build`.

## Conventions

- **Style**: PEP 8, type hints where possible, docstrings on
  functions/classes. Ruff (`line-length = 165`, checks on `lumen/`
  only — `tests` are excluded) plus isort, enforced via pre-commit
  (`.pre-commit-config.yaml`); run `pixi run -e lint lint` before
  pushing. No `breakpoint()` calls in committed code.
- **Tests**: add tests for new features and bug fixes; keep the suite
  green and aim for >80% coverage. Reproduce a bug with a failing test
  before fixing it where possible.
- **Components are `param`-based**: sources, transforms, views,
  filters, and variables are declarative, parameterised classes that
  round-trip through YAML specs — keep new options serializable and
  document them in `docs/`.
- **AI controls contract**: in `lumen/ai/controls` source loaders,
  `outputs["sources"]` must not contain duplicate source objects,
  `outputs["source"]` must reference an item in `outputs["sources"]`,
  and `outputs["table"]` should reflect the most recently active table.
- **Docs**: update `docs/` for user-facing changes and add examples
  where they help; see `docs/extending.md` for extension points.
- **Commits**: clear messages in the present tense, referencing issues
  (`Fixes #123`). One logical change per PR; link the issue it closes.

## Getting help

- Contributing guide: `docs/contributing.md`
- Extension guide: `docs/extending.md`
- Community: [Discourse](https://discourse.holoviz.org/c/lumen/),
  [Discord](https://discord.com/invite/rb6gPXbdAr)
