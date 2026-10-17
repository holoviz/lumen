# Lumen repository architectural map

## Verification record

| Item | Value |
|---|---|
| Repository requested | `https://github.com/zayn/lumen-forkes` |
| Local checkout | `C:\Users\zainp\Desktop\lumen\lumen-forkes` |
| Branch | `main` |
| Commit | `61c4c7f25c5fe5c2d18d63be0fe2cc0a32748d60` |
| Working tree | Clean before this report was added |
| Upstream identity in package metadata | `https://github.com/holoviz/lumen` (`pyproject.toml`, `[project.urls]`) |
| Local Git remote | None configured in this checkout |
| Existing exploration report | None found before this report |
| Verification status | Import passes. Focused collection passes with 89 tests. Core config/pipeline tests pass with 28 passed and 6 skipped when pytest temp files are placed in the writable workspace. |

This report describes the checked-out commit. It does not assume that the `zayn/lumen-forkes` GitHub repository and this local checkout have the same future history. Source references use repository-relative paths and line numbers from this commit.

## 1. Purpose and product shape

Lumen is an open-source, extensible Python framework for chatting with data, retrieval-augmented generation, declarative data pipelines, visualizations, and dashboards. The README identifies the primary capabilities as SQL generation, document context and embeddings, visualization, inspection/editing of model output, reports, and custom agents, analyses, and tools (`README.md:19-34`). It is built on Panel, so the application model is a Bokeh/Panel server application rather than a standalone web server.

There are two related product surfaces:

1. The declarative dashboard framework. A YAML or Python specification connects sources to pipelines and views. The core concepts document summarizes the intended flow as `Sources -> Pipelines -> Views` (`docs/configuration/spec/concepts.md:14-31`).
2. Lumen AI. The `lumen-ai`/`lumen` command starts a chat-oriented Explorer UI. It resolves input tables, selects or configures an LLM provider, creates a coordinator, and executes plans made of agents and tools (`lumen/command/ai.py:40-211`, `lumen/ai/ui.py:716-722`, `lumen/ai/ui.py:2920-2935`).

The package is a fork or derivative of HoloViz Lumen in naming and metadata: the README badge URLs and package URLs point to `holoviz/lumen`, while the requested repository identity is `zayn/lumen-forkes`. The local checkout has no Git remote, so the exact fork relationship cannot be verified from Git configuration.

## 2. Technology stack and package contracts

The implementation is Python 3.11+ (`pyproject.toml:[project]`, `requires-python`). The package metadata declares support classifiers for Python 3.11 through 3.14. The main runtime stack is:

| Layer | Evidence and role |
|---|---|
| Panel / Bokeh / Param | `panel`, `bokeh`, and `param` are required dependencies. `Dashboard`, `Pipeline`, views, widgets, templates, and server lifecycle use them directly (`pyproject.toml:[project].dependencies`, `lumen/dashboard.py:15-23`, `lumen/pipeline.py:10-23`). |
| HoloViews / hvPlot | Rendering and data-view integrations. The view base and tests cover HoloViews and hvPlot (`lumen/views/base.py`, `lumen/tests/views/`). |
| Narwhals / pandas / NumPy / PyArrow / DuckDB | Dataframe portability, tabular operations, file loading, SQL execution, and schema handling. `Pipeline.DataFrame` accepts pandas or Narwhals-supported frames (`lumen/pipeline.py:66-93`); DuckDB is a required dependency. |
| YAML / JSON Schema / Pydantic | Specification loading and validation, structured AI outputs, and provider contracts (`lumen/config.py:232-238`, `lumen/validation.py`, `lumen/ai/schemas.py`, `pyproject.toml`). |
| LLM and tool layer | OpenAI, Instructor, provider SDK adapters, async tool loops, agents, and vector stores (`lumen/ai/llm.py`, `lumen/ai/agents/`, `lumen/ai/tools/`, `lumen/ai/vector_store.py`). Provider SDKs beyond OpenAI are optional extras. |
| Zensical / mkdocstrings | Documentation build. `pixi.toml` defines `docs-build` and `docs-serve`; `zensical.toml` configures the site. |
| Hatch / hatch-vcs | Packaging and VCS-derived versioning (`pyproject.toml:[build-system]`, `[tool.hatch.version]`). |

The public entry points are `lumen-spec = lumen.command:main`, `lumen-ai = lumen.command.ai:main`, and `lumen = lumen.command.ai:main` (`pyproject.toml:[project.scripts]`). The Panel REST entry point is `lumen.rest:lumen_rest_provider`.

Optional extras define integration boundaries rather than unconditional imports: SQLAlchemy, xarray/xarray-sql, Snowflake, BigQuery, geographic packages, multiple AI providers, ChromaDB, MCP, and local model runtimes (`pyproject.toml:[project.optional-dependencies]`). Source modules generally import optional dependencies inside functions or expose optional-import tests (`lumen/tests/sources/test_optional_imports.py`).

## 3. Installation, startup, build, and development commands

### Documented commands

The README documents `conda install -c pyviz lumen`, `pip install lumen`, and `lumen-ai serve data.csv` (`README.md:39-52`). The contributor guide recommends Pixi, with pip as an alternative (`docs/contributing.md:63-83`):

```text
pixi install
pip install -e "[tests]"
pixi run -e test-312 test-unit
pixi run -e lint lint
pixi run -e docs docs-serve
pixi run -e docs docs-build
```

### Configuration-derived commands

`pixi.toml` is the authoritative task definition for local development:

- `install`: editable install without dependencies.
- `test-unit`: `pytest lumen/tests -n logical --dist loadgroup`.
- `docs-build`: `zensical build --clean`.
- `docs-serve`: `zensical serve`.
- `build-pip`: `python -m build .`.
- `build-conda`: `bash scripts/conda/build.sh`.
- `lint`: `pre-commit run --all-files`.
- `typecheck`: `pyright`.

The CLI has two important routing paths. `lumen.command.main` delegates ordinary arguments to Panel's command parser, while explicit `builder`, `validate`, and `precache` commands are handled by Lumen (`lumen/command/__init__.py:62-118`). The AI command extends Panel's `Serve` command with provider, model, agent, endpoint, validation, and code-execution options (`lumen/command/ai.py:40-142`). YAML files are routed through `YamlHandler`, which constructs `Dashboard(<yaml>).servable()` (`lumen/command/__init__.py:20-59`).

### Observed verification

The following commands were attempted in the current host environment:

```text
python -c "import lumen; print(lumen.__version__)"
pytest --collect-only -q lumen/tests/test_config.py lumen/tests/test_pipeline.py lumen/tests/ai/test_coordinator.py
```

Initially, both commands stopped during package import with `ModuleNotFoundError: No module named 'panel'`. `panel` was then installed, followed by the declared editable test setup `python -m pip install -e ".[tests]"`; this installed the repository's runtime and test dependencies. Import now passes and reports `0.0.post1.dev1664+g61c4c7f25`.

Focused collection passes with 89 tests. The combined focused execution was interrupted after early AI tests stopped making progress. The core execution command initially reported 25 passed, 6 skipped, and 3 environment errors because pytest could not create numbered temporary directories under the sandbox temp path. Rerunning with `pytest -q --basetemp .pytest-basetemp lumen/tests/test_config.py lumen/tests/test_pipeline.py` completed successfully: **28 passed, 6 skipped in 4.14s**. The skips are test/environment conditional; no core assertion failures were observed in this run.

## 4. Repository structure

```text
.
├── lumen/                 Python package
│   ├── command/           CLI, YAML handler, AI server handler, validation/build helpers
│   ├── sources/           File, DuckDB, SQL, REST, Intake, Snowflake, BigQuery, xarray sources
│   ├── filters/           Interactive and constant filter abstractions
│   ├── transforms/        In-memory and SQL transforms
│   ├── variables/         Reactive variables and references
│   ├── views/             View base class and concrete visual/table/download views
│   ├── layout.py          Layout composition and lifecycle
│   ├── dashboard.py       Top-level declarative dashboard and Panel template
│   ├── pipeline.py        Source-to-dataframe filtering and transformation
│   ├── state.py           Shared specification/component state and serialization
│   ├── config.py          YAML expansion, global config, registries, defaults
│   ├── ui/                Non-AI dashboard builder/editor UI
│   └── ai/                LLMs, agents, coordinator, tools, vector stores, Explorer UI
├── lumen/tests/           79 pytest files across core, sources, views, transforms, and AI
├── docs/                  Zensical documentation and API/configuration guides
├── examples/              Gallery YAML specifications and AI examples
├── scripts/               Embedding and Conda build helpers
├── .github/workflows/     Test, build, docs, gallery, lock, and nightly lock workflows
├── pyproject.toml         Package metadata, dependencies, entry points, pytest and Ruff config
├── pixi.toml              Reproducible environments and development tasks
└── zensical.toml          Documentation site configuration
```

The largest coupling hotspots are intentionally central: `lumen/ai/llm.py` contains the common LLM protocol and many provider implementations; `lumen/ai/ui.py` coordinates UI, source context, explorations, and plan execution; `lumen/views/base.py` and `lumen/sources/base.py` contain broad registry/serialization behavior. Changes to these files have a wide blast radius.

## 5. Core declarative architecture

### Component and specification model

The specification contract is a YAML-compatible mapping with top-level keys `variables`, `auth`, `defaults`, `config`, `sources`, `pipelines`, and `layouts`. `Dashboard._valid_keys` confirms this contract (`lumen/dashboard.py:528-534`). `load_yaml` expands template variables and parses YAML (`lumen/config.py:232-238`). Validation and `from_spec` methods on each component materialize the mapping into Param-backed Python objects.

The state object is the shared registry for specification, variables, sources, pipelines, and global references (`lumen/state.py:1-60`). It supports serialization through `state.to_spec` and component materialization. This is a key invariant: source and pipeline names in specifications are references into shared state, not only local constructor arguments.

### Dependency direction

```text
YAML/Python specification
          |
          v
config.load_yaml -> Dashboard.validate -> state.spec
          |
          +--> Variables / references
          +--> Sources -> tables / schemas / external data
          +--> Pipelines -> filters -> transforms -> dataframe/query
          +--> Layouts -> Views / Downloads
          |
          v
Panel template -> Bokeh document -> browser session
```

`Dashboard.__init__` validates or loads the specification, sets the root, loads local modules, resolves views, creates `AuthSpec`, `Config`, `Defaults`, variables, and the Panel template (`lumen/dashboard.py:536-600`). `_materialize_specification` loads global sources and pipelines, checks authorization, creates layouts, and starts them; background loading is available for inactive tabs (`lumen/dashboard.py:672-712`). `Dashboard.__panel__` returns the template layout and `servable()` marks it for Panel serving (`lumen/dashboard.py:609-610`, `1126-1130`).

### Sources

`Source` is the input abstraction. `Source.from_spec` resolves a string reference from `state.sources` or constructs the registered source type from a mapping (`lumen/sources/base.py:401-431`). The base class provides schema, table, reference, and lifecycle behavior; concrete implementations include file/REST loading, DuckDB, SQLAlchemy, Intake, Dremio, BigQuery, Snowflake, Prometheus, and xarray SQL (`lumen/sources/`).

The AI CLI also has an input resolver that maps local files, URLs, database connection strings, geospatial files, xarray files, and DuckDB/SQLite databases to source objects. The resolver is exposed by `UI._resolve_data` (`lumen/ai/ui.py:552-713`). It creates an in-memory DuckDB source for ordinary file inputs and installs/loads `httpfs` for remote files when needed (`lumen/sources/base.py:640-674`). This creates a practical boundary: input detection is user-facing and broad, while source classes remain the reusable data access contract.

### Pipelines, filters, and transforms

`Pipeline` consumes a `Source` table or another pipeline and applies filters and transforms (`lumen/pipeline.py:108-121`). `DataFrame.__get__` lazily refreshes stale data when `auto_update` is enabled (`lumen/pipeline.py:66-93`). `_update_data` is the central execution path; it evaluates source data, applies filters and transforms, updates schema/data, and emits reactive events (`lumen/pipeline.py:432-523`). `Pipeline.from_spec` materializes source references, filters, transforms, and parameters (`lumen/pipeline.py:524-560`).

Filters represent user or fixed constraints. Transforms represent structural/data operations, with SQL-specific subclasses where a source can push operations into SQL (`lumen/pipeline.py:24-34`, `lumen/transforms/base.py`, `lumen/transforms/sql.py`). A pipeline can therefore be backed by in-memory dataframe operations or SQL compilation, and a change to filter/transform contracts can affect both execution modes.

### Views and layouts

Views consume a pipeline and turn its data into Panel/HoloViews/hvPlot/table/download output. `View.from_spec` resolves the registered view type and references (`lumen/views/base.py:460-530`). A view listens to pipeline/reference changes and updates its Panel object (`lumen/views/base.py:193-293`). `Layout.from_spec` composes one or more pipelines and views, while `Dashboard._load_layout` starts the layout and watches its `rerender` event (`lumen/dashboard.py:653-666`, `lumen/layout.py:835-880`).

The main reactive path is:

```text
widget / variable event
        -> Filter or reference event
        -> Pipeline stale/update
        -> Source load and transforms
        -> View update / layout rerender
        -> Panel/Bokeh document patch
```

`Dashboard._update_pipelines` explicitly triggers pipeline updates for all materialized layouts when manual update mode is enabled (`lumen/dashboard.py:899-908`). Shared filters are collected into the dashboard sidebar (`lumen/dashboard.py:880-928`).

## 6. AI architecture and execution flows

### AI startup

`lumen-ai` extends Panel's server command. It validates local table paths, skips local validation for URLs and database connection strings, selects a provider explicitly or from environment variables, parses model/model-kwargs/Hugging Face URL options, warms the provider, and installs a Bokeh application handler (`lumen/command/ai.py:101-211`). `AIHandler` renders `lumen/command/app.py.jinja2` into an in-memory Python application that constructs the Explorer UI with the requested configuration (`lumen/command/ai.py:222-326`).

Provider selection is centralized in `LLM_PROVIDERS` and `get_available_llm` (`lumen/ai/llm.py:163-188`, `237-257`). `Llm` is the common Param-based provider abstraction. Provider subclasses adapt SDK clients, structured output modes, streaming deltas, tool calls, model discovery, endpoint configuration, and caching. `OpenAI`, Anthropic, Google, Bedrock, Mistral, local/runtime providers, and CLI-backed providers are all part of this shared file. The optional dependency matrix in `pyproject.toml` is part of the provider contract.

### Explorer UI and coordinator

`UI` configures data context, LLM, agents, tools, analyses, vector stores, and a coordinator. During initialization it adds default agents, appends an analysis agent when analyses are configured, combines configured tools with coordinator tools, constructs the coordinator, and adopts its vector stores (`lumen/ai/ui.py:1198-1227`). A chat message reaches `_chat_invoke`, which passes messages and the current context to `Coordinator.respond` (`lumen/ai/ui.py:2920-2935`).

The coordinator is a planner/executor boundary:

```text
ChatInterface message
       |
       v
Coordinator.respond
       |
       +--> pre-plan / relevance / clarification
       +--> planner chooses agent/tool tasks
       +--> Plan contains ordered Actor/Agent tasks
       v
Plan.execute -> agent/tool async work -> outputs/context
       |
       +--> Pipeline/View/Report outputs
       +--> exploration conversation and view state
```

`Coordinator` owns agent/tool registration, context synchronization, LLM calls, plan generation, and Panel rendering (`lumen/ai/coordinator/base.py:334-671`). `Agent` is the principal extension point. It is a Panel `Viewer`, `ToolUser`, and `ContextProvider`; subclasses implement `applies` and `respond`, commonly using prompt templates and streaming (`lumen/ai/agents/base.py:20-29`, `87-141`). The planner adds dependency resolution and ordered execution (`lumen/ai/coordinator/planner.py:199-659`).

### Tools, vector stores, and external services

Tools are async actors with `applies`, `prepare`, `sync`, and `respond` contracts (`lumen/ai/tools/base.py:1-300`). Vector lookup tools use a shared or document-specific `VectorStore`; the base vector-store API supports add/upsert/query and multiple backends, including NumPy and DuckDB implementations (`lumen/ai/vector_store.py:134-626`, `723-948`, `1127-1526`, `1731-2091`). Metadata and document lookup tools populate stores from source/table descriptions and document chunks (`lumen/ai/tools/metadata_lookup.py`, `document_llm_tools.py`). External integrations include LLM provider APIs, database/filesystem sources, optional MCP servers, and optional provider-specific services.

Code-generating agents have an explicit security boundary. `BaseCode` supports hidden, disabled, prompt, LLM-validated, and allow modes; the CLI documents that prompt/LLM/allow modes execute model-generated code and must not be enabled in production with secrets or sensitive data (`lumen/command/ai.py:70-99`, `lumen/ai/agents/base_code.py:50-248`). This is a major operational invariant and regression-sensitive area.

### Exploration persistence and error recovery

Explorer UI groups work into explorations. `_execute_plan` decides whether to append to an existing exploration or create a new one, executes the plan, then post-processes outputs (`lumen/ai/ui.py:2787-2832`). Successful output synchronizes new sources into global context and updates views. Failed plans retain error information and add retry/rerun/replan controls; `_postprocess_exploration` also asks the LLM to produce a user-facing error explanation (`lumen/ai/ui.py:2847-2919`). This means UI state, plan state, context, and conversation state are coupled at this boundary.

## 7. Public interfaces and extension points

The main extension contracts are:

- `Source`, `Source.from_spec`, source type registration, table/schema loading, and `to_spec`.
- `Pipeline`, `Filter`, and `Transform`, including SQL transform variants.
- `View`, `Layout`, and Panel-compatible `__panel__` methods.
- `Variable`, `Variables`, and reference expansion.
- `Dashboard`, `Config`, `Defaults`, `AuthSpec`, YAML serialization, and `servable()`.
- `Agent`, `Tool`, `ToolUser`, `Coordinator`, `Plan`, and `ContextProvider`.
- `Llm` and provider subclasses with model selection, structured output, streaming, and tool-call behavior.
- `VectorStore` with `add`, `upsert`, `query`, and lifecycle methods.

The documented extension guide explicitly describes Sources, Agents, Tools, and Analyses as subclassable Python classes and recommends the Panel Extension Copier Template (`docs/extending.md:1-31`). The YAML contract is the stable cross-language boundary: names, `type`, references, and `to_spec` output must remain compatible with validation and round-tripping.

## 8. Architectural blast radius and risk assessment

### High-impact areas

| Area | Confirmed consumers | Change risk |
|---|---|---|
| `state` registries and reference resolution | Dashboard, sources, pipelines, views, variables, AI context | Name/reference changes can break YAML materialization, serialization, and AI-generated pipelines. |
| `Source` and schema/dataframe boundary | All pipelines, filters, views, source agents, metadata/vector lookup | Dataframe type, schema, or table-name changes can break both dashboard rendering and AI planning. |
| `Pipeline._update_data` | Views, layouts, UI filters, SQL agents, analyses | Reactive update, lazy loading, and backend parity are central invariants. |
| `View`/`Layout` serialization and Panel lifecycle | Dashboard rendering, UI-generated views, tests, docs examples | A small Panel or Param lifecycle change can affect every dashboard surface. |
| `Llm` / `Message` / tool-loop protocol | All agents, coordinator, providers, usage/trace capture | Provider-specific changes can appear as planner, streaming, or structured-output failures. |
| `Coordinator` / `Plan` | Explorer UI, all agents and tools | Ordering, retries, context propagation, and error handling affect the complete AI product. |
| `ExplorerUI` context and exploration state | Chat, source sync, generated pipelines/views/reports | High cohesion pressure: UI state and execution state are intentionally combined, so regressions can be cross-cutting. |

### Confirmed coupling and possible coupling

Confirmed coupling is visible in imports and calls: `Dashboard` imports and constructs sources, pipelines, layouts, variables, filters, views, and shared `state` (`lumen/dashboard.py:25-43`); `Pipeline` imports source/filter/transform abstractions (`lumen/pipeline.py:24-34`); `UI` constructs coordinator, agents, tools, and vector stores (`lumen/ai/ui.py:1198-1227`).

Possible downstream impacts require targeted tests. For example, changing a provider response shape may affect all provider adapters, but the exact impact depends on the provider mode and the agent's response model. Changing a source schema may affect SQLAgent and metadata indexing only when those tools are enabled. These are risk hypotheses, not confirmed dependencies.

### Fragile boundaries and single points of failure

- The single `llm.py` module is a high-coupling provider hub. A shared message, streaming, or tool-call change can regress many providers.
- `Dashboard` and `ExplorerUI` are composition roots. They combine configuration, state, lifecycle, UI, and error handling, so unit-level confidence does not fully cover integration behavior.
- Global/shared state is convenient for references and serialization but can create cross-session or test-order risks if state is not reset. Tests under `lumen/tests/conftest.py` and `lumen/state.py` should be inspected first when diagnosing leakage.
- YAML parsing uses `yaml.Loader` (`lumen/config.py:232-238`). Specifications are treated as trusted application configuration; loading untrusted YAML should be treated as a security risk.
- Generated code execution is an explicit risk boundary, controlled by CLI/UI settings and provider output. Changes must preserve the default-safe behavior.
- Background layout loading uses a thread pool (`lumen/dashboard.py:693-712`). Changes to shared state or Panel document access must consider thread/session boundaries.

## 9. Testing and CI architecture

There are 79 test files under `lumen/tests`, covering core dashboard/config/auth/schema/layout/pipeline/variables, source backends, filters/transforms, views, and AI coordinator/agents/tools/LLMs/vector stores/UI. The directory structure demonstrates broad coverage areas, but file presence alone does not prove every path is covered.

`pyproject.toml` configures pytest with strict config and markers, doctest modules, warning-as-error behavior, asyncio auto mode, and a minimum pytest version of 7. `pixi.toml` adds pytest-asyncio, xdist, rerunfailures, coverage, and the `test-unit` task. The CI workflow runs pre-commit, then a matrix of unit/core/downstream suites across operating systems and Python 3.11-3.14 (`.github/workflows/test.yaml:1-180`). Coverage is uploaded through Codecov in the unit job.

The test layout is strongest around component contracts and provider/backend-specific behavior. Important regression-sensitive suites include:

- `lumen/tests/test_pipeline.py`, `test_schema.py`, `test_layout.py`, and `test_dashboard.py` for core lifecycle and specification behavior.
- `lumen/tests/sources/` for source materialization, optional imports, SQL, DuckDB, Intake, Snowflake, SQLAlchemy, and xarray.
- `lumen/tests/transforms/` and `lumen/tests/validation/` for transformation and spec validation contracts.
- `lumen/tests/ai/test_coordinator.py`, `test_tool_loop.py`, `test_llm.py`, `test_agents.py`, `test_vector_store.py`, `test_ui.py`, and `test_sql.py` for AI execution paths.

The build workflow creates pip and Conda artifacts and verifies that the built wheel imports (`.github/workflows/build.yaml:1-105`). The docs workflow builds with Zensical and publishes generated documentation (`.github/workflows/docs.yaml:1-75`). Gallery deployment runs after package workflow completion and solves the example project (`.github/workflows/gallery.yaml`).

Recommended pre-change and post-change checks:

```text
pixi install
pixi run -e test-core test-unit
pixi run -e test-312 test-unit
pixi run -e lint lint
pixi run -e lint typecheck
pixi run -e docs docs-build
pixi run -e build build-pip
```

Use the smallest relevant test file first, then the complete environment task. For AI-provider changes, run the AI tests in an environment with the required optional extras and use mocked/provider-independent tests where possible. For source changes, include the relevant backend and optional-import tests.

## 10. Known limitations, assumptions, and open questions

1. The local Git remote is absent. The requested fork's default branch, upstream divergence, and pull-request context are therefore unverified.
2. Runtime verification was blocked by missing `panel`. No claim is made that the full suite passes in a correctly provisioned Pixi environment.
3. Dependency versions are mostly lower bounds or wildcards in `pyproject.toml` and `pixi.toml`; the resolved lockfile was not present in the listed repository files. Exact runtime versions for this commit are therefore unverified.
4. CI uses pinned external GitHub actions and hosted environments. Local behavior can differ from CI because of OS, Python, browser/display, optional SDK, and service availability.
5. Some AI flows depend on external LLM credentials, endpoints, database connections, or vector-store backends. The source code documents contracts, but live service behavior needs integration verification.
6. The architecture uses both the legacy/dashboard `lumen` surface and the newer AI Explorer surface. Their state and serialization boundaries overlap, but not every cross-surface interaction was exercised in this read-only pass.
7. The report identifies likely coupling from imports, constructors, and tests. A full dependency graph, runtime trace, memory profile, and session-concurrency audit would require executing the provisioned application.

## 11. Practical architectural summary

Understand these components first: `lumen/dashboard.py` for the declarative application lifecycle; `lumen/config.py` and `lumen/state.py` for specification and reference semantics; `lumen/sources/base.py`, `lumen/pipeline.py`, and `lumen/views/base.py` for the data-to-visual path; then `lumen/ai/ui.py`, `lumen/ai/coordinator/base.py`, `lumen/ai/agents/`, `lumen/ai/tools/`, and `lumen/ai/llm.py` for the AI path.

The safest issue-investigation starting points are the nearest contract test, the `from_spec`/`to_spec` pair for the affected component, and the composition-root call site in Dashboard or Explorer UI. For source or pipeline problems, trace `Source.from_spec -> Pipeline._update_data -> View._update_panel`. For AI problems, trace `ExplorerUI._chat_invoke -> Coordinator.respond -> Plan.execute -> Agent/Tool -> exploration post-processing`.

The most regression-sensitive areas are shared state and references, source schema/dataframe conversion, pipeline reactivity, view/layout serialization, provider message/tool-loop normalization, coordinator plan execution, background Panel lifecycle, and generated-code safety defaults.

Before changes, run the focused test and inspect the current `to_spec`/`from_spec` contract. After changes, run the focused test, the relevant Pixi test environment, lint/type checks, and documentation/build checks when public configuration or packaging changed. The minimum repository-defined full check is `pixi run -e test-312 test-unit`; the CI matrix provides broader OS/Python coverage.
