# Architecture

This page explains how **Lumen AI** (`lumen/ai`) works internally: how a chat prompt becomes a structured plan, how retrieval (RAG) narrows the schema the model sees, how steps hand data to each other via typed contracts, how the LLM layer and prompts are organised, how generated code is validated before it runs, how interactive editors bridge human and AI decisions, and how results are exported. It is written for contributors, reviewers, and anyone debugging Lumen AI.

Configuration and usage are covered separately in the [Agents](configuration/agents.md), [Context](configuration/context.md), [Coordinators](configuration/coordinators.md), [Tools](configuration/tools.md), and [UI](configuration/ui.md) pages. This page covers how the underlying pieces connect.

> **Scope.** This page covers the AI layer only. The core Lumen engine (`Source`, `Pipeline`, `Filter`, `View`, YAML specs) is only summarised in [Section 15](#15-background-the-core-reactive-engine).

---

## 1. How the pieces fit

Lumen AI does not just answer in chat text. It produces **live, editable Lumen components** (pipelines, views, tables) wrapped in editors that the user can inspect and change directly.

The system has seven layers:

```
+-----------------------------------------------------------------------------------+
|                              1. USER INTERFACE LAYER                              |
|   ExplorerUI / ChatUI (chat, explorations, tabs)  <==>  LumenEditor subclasses    |
|   Source controls (upload, URL, REST, catalog, ...)                               |
+------------------------------------------+----------------------------------------+
                                           | prompt + context
                                           v
+-----------------------------------------------------------------------------------+
|                       2. COORDINATION & PLANNING LAYER                            |
|  Planner (LLM plans whole workflow)  |  DependencyResolver (backward chaining)    |
|  -> Plan (Section of sequential ActorTasks)                                       |
|  planner tools: MetadataLookup, SourceLookup                                      |
+-------------------+------------------------------------+--------------------------+
                    |                                    |
                    v                                    v
+----------------------------------+  +---------------------------------------------+
|    3. RAG & KNOWLEDGE LAYER      |  |           4. ACTORS & CONTRACTS             |
|                                  |  |                                             |
| VectorStore (Numpy / DuckDB /    |  | Actor = Agent | Tool                        |
|   ChromaDB) + Embeddings         |  | ContextModel input/output schemas           |
| Metaset (relevant schema subset) |  | Agents: SQL, hvPlot, VegaLite, DeckGL, ...  |
+----------------------------------+  +----------------------+----------------------+
                                                             |
                    +----------------------------------------+
                    v                                        v
+----------------------------------+  +---------------------------------------------+
|     5. LLM ABSTRACTION LAYER     |  |      6. CODE VALIDATION & EXECUTION         |
|                                  |  |                                             |
| Llm base class + provider        |  | SQL: read-only check, retries, timeout      |
| subclasses, instructor-based     |  | Python/Altair/PyDeck: AST check, optional   |
| structured output, model routing |  | LLM safety review, optional user approval   |
| Jinja2 prompt templates          |  | Monty sandbox tool (optional)               |
+----------------------------------+  +----------------------+----------------------+
                                                             | yields LumenEditors
                                                             v
+-----------------------------------------------------------------------------------+
|                        7. CORE LUMEN & EXPORT                                     |
|    Source --> Pipeline (SQL transforms / filters) --> View                        |
|    export: Jupyter notebook, per-editor YAML / HTML / image / table export        |
+-----------------------------------------------------------------------------------+
```

Lumen AI decides *what to build*. A coordinator plans a sequence of actors, each declaring a contract for what it consumes and produces. Actors run, call the LLM, execute SQL or (optionally) code, and return Lumen components wrapped in `LumenEditor` objects. The user can edit those specs directly, and core Lumen re-renders without another LLM call.

**Observability** (logging, tracing, usage accounting) cuts across all layers and is covered in [Section 12](#12-observability-logging-and-usage).

---

## 2. Coordinators: from prompt to Plan

Follow one prompt, such as *"plot sales by region"*, through the UI:

```
  user prompt
      |
      v
+-----------------------------------------------------------------+
| UI._chat_invoke  ->  coordinator.respond(messages, context)     |
+---------------------------------+-------------------------------+
                                  v
+-----------------------------------------------------------------+
| Planner  (planning phase)                                       |
|  1. _pre_plan      classify follow-up, check clarification,     |
|                    run planner tools (MetadataLookup -> metaset)|
|  2. _make_plan     prune actors, ask the LLM for a RawPlan      |
|  3. _resolve_plan  merge repeated actors, build Plan            |
|  4. plan.validate  static check; ContextError -> plan again     |
+---------------------------------+-------------------------------+
                                  v  Plan
+-----------------------------------------------------------------+
| ExplorerUI._execute_plan                                        |
|   new exploration, or merge into the current one                |
|   plan.param.watch(partial(_add_views, exploration), "views")   |
|   await plan.execute()                                          |
+---------------------------------+-------------------------------+
                                  v  for each task, in order
+-----------------------------------------------------------------+
| Plan._run_task  ->  actor.execute(...)  ->  (outputs, task_ctx) |
|   * check task.output_schema.__required_keys__ in task_ctx      |
|   * merge task_ctx into the shared context                      |
|   * outputs (LumenEditors) -> plan.views -> _add_views -> Tabs  |
+-----------------------------------------------------------------+
```

### Common coordinator behaviour (`lumen/ai/coordinator/base.py`)

`Coordinator` is the base class of `Planner` and `DependencyResolver`. It:

- Instantiates the agents, and pushes the shared `llm` and any `llm_tools` onto each agent.
- Creates a `NumpyVectorStore` if none is supplied. The document store defaults to the same store. Both are placed in `context` so LLM tools can use them.
- Adds `MetadataLookup` automatically if no configured tool provides `metaset`. It adds `SourceLookup` if a `SourceAgent` is present and no tool provides `source_actions`.
- In `respond`, serializes recent chat history, builds the `agents` and `tools` dictionaries, and calls `_pre_plan` and then `_compute_plan`.
- `_pre_plan` (base) drops any agent or tool whose `exclusions` keys are already in the context or whose `applies(context)` returns `False`.
- `validation_enabled` defaults to `False` on the coordinator.

### Planner (`coordinator/planner.py`)

The `Planner` asks the LLM to produce the whole workflow up front.

1. **`_pre_plan`** runs three things in order:
   - **Follow-up classification** (`_check_follow_up_question`). Only runs if `data` is already in the context. The `follow_up` prompt returns one of `"direct"`, `"derived"` or `"new"`. A `"new"` result removes `pipeline` from the context so the plan starts fresh.
   - **Clarification check** (`_check_clarification_needed`). A lightweight yes/no prompt. If it says yes, a clarification LLM tool is made available to the planner. Any clarifications are later appended to the last user message as `[For Clarity]: ...`.
   - **Planner tools** (`_execute_planner_tools`). `planner_tools` defaults to `[MetadataLookup]`. A tool runs if its `applies()` is true and it is either `always_use` or judged relevant by the `tool_relevance` prompt. The resulting contexts are merged with last-write-wins (`LWW`) into the shared context.
2. **`_make_plan`** prunes candidates, then asks the LLM:
   - Computes `all_provides` (every key any agent or tool can output, plus keys already in context).
   - Drops agents whose `applies(context)` is false.
   - Drops agents whose required input keys cannot be satisfied by `all_provides`.
   - Always drops `ValidationAgent` (see [Section 5](#special-handling-of-validationagent)).
   - Drops tools whose required inputs are unsatisfiable.
   - Narrows to candidates that can provide any still-unmet dependency.
   - Builds a Pydantic model with `make_plan_model(agents, tools)` and requests a structured `RawPlan`.
3. **`_resolve_plan`** converts the `RawPlan` to a `Plan`:
   - Consecutive steps that use the same actor are merged into one step.
   - Steps become `ActorTask`s.
   - If `validation_enabled` is true and `ValidationAgent` is available and not already in the plan, a final "Validating results" task is appended.
4. **`plan.validate()`** statically checks contracts (see [Section 6](#static-vs-runtime-validation)). On `ContextError`, the error text is shown, the attempt count increases, and the planner tries again. After more than five failed attempts, planning raises an error.

### DependencyResolver (`coordinator/dependency.py`)

`DependencyResolver` plans backwards from a goal instead of generating the whole plan:

1. It filters agents and tools by `applies()` and asks the LLM to pick one **primary** agent.
2. It loops while the current actor has unmet requirements. `requirements()` defaults to **all keys annotated on `input_schema`** (including `NotRequired` ones), and "unmet" means not already in the context.
3. For each round, it asks the LLM to choose a provider for the unmet keys, and records an `ActorTask` for it.
4. The collected provider tasks are reversed, so producers run first, and the primary agent's task is appended last.

See [Coordinators](configuration/coordinators.md#how-coordinators-work) for guidance on choosing between the two. `Planner` is the default in the UI.

### Execution engine (`coordinator/base.py`, `report.py`)

- `Plan` inherits from `Section`, which is a `TaskGroup` of `ActorTask`s. Tasks run **sequentially in the order fixed during planning**. It is an ordered list, not a dynamic runtime DAG.
- `Plan._run_task` builds a sub-context for the task, renders a roadmap of completed, current and pending tasks into the history, runs `task.execute(...)`, then checks that all keys in `task.output_schema.__required_keys__` are present in the returned `task_context`. A missing key raises `RuntimeError: ... task failed to provide declared context`.
- **Self-healing on `MissingContextError`.** `Plan._handle_task_execution_error` looks for the nearest earlier task whose `output_schema` provides **`pipeline`** (`_find_context_provider`). If found, `_retry_from_provider` resets and re-runs the tasks from there, adding the error as feedback to the user message ("Try a different approach"). If no provider is found, the error is shown in the UI and the plan fails with `__error__` in the context.
- `Plan` also carries `history`, `is_followup`, `abort_on_error`, and `steps_layout` (the live checklist card).
- `Plan.merge` (in `report.py`) lets the UI fold a follow-up plan into the previous exploration's plan.

### Rendering views into the UI

`plan.views` is populated from the **`outputs`** each actor returns (not from `task_context`):

- `TaskGroup._init_views` gathers actor outputs reactively.
- `ExplorerUI._add_views` keeps only `LumenEditor` instances, renders each one, and appends it as a tab in the exploration. The first `SQLEditor` of a new exploration also populates the "Data Source" tab.
- The watcher is attached in `_execute_plan` for the rerun, new-exploration and replan branches.

---

## 3. RAG, schema discovery, and the Metaset

Large databases can have hundreds of tables. Putting the full catalog into the prompt hurts reasoning quality and inflates cost, so Lumen AI retrieves only the relevant part.

```
Sources (tables, columns, descriptions)
       |
       v  index
VectorStore  <-- Embeddings
       |
       +<--- user query: "plot sales by region"
       |
       v  MetadataLookup (similarity search, optional query refinement)
   Metaset   (catalog of relevant tables, optional schemas, optional doc chunks)
       |
       v  context["metaset"]
   SQLAgent / view agents / ChatAgent ...
```

### Vector stores and embeddings (`vector_store.py`, `embeddings.py`)

- Vector stores: `NumpyVectorStore` (**the default** for `Coordinator` and `VectorLookupTool`), `DuckDBVectorStore`, and `ChromaDBVectorStore`. All derive from `VectorStore`.
- Embeddings: `NumpyEmbeddings` (default), `OpenAIEmbeddings`, `AzureOpenAIEmbeddings`, `HuggingFaceEmbeddings`, `LlamaCppEmbeddings`.
- `lumen.ai` exports `NumpyVectorStore` and `DuckDBVectorStore` at the package level.

### `VectorLookupTool` and `MetadataLookup` (`tools/vector_lookup.py`, `tools/metadata_lookup.py`)

- `VectorLookupTool` is the base class for tools that search a vector store. Key parameters: `n` (results, default 5), `min_similarity` (default 0.3), and **query refinement** (`enable_query_refinement`, up to `max_refinement_iterations`, default 3) using the `refine_query` prompt.
- `MetadataLookup` indexes all tables from the available `Source`s (descriptions and column names, controlled by `include_metadata`, `include_columns`, `include_misc`). It has `always_use=True`, is excluded when `dbtsl_metaset` is present, and outputs `metaset`.
- `SourceLookup` (`tools/source_lookup.py`) does the equivalent for external source *actions* (APIs, callables, URL templates) and outputs `source_actions`.
- `DbtslLookup` does the equivalent for dbt Semantic Layer metadata (`dbtsl_metaset`).

### The `Metaset` (`schemas.py`)

`Metaset` is a dataclass with:

- `query`: the query that produced it.
- `catalog`: `dict[str, TableCatalogEntry]`, the discovered tables (descriptions, columns, similarity).
- `schemas`: optional per-table SQL schema data (types, enums, row counts), fetched **lazily** through `await metaset.get_schema(table_slug)` and cached.
- `docs`: optional `DocumentChunk`s retrieved from the document store.
- `schema_tables`: which tables to show full schemas for (if `None`, the top matches by similarity).

---

## 4. The LLM layer

`lumen/ai/llm.py` defines the provider abstraction. Every agent and tool reaches the model through an `Llm` instance.

### The `Llm` base class

- Built on **instructor**-style structured output. Prompts declare a Pydantic `response_model`, and `Llm.invoke(...)` returns a validated instance. `mode` defaults to JSON schema mode, and `create_kwargs` defaults to `max_retries=1` (agents handle their own retries).
- `invoke` and `stream` support multimodal messages (images) where the provider supports vision.
- **Tool calling.** `Llm` accepts `FunctionTool` and `MCPTool` objects and runs a tool loop (`_run_tool_loop`, `_run_tool_calls`), executing tool calls and feeding results back to the model.
- **Model specs.** `model_kwargs` maps a spec key (`default`, `reasoning`, `sql`, and others) to provider settings. Each actor selects a key through `llm_spec_key` (and prompts may override it with `llm_spec`).
- **Model routing.** An entry may declare a `description` and a `routing` model. The routing model is called first to choose which entry to use for a given call.
- **Cost and tracing hooks.** `usage_pricing` (USD per million tokens), `interceptor`, `logfire_tags`, `capture_usage()` and `trace()`. See [Section 12](#12-observability-logging-and-usage).
- Other controls: `temperature`, `timeout`, `select_models` (for UI dropdowns), and `warmup`.

### Providers

| Family | Classes |
| :--- | :--- |
| OpenAI-compatible | `OpenAI`, `AzureOpenAI`, `Ollama`, `Groq`, `OpenRouter`, `Kilo`, `AINavigator`, `AICatalyst` |
| Anthropic | `Anthropic`, `AnthropicBedrock` |
| Others | `Google`, `MistralAI`, `AzureMistralAI`, `Bedrock`, `LiteLLM` |
| Local / in-browser | `LlamaCpp`, `MLX`, `WebLLM` |
| CLI wrappers | `LlmCli`, `ClaudeCode`, `CodexCli` |

`services.py` maps provider names to the API-key environment variables (for example `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `GEMINI_API_KEY`). The UI uses that mapping to pick a default LLM and report connection status.

### Prompts

- Prompts are **Jinja2 templates** under `lumen/ai/prompts/<ActorClassName>/<prompt_name>.jinja2` (for example `Planner/main.jinja2`, `Planner/follow_up.jinja2`, `SQLAgent/revise_output.jinja2`). `prompts/GUIDANCE.md` documents the conventions.
- Each actor declares a `prompts` dictionary mapping a prompt name to a `template` path and an optional `response_model`. `_render_prompt` renders the template with the context, and `_invoke_prompt` renders and calls the LLM in one step.
- Template lookup walks the class hierarchy, so subclasses inherit prompts from their parents unless they override them.
- Users can override a prompt's template, response model, tools, or model spec per actor through the `prompts` parameter.
- `translate.py` converts `param` definitions into Pydantic models, which is used when building response models from Lumen components.

---

## 5. Actors: tools and the agent taxonomy

Everything the coordinator can run is an `Actor` (`lumen/ai/actor.py`):

- `Actor(LLMUser)` defines the abstract `respond(messages, context)` returning `(outputs, context_model)`. Its subclasses automatically get their `respond` wrapped for logfire tracing.
- `ContextProvider` is a **separate mixin** that adds the planning metadata and the `input_schema` / `output_schema`.
- `Agent(Viewer, ToolUser, ContextProvider)` and `Tool(Actor, ContextProvider)` combine them.

### What makes an actor plannable

1. **Soft signals (prompt guidance for the LLM):**
   - `purpose`: what the actor does.
   - `conditions`: when it is preferred (and when it is not).
   - `exclusions`: context keys that mean the actor should *not* be used.
   - `not_with`: actors that must not appear in the same plan.
2. **Hard enforcement (code):**
   - `applies(context)`: an **async classmethod** returning a bool. If false, the actor is dropped before prompt assembly.
   - `input_schema` / `output_schema`: `ContextModel` subclasses. Actors whose required inputs cannot be satisfied are pruned.
   - Static validation also checks `not_with` and `exclusions` across a plan (`validate_taskgroup_exclusions`).

### Tools (`lumen/ai/tools/`)

| Tool | Role |
| :--- | :--- |
| `Tool` | Base class. Has `always_use`, `prepare(context)` (called once at start) and `sync(context)` (called when context changes). |
| `FunctionTool` | Wraps a Python function; its signature becomes a Pydantic model the LLM fills in. |
| `MCPTool` | Exposes tools from an MCP server, converting their JSON Schema to Pydantic models. |
| `VectorLookupTool` | Base for vector-search tools. |
| `MetadataLookup`, `SourceLookup`, `DbtslLookup` | Table, source-action and dbt semantic lookups. |
| LLM tool factories | `make_clarification_llm_tool`, `make_document_vector_llm_tools`, `make_load_metaset_relevant_docs_tool`, `make_monty_llm_tool`. These return functions the LLM can call mid-response. |

`ToolUser` is the mixin that gives an actor (agent, coordinator or tool) its `tools` and initialises them per prompt. Coordinator-level tools (`planner_tools`, `tools`) run in the planning stage. Agent-level `llm_tools` are exposed to the model during that agent's own prompts.

### Agent roster

| Category | Agents | Responsibilities |
| :--- | :--- | :--- |
| **Data and query** | `SQLAgent`, `DbtslAgent`, `AnalysisAgent` | `SQLAgent` generates and runs read-only SQL and builds a `Pipeline`. `DbtslAgent` queries the dbt Semantic Layer. `AnalysisAgent` runs user-defined `Analysis` classes. |
| **Visualization** | `hvPlotAgent`, `VegaLiteAgent`, `DeckGLAgent` | Subclasses of `BaseViewAgent` (`BaseCodeAgent` for VegaLite and DeckGL). Turn a pipeline into a view. |
| **Discovery and documents** | `TableListAgent`, `DocumentListAgent`, `DocumentSummarizerAgent` | List available tables or documents (`BaseListAgent`) or summarise documents. |
| **Sources** | `SourceAgent` | Fetches data from configured external sources (APIs, callables, URL templates), normalises it to DataFrames, and registers it as `DuckDBSource` tables. For HTML pages it also indexes page text as a searchable document. |
| **Conversation and quality** | `ChatAgent`, `ValidationAgent` | General chat, and plan-completeness validation. |

Notes:

- The UI's `default_agents` are `TableListAgent`, `ChatAgent`, `DocumentListAgent`, `DocumentSummarizerAgent`, `SQLAgent`, `SourceAgent`, `VegaLiteAgent`, `ValidationAgent`, `DeckGLAgent`. `hvPlotAgent`, `DbtslAgent`, and `AnalysisAgent` are added by the user. `AnalysisAgent` is added when `analyses` are provided.
- `StoryAgent` (`agents/story.py`) is an `LLMUser`, not a plannable `Agent`. It drives the story feature ([Section 11](#11-reports-stories-and-export)).
- Base classes: `BaseLumenAgent` (adds revise and explain prompts), `BaseViewAgent`, `BaseCodeAgent`, `BaseListAgent`.

#### SQLAgent in more detail

- Validates that the query is read-only (`validate_read_only_sql`, using `sqlglot`) and normalises it (`clean_sql`).
- Executes with retries (`max_retries=2` in `_validate_sql`) and a `query_timeout` (default 60 s; a timed-out query is revised once).
- Can use LLM tools during generation: running exploratory SQL, browsing the data catalog, loading table schemas, and `apply_filter` for narrowing an existing pipeline.
- **Data-quality pass** (`clean_data=True` by default). `data_quality.py` profiles a bounded sample for missing values, duplicates, placeholder numbers, outliers, and similar problems, with **no LLM call**. Only if problems are found does the agent spend one extra LLM call rewriting the query.
- Uses `SQLEditor` as its output editor. It declares `not_with = [DbtslAgent, MetadataLookup, TableListAgent]` and `exclusions = [dbtsl_metaset]`.

#### Special handling of `ValidationAgent`

`ValidationAgent` is never chosen by the planning LLM. `Planner._make_plan` removes it from the candidate list. When `coordinator.validation_enabled` is true, `_resolve_plan` appends it as the final task, where it checks whether the executed plan answered the original query and can offer follow-up suggestions.

---

## 6. Context, contracts, and the dual-channel model

Actors exchange data through a shared context mapping (`TContext = Mapping[str, Any]`, in `lumen/ai/context.py`).

### Declaring contracts with `ContextModel`

Contracts inherit from `ContextModel` (a `TypedDict`). Keys not marked `NotRequired` appear in `__required_keys__`:

```python
# lumen/ai/agents/sql.py
class SQLInputs(ContextModel):
    data: NotRequired[Any]
    source: Source
    sources: Annotated[list[Source], ("accumulate", "source")]
    sql: NotRequired[str]
    metaset: Metaset
    visible_slugs: NotRequired[set[str]]

class SQLOutputs(ContextModel):
    data: Any
    table: str
    sql: str
    pipeline: Pipeline
```

```python
# lumen/ai/agents/base_view.py
class ViewInputs(ContextModel):
    data: Any
    pipeline: Pipeline
    metaset: NotRequired[Metaset]
    table: str

class ViewOutputs(ContextModel):
    view: Any
```

`ViewOutputs.view` is a normal required key, so `view` is enforced like any other declared output.

### Merging contexts

- `Annotated[..., ("accumulate", "from_key")]` marks a field that **accumulates** values from another key across tasks (for example `sources` collects each `source`), with optional de-duplication (`AccumulateSpec`).
- All other keys use **last-write-wins**. The `LWW` model and `merge_contexts(schema, contexts)` implement this.

### The handoff chain

```
MetadataLookup --metaset--> SQLAgent --pipeline, data, table, sql--> hvPlotAgent / VegaLiteAgent
requires: sources           requires: source, sources, metaset      requires: pipeline, data, table
provides: metaset           provides: data, table, sql, pipeline    provides: view
```

### The dual-channel model

Every `respond()` returns two separate things:

```
                      +-------------------+
                      |   actor.respond   |
                      +---------+---------+
               +----------------+----------------+
               v                                 v
        [ Channel 1 ]                     [ Channel 2 ]
       task_context dict                   outputs list
  (metaset, pipeline, data, sql ...)       (LumenEditor instances)
               v                                 v
      Next actor's inputs               plan.views -> UI tabs
```

| Channel | Contents | Destination |
| :--- | :--- | :--- |
| **Task context** | Internal state such as `metaset`, `pipeline`, `data`, `sql`, `table`, `view` | Consumed by downstream actors. Not rendered. |
| **Outputs** | `LumenEditor` components (and Markdown) | Collected by `plan.views` and shown in the UI. |

For view agents, the editor's `render_context()` produces the context part. For a `View` it returns `{"view": <spec dict>}`. For a `Pipeline` it returns `pipeline`, `table`, `source`, and a described sample of `data`. Putting a key in the context never shows anything on screen. Only objects in `outputs` do.

### Static vs runtime validation

- **Static** (`validate_task_inputs` in `context.py`, called by `validate()` before anything runs). Checks, with type compatibility, that each task's `input_schema` is satisfied by the initial context or earlier tasks' `output_schema`, including accumulate fields. Problems are collected as `ValidationIssue`s and raised as one `ContextError` (rendered as a tree). The planner uses this to replan.
- **Runtime** (`Plan._run_task`). After each task, checks that every key in `output_schema.__required_keys__` is in the returned context, otherwise raises `failed to provide declared context`.

---

## 7. Code execution, security, and sandboxing

Lumen AI mostly generates **declarative output** (SQL, Vega-Lite and DeckGL specs). Executing generated Python is an opt-in feature of the code-capable view agents.

### Declarative first

- `BaseViewAgent.respond` generates a YAML/JSON spec through the LLM, builds the Lumen `View` from it, and wraps it in an editor. No code is executed.
- `BaseCodeAgent` (base of `VegaLiteAgent` and `DeckGLAgent`) adds an optional code route controlled by `code_execution`.

### `code_execution` modes

| Mode | Behaviour |
| :--- | :--- |
| `disabled` (agent default) | No code runs. Declarative specs only. Safe for production. |
| `prompt` | Generate code, validate with AST, then **ask the user to approve** before executing. |
| `llm` | Generate code, validate with AST, then have an **LLM safety review** (`CodeSafetyCheck`) approve it. |
| `allow` | Generate and run code without confirmation after AST validation. |

`ExplorerUI.code_execution` additionally accepts `hidden` (the default), which hides the preference in the UI. The modes other than `disabled` execute LLM-generated code and must never be enabled where secrets or sensitive data are reachable. The LLM review reduces accidents. It is not a security boundary against malicious input.

### Execution pipeline (`code_executor.py`)

```
LLM-generated code
   1. AST validation (always)        forbidden imports, attributes, dunders and calls
   2. LLM safety review              only in "llm" mode
   3. User approval                  only in "prompt" mode
   4. Execute via executor           with minimal safe builtins, imports stripped and
                                     required modules injected
```

- `CodeExecutor` is an abstract base. Concrete executors are `AltairExecutor` (Vega-Lite via Altair) and `PyDeckExecutor` (DeckGL via PyDeck).
- AST validation blocks forbidden imports, forbidden function calls, and attribute traversal (including dunder access).
- If the user rejects execution, the code-capable agents return an empty result instead of raising.

### Monty sandbox tool

`make_monty_llm_tool()` (`tools/monty.py`) exposes an optional `run_python` LLM tool backed by the `pydantic_monty` sandbox. It runs a **subset of Python** with no access to Lumen state, files, network or subprocesses, a short timeout, and memory and output limits. It is only available if `pydantic_monty` is installed.

---

## 8. Editors and two-way reactivity

AI-generated artifacts must remain **user-editable**. The UI is built on [Panel](https://panel.holoviz.org/), and each AI output is wrapped in a `LumenEditor` (`lumen/ai/editors.py`).

### Editor types

| Editor | Used for |
| :--- | :--- |
| `LumenEditor` | Base class: holds a Lumen `component`, an editable `spec` string, a title, and footer controls. |
| `SQLEditor` | SQL pipelines. Adds a table explorer, a data preview, filter controls, and the SQL code editor. |
| `VegaLiteEditor` | Vega-Lite views, with spec validation and export. |
| `DeckGLEditor` | DeckGL views. |
| `MultiChartEditor` | Multiple charts combined, with image or SVG stacking on export. |
| `AnalysisOutput` | Output of an `Analysis`, with a re-run button. |
| `DocumentEditor` | Document content. |

Each editor defines `_controls` (by default `RetryControls`, `ExplainControls`, `CopyControls`) and `export_formats`.

### How an edit flows

```
User edits spec in the code editor  (CodeEditor <-> editor.spec, bidirectional)
        |
        v  param.depends("spec") -> _update_component()
_deserialize_component(...)  ->  new component  (reusing the existing pipeline)
        |
        v  param.depends("component") -> render()
Re-rendered output  (cached by spec string; errors show an alert with undo/edit advice)
```

Editing the spec **does not call the LLM**. The editor deserializes the YAML/JSON back into a Lumen component, and the view re-renders. `RetryControls` and `RevisionControls` (`controls/revision.py`) are the path for asking the LLM to revise an output. `ExplainControls` asks it to explain one.

---

## 9. Data ingestion and sources

### Source controls (`lumen/ai/controls/ingest/`)

The UI offers several ways to bring data in, each a Panel control that produces a `SourceResult`:

`UploadSourceControls`, `FileSourceControls`, `DownloadSourceControls`, `URLSourceControls`, `RESTAPISourceControls`, `OpenAPISourceControls`, `ParametricSourceControls`, `CodeSourceControls`, and `CatalogSourceControls`. Shared helpers handle file reading, progress, and metadata detection (`METADATA_EXTENSIONS`, `TABLE_EXTENSIONS`).

`controls/catalog.py` (`SourceCatalog`) lists the available tables and documents. `controls/explorer.py` (`TableExplorer`) is the interactive table explorer.

### Core Lumen sources (`lumen/sources/`)

AI-registered data lives in Lumen `Source` objects, for example `DuckDBSource` (the common in-memory target), `FileSource`, `InMemorySource`, `JSONSource`, `RESTSource`, `WebsiteSource`, `JoinedSource`, and the SQL-capable sources (`BaseSQLSource`, SQLAlchemy, BigQuery, Snowflake, Intake variants, xarray-SQL, Prometheus, AE5). `SQLAgent` requires a SQL-capable `source` in the context.

---

## 10. Sessions, explorations, and follow-ups

State lives at three levels:

1. **Task context.** Ephemeral values passed between actors during one plan run (see [Section 6](#6-context-contracts-and-the-dual-channel-model)).
2. **Exploration.** An `Exploration` (in `ui.py`) holds its own `context`, `plan`, `conversation`, `title`, and a tabbed `view`. `ExplorerUI` shows one active exploration at a time and links explorations to a parent.
3. **Global UI context.** Shared `sources`, the vector stores, and other state available to all explorations. After a plan succeeds, new sources are synced into it.

There is no separate `Memory` class. State is carried in the context dictionaries and `Exploration` objects.

### What `_execute_plan` does

- **Rerun** keeps the same exploration and resets the failed tasks.
- A **new exploration** is created when the plan produces `pipeline` or `view` and the parent already has a `pipeline`, or when starting from the home screen.
- Otherwise the new plan is **merged into the parent exploration's plan** (`parent.plan.merge(plan)`), so the checklist is combined.
- **Replan** (after a failure or user retry) removes the old tasks and asks the coordinator for a new plan from the same history.

### Follow-ups

The planner's follow-up classification (`direct`, `derived`, `new`) tells it whether to build on the existing `pipeline` or start over. A `"new"` classification drops `pipeline` from the context first.

---

## 11. Reports, stories, and export

### Reports and tasks (`report.py`)

`Plan` is one kind of `Section`/`TaskGroup`, built on the generic `Task` / `Action` / `ActorTask` / `TaskGroup` abstractions in `report.py`. The same machinery can run reports outside a chat. `actions.py` provides ready-made actions such as `SQLQuery`.

### Story (`story.py`, `agents/story.py`)

A report can carry an **AI-written, editable story**. `StoryAgent` generates the narrative, and `story.py` renders it as editable prose that is included in exports.

### Export (`export.py`)

- **Notebook.** `export_notebook(outputs, preamble)` turns a list of outputs into a Jupyter notebook. Markdown entries become markdown cells, and each `LumenEditor` becomes a code cell with its spec. `ExplorerUI` exposes this as **Export Notebook** per exploration.
- **Per-editor export.** `LumenEditor.export(fmt)` and `export_formats` (YAML by default, with additional formats on specific editors, such as HTML or image formats for charts).
- **Document export.** `export.py` also contains helpers to build Word (`.docx`) documents from markdown, tables, and charts.
- **Component specs.** Each Lumen component has `to_spec()`. Editors use it to produce the editable spec, and notebook export writes it into the cells.

---

## 12. Observability, logging, and usage

| Module | Purpose |
| :--- | :--- |
| `logfire.py` | Optional [Logfire](https://pydantic.dev/logfire) tracing. `wrap_logfire` decorates planner steps (for example `Make Plan`, `Chat Invoke`) and actor `respond` methods, tagged with `logfire_tags`. Helpers query traces by tag. |
| `logs.py` | `ChatLogs`: an SQLite-backed log of chat sessions and messages (`logs_db_path` on the UI). |
| `interceptor.py` | `Interceptor` for capturing LLM invocations (messages, response, kwargs), for logging or evals. |
| `tool_trace.py` | Scoped collection of `ModelCall` and `ToolCall` records. |
| `usage.py` | `Usage` / `UsageCollector` for provider-reported token counts and optional cost estimates (via `usage_pricing`). |
| `utils.log_debug` | Debug logging controlled by the UI's `log_level`. |

---

## 13. Serving and deployment

- The `lumen-ai` command (`lumen/command/ai.py`) is a console script (`lumen-ai = lumen.command.ai:main`) that starts a Panel/Bokeh server. Running `lumen-ai` with no arguments starts the app without data (`serve no_data`).
- Options configure the provider, API key, provider endpoint, temperature, extra agents, and related settings.
- `ExplorerUI` and `ChatUI` (both subclasses of `UI`) can also be embedded directly in a Python app, for example `lmai.ExplorerUI(data='~/data.csv').servable()`.
- The default LLM is detected from environment variables (see `services.py`) unless one is passed in.

---

## 14. Extension points

| To add... | Subclass / provide | Key points |
| :--- | :--- | :--- |
| An agent | `Agent` (or `BaseLumenAgent`, `BaseViewAgent`) | Set `purpose`, `conditions`, `input_schema`, `output_schema`, `prompts`, and implement `respond`. See [Creating custom agents](configuration/agents.md#creating-custom-agents). |
| A tool | `Tool`, `FunctionTool`, `MCPTool`, or `VectorLookupTool` | Pass via `tools=` (coordinator-level) or `llm_tools=` (model-callable). |
| A custom analysis | `Analysis` (`ParameterizedFunction`) | Implement `applies` and `__call__`. Pass via `analyses=`, which adds `AnalysisAgent`. |
| An LLM provider | `Llm` subclass | Implement `_create_base_client`, optionally set `_instructor_wrapper`. |
| A vector store or embeddings | `VectorStore`, `Embeddings` | Pass `vector_store=` / `document_vector_store=`. |
| A coordinator | `Coordinator` subclass | Provide `_compute_plan` and a `main` prompt. |
| Prompt changes | `prompts=` parameter | Override `template`, `response_model`, `tools`, or `llm_spec`. |

---

## 15. Background: the core reactive engine

Every `Pipeline` created by Lumen AI keeps working without the AI layer:

```
Filter / transform changes --> Pipeline recomputes its data --> View re-renders
```

`Pipeline` (in `lumen/pipeline.py`) has `sql_transforms`, `auto_update` (default `True`), and the data it publishes. Attached `View`s update when it changes. This reactive foundation predates the AI layer and is detailed in [Core Concepts](configuration/spec/concepts.md), [Loading Data](configuration/spec/sources.md), [Transforming Data](configuration/spec/pipelines.md), and [Visualizing Data](configuration/spec/views.md). When charts do not update after a filter change, inspect `pipeline.auto_update` and `pipeline.data`.

---

## 16. Debugging the AI layer

| Symptom | Probable cause | Where to investigate |
| :--- | :--- | :--- |
| **Wrong or missing agent in plan** | Actor eligibility or unmet prerequisites | `agent.applies(context)` (`agents/base.py`); `exclusions` keys already in context; the unmet-input filter in `Planner._make_plan` (`set(agent.input_schema.__required_keys__) - all_provides`); `raw_plan.chain_of_thought`. |
| **Planning fails or loops** | Unfulfilled contract inputs | The `ContextError` from `plan.validate()` (printed as a tree naming the task and key). Planning gives up after more than five attempts. |
| **Hallucinated tables or columns** | Retrieval or metaset mismatch | `context["metaset"]`; whether `MetadataLookup` retrieved the right tables; `min_similarity` and `n` on the lookup tool; the vector store and embeddings in use. |
| **Task fails during execution** | Missing context key or actor error | For `MissingContextError`: `Plan._handle_task_execution_error`, `_find_context_provider` (note: it only searches for a `pipeline` provider). For `failed to provide declared context`: compare the actor's returned keys to `output_schema.__required_keys__`. |
| **SQL errors or slow queries** | Validation, retries, timeout | `SQLAgent._validate_sql` (read-only check, retries), `query_timeout`, and the data-quality pass (`clean_data`). |
| **Code not executed** | Execution mode or rejection | `code_execution` is `disabled` by default. Check AST errors ("Unsafe code detected"), LLM safety review output, and whether the user rejected the prompt (the agent returns an empty result). |
| **View does not appear in the UI** | No `LumenEditor` in `outputs` | Confirm `respond()` returned an editor in the outputs list (channel 2). Check that `_add_views` is attached in `ExplorerUI._execute_plan`. |
| **Edited spec does nothing or errors** | Spec fails deserialization or validation | The editor shows an alert. Check `validate_spec` and `_deserialize_component` in `editors.py`. |
| **Validation step does not run** | Disabled on the coordinator | `coordinator.validation_enabled` (default `False`). See [Coordinators](configuration/coordinators.md#disable-automatic-validation). |
| **LLM errors or wrong model** | Provider or model-spec configuration | `Llm.model_kwargs` spec key used by the actor (`llm_spec_key`), API key environment variables (`services.py`), and the `interceptor` / `tool_trace` / logfire traces. |

---

## Appendix: module map (`lumen/ai/`)

| Path | Contents |
| :--- | :--- |
| `coordinator/` | `base.py` (`Coordinator`, `Plan`), `planner.py` (`Planner`), `dependency.py` (`DependencyResolver`) |
| `actor.py` | `LLMUser`, `Actor`, `ContextProvider` |
| `agents/` | All agents and their base classes (`base`, `base_lumen`, `base_view`, `base_code`, `base_list`, `sql`, `hvplot`, `vega_lite`, `deck_gl`, `chat`, `source`, `validation`, `analysis`, `dbtsl`, `table_list`, `document_list`, `document_summarizer`, `story`) |
| `tools/` | `base`, `vector_lookup`, `metadata_lookup`, `source_lookup`, `dbtsl_lookup`, `mcp`, `monty`, `clarification_llm_tool`, `document_llm_tools`, `metaset_docs_llm_tools` |
| `context.py` | `ContextModel`, merge semantics, static validation, `ContextError` |
| `schemas.py` | `Metaset`, `TableCatalogEntry`, `Column`, `DocumentChunk`, dbt schemas |
| `vector_store.py`, `embeddings.py` | Retrieval backends |
| `llm.py`, `services.py`, `prompts/` | LLM abstraction, provider key mapping, Jinja2 templates |
| `code_executor.py` | AST validation, executors, safety check model |
| `editors.py`, `controls/`, `components.py` | Output editors, ingest and revision controls, UI components |
| `ui.py` | `UI`, `ChatUI`, `ExplorerUI`, `Exploration` |
| `report.py`, `actions.py`, `story.py`, `export.py` | Task framework, built-in actions, stories, exports |
| `analysis.py`, `data_quality.py`, `translate.py` | Custom analyses, deterministic profiling, param-to-Pydantic translation |
| `logfire.py`, `logs.py`, `interceptor.py`, `tool_trace.py`, `usage.py` | Observability |

---
