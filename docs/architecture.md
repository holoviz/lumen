# Lumen Architecture and Developer Guide

This document describes the internal architecture of **Lumen**, detailing how the **Lumen AI** layer (`lumen/ai/`) coordinates with the underlying **Core Lumen** dataflow engine (`lumen/`).

---

## 1. Overview: AI Layer on Top of Core Layer

Modern Lumen provides an AI-assisted analytics experience built on top of a reactive, declarative data engine. Rather than requiring users or developers to manually author YAML specifications, Lumen AI translates natural-language requests into structured task plans and dynamically instantiates Core Lumen primitives.

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                                 LUMEN AI LAYER                                  │
│                                                                                 │
│  ExplorerUI ──> Coordinator / Planner ──> Plan ──> Actors (Agents / Tools)      │
│                                                      │       ▲                  │
│                                                      ▼       │                  │
│                                                 LLM / VectorStore               │
└──────────────────────────────────────┬──────────────────────────────────────────┘
                                       │ Dynamically produces
                                       ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│                                CORE LUMEN LAYER                                 │
│                                                                                 │
│         Source        ───>        Pipeline        ───>        View              │
│      (Data Intake)           (Filters & Transforms)      (hvPlot, Cards, etc.)  │
│                                       ▲                                         │
│                                       │                                         │
│                                    Filter                                       │
│                               (User UI Controls)                                │
└─────────────────────────────────────────────────────────────────────────────────┘
```

### How the Two Layers Connect
- **Lumen AI** acts as the *composer*: it manages user prompts, chat history, semantic search over dataset schemas, and coordinates specialized agents.
- **Core Lumen** acts as the *execution engine*: when agents run (e.g., `SourceAgent`, `SQLAgent`, `hvPlotAgent`), they produce live Core Lumen objects: `Source`, `Pipeline`, and `View` instances.
- **Explorations & Editors**: agents return their results as `LumenEditor` components (e.g., `SQLEditor`, `VegaLiteEditor`), which give users an interactive visualization alongside its live, editable specification. `ExplorerUI` mounts these editors into an `Exploration` container rendered as interactive `Tabs`.

---

## 2. Lumen AI: ExplorerUI $\to$ Coordinator/Planner $\to$ Plan $\to$ Actors $\to$ LLM/VectorStore

The execution flow of Lumen AI proceeds through five major architectural components:

```mermaid
sequenceDiagram
    autonumber
    actor User as User / Browser
    participant UI as ExplorerUI (ui.py)
    participant Planner as Planner (coordinator/planner.py)
    participant Plan as Plan (coordinator/base.py)
    participant Actor as Agent / Tool (agents/ / tools/)
    participant LLM as LLM / VectorStore (llm.py / vector_store.py)

    User->>UI: Submits prompt ("Plot sales by region")
    UI->>Planner: _chat_invoke() -> coordinator.respond(messages, context)

    rect rgb(240, 245, 255)
        Note over Planner,LLM: Planning Phase
        Planner->>Planner: _pre_plan() runs follow-up & clarification checks
        Planner->>Actor: Execute planner tools (default: MetadataLookup)
        Actor-->>Planner: Returns metaset and schema context
        Planner->>LLM: Prompt LLM with dynamic plan schema (RawPlan)
        LLM-->>Planner: Returns task checklist & selected agents
        Planner->>Planner: _resolve_plan() orders tasks by dependencies & merges consecutive actor steps
        Planner->>Planner: plan.validate() (static check; on ContextError the planner retries)
    end

    Planner-->>UI: Returns configured Plan
    UI->>UI: Attach watcher: partial(self._add_views, exploration) on plan "views"
    UI->>Plan: _execute_plan() -> await plan.execute()

    rect rgb(245, 255, 245)
        Note over Plan,Actor: Execution Phase (Sequential Ordered List)
        loop For each task in Plan (ordered list)
            Plan->>Actor: execute(subcontext)
            Actor->>LLM: Invoke prompt or run SQL/Python code
            LLM-->>Actor: Result (code / spec / schema)
            Actor-->>Plan: (outputs, task_context)
            Note over Plan: Check declared output keys & merge task_context into global Context
            Plan-->>UI: outputs (LumenEditors) accumulate in plan.views -> triggers _add_views
        end
    end

    UI->>User: Display rendered exploration tabs & visualizations
```

### 2.1 UI: `ExplorerUI` (`lumen/ai/ui.py`)
`ChatUI` is a thin subclass of the base `UI` with no exploration or plan-execution logic of its own; full conversational data exploration is driven by **`ExplorerUI`**, which is also the entry point used throughout the Lumen docs.
- **Responsibilities**:
  - Manages the split-pane UI: chat feed, table navigation, Graphic Walker data exploration, and exploration tabs.
  - Maintains `Exploration` state, linking exploration trees and tracking the active context.
  - Sets `context["prev_plan"] = exploration.plan` on follow-up queries (when not on the home exploration).
  - In `_chat_invoke()`, calls `await self._coordinator.respond(messages, context)` to obtain a `Plan`, then `await self._execute_plan(plan)`.
  - Handles plan execution (`_execute_plan`), replanning (`_replan`), and reruns.
- **Mounting views**:
  - `_execute_plan()` attaches a watcher to the plan's `"views"` parameter:
    ```python
    watcher = plan.param.watch(partial(self._add_views, exploration), "views")
    ```
  - `_add_views` does **not** wrap anything. It iterates over the new items, **skips any item that is not already a `LumenEditor`**, renders each editor via `_render_view`, and appends it to the exploration's `Tabs`.
  - The watcher is attached only on the rerun, new-exploration and replan paths. When a plan is merged into an existing exploration's plan, no `_add_views` watcher is attached in that branch.

### 2.2 Coordinators: `Planner` and `DependencyResolver` (`lumen/ai/coordinator/`)
- **`Coordinator` (`coordinator/base.py`)**:
  - Base viewer class managing agent/tool registries, vector lookup tools, and prompt rendering. Its `_pre_plan` also filters agents and tools with `applies(context)`.
- **`Planner` (`coordinator/planner.py`)**:
  - The default coordinator (`UI.coordinator` defaults to `Planner`).
  - **`_pre_plan`**: checks whether the turn is a follow-up (`_check_follow_up_question`), checks whether clarification is needed (`_check_clarification_needed`), and runs the planner tools (default: `MetadataLookup`; `SourceLookup` is added automatically when a `SourceAgent` is present).
  - **`_make_plan`**: filters agents via `await agent.applies(context)`, drops agents whose required inputs cannot be provided, generates a Pydantic response model via `make_plan_model(agents, tools)`, and asks the LLM to output a `RawPlan`. `ValidationAgent` is excluded from the candidates here.
  - **`_resolve_plan`**: orders steps according to dependencies and merges consecutive steps that share the same actor.
  - After resolving, the planner calls `plan.validate()`; if that raises `ContextError`, it feeds the error back and retries planning.
- **`DependencyResolver` (`coordinator/dependency.py`)**:
  - An alternative coordinator. It asks the LLM to choose a **primary agent** (`_choose_agent`), then loops: while the current agent has requirements missing from the context, it finds agents whose `provides` covers them and adds them as earlier tasks, repeating until every requirement is satisfied.

### 2.3 Plan (`lumen/ai/coordinator/base.py` & `lumen/ai/report.py`)
- **`Plan`**: Subclass of `Section` (which inherits from `TaskGroup` $\to$ `Task`).
- **Ordered Sequential Execution**: `Plan` is an ordered list of tasks, not an arbitrary runtime DAG. Tasks execute sequentially in order via `_run_task()`.
- **Views**: `plan.views` is not populated from a context key. Each `ActorTask` stores the `outputs` returned by `actor.respond()` (via `_add_outputs`), and `TaskGroup._init_views` reactively concatenates the tasks' views into `plan.views`.
- **UI Progress**: Emits live progress and checklist updates via Panel `ChatStep`.
- **Runtime Error Handling**:
  - If a task raises `MissingContextError`, `_handle_task_execution_error` invokes `_find_context_provider` and `_retry_from_provider` to re-run the responsible upstream actor with corrective feedback.
  - After each task, `_run_task` verifies that every key in `task.output_schema.__required_keys__` is present in the returned `task_context`.

### 2.4 Actors: Agents and Tools (`lumen/ai/actor.py`, `lumen/ai/agents/`, `lumen/ai/tools/`)
- **`Actor` (`lumen/ai/actor.py`)**: Base class inheriting from `LLMUser`. Implements `respond()`. `ContextProvider` adds `purpose`, `conditions`, `not_with`, `input_schema` and `output_schema`.
- **Agents (`lumen/ai/agents/`)**:
  - **`SourceAgent`**: Ingests data sources (DuckDB, SQL, Parquet, CSV) and instantiates a `Source`.
  - **`TableListAgent`**: Browses available tables for a source; requires `source` input.
  - **`SQLAgent`**: Writes and validates SQL queries; produces a `Pipeline`, table, SQL query string, and data.
  - **`hvPlotAgent`**: Generates interactive visualizations using hvPlot and HoloViews.
  - **`VegaLiteAgent` / `DeckGLAgent`**: Generate Vega-Lite specs or geospatial Deck.gl layers.
  - **`AnalysisAgent`**: Runs predefined, user-supplied `Analysis` classes for reliable, repeatable analytical workflows (not loaded by default; enabled when `analyses` are passed to `ExplorerUI`).
  - **`ChatAgent`**: Handles general conversational replies or queries that do not require data operations.
  - **`ValidationAgent`**: A quality gate that checks whether the executed plan fully answered the user's original query and suggests next steps if not. It is appended as a final step when `validation_enabled` is on (the UI exposes this as the "Validation Step" switch) and is not offered as a regular planning candidate.
- **Tools (`lumen/ai/tools/`)**:
  - **`MetadataLookup`**: The default planner tool. Builds a vector store over table metadata and returns the relevant tables, producing the `metaset`.
  - **`SourceLookup`**: Semantic lookup over available sources.
  - **`VectorLookupTool`**: Base class for the vector-search tools above (`MetadataLookup`, `SourceLookup`, `DbtslLookup`).
  - **`make_clarification_llm_tool(...)`**: A factory (in `clarification_llm_tool.py`) that creates the LLM tool the planner uses to ask the user for clarification when a request is ambiguous.
  - **`MCPTool`**: Integrates external tools using the Model Context Protocol.

### 2.5 LLM & VectorStore Integrations (`lumen/ai/llm.py`, `lumen/ai/vector_store.py`)
- **Supported LLM Providers** (all subclasses of `Llm`): `OpenAI`, `AzureOpenAI`, `Anthropic`, `AnthropicBedrock`, `Bedrock`, `Google`, `MistralAI`, `AzureMistralAI`, `Ollama`, `Groq`, `LlamaCpp`, `OpenRouter`, `LiteLLM`, `AINavigator`, `AICatalyst`, among others (e.g. `MLX`, `WebLLM`).
- **Supported Vector Stores**: `NumpyVectorStore` (in-memory), `DuckDBVectorStore` (DuckDB-backed, persistent), and `ChromaDBVectorStore` (requires the `chromadb` package).

---

## 3. Context Model and Step-to-Step Data Handoff

The **Context** (`TContext = Mapping[str, Any]`, defined in `lumen/ai/context.py`) is the shared state passed across tasks. Agents communicate by reading dependencies from, and writing outputs to, the context.

### 3.1 Standard Context Keys

| Context Key | Type | Description |
| :--- | :--- | :--- |
| `"sources"` | `list[Source]` | All instantiated Lumen `Source` objects available in the session. |
| `"source"` | `Source` | The active `Source` selected for the current exploration. |
| `"metaset"` | `Metaset` | Catalog and schema metadata produced by `MetadataLookup`. |
| `"table"` | `str` | The active table name driving the visualization. |
| `"sql"` | `str` | The generated SQL query string produced by `SQLAgent`. |
| `"data"` | `Any` | Materialized or lazy dataframe query result. |
| `"pipeline"` | `Pipeline` | The active Core Lumen `Pipeline` containing filters and transforms. |
| `"view"` | `Any` | The view produced by a view agent, available to downstream actors. |
| `"visible_slugs"` | `set[str]` | Set of table/source slugs currently active and visible in the UI. |
| `"plan"` | `Plan` | The executing `Plan` instance (injected automatically in `Plan.execute`). |
| `"prev_plan"` | `Plan` | The previous exploration plan (set by `ExplorerUI` for follow-ups). |
| `"__error__"` | `str` | Stored error message if a task encounters an unrecoverable failure. |

> **Note**: `plan.views` (what `ExplorerUI` renders) is separate from the `"view"` context key. `plan.views` is built from the **outputs** that each actor returns from `respond()`, i.e. the first element of the `(outputs, task_context)` tuple, not from the context dictionary.

### 3.2 Schema Contracts: `ContextModel`
Every `Actor` declares typing contracts by subclassing `ContextModel` (a `TypedDict` subclass in `lumen/ai/context.py`).

For example, `SQLAgent` defines:
```python
# lumen/ai/agents/sql.py
class SQLInputs(ContextModel):
    source: Source
    sources: Annotated[list[Source], ("accumulate", "source")]
    metaset: Metaset
    data: NotRequired[Any]
    sql: NotRequired[str]
    visible_slugs: NotRequired[set[str]]

class SQLOutputs(ContextModel):
    data: Any
    table: str
    sql: str
    pipeline: Pipeline
```

View agents (such as `hvPlotAgent`, `VegaLiteAgent`, `DeckGLAgent`) subclass `BaseViewAgent`, which declares:
```python
# lumen/ai/agents/base_view.py
class ViewInputs(ContextModel):
    data: Any
    pipeline: Pipeline
    table: str
    metaset: NotRequired[Metaset]

class ViewOutputs(ContextModel):
    view: Any
```

### 3.3 The Real Handoff Chain
The standard end-to-end dataflow chain operates as follows:

```
┌────────────────────┐      Provides: {"metaset"}      ┌────────────────────┐      Provides: {"pipeline", "data", "table", "sql"}      ┌────────────────────┐
│   MetadataLookup   │ ──────────────────────────────> │      SQLAgent      │ ───────────────────────────────────────────────────────> │    hvPlotAgent     │
│   (Planner Tool)   │                                 │                    │                                                          │   (View Agent)     │
└────────────────────┘                                 └────────────────────┘                                                          └────────────────────┘
Requires: {sources}                                    Requires: {source, sources, metaset}                                            Requires: {pipeline, data, table}
                                                                                                                                       Provides: {"view"}
```

1. **Pre-planning / Metadata**: `MetadataLookup` inspects catalog schemas and populates `"metaset"`.
2. **Data & Pipeline Generation**: `SQLAgent` consumes `"source"`, `"sources"` and `"metaset"`, writes a SQL query, and provides `"pipeline"`, `"data"`, `"table"`, and `"sql"`.
3. **Visualization Generation**: `hvPlotAgent` consumes `"pipeline"`, `"data"`, and `"table"`, and provides `"view"`. It also returns its `LumenEditor` as an output.
4. **UI Presentation**: the returned outputs flow into `plan.views`, and `ExplorerUI`'s watcher (`partial(self._add_views, exploration)`) mounts each `LumenEditor` into the `Tabs` layout.

### 3.4 Static vs. Runtime Validation
Lumen enforces contract safety at two separate stages:
- **Static Validation (`validate_task_inputs`)**:
  - Called during plan construction from `ExecutableTask.validate()` (the `Planner` calls `plan.validate()` after `_resolve_plan`).
  - Statically analyzes whether each task's `input_schema` is satisfied by the current context or promised by upstream task output schemas. Raises `ContextError` if types are incompatible or dependencies are missing; the planner catches this and retries.
- **Runtime Validation**:
  - `validate_task_inputs` is not rerun before each task. Instead, tasks raise `MissingContextError` if an expected value is absent at runtime.
  - Upon task completion, `Plan._run_task()` verifies that the task returned all keys listed in `task.output_schema.__required_keys__`.

---

## 4. The Core Layer (Dataflow Engine)

Once Lumen AI generates components, they operate according to **Core Lumen's** reactive architecture (`Source` $\to$ `Pipeline` $\to$ `Filter` $\to$ `View`).

```
+---------------+        +------------------+        +-----------------+
|    Source     | -----> |     Pipeline     | -----> |      View       |
| (Data Intake) |        | (Reactive Engine)|        | (Visualization) |
+---------------+        +------------------+        +-----------------+
                                  ^
                                  |
                         +------------------+
                         |      Filter      |
                         |  (User Controls) |
                         +------------------+
```

### 4.1 Component Roles
- **`Source` (`lumen/sources/base.py`)**: Ingests raw data (DuckDB, SQL, Parquet). Exposes `get(table, **query)` and `get_schema(table)`.
- **`Pipeline` (`lumen/pipeline.py`)**: Central coordinator (`class Pipeline(Viewer, Component)`). Holds collections of `filters` and `transforms`. Contains the reactive parameter `data = DataFrame(...)`.
  - **Pipeline Chaining**: a `Pipeline` can consume another pipeline through its `pipeline` parameter. The constructor still requires `source` and `table` as keyword arguments, so pass them explicitly:
    ```python
    downstream = Pipeline(pipeline=upstream, source=upstream.source, table=upstream.table)
    ```
    (In a YAML/dict spec, a `pipeline:` key fills these in automatically.) A chained pipeline computes its data from `upstream.data` instead of calling `source.get()`, and it watches the upstream pipeline's `data` and `_stale`.
- **`Filter` (`lumen/filters/base.py`)**: Interactive UI widgets (sliders, selectors). Exposes `value = param.Parameter(...)`. The `widget` attribute exists on widget-based filters (`BaseWidgetFilter` and its subclasses).
- **`View` (`lumen/views/base.py`)**: Renders visual representations. In `View.__init__`, it watches both the pipeline data and its own parameters:
  ```python
  pipeline.param.watch(self.update, 'data')
  self.param.watch(self.update, [p for p in self.param if p not in ('rerender', 'selection_expr', 'name')])
  ```
  - `hvPlotView` overrides `update()`. It ignores events caused by its own `ParamFilter`/selection, and when `streaming` is enabled it pushes fresh data through its data stream (`self._data_stream.send(self.get_data())`). Otherwise it falls back to `_update_panel()` and, if needed, `rerender`.

### 4.2 The Reactive Update Flow
When a user interacts with a filter control in the UI, the update travels through this exact chain:

$$\text{Filter.value changes} \longrightarrow \text{Pipeline._init_callbacks triggers } \texttt{\_update\_data} \longrightarrow \text{Pipeline.data updates} \longrightarrow \text{View.update} \longrightarrow \text{Re-render}$$

1. **`Filter.value` changes**: the user moves a slider or selects a dropdown value.
2. **`Pipeline._init_callbacks`**: the pipeline registered `filt.param.watch(self._update_data, ['value'])`.
3. **`Pipeline._update_data`**: checks `auto_update`, sets `loading = True`, calls `_compute_data()` (querying `source.get()` and applying in-memory transforms), then assigns `self.data = new_data`.
4. **`View.update`**: the observer registered on `pipeline.param.data` fires.
5. **Re-render / In-place Patch**: `View.update()` clears the internal cache (`self._cache = None`) and calls `self._update_panel()`. If the view cannot update parameters in place or stream new rows, it triggers `self.param.trigger('rerender')`.

---

## 5. Contributor Debugging Guide

### 5.1 Debugging the AI Layer

#### "Why did the planner pick the wrong agent?"
- **Check `agent.applies(context)`**:
  - File: `lumen/ai/agents/base.py` (a classmethod, overridable per agent).
  - In `Planner._make_plan()` (and the base `Coordinator._pre_plan()`), candidates are filtered out before LLM prompting if `await agent.applies(context)` returns `False`.
- **Check Unmet Dependencies**:
  - The planner computes `set(agent.input_schema.__required_keys__) - all_provides`. If any required input key cannot be provided by the context or by other actors, the agent is deemed unsatisfiable and excluded.
- **Inspect the `_make_plan` Prompt and Chain-of-Thought**:
  - File: `lumen/ai/coordinator/planner.py`
  - Set a breakpoint in `_make_plan()` to inspect `system` and the streamed `raw_plan.chain_of_thought`.

#### "Why did planning retry or fail before anything ran?"
- After `_resolve_plan`, the planner calls `plan.validate()`. A `ContextError` (missing dependency or incompatible types between steps) is streamed to the user and triggers another planning attempt. Inspect the `ContextError` text to see which task's `input_schema` was not satisfied.

#### "Why did this step fail during execution?"
- **Missing Context at Runtime**:
  - If a task raises `MissingContextError`, inspect `Plan._handle_task_execution_error` and `Plan._find_context_provider`.
- **Unprovided Output Contract**:
  - If an actor completes without returning all keys required by its `output_schema`, execution raises a `RuntimeError` ending in `"failed to provide declared context"` (from `Plan._run_task` or `ActorTask._execute`); the chat step title lists the missing keys. Check the context dictionary returned by `agent.respond()`.
- **Code Execution Error**:
  - For code-generating agents (`SQLAgent`, `BaseCodeAgent`), check `lumen/ai/code_executor.py` or the SQL validation logs.

#### "Why did newly generated views not show up in the UI?"
- **Outputs, not context**: verify that the agent's `respond()` returned a `LumenEditor` in its `outputs` list. `ExplorerUI._add_views` silently skips anything that is not a `LumenEditor`, and putting a `"view"` in the context does not add it to `plan.views`.
- **Watcher**: in `ExplorerUI._execute_plan()`, check which branch ran. The watcher `plan.param.watch(partial(self._add_views, exploration), "views")` is attached for reruns, new explorations and replans, but not when the plan is merged into an existing exploration's plan.

### 5.2 Debugging the Core Layer

#### "Why didn't my view update when a filter changed?"

Follow this checklist:
1. **Filter Watcher**: check `Filter.value`. Was the filter appended to `pipeline.filters` after initialization without calling `pipeline.add_filter()`?
2. **Pipeline Gate**: set a breakpoint at `lumen/pipeline.py:Pipeline._update_data`. Is `self.auto_update` set to `False`? Is `self._update_widget.loading` stuck on `True`?
3. **Source Fetch**: in `Pipeline._compute_data()`, does `source.get()` return cached data? (Verify via `source.clear_cache()`.) For a chained pipeline, check whether the upstream pipeline's `data` actually changed.
4. **Data Assignment (User Transforms)**: in-place DataFrame mutations in custom transforms (e.g. `df['x'] = ...`) may not trigger Param change events if the dataframe object identity does not change. Always return fresh copies.
5. **View Watcher**: set a breakpoint in `lumen/views/base.py:View.update`. Ensure `pipeline.param.watch(self.update, 'data')` fired. Remember that changing the view's own parameters also calls `update()`.
6. **Panel Rendering**: in `View._update_panel()`, verify whether an in-place param update or stream succeeded, or whether `self.param.trigger('rerender')` was required. For `hvPlotView`, also check the `streaming` flag and the own-event filter in its `update()` override.
