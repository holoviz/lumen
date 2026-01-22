# :material-file-document: Reports

Reports execute a sequence of tasks in order, pass data between tasks, and render as a document. Use reports when you need reproducible, automatable analytical workflows instead of free-form chat.

Reports work best for reproducibility, automation, structure, export to notebooks, and programmatic control. For exploratory analysis, use the chat interface instead.

## Quick start

Create a report with two tasks. The first produces a greeting, the second reads it from context:

```python
import lumen.ai as lmai
from lumen.ai.report import Action, Report, Section
from panel.pane import Markdown

class HelloAction(Action):
    async def _execute(self, context, **kwargs):
        return [Markdown("**Hello, World!**")], {"greeting": "Hello"}

class SummaryAction(Action):
    async def _execute(self, context, **kwargs):
        greeting = context.get("greeting", "Hi")
        return [Markdown(f"Summary: {greeting}")], {}

report = Report(
    Section(
        HelloAction(title="Greeting"),
        SummaryAction(title="Summary"),
        title="My Section"
    ),
    title="My Report"
)

await report.execute()
report.servable()
```

## Structure

Reports follow a three-level hierarchy:

```
Report (level 1)
└── Section (level 2)
    └── Task (level 3): Action or ActorTask
```

**Report** is the top-level container with buttons to execute, clear, collapse/expand, export, and configure. **Section** groups related tasks as collapsible accordions. **Task** is a unit of work—either an `Action` (custom Python) or `ActorTask` (wraps an Agent).

```python
report = Report(
    Section(
        LoadDataAction(),
        AnalyzeAction(),
        title="Data Processing"
    ),
    Section(
        VisualizeAction(),
        title="Visualization"
    ),
    title="Quarterly Review"
)
```

## Actions

Subclass `Action` and implement `_execute` to create a task. Return a tuple of (outputs, context_updates):

```python
from lumen.ai.report import Action
from panel.pane import Markdown

class MyAction(Action):
    async def _execute(self, context, **kwargs):
        result = do_something()
        return [Markdown(f"Result: {result}")], {"my_result": result}
```

The outputs list contains renderable objects (Markdown, plots, tables). The context dict shares values with downstream tasks. Return `[]` for no visible output or `{}` for no context updates.

**Parameters:** `title`, `instruction`, `context` (initial values), `abort_on_error` (default True), `render_outputs` (default True).

??? example "Complete action example"
    Load a parquet file, create a pipeline, and return a visualization:

    ```python
    import pandas as pd
    import panel as pn
    from lumen.ai.report import Action, Report, Section
    from lumen.ai.views import LumenOutput
    from lumen.pipeline import Pipeline
    from lumen.sources.duckdb import DuckDBSource
    from lumen.views.base import VegaLiteView

    pn.extension("vega")

    class ProcessDataAction(Action):
        async def _execute(self, context, **kwargs):
            source = DuckDBSource(tables={"data": "data/windturbines.parquet"})
            avg_source = source.create_sql_expr_source(tables={"avg_data": "SELECT t_state, AVG(p_cap) as avg_p_cap FROM data GROUP BY t_state"})
            pipeline = Pipeline(source=avg_source, table="avg_data")

            view = VegaLiteView(pipeline=pipeline, spec={
                "$schema": "https://vega.github.io/schema/vega-lite/v5.json",
                "mark": "bar",
                "encoding": {
                    "x": {"field": "t_state", "type": "nominal"},
                    "y": {"field": "avg_p_cap", "type": "quantitative"}
                },
            })
            output = LumenOutput(component=view, title="Data Overview")

            df = pipeline.data
            summary = f"Loaded {len(df)} rows, {len(df.columns)} columns"
            return [df, output], {"pipeline": pipeline, "summary": summary}


    report = Report(Section(ProcessDataAction(), title="Average Power Capacity by State"))
    report.show()
    ```

### Built-in actions

Lumen provides `SQLQuery`, a ready-to-use action that executes SQL and renders results:

```python
from lumen.ai.actions import SQLQuery
from lumen.sources.duckdb import DuckDBSource

source = DuckDBSource(uri="data.db")

section = Section(
    SQLQuery(
        source=source,
        table="sales_summary",
        sql_expr="SELECT region, SUM(amount) as total FROM sales GROUP BY region",
        title="Sales by Region"
    ),
    title="Analysis"
)
```

**Parameters:** `source`, `table` (required), `sql_expr`, `generate_caption` (default True), `user_content` (caption instructions).

`SQLQuery` outputs `source`, `pipeline`, `data`, `metaset`, and `table` to context for downstream tasks.

## ActorTask

Wrap any Agent as a task using `ActorTask`:

```python
from lumen.ai.report import ActorTask, Report, Section
from lumen.ai.agents import SQLAgent, ChatAgent

section = Section(
    ActorTask(SQLAgent(), title="Query Data", instruction="Get sales by region"),
    ActorTask(ChatAgent(), title="Analyze", instruction="Explain the trends"),
    title="Sales Analysis"
)
```

**Parameters:** `actor` (the Agent instance), `title`, `instruction`, `history`, `context`.

Use `ActorTask` when you want LLM capabilities (SQL generation, chart creation). Use `Action` for custom Python logic that doesn't need an LLM.

??? example "Complete ActorTask example with SQLAgent"
    Set up an SQLAgent with the required context (source, sources list, and metaset):

    ```python
    from lumen.ai.report import ActorTask, Report, Section
    from lumen.ai.agents import SQLAgent
    from lumen.sources.duckdb import DuckDBSource
    from lumen.ai.schemas import get_metaset
    from lumen.ai.llm import OpenAI

    llm = OpenAI()
    source = DuckDBSource(tables={"turbines": "data/windturbines.parquet"})
    metaset = await get_metaset([source], tables=source.get_tables())

    section = Section(
        ActorTask(
            SQLAgent(llm=llm),
            title="Query Data",
            instruction="Get average turbines pcap",
            context={"source": source, "sources": [source], "metaset": metaset}
        ),
        title="Sales Analysis",
    )
    report = Report(section)
    await report.execute()
    report.servable()
    ```

    The `metaset` provides table metadata that the SQLAgent uses to understand the schema and generate correct SQL queries.

## Context

Tasks communicate through a shared context dictionary. Each task reads values and adds new values for downstream tasks.

### Declaring dependencies

Declare dependencies with schemas:

```python
from lumen.ai.context import ContextModel
from typing import NotRequired

class AnalysisInputs(ContextModel):
    pipeline: object  # Required
    sql: NotRequired[str]  # Optional

class AnalysisOutputs(ContextModel):
    insights: str
    metrics: dict

class AnalysisAction(Action):
    input_schema = AnalysisInputs
    output_schema = AnalysisOutputs
    
    async def _execute(self, context, **kwargs):
        pipeline = context["pipeline"]
        sql = context.get("sql", "")  # Use .get() for optional keys
        return [], {"insights": "...", "metrics": {...}}
```

Tasks with unsatisfied required inputs won't execute until a previous task provides them.

### Accumulating values

When multiple tasks provide the same key, last writer wins. To accumulate values into a list:

```python
from typing import Annotated

class MyInputs(ContextModel):
    all_tables: Annotated[list[str], ("accumulate", "table")]
```

## Invalidation

When a task's output changes, downstream tasks automatically re-run:

```python
await report.execute()
report[0][0].out_context = {"data": new_data}  # TaskB and TaskC re-run
```

Manually invalidate tasks that depend on specific keys:

```python
invalidated_tasks, affected_keys = report.invalidate(["data"])
await report.execute()
```

## Error handling

Reports abort on first error by default. Set `abort_on_error=False` to continue:

```python
section = Section(
    RiskyAction(),
    SafeAction(),
    title="Processing",
    abort_on_error=False
)
```

Check status after execution:

```python
await report.execute()
for section in report:
    for task in section:
        print(f"{task.title}: {task.status}")  # idle, running, success, error
```

## Export

Convert an executed report to a Jupyter notebook:

```python
await report.execute()
notebook_json = report.to_notebook()

with open("report.ipynb", "w") as f:
    f.write(notebook_json)
```

In the UI, click the **Export** button. The report must be executed first.

## Validation

Check that all task dependencies can be satisfied before running:

```python
from lumen.ai.context import ContextError

try:
    issues, output_types = report.validate(context={"source": my_source})
except ContextError as e:
    print(f"Validation failed: {e}")
```

## Modifying reports

```python
section.append(LoadAction())
section.insert(0, ValidationAction())
section.remove(task_to_remove)

report1.merge(report2)  # Adds report2's sections to report1
```

## Lifecycle methods

| Method | When called | Purpose |
|--------|-------------|---------|
| `prepare(context)` | Before first execution | Async setup |
| `execute(context)` | Each run | Entry point |
| `_execute(context)` | Each run | Your implementation |
| `reset()` | On clear | Clear outputs |
| `cleanup()` | On removal | Final cleanup |

```python
class MyAction(Action):
    async def prepare(self, context):
        self.model = load_model()
        await super().prepare(context)
    
    def cleanup(self):
        super().cleanup()
        self.model = None
```

## Troubleshooting

**"Task has unmet requirements"** — The task's `input_schema` requires a key no previous task provides. Add a task that provides it, or mark the field as `NotRequired`.

**Context not updating** — `_execute` must return a tuple: `return [output], {"key": value}`, not just `return [output]`.

**Notebook export fails** — Call `execute()` before `to_notebook()`.
