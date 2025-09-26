# Generating AI-powered Reports

This guide walks you through the basics of creating an interactive, multi-step `Report` powered by your LLM, a SQL source (DuckDB), and a couple of agents/tools for analysis and visualization. You’ll:

- wire up data and memory,
- define a simple plotting function,
- compose a `Report` from `Section`s and `TaskGroup`s

run it and export to a Jupyter notebook.

## Background

The [Lumen AI Explorer](./using_lumen_ai) provides a freeform tool to chat with your data, generate artifacts, such as tables, plots and text, but sometimes you want to declare a well structured report that performs specific SQL queries, generates plots, summarizes the results and more. This is where Lumen's `Report` functionality steps in. Using a small number of primitives such as `Task`s and `TaskGroup`s you can declaratively put together a report that runs pre-defined actions or invokes LLMs to generate artifacts that can be combined into a series of `Section` that make up the overall `Report`.

## Minimal “Hello, Report” (tiny in-memory table)

```python
import lumen as lm
import lumen.ai as lmai
import panel as pn

pn.extension('tabulator')

# 1) LLM (used by optional captioning, agents, etc.)
llm = lmai.llm.OpenAI()

# 2) Tiny DuckDB source (no files!)
source = lm.sources.duckdb.DuckDBSource(tables={
    'tiny': """
        SELECT * FROM (
          VALUES (1,'A',10.5),
                 (2,'B',20.0),
                 (3,'A', 7.5)
        ) AS t(id, category, value)
    """
})

# 3) Seed the shared memory with a default table/pipeline
lmai.memory['source'] = source

# 4) Build a one-section Report with a single SQL step
from lumen.ai.report import Report, Section, SQLQuery

report = Report(
    Section(
        SQLQuery(
            title="By Category",
            table="by_cat",
            sql_expr="""
                SELECT category, COUNT(*) AS n, AVG(value) AS avg_value
                FROM tiny GROUP BY category
            """
        ),
        title="Explore tiny table"
    ),
    llm=llm,
    memory=lmai.memory,
    title="Hello, Report"
)

report
```

**What happens here?**

`SQLQuery` runs the SQL, creates a working table `by_cat`, updates memory (source, pipeline, table, etc.), and emits a renderable output.

The `Report` gives you a toolbar: ▶ Run, Clear, Collapse/Expand, Settings, Export to .ipynb.

## Adding a custom step

```python
from lumen.ai.tools import FunctionTool


def describe(pipeline):
    return pn.pane.DataFrame(pipeline.data.describe())

report = Report(
    Section(
        SQLQuery(
            title="By Category",
            table="by_cat",
            sql_expr="SELECT category, COUNT(*) AS n, AVG(value) AS avg_value FROM tiny GROUP BY category"
        ),
        FunctionTool(decribe, requires=['pipeline']),
        title="Explore tiny table"
    ),
    llm=llm,
    memory=lmai.memory,
    title="Hello, Report (+sample)"
)

report
```

## How to think about composition

- Start with a single `Task` (e.g., `SQLQuery`) to get data ready.
- Wrap simple Python helpers with FunctionTool for quick checks, charts, or tables.
- When you want sequences, use a `TaskGroup` (or just put multiple items into a `Section`).
- Group related steps into a `Section`; put multiple sections into a `Report`.

## Run, serve, export

- **Notebook**: just display report and click ▶ Run.
- **Programmatic**: `await report.execute()` in an async context.
- **App**: `pn.serve(report, show=True)`.
- **Export**: Click Export to .ipynb in the `Report` toolbar or call `.to_notebook()`.
