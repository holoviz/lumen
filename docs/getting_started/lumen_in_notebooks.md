# :material-notebook: Lumen in notebooks

Ask your data questions from a notebook cell and get back an answer, the SQL that produced it and a chart.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/holoviz/lumen/blob/main/examples/notebooks/ask_in_a_notebook.ipynb)

## Load data and ask

``` bash
pip install "lumen[ai-openai]" hvsampledata
```

``` py title="Ask a question"
import hvsampledata
import lumen

from lumen.ai.llm import Kilo

source = lumen.read({"penguins": hvsampledata.penguins("pandas")})
answer = source.ask("Which species is heaviest on average?", llm=Kilo())  # (1)!
answer
```

1. `Kilo()` serves free models without an account. With an API key such as `OPENAI_API_KEY` set, leave out `llm=` and Lumen uses that provider.

`lumen.read` takes file paths and URLs, pandas, polars and GeoPandas DataFrames, pyarrow Tables, xarray Datasets, DuckDB connections, Sources and Pipelines. A dict names the tables, and everything in one dict except a ready-made Source lands in one source, so a question can join them. See [Data Sources](../configuration/sources.md#in-memory-data).

`pipeline.ask(...)` limits the question to one table and its filters, and `await source.aask(...)` is the async form.

## The answer

An `Answer` renders as a card with the text, the chart and a collapsed SQL section. It also holds:

| Field | What it is |
|-------|------------|
| `value` | The written answer |
| `sql` | The query that produced the data |
| `pipeline` | A Pipeline over the result, e.g. `answer.pipeline.data` |
| `view` / `spec` | The chart as a Lumen View and its Vega-Lite spec |
| `plan` / `status` | The steps the planner ran and how they ended |

## Follow up

``` py
chat = source.chat(llm=Kilo())
chat.ask("What is the average body mass per species?")
chat.ask("Only keep the species above 4000 g")  # (1)!
```

1. Each follow-up sees the earlier questions and runs on the data the last answer left.

## Rerun, export, explore

- `answer.code` is Lumen code that rebuilds the result without an LLM. In-memory data is embedded, so keep it for small results. Source settings go in as they are, so check it for credentials (a database URI with a password, say) before sharing it.
- `answer.export("my_answer")` writes `my_answer/answer.py` and the data as Parquet files under `my_answer/data/`, so large answers stay small and portable.
- `answer.explore()` opens the chat UI on the same source and LLM, with the answer as its first exploration.

## Where answers render

| Front end | What you get |
|-----------|--------------|
| JupyterLab, Notebook 7 | The interactive card: answer, chart and an expandable SQL section |
| Google Colab | The interactive card. Panel warns that `jupyter_bokeh` is missing; the card still works without it. With panel-material-ui 0.16.7 or earlier the card stays light in Colab's dark theme |
| VS Code | The interactive card once `jupyter_bokeh` is installed in the kernel's environment. Without it, Panel prints an install hint and you get the answer, the Vega-Lite chart and the SQL as plain outputs |
| marimo | A static card with the answer, a PNG and the SQL. Pick the Vega-Lite output in the cell's output menu for the interactive chart |
| Python REPL or script | Echoing `answer` shows its status and first line; `answer.value` and `answer.sql` hold the full text |

For VS Code, install the widget bridge into the same environment as the kernel:

``` bash
pip install jupyter_bokeh
```

## What Lumen runs

Lumen only writes SQL and declarative chart specs, and shows both to you. It never runs Python written by the LLM unless you turn code execution on. The SQL runs in DuckDB, which can read files on your machine (`read_text('/...')`, say), and query results go to the LLM.

## What goes to the LLM provider

`ask` uses the first provider whose API key is set in the environment, unless you pass `llm=`. That provider receives your tables' schemas, column statistics and sample rows, and a sample of each query result, so only ask about data you may send there.
