# :material-notebook: Lumen in notebooks

Ask your data questions from a notebook cell and get back an answer, the SQL that produced it and a chart.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/holoviz/lumen/blob/main/examples/notebooks/ask_in_a_notebook.ipynb)

## Load data and ask

``` py title="Ask a question"
import hvsampledata
import lumen

from lumen.ai.llm import Kilo

source = lumen.read({"penguins": hvsampledata.penguins("pandas")})
answer = source.ask("Which species is heaviest on average?", llm=Kilo())  # (1)!
answer
```

1. `Kilo()` serves free models without an account. With an API key such as `OPENAI_API_KEY` set, leave out `llm=` and Lumen uses that provider.

`lumen.read` takes file paths and URLs, pandas, polars and GeoPandas DataFrames, pyarrow Tables, xarray Datasets, DuckDB connections, Sources and Pipelines. A dict names the tables, and everything in one dict lands in one source so a question can join them. See [Data Sources](../configuration/sources.md#in-memory-data).

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

- `answer.code` is Lumen code that rebuilds the result without an LLM. In-memory data is embedded, so keep it for small results.
- `answer.export("my_answer")` writes `my_answer/answer.py` and the data as Parquet files under `my_answer/data/`, so large answers stay small and portable.
- `answer.explore()` opens the chat UI on the same source and LLM, with the answer as its first exploration.

## Where answers render

<!-- filled in from the front-end checks -->

## What Lumen runs

Lumen only writes SQL and declarative chart specs, and shows both to you. It never runs Python written by the LLM unless you turn code execution on, so asking a question cannot execute arbitrary code on your machine.
