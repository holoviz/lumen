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

`lumen.read` takes file paths and URLs, pandas, polars and GeoPandas DataFrames, pyarrow Tables, xarray Datasets, DuckDB connections, Sources and Pipelines. A dict names the tables, and everything in one dict lands in one source so a question can join them. See [Data Sources](../configuration/sources.md#in-memory-data).

`pipeline.ask(...)` limits the question to one table and its filters, and `await source.aask(...)` is the async form.

## The answer

An `Answer` renders as a card with the text, the chart, and collapsed SQL and Spec sections: the query, and the Vega-Lite or hvPlot spec the chart was drawn from. It also holds:

| Field | What it is |
|-------|------------|
| `value` | The written answer |
| `sql` | The query that produced the data |
| `pipeline` | A Pipeline over the result, e.g. `answer.pipeline.data` |
| `view` / `spec` | The chart as a Lumen View and its Vega-Lite spec |
| `plan` / `status` | The steps the planner ran and how they ended |
| `error` | Why the question failed, when `status` is `"error"` |

While a question runs, one line under the cell shows the step it is on. Interrupt the kernel to stop it; Lumen cancels the question and its LLM calls.

## Follow up

``` py
chat = source.chat(llm=Kilo())
chat.ask("What is the average body mass per species?")
chat.ask("Only keep the species above 4000 g")  # (1)!
```

1. Each follow-up sees the earlier questions and runs on the data the last answer left.

## Change the chart

``` py
answer = source.ask("Plot the average body mass per species as a bar chart", llm=Kilo())
answer.refine("Color the bars by species and sort them by mass")  # (1)!
```

1. `refine` changes only the chart, with one LLM call and no new SQL. It returns a new answer and leaves the old one as it was.

## When a question fails

A question that cannot be answered still returns an `Answer`, with `status` set to `"error"`. Its card shows the error and the failed SQL, open. Ask again with a hint:

``` py
answer = source.ask("Average flipper length by island in cm?", llm=Kilo())
if answer.status == "error":
    answer = answer.retry(hint="Flipper length is in mm, divide by 10")
```

`retry()` without a hint asks the same question again. In a script, check `answer.status` before using the result.

## Rerun, export, explore

- `answer.code` is Lumen code that rebuilds the result without an LLM. In-memory data is embedded, so keep it for small results.
- `answer.to_cell()` puts that code into a new cell below, in Jupyter.
- `answer.export("my_answer")` writes `my_answer/answer.py` and the data as Parquet files under `my_answer/data/`, so large answers stay small and portable.
- `answer.explore()` opens the chat UI on the same source and LLM, with the answer as its first exploration.

## Spatial and gridded questions

GeoDataFrames and GeoParquet files load with their geometry and CRS, and xarray Datasets load as grids:

``` py
import geopandas as gpd
import xarray as xr

source = lumen.read({"places": gpd.read_file("places.geojson"), "regions": gpd.read_file("regions.geojson")})
source.ask("How many places are within 750 km of Berlin?")

grid = lumen.read(xr.open_dataset("era5_t2m.nc"))
grid.ask("Which month had the highest area-weighted global mean temperature?")
```

When a table has a geometry column or is a grid, Lumen gives the model extra rules for it: distances and areas in metres on a sphere or an equal-area CRS rather than in degrees, longitude before latitude, a spatial predicate in every join; filters on the grid's dimensions, latitude weights for means and Kelvin to Celsius.

## Settings

`lumen.ai.configure` sets the defaults for every question in the session. A question's own `llm=` or `agents=` still wins.

``` py
from lumen.ai import configure
from lumen.ai.llm import Kilo

configure(llm=Kilo(), max_rows=100_000, share="schema", mask_pii=True)
```

| Setting | Environment variable | What it does |
|---------|----------------------|--------------|
| `llm` | `LUMEN_AI_PROVIDER`, `LUMEN_AI_MODEL` | The LLM, e.g. `LUMEN_AI_PROVIDER=kilo`. Without either, the first provider with an API key |
| `agents` | | The agents the planner may use |
| `max_rows` | | Rows an answer's data holds, as a LIMIT on its query (1,000,000 by default, `None` for all). The card notes when a result hits it |
| `share` | `LUMEN_AI_SHARE` | What of the data prompts contain, see below |
| `mask_pii` | `LUMEN_AI_MASK_PII` | Withhold the values of columns that look like personal data |

Files, globs and URLs stay on disk and are queried in place, so `lumen.read({"sales": "data/*.parquet"})` does not load the data into memory. Name tables with a dict, as there; a bare path names the table after the whole path.

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

Lumen only writes SQL and declarative chart specs, and shows both to you. It never runs Python written by the LLM unless you turn code execution on, so asking a question cannot execute arbitrary code on your machine.

What the LLM sees of your data is up to `share`:

| `share` | The LLM sees |
|---------|--------------|
| `"sample"` (default) | Values: summaries of results, value counts, ranges, examples and chart images |
| `"schema"` | Table and column names, types, row counts and null fractions, but no values |
| `"none"` | Table and column names only |

The query always runs on the full data on your machine, so tables and charts are the same at every level; with less to go on, the written answers and the SQL the model writes can be plainer. Error messages from the database may still quote a value.
