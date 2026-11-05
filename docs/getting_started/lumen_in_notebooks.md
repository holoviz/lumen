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

## Look at the data first

``` py
source.overview()
```

The overview lists each table's columns with their types, nulls, distinct counts and ranges, flags what looks wrong in the first 5,000 rows (missing values, duplicate rows, padded text, numbers stored as text, constant columns, outliers) and suggests five questions. In a notebook each question is a button that asks it and shows the answer below. Writing the questions is one LLM call, sent only what `share` allows; `source.overview(questions=False)` skips it. For a large table the distinct counts and ranges come from a sample, and the overview says so.

## The answer

An `Answer` renders as a card with the text, the chart, and collapsed SQL, Spec and Steps sections: the query, the Vega-Lite or hvPlot spec the chart was drawn from, and each step with how long it took. It also holds:

| Field | What it is |
|-------|------------|
| `value` | The written answer |
| `sql` | The query that produced the data |
| `data` | The result in the type you asked with: polars for polars input, pyarrow for pyarrow, a GeoDataFrame or xarray Dataset for GeoPandas or xarray, pandas otherwise |
| `pipeline` | A Pipeline over the result, e.g. `answer.pipeline.data` |
| `chart` | The chart to keep working with: an Altair chart for Vega-Lite (needs `altair`), the HoloViews object for hvPlot, so `answer.chart + other` and `.opts(...)` work |
| `to_vega()` | The chart as a self-contained Vega-Lite spec with its data |
| `view` / `spec` | The chart as a Lumen View and its Vega-Lite spec |
| `plan` / `status` | The steps the planner ran and how they ended: `"success"`, `"error"` or `"unanswerable"` |
| `error` | Why the question failed or cannot be answered |
| `usage` | Tokens and, when the provider prices them, the cost of the answer's LLM calls, e.g. `answer.usage.input_tokens`; the card ends with a one-line summary |
| `result` | The answer in the type asked for with `returns=`, see below |
| `why` / `trace` | The planner's reasoning, and each step as it ran: actor, instruction, what it needed and provided, seconds, LLM calls and its SQL. `answer.plan.graph()` and `answer.plan.to_mermaid()` give how the steps feed each other |

While a question runs, one line under the cell shows the step it is on. Interrupt the kernel to stop it; Lumen cancels the question and its LLM calls.

## Follow up

``` py
chat = source.chat(llm=Kilo())
chat.ask("What is the average body mass per species?")
chat.ask("Only keep the species above 4000 g")  # (1)!
```

1. Each follow-up sees the earlier questions and runs on the data the last answer left.

Displaying `chat` in a notebook shows it as a chat widget, starting with the answers so far. A question typed into it is asked with `chat.ask`, so the widget and your code share one conversation.

To keep a conversation:

| Method | What you get |
|--------|--------------|
| `chat.to_html("chat.html")` | One page: each question over its answer, the chart as an image, the SQL and the spec |
| `chat.to_notebook("chat.ipynb")` | Each question and answer, then Lumen code that rebuilds its chart or data with no LLM |
| `chat.to_report("Penguins")` | A Lumen Report of the answers and their charts, with its own `to_html`, `to_notebook` and `to_docx` |

## Check the data

`source.check` takes a rule the data should keep and returns the rows that break it:

``` py
source.check("SELECT * FROM penguins WHERE body_mass_g <= 0")  # (1)!
source.check("no penguin weighs under 2 kg")  # (2)!
```

1. A SELECT returning the offending rows runs as written, with no LLM.
2. A rule in plain language costs one LLM call, which writes that SELECT; `result.sql` holds it.

The result is falsy when rows break the rule, so `assert source.check(...)` works in pytest and its failure message shows the rule, the query, the count and the first rows. `result.rows` holds up to 1,000 of them in the type the data was read in, and `result.raise_()` raises the same message outside pytest. Only read-only queries run. To keep a test free of LLM calls, paste the `result.sql` of a plain-language rule into the test.

## Test code that asks

`lumen.ai.testing` records a real LLM's answers once and replays them, so tests of code built on `ask` run offline, fast and the same every time:

``` py
from lumen.ai.testing import cassette

def test_penguins_in_2008():
    with cassette("tests/cassettes/penguins_2008.json") as llm:
        answer = source.ask("How many penguins were measured in 2008?", llm=llm, returns=int)
    assert answer.result == 114
```

`cassette` replays by default. Run the tests once with `LUMEN_AI_RECORD=1` to record with your LLM (or pass one, `cassette(path, llm)`), and commit the JSON file. A call that was not recorded, e.g. after a prompt changed, raises `MissingRecording` naming the closest recorded one; record again. `record(llm, path)` and `ReplayLLM(path)` are the two halves, for use outside a `with` block. Calls the LLM makes to tools are replayed as their final answer, without running the tools again.

## Ask about a point

Click a point, bar or shape on an answer's chart, or drag a box over several on an hvPlot chart or map, and a box under the chart asks about it: "what is this?", "why is it so high?", "show its history". On a grid or a density image, a click picks the nearest cell or point. The clicked rows are in `answer.selection`, and the same works from code:

``` py
answer.ask_about("Why is this one so high?")             # (1)!
answer.ask_about("Show its history", rows={"island": "Dream"})
```

1. Asks about `answer.selection`; `rows=` takes a DataFrame, a dict, a list of dicts or row positions instead.

The LLM gets the question, the rows (up to 20, with personal-looking columns withheld under `mask_pii`) and the answer before, and answers from the rows when they hold what is asked, querying the data filtered on them otherwise. In a chat, or in the chat UI's explorations, the follow-up joins the conversation. It sends row values, so it needs `share="sample"`. Static outputs (PNG, a saved notebook) have no box.

## Maps over time

A map of a grid with several time steps, such as `air_temperature`'s 20, gets a slider over time instead of drawing every step on top of the others; the card notes the slider, which a saved notebook or PNG cannot show. A grid is rasterized by the cells of one step, not all of them.

## Large results

A scatter or line chart of more than 5,000 points is drawn with hvPlot and datashader as a density image, which stays fast however many rows there are, instead of a Vega-Lite chart with one mark per row; a chart coloured by a category blends the categories. The card says so. Without datashader installed the chart stays Vega-Lite and the card suggests installing it.

## Change the chart

``` py
answer = source.ask("Plot the average body mass per species as a bar chart", llm=Kilo())
answer.refine("Color the bars by species and sort them by mass")  # (1)!
```

1. `refine` changes only the chart, with one LLM call and no new SQL. It returns a new answer and leaves the old one as it was.

To change the chart by hand, open the card's Spec section, press Edit, change the spec and press Redraw. `answer.with_spec(...)` does the same from code, taking the spec as a dict or as JSON or YAML text, and needs no LLM.

## When a question fails

A question that cannot be answered still returns an `Answer`, with `status` set to `"error"`. Its card shows the error and the failed SQL, open. Ask again with a hint:

``` py
answer = source.ask("Average flipper length by island in cm?", llm=Kilo())
if answer.status == "error":
    answer = answer.retry(hint="Flipper length is in mm, divide by 10")
```

`retry()` without a hint asks the same question again. In a script, check `answer.status` before using the result.

When the data does not hold what a question needs, such as a column or a year it lacks, the answer comes back with `status` set to `"unanswerable"` and `error` saying why, without running a query:

``` py
answer = source.ask("What is the average beak colour of each species?", llm=Kilo())
answer.status  # "unanswerable"
answer.error   # e.g. "No table records beak colour."
```

`answer.data` follows the input only for data read with `lumen.read`; for a Source built by hand it is pandas. Claude Code reports the cost of its calls at API prices, which is not what a subscription is billed.

## Typed answers

Ask with `returns=` to get the answer as a value rather than prose:

``` py
from pydantic import BaseModel

count = source.ask("How many penguins were measured in 2008?", returns=int)
count.result  # 114

class Heaviest(BaseModel):
    species: str
    body_mass_g: float

source.ask("Which species is heaviest on average?", returns=Heaviest).result
```

A result already in the shape of the type is used as it is, with no extra LLM call: one cell for a number or a string, one column for a list, one row for a model. Otherwise one more LLM call reads the type from the written answer, and from the rows when `share="sample"`. It can round: asking for `int` when the data says 38.5 gives 38 or 39. An answer that cannot be read as the type has status `"error"`.

## Rerun, export, explore

- `answer.compile(params=["year"])` turns the query into a function with no LLM: `count = answer.compile(["year"])`, then `count(year=2009)` runs the same query for 2009. Each parameter is a column the query compares with values: one value, a `(low, high)` pair for `BETWEEN`, or a list for `IN`. Values are bound as query parameters, never written into the SQL. It works on DuckDB sources, which `lumen.read` gives.
- `answer.code` is Lumen code that rebuilds the result without an LLM. In-memory data is embedded, so keep it for small results.
- `answer.to_cell()` puts that code into a new cell below, in Jupyter.
- `answer.export("my_answer")` writes `my_answer/answer.py` and the data as Parquet files under `my_answer/data/`, so large answers stay small and portable.
- `answer.explore()` opens the chat UI on the same source and LLM, with the answer as its first exploration.

## Workflow

**Describe your data.** The LLM writes better SQL when it knows what tables and columns mean:

``` py
source.describe(
    "penguins", "One row per penguin measured on the Palmer Archipelago, 2007 to 2009",
    columns={"body_mass_g": "Body mass in grams", "sex": "male or female, missing for some birds"},
)
```

Your text replaces any description the database has, and the next question sees it.

**Check the plan first.** `confirm=True` stops once the question is planned. The card lists the steps with a Run button, and `run()` runs exactly those steps without planning again:

``` py
plan = source.ask("Which island has the heaviest penguins?", confirm=True)
plan.status   # "planned"
plan.run()    # the same answer object, now with its result
```

**Cache answers.** With `configure(cache=True)`, asking the same question of the same data and model again returns the earlier answer straight away, marked "From the answer cache". The key covers the question, the data's columns, types, row counts and descriptions, the model, the agents, the sharing settings and Lumen's prompts. Frames are copied when `lumen.read` reads them, so editing a DataFrame afterwards changes no answer; a file or database whose values change while its row count stays the same is not noticed, so call `lumen.ai.clear_cache()` after such a change. Up to 128 answers and 2,000,000 rows of their data are kept, oldest dropped first.

**Ask from a cell.** The `%%lumen` magic takes the source and the question; `{name}` fills in a variable:

``` py
%load_ext lumen.ai.notebook
```

``` py
%%lumen source --as heaviest
Which species was heaviest in {year}?
```

`%lumen source How many rows?` asks in one line, and `--confirm` previews the plan.

## Maps and grids

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/holoviz/lumen/blob/main/examples/notebooks/maps_and_grids.ipynb)

``` bash
pip install "lumen[ai-openai,xarray]" hvsampledata geopandas geoviews lonboard
```

``` py title="Shapes, points and a grid"
import hvsampledata
import lumen

from lumen.ai.llm import Kilo

states = lumen.read({"states": hvsampledata.us_states("geopandas")})
quakes = lumen.read({"quakes": hvsampledata.earthquakes("pandas")})
air = lumen.read({"air": hvsampledata.air_temperature("xarray")})

income = states.ask("Map median household income by state", llm=Kilo())  # (1)!
quake_map = quakes.ask("Map the earthquakes, coloured by magnitude", llm=Kilo())  # (2)!
mean_air = air.ask("Map the mean air temperature in degrees Celsius", llm=Kilo())  # (3)!
```

1. US states are polygons in a GeoDataFrame, so the answer is a choropleth on a basemap.
2. A table with `lat` and `lon` columns is drawn as points on a basemap.
3. The air temperature grid is drawn as a quadmesh over coastlines.

An answer over a table with a geometry column, or over an xarray Dataset, has a few more fields:

| Field | What it is |
|-------|------------|
| `gdf` | The result as a GeoDataFrame with its CRS, or `None` without geometry. Warns once when the geometry looks like longitude and latitude but has no CRS |
| `ds` | The result as an xarray Dataset, for answers over gridded data, or `None` |
| `to_file(path)` | Writes the result by extension: `.parquet` (GeoParquet when there is geometry), `.geojson`, `.gpkg`, `.csv`, or `.zarr` and `.nc` for gridded answers |
| `map()` | The geometry on an interactive map: lonboard when installed, otherwise hvPlot on a basemap |

``` py
income.gdf.head()               # a GeoDataFrame in EPSG:4326
income.to_file("income.parquet")  # GeoParquet, CRS kept
income.map()                    # lonboard, or hvPlot on a basemap
mean_air.ds                     # an xarray Dataset on lat and lon
```

Maps need geoviews: without it they are drawn on plain axes. Large grids are rasterized. Where the card cannot render, such as a saved notebook, a map answer falls back to a static Vega-Lite map.

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
| `cache` | `LUMEN_AI_CACHE` | Keep answers in memory, so the same question of the same data and model returns without LLM calls; `lumen.ai.clear_cache()` empties it |

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
