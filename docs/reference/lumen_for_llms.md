# Lumen for LLMs

One page for coding agents and LLMs writing Python with Lumen: the smallest working example, the API at a glance, and the errors people hit with their fixes. The full guide is [Lumen in notebooks](../getting_started/lumen_in_notebooks.md).

## Smallest working example

``` bash
pip install lumen
```

``` py
import lumen
from lumen.ai.llm import Kilo  # free models, no account; or set OPENAI_API_KEY and leave out llm=

source = lumen.read("sales.csv")  # also URLs, DataFrames, xarray Datasets, DuckDB connections
answer = source.ask("Which region grew fastest last quarter?", llm=Kilo())

answer.status   # "success", "error" or "unanswerable"
answer.value    # the written answer
answer.sql      # the query that produced the data
answer.data     # the data, in the type it was read as (pandas, polars, GeoDataFrame, xarray)
answer.chart    # an Altair or HoloViews object to keep working with
```

## API at a glance

**Load data**

- `lumen.read(data)` returns a Source. `data` is a path, URL, DataFrame, pyarrow Table, xarray Dataset, GeoDataFrame or DuckDB connection, or a list of them, or a dict that names the tables and puts them in one source so a question can join them.
- `source.describe(table, description, columns={...})` tells the LLM what a table and its columns mean.
- `source.overview()` shows each table's columns, data problems and five questions to ask.

**Ask**

- `source.ask(question, llm=None, confirm=False, returns=None)` returns an Answer. `await source.aask(...)` is the async form, and `pipeline.ask(...)` limits the question to one table and its filters.
- `source.ask_many([q1, q2, ...], concurrency=4)` asks independent questions at once.
- `chat = source.chat()`, then `chat.ask(...)`, asks follow-ups that build on earlier answers.
- `returns=` takes a type or a Pydantic model and fills `answer.result` with a typed value.
- `confirm=True` returns a planned answer; `answer.run()` runs it.

**Use the answer**

- Text and query: `value`, `sql`, `status`, `error`, `why`, `question`.
- Data: `data`, `gdf` (GeoDataFrame or None), `ds` (xarray Dataset or None), `truncated_at` (the row cap, when the data was cut).
- Chart: `chart`, `spec` (Vega-Lite), `view`, `map()`, `tweak()`, `to_vega()`, `with_spec(spec)`.
- Follow up: `refine(instruction)` changes the chart, `retry(hint=None)` asks again, and `ask_about(question, rows=None)` asks about selected rows.
- Keep it: `code` (Lumen code that rebuilds it with no LLM), `to_cell()`, `to_file(path)`, `export(directory)`, `explore()` (opens the chat app).
- Cost: `usage`, e.g. `answer.usage.input_tokens`.

**Settings**

- `lumen.ai.configure(llm=..., agents=..., max_rows=1_000_000, share="sample", mask_pii=False, cache=False)`.
- `share` sets what prompts may contain: `"sample"` (row values), `"schema"` (structure only) or `"none"` (names and row counts).
- `mask_pii=True` withholds columns that look like personal data.

**Apps**

- `lumen-ai serve data.csv` (also `lumen serve`) opens the chat app on the same agents.
- A YAML spec served with `lumen serve app.yaml` builds a dashboard, a separate and older API.

## Errors and fixes

| Error or symptom | Fix |
|---|---|
| `RuntimeError: No LLM provider found` | Pass `llm=lumen.ai.llm.Kilo()`, set a provider key such as `OPENAI_API_KEY` and restart the kernel, or call `lumen.ai.configure(llm=...)` |
| An `ImportError` for a provider's client, e.g. `anthropic` | Install its extra: `pip install 'lumen[ai-anthropic]'`, `lumen[ai-google]`, `lumen[ai-mistralai]`. OpenAI-compatible providers (Ollama, Groq, OpenRouter, Kilo) need none |
| Reading NetCDF, Zarr or an xarray Dataset asks for `lumen[xarray]` | `pip install 'lumen[xarray]'`; maps and geometry also want `lumen[geo]` |
| `answer.status == "unanswerable"` | The data does not hold what was asked; `answer.why` says what is missing |
| `answer.status == "error"` | `answer.error` says why; `answer.retry(hint="...")` asks again with a hint |
| `answer.truncated_at` is set | The result passed the row cap; aggregate in the question, or `lumen.ai.configure(max_rows=None)` |
| `Asking about rows sends them to the LLM, which share='schema' does not allow` | `lumen.ai.configure(share="sample")`, if those values may go to the provider |
| A question about two tables cannot join them | Read them together, `lumen.read({"orders": a, "customers": b})`, so they share one source |
