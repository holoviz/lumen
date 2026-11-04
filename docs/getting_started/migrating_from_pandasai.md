# :material-swap-horizontal: Migrating from PandasAI

If you ask your data questions with [PandasAI](https://docs.pandas-ai.com/v3/getting-started), this page shows the same steps in Lumen. The PandasAI side follows its v3 documentation (`pandasai` 3.0.0, which its docs mark as beta); the Lumen side was run on hvsampledata's penguins, with `customers` and `orders` standing for your own frames.

## Install

=== "PandasAI"

    ``` bash
    pip install pandasai pandasai-litellm
    ```

=== "Lumen"

    ``` bash
    pip install "lumen[ai-openai]" hvsampledata
    ```

Lumen also runs keyless: `Kilo()` serves free models without an account, and a signed-in coding CLI such as Claude Code works too. See [LLM Providers](../configuration/llm_providers.md).

## Load data and pick the LLM

=== "PandasAI"

    ``` py
    import pandasai as pai
    from pandasai_litellm.litellm import LiteLLM

    pai.config.set({"llm": LiteLLM(model="gpt-4.1-mini", api_key="...")})
    df = pai.read_csv("penguins.csv")
    ```

=== "Lumen"

    ``` py
    import hvsampledata
    import lumen
    from lumen.ai import configure
    from lumen.ai.llm import Kilo

    configure(llm=Kilo())
    source = lumen.read({"penguins": hvsampledata.penguins("pandas")})
    ```

`lumen.read` takes file paths and URLs, pandas, polars and GeoPandas DataFrames, pyarrow Tables, xarray Datasets and DuckDB connections. Files stay on disk and are queried in place.

## Ask a question

=== "PandasAI"

    ``` py
    response = df.chat("Which species is heaviest on average?")
    print(response)
    ```

=== "Lumen"

    ``` py
    answer = source.ask("Which species is heaviest on average?")
    answer
    ```

In a notebook the answer renders as a card with the text, a chart when one helps and the SQL. `answer.value` is the text.

## Several tables

=== "PandasAI"

    ``` py
    response = pai.chat("Who are the top customers and what do they buy?", customers, orders)
    ```

=== "Lumen"

    ``` py
    shop = lumen.read({"customers": customers, "orders": orders})
    answer = shop.ask("Who are the top customers and what do they buy?")
    ```

Every table in one `lumen.read` dict lands in one DuckDB database, so a question can join them.

## Follow-ups

=== "PandasAI"

    ``` py
    from pandasai import Agent

    agent = Agent([df])
    agent.chat("Which species is heaviest on average?")
    agent.follow_up("And on each island?")
    ```

=== "Lumen"

    ``` py
    chat = source.chat()
    chat.ask("Which species is heaviest on average?")
    chat.ask("And on each island?")
    ```

## What you get back

| | PandasAI | Lumen |
|---|---|---|
| The answer | `response.value` | `answer.value` |
| What ran | `response.last_code_executed`, the generated Python | `answer.sql`, the query; the card's Spec section for the chart |
| The data | the response when it is a DataFrame | `answer.data`, in the type you asked with (pandas, polars, pyarrow, GeoDataFrame, xarray) |
| Rerun without an LLM | | `answer.code`, a Lumen script |
| Change the chart | ask again | `answer.refine("...")`, or edit the spec and Redraw |

## Running generated code

PandasAI [executes the Python code](https://docs.pandas-ai.com/v3/privacy-security) the LLM writes, and offers an optional Docker sandbox (`pandasai-docker`) to isolate it.

Lumen writes SQL and declarative chart specs, shows both, and runs no Python written by the LLM unless you turn code execution on. The query runs in DuckDB, or in your own database, on your machine.

## Settings that have no PandasAI counterpart here

``` py
configure(
    share="schema",  # (1)!
    mask_pii=True,   # (2)!
    cache=True,      # (3)!
)
answer.usage         # (4)!
```

1. Keep data values out of every prompt: the LLM sees names, types and counts only.
2. Withhold the values of columns that look like personal data.
3. Asking the same question of the same data again returns without LLM calls.
4. The tokens and, when priced, the cost of the answer's LLM calls.

`source.describe(table, columns={...})` tells the LLM what columns mean, and `source.ask(..., confirm=True)` shows the plan before anything runs. See [Lumen in notebooks](lumen_in_notebooks.md).

## What does not carry over

- **Analysis that needs Python.** PandasAI can answer with any pandas code. Lumen answers with SQL and charts, so a question SQL cannot express needs code execution turned on, or `answer.data` and your own pandas.
- **Charts as images.** PandasAI returns charts as images you can `save()`. Lumen draws interactive Vega-Lite or hvPlot charts; the static output carries a PNG.
- **PandasAI's semantic layer.** Its dataset definitions have no direct equivalent; table and column descriptions in Lumen come from `source.describe` or a source's `metadata`.
