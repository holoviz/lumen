# Custom Data Sources

```{admonition} What does this guide solve?
---
class: important
---
This guide shows you how to configure custom data sources for Lumen AI.
```

## Overview

We will be using a local LLM to understand how to load custom data sources in Lumen. You do not need
to use a local LLM, and can instead opt for using one you have an API key for, ensure your API key
is in the environment of the terminal you run your commands in.

## Local and remote files using the command line

You can download the standard penguins data set
[here](https://datasets.holoviz.org/penguins/v1/penguins.csv). To start Lumen AI, run the following
command (replacing the path where you downloaded the data to).

```bash
lumen-ai serve penguins.csv --provider llama-cpp --show
```

If instead you do not want to download data, you can tell Lumen where the data is on the web, and
start a chat.

```bash
lumen-ai serve "https://datasets.holoviz.org/penguins/v1/penguins.csv" --provider llama-cpp --show
```

## Local and remote files using a Panel app

Download the [earthquakes](https://earthquake.usgs.gov/fdsnws/event/1/query?format=csv) dataset and
make a note of where it is on your system. Create a file called `app.py` and update the path to
where the earthquakes data was downloaded to. We will use both a local file, and a remote file with
the app.

```python
# app.py
import lumen.ai as lmai
import panel as pn

pn.extension("vega")
llm = lmai.llm.LlamaCpp()

lmai.ExplorerUI(
    data=[
        "/LOCAL/PATH/TO/earthquakes.csv",
        "https://datasets.holoviz.org/penguins/v1/penguins.csv",
    ],
    llm=llm,
    agents=[lmai.agents.SQLAgent, lmai.agents.VegaLiteAgent],
).servable()
```

Run the `app.py` file with the following command.

```bash
panel serve app.py --show
```

## Custom data sources

You can also create apps using custom Lumen sources. Below are examples connecting to different
sources.

### Snowflake

```python
import lumen.ai as lmai
from lumen.sources.snowflake import SnowflakeSource

source = SnowflakeSource(
    account="...",
    authenticator="externalbrowser",
    database="...",
    user="...",
)
lmai.ExplorerUI(source).servable()
```

### DuckDB

One thing to note about using DuckDB as a custom data source is that if your parquet files have a
non-standard extension name _e.g._ `.parq`, then you need to wrap the path to those files with the
directive `read_parquet(...)`. DuckDB has many native ways for reading parquet files, see
[https://duckdb.org/docs/data/parquet/overview.html](https://duckdb.org/docs/data/parquet/overview.html)
for an overview of the methods available to you, and which ones you will need to use the directive
for when using Lumen AI.

```python
import lumen.ai as lmai
from lumen.sources.duckdb import DuckDBSource

# Use a list
tables = ["path/to/parquet/dataset/file.parquet", "read_parquet('file.parq')", "penguins.csv"]
# Use a dictionary
#tables = {
#    "penguins": "path/to/penguins.csv",
#    "earthquakes": "read_parquet('path/to/earthquakes.parq')",
#}

source = DuckDBSource(tables=tables)
lmai.ExplorerUI(source).servable()
```

## No local or remote data files

The Panel apps and terminal commands above use local paths or URIs to hosted data files, however,
you are not required to specify data files when starting Lumen AI. You can run the following command
in your terminal.

```bash
lumen-ai serve --provider llama-cpp --show
```

Lumen AI will start up in your default browser with no data available. You can use the `Drag & Drop`
area to upload your local data files, or select the `Browse` link to open a file dialog where you
can select the data you wish to upload to Lumen AI.
